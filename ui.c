/* Copyright (c) 2014-2015, TAKAHASHI Tomohiro (TTRFTECH) edy555@gmail.com
 * All rights reserved.
 *
 * This is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 3, or (at your option)
 * any later version.
 *
 * The software is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with GNU Radio; see the file COPYING.  If not, write to
 * the Free Software Foundation, Inc., 51 Franklin Street,
 * Boston, MA 02110-1301, USA.
 */

#include "ch.h"
#include "hal.h"
#include "chprintf.h"
#include "nanovna.h"
#include <string.h>
#include <stdlib.h>
#include <math.h>

#pragma GCC push_options
#pragma GCC optimize ("Os")

uistat_t uistat = {
 current_trace: 0,
 lever_mode: LM_MARKER,
 marker_delta: FALSE,
 marker_noise: FALSE,
 marker_tracking : FALSE,
 auto_center_marker : FALSE,
 text : "",
};

#define NO_EVENT                    0
#define EVT_BUTTON_SINGLE_CLICK     0x01
#define EVT_BUTTON_DOUBLE_CLICK     0x02
#define EVT_BUTTON_DOWN_LONG        0x04
#define EVT_UP                  0x10
#define EVT_DOWN                0x20
#define EVT_REPEAT              0x40

#define BUTTON_DOWN_LONG_TICKS      MS2ST(500)   // 500ms
#define BUTTON_DOUBLE_TICKS         MS2ST(250)   // 250ms
#define BUTTON_REPEAT_TICKS         MS2ST( 40)   //  40ms
#define BUTTON_DEBOUNCE_TICKS       MS2ST(  2)   //   2ms

/* lever switch assignment */
#define BIT_UP1     3
#define BIT_PUSH    2
#define BIT_DOWN1   1

#define READ_PORT() palReadPort(GPIOA)
#define BUTTON_MASK 0b1110

static uint16_t last_button = 0b0000;
static uint32_t last_button_down_ticks;
static uint32_t last_button_repeat_ticks;

#define MENU_USE_AUTOHEIGHT
#ifdef MENU_USE_AUTOHEIGHT
static uint16_t menu_button_height = MENU_BUTTON_HEIGHT_N(MENU_BUTTON_MIN);
#endif

volatile uint8_t operation_requested = OP_NONE;

int8_t previous_marker = MARKER_INVALID;

enum {
  UI_NORMAL, UI_MENU, UI_KEYPAD
};

#define NUMINPUT_LEN 12
static uint8_t ui_mode = UI_NORMAL;
static uint8_t keypad_mode;
static char    kp_buf[NUMINPUT_LEN+1];
static int8_t  kp_index = 0;
static char   *kp_help_text = NULL;
static uint8_t menu_current_level = 0;
static int  selection = 0;

static const uint8_t slider_bitmap[]=
{
  _BMP8(0b11111110),
  _BMP8(0b11111110),
  _BMP8(0b01111100),
  _BMP8(0b00111000),
  _BMP8(0b00010000)
};

// Button definition (used in MT_ADV_CALLBACK for custom)
#define BUTTON_ICON_NONE            -1
#define BUTTON_ICON_NOCHECK          0
#define BUTTON_ICON_CHECK            1
#define BUTTON_ICON_CHECK_AUTO       2
#define BUTTON_ICON_CHECK_MANUAL     3
#define BUTTON_ICON_GROUP            4
#define BUTTON_ICON_GROUP_CHECKED    5

#define CHECK_ICON(S) ((S) ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK)
#define GROUP_ICON(S) ((S) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP)
#define AUTO_ICON(S)  (S>=2?BUTTON_ICON_CHECK_AUTO:S)            // Depends on order of ICONs!!!!!

#define BUTTON_BORDER_NONE           0x00
#define BUTTON_BORDER_WIDTH_MASK     0x0F

// Define mask for draw border (if 1 use light color, if 0 dark)
#define BUTTON_BORDER_TYPE_MASK      0xF0
#define BUTTON_BORDER_TOP            0x10
#define BUTTON_BORDER_BOTTOM         0x20
#define BUTTON_BORDER_LEFT           0x40
#define BUTTON_BORDER_RIGHT          0x80

#define BUTTON_BORDER_FLAT           0x00
#define BUTTON_BORDER_RISE           (BUTTON_BORDER_TOP|BUTTON_BORDER_RIGHT)
#define BUTTON_BORDER_FALLING        (BUTTON_BORDER_BOTTOM|BUTTON_BORDER_LEFT)

// Touch screen
#define EVT_TOUCH_NONE     0
#define EVT_TOUCH_DOWN     1
#define EVT_TOUCH_PRESSED  2
#define EVT_TOUCH_RELEASED 3

#define TOUCH_INTERRUPT_ENABLED   1
static uint8_t touch_status_flag = 0;
static int8_t last_touch_status = EVT_TOUCH_NONE;
static int16_t last_touch_x;
static int16_t last_touch_y;

#define KP_CONTINUE 0
#define KP_DONE 1
#define KP_CANCEL 2

static void ui_mode_keypad(int _keypad_mode);
// static void draw_menu(void);
static void leave_ui_mode(void);
static void erase_menu_buttons(void);
static void ui_process_keypad(void);
static void choose_active_marker(void);
static void menu_move_back(bool leave_ui);
//static const menuitem_t menu_marker_type[];

static int btn_check(void)
{
  systime_t ticks;
  // Debounce input
  while(TRUE){
    ticks = chVTGetSystemTimeX();
    if(ticks - last_button_down_ticks > BUTTON_DEBOUNCE_TICKS)
      break;
    chThdSleepMilliseconds(10);
  }
  int status = 0;
  uint16_t cur_button = READ_PORT() & BUTTON_MASK;
  // Detect only changed and pressed buttons
  uint16_t button_set = (last_button ^ cur_button) & cur_button;
  last_button_down_ticks = ticks;
  last_button = cur_button;

  if (button_set & (1<<BIT_PUSH))
    status |= EVT_BUTTON_SINGLE_CLICK;
  if (button_set & (1<<BIT_UP1))
    status |= EVT_UP;
  if (button_set & (1<<BIT_DOWN1))
    status |= EVT_DOWN;
  return status;
}

static int btn_wait_release(void)
{
  while (TRUE) {
    systime_t ticks = chVTGetSystemTimeX();
    systime_t dt = ticks - last_button_down_ticks;
    // Debounce input
//    if (dt < BUTTON_DEBOUNCE_TICKS){
//      chThdSleepMilliseconds(10);
//      continue;
//    }
    chThdSleepMilliseconds(1);
    uint16_t cur_button = READ_PORT() & BUTTON_MASK;
    uint16_t changed = last_button ^ cur_button;
    if (dt >= BUTTON_DOWN_LONG_TICKS && (cur_button & (1<<BIT_PUSH)))
      return EVT_BUTTON_DOWN_LONG;
    else if (changed & (1<<BIT_PUSH)) { // release
      last_button = cur_button;
      last_button_down_ticks = ticks;
      return EVT_BUTTON_SINGLE_CLICK;
    }
    if (changed) {
      // finished
      last_button = cur_button;
      last_button_down_ticks = ticks;
      return 0;
    }

    if (dt > BUTTON_DOWN_LONG_TICKS &&
      ticks > last_button_repeat_ticks) {
      int status = 0;
      if (cur_button & (1<<BIT_DOWN1))
        status |= EVT_DOWN | EVT_REPEAT;
      if (cur_button & (1<<BIT_UP1))
        status |= EVT_UP | EVT_REPEAT;
      last_button_repeat_ticks = ticks + BUTTON_REPEAT_TICKS;
      return status;
    }
  }
}


#define SOFTWARE_TOUCH
//*******************************************************************************
// Software Touch module
//*******************************************************************************
#ifdef SOFTWARE_TOUCH
// ADC read count for measure X and Y (2^N count)
#define TOUCH_X_N 3
#define TOUCH_Y_N 3
static int
touch_measure_y(void)
{
  // drive low to high on X line (At this state after touch_prepare_sense)
//  palSetPadMode(GPIOB, GPIOB_XN, PAL_MODE_OUTPUT_PUSHPULL); //
//  palSetPadMode(GPIOA, GPIOA_XP, PAL_MODE_OUTPUT_PUSHPULL); //
  // drive low to high on X line (coordinates from top to bottom)
  palClearPad(GPIOB, GPIOB_XN);
//  palSetPad(GPIOA, GPIOA_XP);

  // open Y line (At this state after touch_prepare_sense)
//  palSetPadMode(GPIOB, GPIOB_YN, PAL_MODE_INPUT);        // Hi-z mode
  palSetPadMode(GPIOA, GPIOA_YP, PAL_MODE_INPUT_ANALOG);   // <- ADC_TOUCH_Y channel

//  chThdSleepMilliseconds(20);
  uint32_t v = 0, cnt = 1<<TOUCH_Y_N;
  do{v+=adc_single_read(ADC_TOUCH_Y);}while(--cnt);
  return v>>TOUCH_Y_N;
}

static int
touch_measure_x(void)
{
  // drive high to low on Y line (coordinates from left to right)
  palSetPad(GPIOB, GPIOB_YN);
  palClearPad(GPIOA, GPIOA_YP);
  // Set Y line as output
  palSetPadMode(GPIOB, GPIOB_YN, PAL_MODE_OUTPUT_PUSHPULL);
  palSetPadMode(GPIOA, GPIOA_YP, PAL_MODE_OUTPUT_PUSHPULL);
  // Set X line as input
  palSetPadMode(GPIOB, GPIOB_XN, PAL_MODE_INPUT);        // Hi-z mode
  palSetPadMode(GPIOA, GPIOA_XP, PAL_MODE_INPUT_ANALOG); // <- ADC_TOUCH_X channel

  uint32_t v = 0, cnt = 1<<TOUCH_X_N;
  do{v+=adc_single_read(ADC_TOUCH_X);}while(--cnt);
  return v>>TOUCH_X_N;
}
// Manually measure touch event
static inline int
touch_status(void)
{
  return adc_single_read(ADC_TOUCH_Y) > TOUCH_THRESHOLD;
}

static void
touch_prepare_sense(void)
{
  // Set Y line as input
  palSetPadMode(GPIOB, GPIOB_YN, PAL_MODE_INPUT);          // Hi-z mode
  palSetPadMode(GPIOA, GPIOA_YP, PAL_MODE_INPUT_PULLDOWN); // Use pull
  // drive high on X line (for touch sense on Y)
  palSetPad(GPIOB, GPIOB_XN);
  palSetPad(GPIOA, GPIOA_XP);
  // force high X line
  palSetPadMode(GPIOB, GPIOB_XN, PAL_MODE_OUTPUT_PUSHPULL);
  palSetPadMode(GPIOA, GPIOA_XP, PAL_MODE_OUTPUT_PUSHPULL);

//  chThdSleepMilliseconds(10); // Wait 10ms for denounce touch
}

static void
touch_start_watchdog(void)
{
  if (touch_status_flag&TOUCH_INTERRUPT_ENABLED) return;
  touch_status_flag^=TOUCH_INTERRUPT_ENABLED;
  adc_start_analog_watchdog();
#ifdef __REMOTE_DESKTOP__
  remote_mouse_down = 0;
#endif
}
static void
touch_stop_watchdog(void)
{
  if (!(touch_status_flag&TOUCH_INTERRUPT_ENABLED)) return;
  touch_status_flag^=TOUCH_INTERRUPT_ENABLED;
  adc_stop_analog_watchdog();
}

// Touch panel timer check (check press frequency 20Hz)
static const GPTConfig gpt3cfg = {
  20,     // 200Hz timer clock. 200/10 = 20Hz touch check
  NULL,   // Timer callback.
  0x0020, // CR2:MMS=02 to output TRGO
  0
};

//
// Touch init function init timer 3 trigger adc for check touch interrupt, and run measure
//
static void touch_init(void){
  // Prepare pin for measure touch event
  touch_prepare_sense();
  // Start touch interrupt, used timer_3 ADC check threshold:
  gptStart(&GPTD3, &gpt3cfg);         // Init timer 3
  gptStartContinuous(&GPTD3, 10);     // Start timer 3 vs timer 10 interval
  touch_start_watchdog();             // Start ADC watchdog (measure by timer 3 interval and trigger interrupt if touch pressed)
}

// Main software touch function, should:
// set last_touch_x and last_touch_x
// return touch status
static int
touch_check(void)
{
  touch_stop_watchdog();

  int stat = touch_status();
  if (stat) {
    int y = touch_measure_y();
    int x = touch_measure_x();
    touch_prepare_sense();
    if (touch_status())
    {
      last_touch_x = x;
      last_touch_y = y;
    }
#ifdef __REMOTE_DESKTOP__
    remote_mouse_down = 0;
  } else {
    stat = remote_mouse_down == 1;
#endif
  }
  if (stat != last_touch_status) {
    last_touch_status = stat;
    return stat ? EVT_TOUCH_PRESSED : EVT_TOUCH_RELEASED;
  }
  return stat ? EVT_TOUCH_DOWN : EVT_TOUCH_NONE;
}
//*******************************************************************************
// End Software Touch module
//*******************************************************************************
#endif // end SOFTWARE_TOUCH

void touch_set(int16_t x, int16_t y) {
  last_touch_x = x;
  last_touch_y = y;
}

void
touch_wait_release(void)
{
  while (touch_check() != EVT_TOUCH_NONE)
    chThdSleepMilliseconds(20);
}
#if 0
static inline void
touch_wait_pressed(void)
{
  while (touch_check() != EVT_TOUCH_PRESSED)
    ;
}
#endif

static inline void
touch_wait_released(void)
{
  while (touch_check() != EVT_TOUCH_RELEASED)
    ;
}

void
touch_cal_exec(void)
{
  int x1, x2, y1, y2;
  int old_flip = config.flip;
  config.flip = 0;
  ili9341_set_foreground(LCD_FG_COLOR);
  ili9341_set_background(LCD_BG_COLOR);
  ili9341_clear_screen();
  ili9341_line(0, 0, 0, 32);
  ili9341_line(0, 0, 32, 0);
  ili9341_line(0, 0, 32, 32);
  ili9341_drawstring("TOUCH UPPER LEFT", 40, 40);
  touch_wait_released();
//  touch_wait_release();
  x1 = last_touch_x;
  y1 = last_touch_y;

  ili9341_clear_screen();
  ili9341_line(LCD_WIDTH-1, LCD_HEIGHT-1, LCD_WIDTH-1, LCD_HEIGHT-32);
  ili9341_line(LCD_WIDTH-1, LCD_HEIGHT-1, LCD_WIDTH-32, LCD_HEIGHT-1);
  ili9341_line(LCD_WIDTH-1, LCD_HEIGHT-1, LCD_WIDTH-32, LCD_HEIGHT-32);
  ili9341_drawstring("TOUCH LOWER RIGHT", LCD_WIDTH-17*(FONT_WIDTH)-30, LCD_HEIGHT-FONT_GET_HEIGHT-35);

  touch_wait_released();
 // touch_wait_release();
  x2 = last_touch_x;
  y2 = last_touch_y;

  config.touch_cal[0] = x1;
  config.touch_cal[1] = y1;
  config.touch_cal[2] = (x2 - x1) * 16 / LCD_WIDTH;
  config.touch_cal[3] = (y2 - y1) * 16 / LCD_HEIGHT;

  config.flip = old_flip;

  config_save();            // Auto save touch calibration

  //redraw_all();
}

void
touch_draw_test(void)
{
  int x0, y0;
  int x1, y1;

  ili9341_set_foreground(LCD_FG_COLOR);
  ili9341_set_background(LCD_BG_COLOR);
  ili9341_clear_screen();
  ili9341_drawstring("TOUCH TEST: DRAG PANEL, PRESS BUTTON TO FINISH", OFFSETX, LCD_HEIGHT - FONT_GET_HEIGHT);

  int old_button_state = 0;
  while (touch_check() != EVT_TOUCH_PRESSED) {
    int button_state = READ_PORT() & BUTTON_MASK;
    if (button_state != old_button_state) {
      char buf[20];
      plot_printf(buf, sizeof buf, "STATE: %4d       ", button_state);
      ili9341_drawstring_7x13(buf, 120, 120);
      old_button_state = button_state;
    }

  }

  do {
    if (touch_check() == EVT_TOUCH_PRESSED){
      touch_position(&x0, &y0);
      do {
        chThdSleepMilliseconds(50);
        touch_position(&x1, &y1);
        ili9341_line(x0, y0, x1, y1);
        x0 = x1;
        y0 = y1;
      } while (touch_check() != EVT_TOUCH_RELEASED);
    }
  }while (!(btn_check() & EVT_BUTTON_SINGLE_CLICK));
}


void
touch_position(int *x, int *y)
{
#ifdef __REMOTE_DESKTOP__
  if (remote_mouse_down) {
    *x = last_touch_x;
    *y = last_touch_y;
    return;
  }
#endif
  int tx = (last_touch_x - config.touch_cal[0]) * 16 / config.touch_cal[2];
  int ty = (last_touch_y - config.touch_cal[1]) * 16 / config.touch_cal[3];

  if (config.flip) {
    tx = LCD_WIDTH - 1 - tx;
    ty = LCD_HEIGHT - 1 - ty;
  }
  *x = tx;
  *y = ty;
}

void
show_version(void)
{
  int x = 5, y = 5, i = 0;
  ili9341_set_foreground(LCD_FG_COLOR);
  ili9341_set_background(LCD_BG_COLOR);

  ili9341_clear_screen();
  uint16_t shift = 0b00000100001;
// Version text for tinySA3
#ifdef TINYSA3
  ili9341_drawstring_10x14(info_about[i++], x , y);
  y+=FONT_GET_HEIGHT*3+3-5;
  while (info_about[i]) {
    do {shift>>=1; y+=5;} while (shift&1);
    ili9341_drawstring(info_about[i++], x, y+=FONT_STR_HEIGHT+3-5);
  }
  if (has_esd)
    ili9341_drawstring("ESD protected", x, y+=FONT_STR_HEIGHT + 2);

  y+=FONT_STR_HEIGHT + 1;
#endif
// Version text for tinySA4
#ifdef TINYSA4
  ili9341_drawstring_10x14(info_about[i++], x , y);
  y+=FONT_GET_HEIGHT*3+2-5;
  ili9341_drawstring_7x13(info_about[i++], x , y);
  while (info_about[i]) {
    do {shift>>=1; y+=5;} while (shift&1);
    ili9341_drawstring_7x13(info_about[i++], x, y+=bFONT_STR_HEIGHT+2-5);
  }
  char buf[96];
  plot_printf(buf, sizeof(buf), "HW Version:%d", adc1_single_read(0));
  ili9341_drawstring_7x13(buf, x, y+=bFONT_STR_HEIGHT);

extern const char *states[];
#define ENABLE_THREADS_COMMAND
#ifdef ENABLE_THREADS_COMMAND
  y+=FONT_STR_HEIGHT + 1;
  thread_t *tp;
  tp = chRegFirstThread();
  do {
    uint32_t max_stack_use = 0U;
#if (CH_DBG_ENABLE_STACK_CHECK == TRUE) || (CH_CFG_USE_DYNAMIC == TRUE)
    uint32_t stklimit = (uint32_t)tp->wabase;
#if CH_DBG_FILL_THREADS == TRUE
    uint8_t *p = (uint8_t *)tp->wabase; while(p[max_stack_use]==CH_DBG_STACK_FILL_VALUE) max_stack_use++;
#endif
#else
    uint32_t stklimit = 0U;
#endif
    char buf[96];
    plot_printf(buf, sizeof(buf), "%08x|%08x|%08x|%08x|%4u|%4u|%9s|%12s",
             stklimit, (uint32_t)tp->ctx.sp, max_stack_use, (uint32_t)tp,
             (uint32_t)tp->refs - 1, (uint32_t)tp->prio, states[tp->state],
             tp->name == NULL ? "" : tp->name);
    ili9341_drawstring_7x13(buf, x, y+=bFONT_STR_HEIGHT);
    tp = chRegNextThread(tp);
  } while (tp != NULL);
#endif
  y+=bFONT_STR_HEIGHT + 1;
#endif  // TINYSA4
  uint16_t cnt = 0;
  while (true) {
    if (touch_check() == EVT_TOUCH_PRESSED)
      break;
    if (btn_check() & EVT_BUTTON_SINGLE_CLICK)
      break;
    chThdSleepMilliseconds(40);
    if ((cnt++)&0x07) continue; // Not update time so fast

#ifdef TINYSA4
#ifdef __USE_RTC__
    uint32_t tr = rtc_get_tr_bin(); // TR read first
    uint32_t dr = rtc_get_dr_bin(); // DR read second
    char buf[96];
    plot_printf(buf, sizeof(buf), "Time: 20%02d/%02d/%02d %02d:%02d:%02d" " (LS%c)",
      RTC_DR_YEAR(dr),
      RTC_DR_MONTH(dr),
      RTC_DR_DAY(dr),
      RTC_TR_HOUR(dr),
      RTC_TR_MIN(dr),
      RTC_TR_SEC(dr),
      (RCC->BDCR & STM32_RTCSEL_MASK) == STM32_RTCSEL_LSE ? 'E' : 'I');
    ili9341_drawstring_7x13(buf, x, y);
#endif
#if 0
    uint32_t vbat=adc_vbat_read();
    plot_printf(buf, sizeof(buf), "Batt: %d.%03dV", vbat/1000, vbat%1000);
    ili9341_drawstring_7x13(buf, x, y + bFONT_STR_HEIGHT + 1);
#endif
#endif // TINYSA4
  }
}

#ifndef TINYSA4
void
enter_dfu(void)
{
  int x = 5, y = 5;
  ili9341_set_foreground(LCD_FG_COLOR);
  ili9341_set_background(LCD_BG_COLOR);
  // leave a last message 
  ili9341_clear_screen();
  ili9341_drawstring_7x13("DFU: Device Firmware Update Mode\n"
                          "To exit DFU mode, please reset device yourself.", x, y);
  // see __early_init in ./NANOVNA_STM32_F072/board.c
  *((unsigned long *)BOOT_FROM_SYTEM_MEMORY_MAGIC_ADDRESS) = BOOT_FROM_SYTEM_MEMORY_MAGIC;
  NVIC_SystemReset();
}
#endif

static void
select_lever_mode(int mode)
{
  if (uistat.lever_mode != mode) {
    uistat.lever_mode = mode;
    redraw_request |= REDRAW_FREQUENCY | REDRAW_MARKER;
  }
}

// type of menu item 
enum {
  MT_NONE,                      // sentinel menu
//  MT_BLANK,                     // blank menu (nothing draw)
  MT_SUBMENU,                   // enter to submenu
  MT_CALLBACK,                  // call user function
  MT_ADV_CALLBACK,              // adv call user function
  MT_CANCEL,                    // menu, step back on one level up
  MT_TITLE,                     // Title
  MT_KEYPAD,
  MT_REPEATS = 0x08,
  MT_ICON = 0x10,
  MT_HIGH = 0x20,               // Only applicable to high mode
  MT_LOW = 0x40,                // Only applicable to low mode
  MT_FORM = 0x80,               // Large button menu
};
//#define MT_BACK     0x40
//#define MT_LEAVE    0x20
#define MT_MASK(x) (0x7 & (x))
#define DATA_STARTS_REPEATS(S,R)  (((R)<<4)+(S))

#define MT_CUSTOM_LABEL  0

// Call back functions for MT_CALLBACK type
typedef void (*menuaction_cb_t)(int item, uint16_t data);
#define UI_FUNCTION_CALLBACK(ui_function_name) void ui_function_name(int item, uint16_t data)

typedef void (*menuaction_acb_t)(int item, uint16_t data, ui_button_t *b);
#define UI_FUNCTION_ADV_CALLBACK(ui_function_name) void ui_function_name(int item, uint16_t data, ui_button_t *b)

static freq_t
get_marker_frequency(int marker)
{
  if (marker < 0 || marker >= MARKERS_MAX)
    return 0;
  if (!markers[marker].enabled)
    return 0;
  return getFrequency(markers[marker].index);
}

static UI_FUNCTION_CALLBACK(menu_marker_op_cb)
{
  (void)item;
  freq_t freq = get_marker_frequency(active_marker);
  if (freq == 0)
    return; // no active marker

  switch (data) {
  case 0: /* MARKER->START */
  case 1: /* MARKER->STOP */
  case 2: /* MARKER->CENTER */
    set_sweep_frequency(data, freq);
    if (data == 2) {
      uistat.lever_mode = LM_SPAN;
      uistat.auto_center_marker = true;
    }
    break;
  case 3: /* MARKERS->SPAN */
    {
      if (previous_marker == MARKER_INVALID || active_marker == previous_marker) {
        // if only 1 marker is active, keep center freq and make span the marker comes to the edge
        freq_t center = get_sweep_frequency(ST_CENTER);
        freq_t span = center > freq ? center - freq : freq - center;
        set_sweep_frequency(ST_SPAN, span * 2);
      } else {
        // if 2 or more marker active, set start and stop freq to each marker
        freq_t freq2 = get_marker_frequency(previous_marker);
        if (freq2 == 0)
          return;
        if (freq > freq2) {
          freq2 = freq;
          freq = get_marker_frequency(previous_marker);
        }
        set_sweep_frequency(ST_START, freq);
        set_sweep_frequency(ST_STOP, freq2);
      }
    }
    break;
  case 4: // marker -> ref level
    {
    float l = actual_t[markers[active_marker].index];
    float s_max = value(l)/setting.scale;
    user_set_reflevel(setting.scale*(floorf(s_max)+2));
    }
    break;
#ifdef __VNA__
  case 4: /* MARKERS->EDELAY */
    { 
      if (uistat.current_trace == -1)
        break;
      float (*array)[2] = measured[trace[uistat.current_trace].channel];
      float v = groupdelay_from_array(markers[active_marker].index, array);
      set_electrical_delay(electrical_delay + (v / 1e-12));
    }
    break;
#endif
  }
  menu_move_back(true);
  redraw_request |= REDRAW_CAL_STATUS;
  //redraw_all();
}

static UI_FUNCTION_CALLBACK(menu_markers_reset_cb)
{
  (void)item;
  (void)data;
  markers_reset();
}

static UI_FUNCTION_CALLBACK(menu_marker_search_cb)
{
  (void)item;
  int i = -1;
  if (active_marker == MARKER_INVALID)
    return;
  markers[active_marker].mtype &= ~M_TRACKING;
  switch (data) {
  case 0: /* search Left */
    i = marker_search_left_min(active_marker);
    break;
  case 1: /* search right */
    i = marker_search_right_min(active_marker);
    break;
  case 2: /* search Left */
    i = marker_search_left_max(active_marker);
    break;
  case 3: /* search right */
    i = marker_search_right_max(active_marker);
    break;
  case 4: /* peak search */
    i = marker_search_max(active_marker);
    break;
  }
  if (i != -1) {
    markers[active_marker].index = i;
    if (data > 1) // Maximum related
      interpolate_maximum(active_marker);
    else
      markers[active_marker].frequency = getFrequency(i);
  }
  redraw_marker(active_marker);
//  if (data == 4)
    select_lever_mode(LM_MARKER);   // Allow any position with level
//  else
//    select_lever_mode(LM_SEARCH); // Jump from maximum to maximum
}

#if 0
static UI_FUNCTION_ADV_CALLBACK(menu_marker_tracking_acb){
  (void)item;
  (void)data;
  if (active_marker == MARKER_INVALID) return;
  if(b){
    b->icon = markers[active_marker].mtype & M_TRACKING ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  markers[active_marker].mtype ^= M_TRACKING;
}
#endif

#ifdef __VNA__
static void
menu_marker_smith_cb(int item, uint8_t data)
{
  (void)item;
  marker_smith_format = data;
  redraw_marker(active_marker);
  draw_menu();
}
#endif

static void
active_marker_select(int item)  // used only to select an active marker from the modify marker selection menu
{
  if (item == -1) {
    active_marker = previous_marker;
    previous_marker = MARKER_INVALID;
    if (active_marker == MARKER_INVALID) {
      choose_active_marker();
    }
  } else {
    if (previous_marker != active_marker) {
      previous_marker = active_marker;
      active_marker = item;
    } else {
      active_marker = item;
    }
  }
}

#include "ui_sa.c"

#define MENU_STACK_DEPTH_MAX 7
const menuitem_t *menu_stack[MENU_STACK_DEPTH_MAX] = {
  menu_top, NULL, NULL, NULL, NULL, NULL, NULL
};

int current_menu_is_form(void)
{
  return menu_stack[menu_current_level]->type & MT_FORM;
}

static bool menuDisabled(uint8_t type){
  if ((type & MT_LOW) && !MODE_LOW(setting.mode))
    return true;
  if ((type & MT_HIGH) && !MODE_HIGH(setting.mode))
    return true;
//  if (type == MT_BLANK)
//    return true;
  return false;
}

static const menuitem_t *menu_next_item(const menuitem_t *m, int *sub_item){
  do{
    if (m->type & MT_REPEATS) {
      (*sub_item)++;
      if (*sub_item < ((m->data>>4) & 0x0f))
        return m;
      *sub_item = 0;
    }
    m++;
    m = MT_MASK(m->type) == MT_NONE ? (menuitem_t *)m->reference : m;
  } while(m!=NULL && menuDisabled(m->type));
  return m;
}

static const menuitem_t *current_menu_item(int i, int *sub_item){
  *sub_item = 0;
  const menuitem_t * m = menu_stack[menu_current_level];
  while (i--) m = menu_next_item(m,sub_item);
  return m;
}

static int current_menu_get_count(void){
  int i = 0,sub_item = 0;
  const menuitem_t *m = menu_stack[menu_current_level];
  while (m){m = menu_next_item(m, &sub_item); i++;}
  return i;
}

static void
ensure_selection(void)
{
  const menuitem_t *menu = menu_stack[menu_current_level];
  int i = current_menu_get_count();
  if (selection <  0) selection =  -1;
  if (selection >= i) selection = i-1;
  if (MT_MASK(menu[0].type) == MT_TITLE && selection == 0) selection = 1;
  if (i <  MENU_BUTTON_MIN) i = MENU_BUTTON_MIN;
  if (i >= MENU_BUTTON_MAX) i = MENU_BUTTON_MAX;
#ifdef MENU_USE_AUTOHEIGHT
  menu_button_height = MENU_BUTTON_HEIGHT_N(i);
#endif
}

static void
menu_move_back(bool leave_ui)
{
  if (menu_current_level == 0)
    return;
  erase_menu_buttons();
  bool form = current_menu_is_form();
  menu_current_level--;

  // redraw all if switch from form to normal menu mode or back
  if (form != current_menu_is_form())
    redraw_request|=REDRAW_AREA|REDRAW_BATTERY|REDRAW_FREQUENCY|REDRAW_CAL_STATUS;

  selection = -1;
  if (leave_ui)
    ui_mode_normal();
  else
    ui_mode_menu();
}

void
menu_push_submenu(const menuitem_t *submenu)
{
  erase_menu_buttons();
  if (menu_current_level < MENU_STACK_DEPTH_MAX-1)
    menu_current_level++;
  menu_stack[menu_current_level] = submenu;
  ui_mode_menu();
}

void
menu_push_lowoutput(void)
{
  menu_push_submenu(menu_lowoutputmode);
}

void
menu_push_highoutput(void)
{
  menu_push_submenu(menu_highoutputmode);
}

/*
static void
menu_move_top(void)
{
  if (menu_current_level == 0)
    return;
  menu_current_level = 0;
  ensure_selection();
  erase_menu_buttons();
  draw_menu();
}
*/

static void
menu_invoke(int item)
{
  int sub_item;
  const menuitem_t *menu = current_menu_item(item, &sub_item);
  if (menu == NULL) return;
  switch (MT_MASK(menu->type)) {
//  case MT_NONE:
//  case MT_BLANK:
//    ui_mode_normal();
//    break;

  case MT_CANCEL:
    menu_move_back(false);
    break;

  case MT_CALLBACK: {
    uistat.auto_center_marker = false;
    menuaction_cb_t cb = (menuaction_cb_t)menu->reference;
    if (cb) (*cb)(item, (menu->type & MT_REPEATS) ? (menu->data & 0x0f)+sub_item : menu->data);
//    if (!(menu->type & MT_FORM))
    redraw_request |= REDRAW_CAL_STATUS;
    break;
  }
  case MT_ADV_CALLBACK: {
    uistat.auto_center_marker = false;
    menuaction_acb_t cb = (menuaction_acb_t)menu->reference;
    if (cb) (*cb)(item, (menu->type & MT_REPEATS) ? (menu->data & 0x0f)+sub_item : menu->data, NULL);
//    if (!(menu->type & MT_FORM))
    redraw_request |= REDRAW_CAL_STATUS | REDRAW_BATTERY;
    break;
  }
  case MT_SUBMENU:
    menu_push_submenu((const menuitem_t*)menu->reference);
    break;

  case MT_KEYPAD:
    uistat.auto_center_marker = false;
    if (current_menu_is_form()) {
      redraw_frame();         // Remove form numbers
    }
    kp_help_text = (char *)menu->reference;
    if (menu->data <= KM_CW) {      // One of the frequency input keypads
      if (MODE_LOW(setting.mode))
        kp_help_text = VARIANT("0..350MHz",range_text);
      else if (menu->data == KM_SPAN)
        kp_help_text = VARIANT("0..720Mhz",range_text);
      else
        kp_help_text = VARIANT("240..960Mhz",range_text);
    }
    ui_mode_keypad(menu->data);
    redraw_request |= REDRAW_CAL_STATUS;
    break;
  }
  // Redraw menu after if UI in menu mode
  if (ui_mode == UI_MENU)
    draw_menu();
}

static const char * const keypad_scale_text[] = {"0", "1", "2", "5", "10", "20" , "50", "100", "200", "500"};
//static const int  keypad_scale_value[] = { 1, 2, 5, 10, 20 , 50, 100, 200, 500};

static void
draw_button(uint16_t x, uint16_t y, uint16_t w, uint16_t h, ui_button_t *b)
{
  uint16_t bw = b->border&BUTTON_BORDER_WIDTH_MASK;
  ili9341_set_foreground(b->fg);
  ili9341_set_background(b->bg);ili9341_fill(x + bw, y + bw, w - (bw * 2), h - (bw * 2));
  if (bw==0) return;
  uint16_t br = LCD_RISE_EDGE_COLOR;
  uint16_t bd = LCD_FALLEN_EDGE_COLOR;
  uint16_t type = b->border;
  ili9341_set_background(type&BUTTON_BORDER_TOP    ? br : bd);ili9341_fill(x,          y,           w, bw); // top
  ili9341_set_background(type&BUTTON_BORDER_RIGHT  ? br : bd);ili9341_fill(x + w - bw, y,          bw,  h); // right
  ili9341_set_background(type&BUTTON_BORDER_LEFT   ? br : bd);ili9341_fill(x,          y,          bw,  h); // left
  ili9341_set_background(type&BUTTON_BORDER_BOTTOM ? br : bd);ili9341_fill(x,          y + h - bw,  w, bw); // bottom
  // Set colors for button text after
  ili9341_set_background(b->bg);
}

void drawMessageBox(char *header, char *text, uint32_t delay){
  ui_button_t b;
  b.bg = LCD_MENU_COLOR;
  b.fg = LCD_MENU_TEXT_COLOR;
  b.border = BUTTON_BORDER_FLAT|1;
  // Draw header
  draw_button((LCD_WIDTH-MESSAGE_BOX_WIDTH)/2, LCD_HEIGHT/2-40, MESSAGE_BOX_WIDTH, 60, &b);
  ili9341_drawstring_7x13(header, (LCD_WIDTH-MESSAGE_BOX_WIDTH)/2 + 10, LCD_HEIGHT/2-40 + 5);
  // Draw window
  ili9341_set_background(LCD_FG_COLOR);
  ili9341_fill((LCD_WIDTH-MESSAGE_BOX_WIDTH)/2+3, LCD_HEIGHT/2-40+bFONT_STR_HEIGHT+8, MESSAGE_BOX_WIDTH-6, 60-bFONT_STR_HEIGHT-8-3);
  ili9341_drawstring_7x13(text, (LCD_WIDTH-MESSAGE_BOX_WIDTH)/2 + 20, LCD_HEIGHT/2-40 + bFONT_STR_HEIGHT + 8 + 14);
  chThdSleepMilliseconds(delay);
}

static void
draw_keypad(uint32_t mask)
{
  int i;
  ui_button_t button;
  button.fg = LCD_MENU_TEXT_COLOR;
  for(i = 0; keypads[i].c >= 0; i++) {
    if ((mask&(1<<i)) == 0) continue;
    button.bg = LCD_MENU_COLOR;
    if (i == selection){
      button.bg = LCD_MENU_ACTIVE_COLOR;
      button.border = KEYBOARD_BUTTON_BORDER|BUTTON_BORDER_FALLING;
    }
    else
      button.border = KEYBOARD_BUTTON_BORDER|BUTTON_BORDER_RISE;
    int x = KP_GET_X(keypads[i].x);
    int y = KP_GET_Y(keypads[i].y);
    draw_button(x, y, KP_WIDTH, KP_HEIGHT, &button);
    if (keypads[i].c < KP_0) { // KP_0
      ili9341_drawfont(keypads[i].c,
                     x + (KP_WIDTH - NUM_FONT_GET_WIDTH) / 2,
                     y + (KP_HEIGHT - NUM_FONT_GET_HEIGHT) / 2);
    } else {
      const char *t = keypad_scale_text[keypads[i].c - KP_0];
      ili9341_drawstring_10x14(t,
                     x + (KP_WIDTH  - wFONT_MAX_WIDTH*strlen(t)) / 2,
                     y + (KP_HEIGHT - wFONT_GET_HEIGHT) / 2);
    }
  }
}

static int
menu_is_multiline(const char *label)
{
  int n = 1;
  if (label)
    while (*label)
      if (*label++ == '\n')
        n++;
  return n;
}

static int period_pos(void) {int j; for (j = 0; j < kp_index && kp_buf[j] != '.'; j++); return j;}

static void
draw_numeric_input(const char *buf)
{
  uint16_t i;
  uint16_t x = 10 + 10 * FONT_WIDTH + 4;
  uint16_t xsim;
#ifdef __USE_RTC__
  if (keypad_mode == KM_RTC_DATE || keypad_mode == KM_RTC_TIME)
    xsim = 0b01010100;
  else
#endif
  xsim = (0b00100100100100100 >>(2-(period_pos()%3)))&(~1);
  ili9341_set_foreground(LCD_INPUT_TEXT_COLOR);
  ili9341_set_background(LCD_INPUT_BG_COLOR);
  for (i = 0; buf[i]; i++) {
    int c = buf[i];
         if (c == '.'){c = KP_PERIOD;xsim<<=4;}
    else if (c == '-'){c = KP_MINUS; xsim&=~3;}
    else// if (c >= '0' && c <= '9')
      c = c - '0';
    if (c < 0) c = 0;
    // Add space before char
    int16_t space = xsim&1 ? 2 + 10 : 2;
    xsim>>=1;
    ili9341_fill(x, LCD_HEIGHT-NUM_INPUT_HEIGHT+4, space, NUM_INPUT_HEIGHT);
    x+=space;
    if (c >= 0) // c is number
      ili9341_drawfont(c, x, LCD_HEIGHT-NUM_INPUT_HEIGHT+4);
    else
      break;
    x+=NUM_FONT_GET_WIDTH;
  }

  ili9341_fill(x, LCD_HEIGHT-NUM_INPUT_HEIGHT+4, LCD_WIDTH - 1 - x, NUM_INPUT_HEIGHT);
  if (buf[0] == 0 && kp_help_text != NULL) {
    int  lines = menu_is_multiline(kp_help_text);
    ili9341_set_foreground(LCD_INPUT_TEXT_COLOR);
    ili9341_drawstring_7x13(kp_help_text, 64+NUM_FONT_GET_WIDTH+2, LCD_HEIGHT-(lines*bFONT_GET_HEIGHT+NUM_INPUT_HEIGHT)/2);
  }
}

static void
draw_numeric_area_frame(void)
{
  ili9341_set_foreground(LCD_INPUT_TEXT_COLOR);
  ili9341_set_background(LCD_INPUT_BG_COLOR);

  ili9341_fill(0, LCD_HEIGHT-NUM_INPUT_HEIGHT, LCD_WIDTH, NUM_INPUT_HEIGHT);
  char *name = keypads_mode_tbl[keypad_mode].name;
  int lines = menu_is_multiline(name);
  ili9341_drawstring_7x13(name, 10, LCD_HEIGHT-NUM_INPUT_HEIGHT + (NUM_INPUT_HEIGHT-lines*bFONT_STR_HEIGHT)/2);
  //ili9341_drawfont(KP_KEYPAD, 300, 216);
  draw_numeric_input("");
}

#define ICON_WIDTH        16
#define ICON_HEIGHT       11

static const uint8_t check_box[] = {
  _BMP16(0b0011111111110000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0011111111110000),

  _BMP16(0b0011111111110000),
  _BMP16(0b0010000000001000),
  _BMP16(0b0010000000011000),
  _BMP16(0b0010000000110000),
  _BMP16(0b0010000001100000),
  _BMP16(0b0010100011010000),
  _BMP16(0b0010110110010000),
  _BMP16(0b0010011100010000),
  _BMP16(0b0010001000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0011111111110000),

  _BMP16(0b0011111111111000),
  _BMP16(0b0010000000001000),
  _BMP16(0b0010001111101000),
  _BMP16(0b0010011001101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010111111101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010000000001000),
  _BMP16(0b0011111111111000),

  _BMP16(0b0011111111111000),
  _BMP16(0b0010000000001000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010111011101000),
  _BMP16(0b0010111111101000),
  _BMP16(0b0010110101101000),
  _BMP16(0b0010110101101000),
  _BMP16(0b0010110001101000),
  _BMP16(0b0010000000001000),
  _BMP16(0b0011111111111000),

  _BMP16(0b0000000000000000),
  _BMP16(0b0000011110000000),
  _BMP16(0b0000100001000000),
  _BMP16(0b0001000000100000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0010000000010000),
  _BMP16(0b0001000000100000),
  _BMP16(0b0000100001000000),
  _BMP16(0b0000011110000000),

  _BMP16(0b0000000000000000),
  _BMP16(0b0000011110000000),
  _BMP16(0b0000100001000000),
  _BMP16(0b0001001100100000),
  _BMP16(0b0010011110010000),
  _BMP16(0b0010111111010000),
  _BMP16(0b0010111111010000),
  _BMP16(0b0010011110010000),
  _BMP16(0b0001001100100000),
  _BMP16(0b0000100001000000),
  _BMP16(0b0000011110000000),
};

#ifndef MENU_USE_AUTOHEIGHT
#ifdef TINYSA4
#define menu_button_height  ((menu[i].type & MT_FORM) || menu_is_multiline(menu[i].label) == 2 ? LCD_HEIGHT/10 : LCD_HEIGHT/12 )
#else
#define menu_button_height  ((menu[i].type & MT_FORM) || menu_is_multiline(menu[i].label) == 2 ? LCD_HEIGHT/8 : LCD_HEIGHT/10 )
#endif
#endif

static void
draw_menu_buttons(const menuitem_t *menu, uint32_t mask)
{
  int i, y;
  ui_button_t button;
  const menuitem_t *m = menu;
  int sub_item = 0;
  for (i = 0, y = 0; m; m = menu_next_item(m, &sub_item), i++, y += menu_button_height) {
    if ((mask&(1<<i)) == 0)
      continue;
    button.icon = BUTTON_ICON_NONE;
    // Border width
    button.border = MENU_BUTTON_BORDER;

    if (MT_MASK(m->type) == MT_TITLE) {
      button.fg = LCD_FG_COLOR;
      button.bg = LCD_BG_COLOR;
      button.border = 0; // no border for title
    } else {
      button.bg = LCD_MENU_COLOR;
      button.fg = LCD_MENU_TEXT_COLOR;
    }

    if (i == selection){
      button.bg = LCD_MENU_ACTIVE_COLOR;
      button.border|= BUTTON_BORDER_FALLING;
    }
    else
      button.border|= BUTTON_BORDER_RISE;

    // Need replace this obsolete bad function on new MT_ADV_CALLBACK variant
    menu_item_modify_attribute(menu, i, &button);      // before plot_printf to create status text
    char *text;
    // MT_ADV_CALLBACK - allow change button data in callback, more easy and correct
    if (MT_MASK(m->type) == MT_ADV_CALLBACK){
      menuaction_acb_t cb = (menuaction_acb_t)m->reference;
      if (cb) (*cb)(i, (m->type & MT_REPEATS) ? (m->data & 0x0f)+sub_item : m->data, &button);
      // Apply custom text, from button label and
      if (m->label != MT_CUSTOM_LABEL)
        plot_printf(button.text, sizeof(button.text), m->label, button.param_1.u);
      text = button.text;
    }
    else
      text = m->label;
    // Only keypad retrieves value
    if (MT_MASK(m->type) == MT_KEYPAD) {
      fetch_numeric_target(m->data);
      plot_printf(button.text, sizeof button.text, m->label, uistat.text);
      text = button.text;
    }

    int button_height = menu_button_height;
    if (current_menu_is_form()) {
      int button_width = MENU_FORM_WIDTH;
      int button_start = (LCD_WIDTH - MENU_FORM_WIDTH)/2; // At center of screen
      draw_button(button_start, y, button_width, button_height, &button);
      uint16_t text_offs = button_start + 6;
      if (button.icon >=0){
        ili9341_blitBitmap(button_start+3, y+(button_height-ICON_HEIGHT)/2, ICON_WIDTH, ICON_HEIGHT, &check_box[button.icon*2*ICON_HEIGHT]);
        text_offs = button_start+6+ICON_WIDTH+1;
      }
#ifdef __ICONS__
      if (m->type & MT_ICON) {
        ili9341_blitBitmap(button_start+MENU_FORM_WIDTH-2*FORM_ICON_WIDTH-8,y+(button_height-FORM_ICON_HEIGHT)/2,FORM_ICON_WIDTH,FORM_ICON_HEIGHT,& left_icons[((menu[i].data >>4)&0xf)*2*FORM_ICON_HEIGHT]);
        ili9341_blitBitmap(button_start+MENU_FORM_WIDTH-  FORM_ICON_WIDTH-8,y+(button_height-FORM_ICON_HEIGHT)/2,FORM_ICON_WIDTH,FORM_ICON_HEIGHT,&right_icons[((menu[i].data >>0)&0xf)*2*FORM_ICON_HEIGHT]);
      }
#endif
      int local_text_shift = 0;
      if (MT_MASK(m->type) == MT_KEYPAD) {
        int local_slider_positions = 0;
        if (m->data == KM_CENTER) {
          local_slider_positions =  LCD_WIDTH/2+setting.slider_position;
          lcd_printf(button_start+12 + 0 * MENU_FORM_WIDTH/5, y+button_height-9, "%+3.0FHz", -(float)setting.slider_span);
          lcd_printf(button_start+12 + 1 * MENU_FORM_WIDTH/5, y+button_height-9, "%+3.0FHz", -(float)setting.slider_span/10);
          lcd_printf(button_start+12 + 2 * MENU_FORM_WIDTH/5, y+button_height-9, "Set");
          lcd_printf(button_start+12 + 3 * MENU_FORM_WIDTH/5, y+button_height-9, "%+3.0FHz",  (float)setting.slider_span/10);
          lcd_printf(button_start+12 + 4 * MENU_FORM_WIDTH/5, y+button_height-9, "%+3.0FHz",  (float)setting.slider_span);
        } else if (m->data == KM_LOWOUTLEVEL) {
          local_slider_positions = ((get_level() - level_min()) * (MENU_FORM_WIDTH-8)) / level_range() + OFFSETX+4;
          lcd_printf(button_start+12 + 0 * MENU_FORM_WIDTH/5, y+button_height-9, "%+ddB", -10);
          lcd_printf(button_start+12 + 1 * MENU_FORM_WIDTH/5, y+button_height-9, "%+ddB",  -1);
          lcd_printf(button_start+12 + 2 * MENU_FORM_WIDTH/5, y+button_height-9, "Set");
          lcd_printf(button_start+12 + 3 * MENU_FORM_WIDTH/5, y+button_height-9, "%+ddB",   1);
          lcd_printf(button_start+12 + 4 * MENU_FORM_WIDTH/5, y+button_height-9, "%+ddB",  10);
        } else if (m->data == KM_HIGHOUTLEVEL) {
          local_slider_positions = ((get_level() - level_min() ) * (MENU_FORM_WIDTH-8)) / level_range() + OFFSETX+4;
        }
        if (local_slider_positions){
          if (local_slider_positions < button_start)
            local_slider_positions = button_start;
          // Shift down text if slider present
          local_text_shift = 2;
          ili9341_blitBitmap(local_slider_positions - 4, y, 7, 5, slider_bitmap);
          // Draw divider for sliders
          for (int i = 1; i <= 4; i++)
            ili9341_line(button_start + i * MENU_FORM_WIDTH/5, y+button_height-9, button_start + i * MENU_FORM_WIDTH/5, y+button_height);
        }
      }
//      ili9341_drawstring_size(text, text_offs, y+(button_height-2*FONT_GET_HEIGHT)/2-local_text_shift, 2);
      ili9341_drawstring_10x14(text, text_offs, y+(button_height-wFONT_GET_HEIGHT)/2-local_text_shift);
    } else {
      int button_width = MENU_BUTTON_WIDTH;
      int button_start = LCD_WIDTH - MENU_BUTTON_WIDTH;
      draw_button(button_start, y, button_width, button_height, &button);
      uint16_t text_offs = button_start + 7;
      if (button.icon >=0){
        ili9341_blitBitmap(button_start+2, y+(button_height-ICON_HEIGHT)/2, ICON_WIDTH, ICON_HEIGHT, &check_box[button.icon*2*ICON_HEIGHT]);
        text_offs = button_start+2+ICON_WIDTH;
      }
      int lines = menu_is_multiline(text);
#define BIG_BUTTON_FONT 1
#ifdef BIG_BUTTON_FONT
      ili9341_drawstring_7x13(text, text_offs, y+(button_height-lines*bFONT_GET_HEIGHT)/2);
#else
      ili9341_drawstring(text, text_offs, y+(button_height-linesFONT_GET_HEIGHT)/2);
#endif
    }
  }
  // Cleanup other buttons (less flicker)
  // Erase empty buttons
  if (NO_WATERFALL - y > 0){
    ili9341_set_background(LCD_BG_COLOR);
    ili9341_fill(LCD_WIDTH-MENU_BUTTON_WIDTH, y, MENU_BUTTON_WIDTH, NO_WATERFALL - y);
  }
//  if (current_menu_is_form())
//    draw_battery_status();
}

enum { SL_UNKNOWN, SL_SPAN, SL_MOVE};

void set_keypad_value(int v)
{
  keypad_mode = v;
  set_numeric_value();
}

void check_frequency_slider(freq_t slider_freq)
{

  if ( (maxFreq - minFreq) < (freq_t)setting.slider_span) {
    setting.slider_span = maxFreq - minFreq;                         // absolute mode with max step size
  }
  freq_t half_span = setting.slider_span >> 1;
  int temp = (setting.slider_span / (MENU_FORM_WIDTH-8));
  if (minFreq + half_span > slider_freq) {
    setting.slider_position -= (minFreq + half_span - slider_freq) / temp;            // reposition if needed
  }
  if (maxFreq < slider_freq + half_span) {
    setting.slider_position += (slider_freq + half_span - maxFreq) / temp;            // reposition if needed
  }
}

#define TOUCH_DEAD_ZONE 40
static void
menu_select_touch(const menuitem_t * m, int i, int pos)
{
  uint32_t mask = (1<<i)|(1<<selection);
  selection = i;
  draw_menu_mask(mask);
#if 1               // drag values
  int keypad = m->data;
  int touch_x, touch_y,  prev_touch_x = 0;
  systime_t dt = 0;
  int mode = SL_UNKNOWN;

  systime_t ticks = chVTGetSystemTimeX();
  while (touch_check() != EVT_TOUCH_NONE){
    dt = chVTGetSystemTimeX() - ticks;
    if (dt > BUTTON_DOWN_LONG_TICKS) break;
  }

  if (current_menu_is_form() && MT_MASK(m->type) == MT_KEYPAD && dt >= BUTTON_DOWN_LONG_TICKS){
    // Wait release touch and process it
    while (touch_check() != EVT_TOUCH_NONE){
      touch_position(&touch_x, &touch_y);
      if (abs(touch_x -  prev_touch_x) < 2) continue;

      fetch_numeric_target(keypad);
      int new_slider = touch_x - LCD_WIDTH/2;   // Can have negative outcome
      if (new_slider < - (MENU_FORM_WIDTH-8)/2 - 1)
        new_slider = -(MENU_FORM_WIDTH-8)/2 - 1;
      if (new_slider > (MENU_FORM_WIDTH-8)/2 + 1)
        new_slider = (MENU_FORM_WIDTH-8)/2 + 1;
      if (keypad == KM_CENTER) {
        if (mode == SL_UNKNOWN ) {
          if (abs(setting.slider_position - new_slider) < TOUCH_DEAD_ZONE) // Pick up slider
            mode = SL_MOVE;
          else
            mode = SL_SPAN;
        }
        if (mode == SL_MOVE ) {
          long_t freq_delta = (setting.slider_span/(MENU_FORM_WIDTH-8))*(new_slider - setting.slider_position);
          if (freq_delta < 0 && uistat.freq_value < (freq_t)(-freq_delta))
            uistat.freq_value = 0;
          else
            uistat.freq_value+= freq_delta;
          if (uistat.freq_value < minFreq)
            uistat.freq_value = minFreq;
          if (uistat.freq_value > maxFreq)
            uistat.freq_value = maxFreq;
          setting.slider_position = new_slider;
          set_keypad_value(keypad);
          dirty = false;
          perform(false, 0, uistat.freq_value, false);
          draw_menu_mask(1<<i);
        }
        else if (mode == SL_SPAN ){
          freq_t slider_freq;
          slider_freq = uistat.freq_value;
          int pw=new_slider + LCD_WIDTH/2;
          setting.slider_position = pw - LCD_WIDTH/2;   // Show delta on slider
          setting.slider_span = 10;
          while (pw>0) {
            setting.slider_span += setting.slider_span;
            pw -= 12;
            if (pw <=0)
              break;
            setting.slider_span += setting.slider_span + (setting.slider_span >>1);
            pw -= 12;
            if (pw<=0)
              break;
            setting.slider_span *= 2;
            pw -= 12;
          }
          if ((freq_t)setting.slider_span > (maxFreq - minFreq))
            setting.slider_span = (maxFreq - minFreq);
          freq_t old_minFreq = minFreq;           // Save when in high mode
          minFreq = 0;                              // And set minFreq to 0 for span display
          uistat.freq_value = setting.slider_span;
          set_keypad_value(keypad);
          plot_printf(center_text, sizeof center_text, "RANGE: %%s");
          draw_menu_mask(1<<i);                      // Show slider span
          minFreq = old_minFreq;                     // and restore minFreq
          uistat.freq_value = slider_freq;        // and restore current slider freq
          set_keypad_value(keypad);
          plot_printf(center_text, sizeof center_text, "FREQ: %%s");
          setting.slider_position = 0;                         // reset slider after span change
          check_frequency_slider(slider_freq);
        }
        chThdSleepMilliseconds(100);
      } else if (keypad == KM_LOWOUTLEVEL) {
        uistat.value =  setting.external_gain + ((touch_x - OFFSETX+4) * level_range() ) / (MENU_FORM_WIDTH-8) + level_min() ;
        set_keypad_value(keypad);
        draw_menu_mask(1<<i);
        perform(false, 0, get_sweep_frequency(ST_CENTER), false);
      } else if (keypad == KM_HIGHOUTLEVEL) {
        set_level( (touch_x - OFFSETX+4) *(level_range()) / (MENU_FORM_WIDTH-8) + level_min() );
        draw_menu_mask(1<<i);
        perform(false, 0, get_sweep_frequency(ST_CENTER), false);
      }
      prev_touch_x = touch_x;

    }
    selection = -1;
    draw_menu_mask(1<<i);
    return;
  }
  if (current_menu_is_form() && MT_MASK(m->type) == MT_KEYPAD){
    bool do_exit = false;
    long_t step = 0;
    if (keypad == KM_LOWOUTLEVEL) {
      switch (pos) {
        case 0:step = -10;break;
        case 1:step =  -1;break;
        case 2:goto nogo;
        case 3:step =  +1;break;
        case 4:step = +10;break;
      }
      uistat.value = setting.external_gain + get_level() + step;
      do_exit = true;
    }
    else if (keypad == KM_CENTER) {
      switch (pos) {
        case 0: step = setting.slider_span;   step=-step; break;
        case 1: step = setting.slider_span/10;step=-step; break;
        case 2: goto nogo;
        case 3: step = setting.slider_span/10;break;
        case 4: step = setting.slider_span;   break;
      }
      if (step < 0 && get_sweep_frequency(ST_CENTER) < (freq_t)(-step))
        uistat.freq_value = 0;
      else
        uistat.freq_value = get_sweep_frequency(ST_CENTER) + step;
      do_exit = true;
      setting.slider_position = 0;                         // reset slider after step
      check_frequency_slider(uistat.freq_value);
    }
    if (do_exit){
      set_keypad_value(keypad);
      selection = -1;
      draw_menu_mask(1<<i);
      perform(false, 0, get_sweep_frequency(ST_CENTER), false);
      return;
    }
  }
nogo:
  setting.slider_position = 0;            // Reset slider when entering frequency
#endif
//  touch_wait_release();
  selection = -1;
  menu_invoke(i);
}


static void
menu_apply_touch(int touch_x, int touch_y)
{
  const menuitem_t *m = menu_stack[menu_current_level];
  int i;
  int y = 0;
  int active_button_start;
  if (current_menu_is_form()) {
    active_button_start = (LCD_WIDTH - MENU_FORM_WIDTH)/2;
//  active_button_stop = LCD_WIDTH - active_button_start;
  } else {
    active_button_start = LCD_WIDTH - MENU_BUTTON_WIDTH;
//  active_button_stop = LCD_WIDTH;
  }
  int sub_item = 0;
  for (i = 0; m; m = menu_next_item(m,&sub_item), i++, y+= menu_button_height) {
    if (MT_MASK(m->type) == MT_TITLE) continue;
    if (y < touch_y && touch_y < y+menu_button_height && touch_x > active_button_start) {
      menu_select_touch(m, i, (( touch_x - active_button_start) * 5 ) / MENU_FORM_WIDTH);
      return;
    }
  }
  if (current_menu_is_form())
    return;
  touch_wait_release();
  ui_mode_normal();
}

void
draw_menu(void)
{
  draw_menu_buttons(menu_stack[menu_current_level], -1);
}

void
draw_menu_mask(uint32_t mask)
{
  draw_menu_buttons(menu_stack[menu_current_level], mask);
}

#ifdef __SWEEP_RESTART__
systime_t old_sweep_time;

void
refresh_sweep_menu(int i)
{
  current_index = i;
  systime_t new_sweep_time = chVTGetSystemTimeX();
  if (new_sweep_time - old_sweep_time > ONE_SECOND_TIME/200 && i >= 0) {
    old_sweep_time = new_sweep_time;
    if (menu_stack[menu_current_level] == menu_lowoutputmode)
      draw_menu_buttons(menu_stack[menu_current_level], 1 << 5);
    if (menu_stack[menu_current_level] == menu_highoutputmode)
      draw_menu_buttons(menu_stack[menu_current_level], 1 << 5);
  }
}
#endif

static void
erase_menu_buttons(void)
{
// Not need, screen redraw in all cases
//  ili9341_fill(area_width, 0, LCD_WIDTH - area_width, area_height, LCD_BG_COLOR);
 // if (current_menu_is_form())
 //   ili9341_fill(OFFSETX, 0,LCD_WIDTH-OFFSETX, menu_button_height*MENU_BUTTON_MAX, LCD_BG_COLOR);
 // else
 //   ili9341_fill(LCD_WIDTH-MENU_BUTTON_WIDTH, 0, MENU_BUTTON_WIDTH, menu_button_height*MENU_BUTTON_MAX, LCD_BG_COLOR);
  draw_frequencies();
}

#if 0
static void
erase_numeric_input(void)
{
  ili9341_set_background(LCD_BG_COLOR);
  ili9341_fill(0, LCD_HEIGHT-NUM_INPUT_HEIGHT, LCD_WIDTH, NUM_INPUT_HEIGHT);
}
#endif

static void
leave_ui_mode()
{
//  if (ui_mode == UI_MENU) {
//    request_to_draw_cells_behind_menu();
//    erase_menu_buttons();
//  }
  ili9341_set_background(LCD_BG_COLOR);
  // Erase bottom area (not redraw on area update)
//  if (menu_button_height*MENU_BUTTON_MAX - area_height > 0)
//    ili9341_fill(LCD_WIDTH-MENU_BUTTON_WIDTH, area_height, MENU_BUTTON_WIDTH, menu_button_height*MENU_BUTTON_MAX - area_height);
  if (setting.waterfall)
    set_waterfall();
  redraw_request|=REDRAW_AREA | REDRAW_FREQUENCY | REDRAW_CAL_STATUS | REDRAW_BATTERY;
}

void
ui_mode_menu(void)
{
//  if (ui_mode == UI_MENU)
//    return;
  ui_mode = UI_MENU;
  ensure_selection();
  if (current_menu_is_form()) {
    redraw_frame();
    area_width = 0;
    area_height = 0;
  } else {
    area_width = AREA_WIDTH_NORMAL - MENU_BUTTON_WIDTH;
    area_height = AREA_HEIGHT_NORMAL;
  }
  draw_menu();
  redraw_request|=REDRAW_BATTERY|REDRAW_CAL_STATUS;
}

static void
ui_mode_keypad(int _keypad_mode)
{
  if (ui_mode == UI_KEYPAD && keypad_mode == _keypad_mode )
    return;

  // keypads array
  keypad_mode = _keypad_mode;
  keypads = keypads_mode_tbl[_keypad_mode].keypad_type;

  ui_mode = UI_KEYPAD;
  if (!current_menu_is_form())
    draw_menu();
  draw_keypad(-1);
  draw_numeric_area_frame();
  ui_process_keypad();
}

void
ui_mode_normal(void)
{
  if (ui_mode == UI_NORMAL)
    return;
  if (current_menu_is_form())
    return;
  leave_ui_mode();
  area_width  = AREA_WIDTH_NORMAL;
  area_height = AREA_HEIGHT_NORMAL;
  ui_mode = UI_NORMAL;
}

static void
lever_move_marker(int status)
{
  uint16_t step = 1<<2;
  do {
    if (active_marker != MARKER_INVALID && markers[active_marker].enabled) {
      int idx = (int)markers[active_marker].index;
      if (status & EVT_DOWN) {
        idx -= step>>2;
        if (idx < 0) idx = 0 ;
      }
      if (status & EVT_UP) {
        idx += step>>2;
        if (idx  > sweep_points-1) idx = sweep_points-1 ;
      }
      markers[active_marker].index = idx;
      markers[active_marker].frequency = getFrequency(idx);
      redraw_marker(active_marker);
      markers[active_marker].mtype &= ~M_TRACKING;    // Disable tracking when dragging marker
      step++;
    }
    status = btn_wait_release();
  } while (status != 0);
}

static void
lever_search_marker(int status)
{
  int i = -1;
  if (active_marker != MARKER_INVALID) {
    if (status & EVT_DOWN)
      i = marker_search_left_max(markers[active_marker].index);
    else if (status & EVT_UP)
      i = marker_search_right_max(markers[active_marker].index);
    if (i != -1) {
      markers[active_marker].index = i;
      interpolate_maximum(active_marker);
//      markers[active_marker].frequency = frequencies[i];
    }
    redraw_marker(active_marker);
  }
}

// ex. 10942 -> 10000
//      6791 ->  5000
//       341 ->   200
static freq_t
step_round(freq_t v)
{
  // decade step
  freq_t x = 1;
  for (x = 1; x*10 <= v; x*= 10)
    ;

  // 1-2-5 step
  if (x * 2 > v)
    return x;
  else if (x * 5 > v)
    return x * 2;
  else
    return x * 5;
}

static void
lever_zoom_time(int status)
{
  uint32_t time = setting.sweep_time_us; // in uS
  if (time < MINIMUM_SWEEP_TIME)
    time = MINIMUM_SWEEP_TIME;
  if (status & EVT_UP) {
    time = time*10/25;
  } else if (status & EVT_DOWN) {
    time = time*25/10;
  }
  time = step_round(time);
  set_sweep_time_us(time);
}

static void
lever_move(int status, int mode)
{
  freq_t freq = get_sweep_frequency(mode);
  if (mode == ST_SPAN){
    if (uistat.auto_center_marker) {
      freq = get_marker_frequency(active_marker);
      search_maximum(active_marker, freq, 10 );
      if (freq == 0) return;
      set_sweep_frequency(ST_CENTER, freq);
      return;
    }
    if (status & EVT_UP  ) freq = setting.frequency_var ? (freq + setting.frequency_var) : step_round(freq*4 + 1);
    if (status & EVT_DOWN) freq = setting.frequency_var ? (freq - setting.frequency_var) : step_round(freq   - 1);
  }
  else {
    freq_t span = setting.frequency_var ? setting.frequency_var : step_round(get_sweep_frequency(ST_SPAN) / 4);
    if (status & EVT_UP  ) freq+= span;
    if (status & EVT_DOWN) freq-= span;
  }
  if (freq > STOP_MAX || freq < START_MIN) return;
  set_sweep_frequency(mode, freq);
}

#define STEPRATIO 0.2
static void
ui_process_normal_lever(void)
{
  int status = btn_check();
  if (status != 0) {
    if (status & EVT_BUTTON_SINGLE_CLICK) {
      ui_mode_menu();
    } else {
      switch (uistat.lever_mode) {
      case LM_MARKER: lever_move_marker(status);   break;
      case LM_SEARCH: lever_search_marker(status); break;
      case LM_CENTER: lever_move(status, FREQ_IS_STARTSTOP() ? ST_START : ST_CENTER); break;
      case LM_SPAN:
        if (FREQ_IS_CW()) 
          lever_zoom_time(status);
        else
          lever_move(status, FREQ_IS_STARTSTOP() ? ST_STOP : ST_SPAN);
        break;
      }
    }
  }
}

#ifdef __LISTEN__
bool
ui_process_listen_lever(void)
{
  int status = btn_check();
  if (status != 0) {
    if (status & EVT_BUTTON_SINGLE_CLICK) {
      return false;
    } else {
      lever_move_marker(status);
    }
  }
  return true;
}
#endif

static void
ui_process_menu_lever(void)
{
  // Flag show, can close menu if user come out from it
  // if false user must select some thing
  const menuitem_t *menu = menu_stack[menu_current_level];
  int status = btn_check();
  if (status == 0) return;
  if (selection >=0 && status & EVT_BUTTON_SINGLE_CLICK) {
    menu_invoke(selection);
    return;
  }
  uint16_t count = current_menu_get_count();
  do {
    uint32_t mask = 1<<selection;
    if (status & EVT_UP  ) selection++;
    if (status & EVT_DOWN) selection--;
    // not close if type = form menu
    if ((uint16_t)selection >= count && !(menu[0].type & MT_FORM)){
      ui_mode_normal();
      return;
    }
    ensure_selection();
    draw_menu_mask(mask|(1<<selection));
    chThdSleepMilliseconds(100); // Add delay for not move so fast in menu
  } while ((status = btn_wait_release()) != 0);
  return;
}

static int
keypad_click(int key)
{
  int c = keypads[key].c;
  if ((c >= KP_X1 && c <= KP_G) || c == KP_m || c == KP_u || c == KP_n) {
#if 0
    float scale = 1.0;
    if (c >= KP_X1 && c <= KP_G) {
      int n = c - KP_X1;
      while (n-- > 0)
        scale *= 1000.0;
    } else if (c == KP_m) {
      scale /= 1000.0;
    } else if (c == KP_u) {
      scale /= 1000000.0;
    } else if (c == KP_n) {
      scale /= 1000000000.0;
    }
    /* numeric input done */
    uistat.value = my_atof(kp_buf) * scale;
#else
    char modifier = 0;
    if (c == KP_K) modifier = 'k';
    else if (c == KP_M) modifier = 'M';
    else if (c == KP_G) modifier = 'G';
    else if (c == KP_m) modifier = 'm';
    else if (c == KP_u) modifier = 'u';
    else if (c == KP_n) modifier = 'n';
    if (modifier) kp_buf[kp_index++] = modifier;
    kp_buf[kp_index++] = 0;
    uistat.value = my_atof(kp_buf);
    uistat.freq_value = my_atoui(kp_buf);
#endif
    set_numeric_value();
    return KP_DONE;
  } else if (c <= 9 && kp_index < NUMINPUT_LEN) {
    kp_buf[kp_index++] = '0' + c;
  } else if (c>=KP_0) {
    kp_buf[kp_index++] = keypad_scale_text[c-KP_0][0];
    if (c >=KP_10)
      kp_buf[kp_index++] = '0';
    if (c >=KP_100)
      kp_buf[kp_index++] = '0';
  } else if (c == KP_PERIOD && kp_index < NUMINPUT_LEN) {
    // check period in former input
    int j;
    for (j = 0; j < kp_index && kp_buf[j] != '.'; j++)
      ;
    // append period if there are no period
    if (kp_index == j)
      kp_buf[kp_index++] = '.';
  } else if (c == KP_MINUS && kp_index < NUMINPUT_LEN) {
    if (kp_index == 0)
      kp_buf[kp_index++] = '-';
    else {
      // always allow sign change, even when not on first position
      if (kp_buf[0] == '-') {
        kp_index = 0;
        do {
          kp_buf[kp_index] = kp_buf[kp_index+1];
          kp_index++;
        } while (kp_buf[kp_index]);
      } else {
        int j = kp_index;
        do {
          kp_buf[j+1] = kp_buf[j];
          j--;
        } while (j >= 0);
        kp_buf[0] = '-';
        kp_index++;
      }

    }
  } else if (c == KP_BS) {
    if (kp_index == 0) {
      return KP_CANCEL;
    }
    --kp_index;
  }
  kp_buf[kp_index] = '\0';
  draw_numeric_input(kp_buf);
  return KP_CONTINUE;
}

static int
keypad_apply_touch(void)
{
  int touch_x, touch_y;
  int i = 0;

  touch_position(&touch_x, &touch_y);

  while (keypads[i].c >= 0) {
    int x = KP_GET_X(keypads[i].x);
    int y = KP_GET_Y(keypads[i].y);
    if (x < touch_x && touch_x < x+KP_WIDTH && y < touch_y && touch_y < y+KP_HEIGHT) {
      uint32_t mask = (1<<i)|(1<<selection);
      // draw focus
      selection = i;
      draw_keypad(mask);
      touch_wait_release();
      // erase focus
      selection = -1;
      draw_keypad((1<<i));
      return i;
    }
    i++;
  }
  return -1;
}

static void
ui_process_keypad(void)
{
  int status;
  kp_index = 0;
  int keypads_last_index;
  for (keypads_last_index = 0; keypads[keypads_last_index+1].c >= 0; keypads_last_index++)
    ;
  while (TRUE) {
    status = btn_check();
    if (status & (EVT_UP|EVT_DOWN)) {
      int s = status;
      do {
        uint32_t mask = (1<<selection);
        if (s & EVT_UP)
          if (--selection < 0)
            selection = keypads_last_index;
        if (s & EVT_DOWN)
          if (++selection > keypads_last_index)
            selection = 0;
        draw_keypad(mask|(1<<selection));
        chThdSleepMilliseconds(100);
      } while ((s = btn_wait_release()) != 0);
    }

    if (status == EVT_BUTTON_SINGLE_CLICK) {
      if (keypad_click(selection))
        /* exit loop on done or cancel */
        break;
    }

    if (touch_check() == EVT_TOUCH_PRESSED) {
      int key = keypad_apply_touch();
      if (key >= 0 && keypad_click(key))
        /* exit loop on done or cancel */
        break;
    }
  }
  kp_help_text = NULL;
  redraw_frame();
  if (current_menu_is_form()) {
    ui_mode_menu(); //Reactivate menu after keypad
    selection = -1;
  } else {
    ui_mode_normal();
  }
  //redraw_all();
}

static void
ui_process_lever(void)
{
  switch (ui_mode) {
  case UI_NORMAL:
    ui_process_normal_lever();
    break;
  case UI_MENU:
    ui_process_menu_lever();
    break;
//  case UI_KEYPAD:
//    ui_process_keypad();
//    break;
  }
}

static void
drag_marker(int t, int m)
{
  /* wait touch release */
  do {
    int touch_x, touch_y;
    int index;
    touch_position(&touch_x, &touch_y);
    touch_x -= OFFSETX;
    touch_y -= OFFSETY;
    index = search_nearest_index(touch_x, touch_y, t);
    if (index >= 0) {
      markers[m].index = index;
      markers[m].frequency = getFrequency(index);
      redraw_marker(m);
    }
  } while (touch_check()!= EVT_TOUCH_RELEASED);
}

static int
touch_pickup_marker(int touch_x, int touch_y)
{
  int m, t;
  touch_x -= OFFSETX;
  touch_y -= OFFSETY;

  int i = MARKER_INVALID, mt;
  int min_dist = MARKER_PICKUP_DISTANCE * MARKER_PICKUP_DISTANCE;
  // Search closest marker to touch position
  for (t = 0; t < TRACES_MAX; t++) {
    if (IS_TRACE_DISABLE(t))
      continue;
    for (m = 0; m < MARKERS_MAX; m++) {
      if (!markers[m].enabled)
        continue;
      // Get distance to marker from touch point
      int dist = distance_to_index(t, markers[m].index, touch_x, touch_y);
      if (dist < min_dist) {
        min_dist = dist;
        i  = m;
        mt = t;
      }
    }
  }
  // Marker not found
  if (i == MARKER_INVALID)
    return FALSE;
  // Marker found, set as active and start drag it
  if (active_marker != i) {
    previous_marker = active_marker;
    active_marker = i;
  }
  // Disable tracking
  markers[i].mtype &= ~M_TRACKING;    // Disable tracking when dragging marker
  // Leveler mode = marker move
  select_lever_mode(LM_MARKER);
  // select trace
  uistat.current_trace = mt;
  // drag marker until release
  drag_marker(mt, i);
  return TRUE;
}

static int touch_quick_menu(int touch_x, int touch_y)
{
  if (touch_x <OFFSETX)
  {
    touch_wait_release();
    return invoke_quick_menu(touch_y);
  }
  return FALSE;
}

#ifdef __USE_SD_CARD__
//*******************************************************************************************
// Bitmap file header for LCD_WIDTH x LCD_HEIGHT image 16bpp (v4 format allow set RGB mask)
//*******************************************************************************************
#define BMP_UINT32(val)  ((val)>>0)&0xFF, ((val)>>8)&0xFF, ((val)>>16)&0xFF, ((val)>>24)&0xFF
#define BMP_H1_SIZE      (14)                        // BMP header 14 bytes
#define BMP_V4_SIZE      (56)                        // v4  header 56 bytes
#define BMP_HEAD_SIZE    (BMP_H1_SIZE + BMP_V4_SIZE) // Size of all headers
#define BMP_SIZE         (2*LCD_WIDTH*LCD_HEIGHT)    // Bitmap size = 2*w*h
#define BMP_FILE_SIZE    (BMP_SIZE + BMP_HEAD_SIZE)  // File size = headers + bitmap
static const uint8_t bmp_header_v4[14+56] = {
// BITMAPFILEHEADER (14 byte size)
  0x42, 0x4D,                // BM signature
  BMP_UINT32(BMP_FILE_SIZE), // File size (h + v4 + bitmap)
  0x00, 0x00,                // reserved
  0x00, 0x00,                // reserved
  BMP_UINT32(BMP_HEAD_SIZE), // Size of all headers (h + v4)
// BITMAPINFOv4 (56 byte size)
  BMP_UINT32(BMP_V4_SIZE),   // Data offset after this point (v4 size)
  BMP_UINT32(LCD_WIDTH),     // Width
  BMP_UINT32(LCD_HEIGHT),    // Height
  0x01, 0x00,                // Planes
  0x10, 0x00,                // 16bpp
  0x03, 0x00, 0x00, 0x00,    // Compression (BI_BITFIELDS)
  BMP_UINT32(BMP_SIZE),      // Bitmap size (w*h*2)
  0xC4, 0x0E, 0x00, 0x00,    // x Resolution (96 DPI = 96 * 39.3701 inches per metre = 0x0EC4)
  0xC4, 0x0E, 0x00, 0x00,    // y Resolution (96 DPI = 96 * 39.3701 inches per metre = 0x0EC4)
  0x00, 0x00, 0x00, 0x00,    // Palette size
  0x00, 0x00, 0x00, 0x00,    // Palette used
// Extend v4 header data (color mask for RGB565)
  0x00, 0xF8, 0x00, 0x00,    // R mask = 0b11111000 00000000
  0xE0, 0x07, 0x00, 0x00,    // G mask = 0b00000111 11100000
  0x1F, 0x00, 0x00, 0x00,    // B mask = 0b00000000 00011111
  0x00, 0x00, 0x00, 0x00     // A mask = 0b00000000 00000000
};

FRESULT open_file(char *ext)
{
  FRESULT res = f_mount(fs_volume, "", 1);
  // fs_volume, fs_file and fs_filename stored at end of spi_buffer!!!!!
//  shell_printf("Mount = %d\r\n", res);
  if (res != FR_OK)
    return res;
#if FF_USE_LFN >= 1
  uint32_t tr = rtc_get_tr_bcd(); // TR read first
  uint32_t dr = rtc_get_dr_bcd(); // DR read second
  plot_printf(fs_filename, FF_LFN_BUF, "SA_%06x_%06x.%s", dr, tr, ext);
#else
  plot_printf(fs_filename, FF_LFN_BUF, "%08x.%s", rtc_get_FAT(), ext);
#endif
  res = f_open(fs_file, fs_filename, FA_CREATE_ALWAYS | FA_READ | FA_WRITE);
 return res;
}

void close_file(FRESULT res)
{
  if (res == FR_OK)
    res = f_close(fs_file);
//  time = chVTGetSystemTimeX() - time;
//  shell_printf("Total time: %dms (write %d byte/sec)\r\n", time/10, (LCD_WIDTH*LCD_HEIGHT*sizeof(uint16_t)+sizeof(bmp_header_v4))*10000/time);
  drawMessageBox("Save:", res == FR_OK ? fs_filename : "  Write failed  ", 2000);
  redraw_request|= REDRAW_AREA;
}
static bool
made_screenshot(int touch_x, int touch_y)
{
  int y, i;
  UINT size;
  if (touch_y < SD_CARD_START || touch_y > SD_CARD_START + 20 || touch_x > OFFSETX)
    return FALSE;
  ili9341_set_background(LCD_BG_COLOR);
  ili9341_fill(4, SD_CARD_START, 16, 16);
  touch_wait_release();
//  uint32_t time = chVTGetSystemTimeX();
//  shell_printf("Screenshot\r\n");
  FRESULT res = open_file("bmp");
  uint16_t *buf = (uint16_t *)spi_buffer;
//  shell_printf("Open %s, result = %d\r\n", fs_filename, res);
  if (res == FR_OK){
    res = f_write(fs_file, bmp_header_v4, sizeof(bmp_header_v4), &size);
    for (y = LCD_HEIGHT-1; y >= 0 && res == FR_OK; y--) {
      ili9341_read_memory(0, y, LCD_WIDTH, 1, buf);
      for (i = 0; i < LCD_WIDTH; i++)
        buf[i] = __REVSH(buf[i]); // swap byte order (example 0x10FF to 0xFF10)
      res = f_write(fs_file, buf, LCD_WIDTH*sizeof(uint16_t), &size);
    }
//   res = f_close(fs_file);
//    shell_printf("Close %d\r\n", res);
//    testLog();
  }
  close_file(res);
  return TRUE;
}

void save_to_sd(int mask)
{
  FRESULT res = open_file("csv");
  UINT size;
  if (res == FR_OK) {
    for (int i = 0; i < sweep_points; i++) {
      char *buf = (char *)spi_buffer;
      if (mask & 1) buf += plot_printf(buf, 100, "%U, ", getFrequency(i));
      if (mask & 2) buf += plot_printf(buf, 100, "%f ", value(measured[TRACE_ACTUAL][i]));
      if (mask & 4) buf += plot_printf(buf, 100, "%f ", value(measured[TRACE_STORED][i]));
      if (mask & 8) buf += plot_printf(buf, 100, "%f", value(measured[TRACE_TEMP][i]));
      buf += plot_printf(buf, 100, "\r\n");
      res = f_write(fs_file, (char *)spi_buffer, buf - (char *)spi_buffer, &size);
      if (res != FR_OK)
        break;
    }
  }
  close_file(res);
}

#endif

static int
touch_lever_mode_select(int touch_x, int touch_y)
{
  if (touch_y > HEIGHT) {
    touch_wait_release();
    // Touch on left frequency field side
    if (touch_x < FREQUENCIES_XPOS2 - 50) {
      if (uistat.lever_mode == LM_CENTER){
        ui_mode_keypad(FREQ_IS_CENTERSPAN() ? KM_CENTER : KM_START);
        return TRUE;
      }
    }
    else if (touch_x <  FREQUENCIES_XPOS2 + 50) {
      // toggle frequency mode start/stop <=> center/span
      setting.freq_mode^= FREQ_MODE_CENTER_SPAN;
      redraw_request |= REDRAW_FREQUENCY;
      return true;
    }
    else if (uistat.lever_mode == LM_SPAN) {
      ui_mode_keypad(FREQ_IS_CW() ? KM_SWEEP_TIME : (FREQ_IS_CENTERSPAN() ? KM_SPAN : KM_STOP));
      return TRUE;
    }
    select_lever_mode(touch_x < FREQUENCIES_XPOS2 ? LM_CENTER : LM_SPAN);
    return TRUE;
  }
  if (touch_x < OFFSETX)
  {
    return invoke_quick_menu(touch_y);
  }
  else if (touch_y < 25) {
#ifdef __VNA__
    if (touch_x < FREQUENCIES_XPOS2 && get_electrical_delay() != 0.0) {
      select_lever_mode(LM_EDELAY);
    } else {
#endif
      select_lever_mode(LM_MARKER);
#ifdef __VNA__
      }
#endif
    return TRUE;
  }
  return FALSE;
}

static int
touch_marker_select(int touch_x, int touch_y)
{
  int selected_marker = 0;
  if (current_menu_is_form() || touch_x > LCD_WIDTH-MENU_BUTTON_WIDTH || touch_x < 25 || touch_y > 30)
    return FALSE;
  if (touch_y > 15)
    selected_marker = 2;
  selected_marker += (touch_x >150 ? 1 : 0);
  for (int i = 0; i < MARKERS_MAX; i++) {
    if (markers[i].enabled) {
      if (selected_marker == 0) {
        if (active_marker == i) {
          extern const menuitem_t menu_marker_modify[];
          touch_wait_release();
          selection = -1;
          menu_current_level = 0;
          menu_push_submenu(menu_marker_modify);
          break;
        }
        active_marker = i;
        redraw_marker(active_marker);
        break;
      }
      selected_marker --;
    }
  }
  if (touch_y < 25) {
#ifdef __VNA__
    if (touch_x < FREQUENCIES_XPOS2 && get_electrical_delay() != 0.0) {
      select_lever_mode(LM_EDELAY);
    } else {
#endif
      select_lever_mode(LM_MARKER);
#ifdef __VNA__
      }
#endif
    return TRUE;
  }
  return FALSE;
}

static
void ui_process_touch(void)
{
  int touch_x, touch_y;
  int status = touch_check();
  if (status == EVT_TOUCH_PRESSED || status == EVT_TOUCH_DOWN) {
    touch_position(&touch_x, &touch_y);
    switch (ui_mode) {
    case UI_NORMAL:
#ifdef __USE_SD_CARD__
      if (made_screenshot(touch_x, touch_y))
        break;
#endif
      if (touch_quick_menu(touch_x, touch_y))
        break;
      // Try drag marker
      if (touch_pickup_marker(touch_x, touch_y))
        break;
      if (touch_marker_select(touch_x, touch_y))
        break;
      // Try select lever mode (top and bottom screen)
      if (touch_lever_mode_select(touch_x, touch_y)) {
//        touch_wait_release();
        break;
      }

      // switch menu mode after release
      touch_wait_release();
      selection = -1; // hide keyboard mode selection
      ui_mode_menu();
      break;
    case UI_MENU:
      menu_apply_touch(touch_x, touch_y);
      break;
    }
  }
}

static uint16_t previous_button_state = 0;

void
ui_process(void)
{
  int button_state = READ_PORT() & BUTTON_MASK;
  if (ui_mode == UI_NORMAL && current_menu_is_form()) {     //   Force into menu mode
    selection = -1; // hide keyboard mode selection
    ui_mode_menu();
  }
  if ((operation_requested&OP_LEVER) || previous_button_state != button_state) {
    ui_process_lever();
    previous_button_state = button_state;
    operation_requested = OP_NONE;
  }
  if (operation_requested&OP_TOUCH) {
   ui_process_touch();
    operation_requested = OP_NONE;
  }
  touch_start_watchdog();
}

/* Triggered when the button is pressed or released. The LED4 is set to ON.*/
static void extcb1(EXTDriver *extp, expchannel_t channel)
{
  (void)extp;
  (void)channel;
  if (channel != 9)
    operation_requested|=OP_LEVER;
  // cur_button = READ_PORT() & BUTTON_MASK;
}

static const EXTConfig extcfg = {
  {
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_RISING_EDGE | EXT_CH_MODE_AUTOSTART | EXT_MODE_GPIOA, extcb1},
    {EXT_CH_MODE_RISING_EDGE | EXT_CH_MODE_AUTOSTART | EXT_MODE_GPIOA, extcb1},
    {EXT_CH_MODE_RISING_EDGE | EXT_CH_MODE_AUTOSTART | EXT_MODE_GPIOA, extcb1},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
#ifdef __WAIT_CTS_WHILE_SLEEPING__
    {EXT_CH_MODE_RISING_EDGE | EXT_CH_MODE_AUTOSTART | EXT_MODE_GPIOB, extcb1},
#else
    {EXT_CH_MODE_DISABLED, NULL},
#endif
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL},
    {EXT_CH_MODE_DISABLED, NULL}
  }
};

void
handle_touch_interrupt(void)
{
  operation_requested|= OP_TOUCH;
}

void
ui_init()
{
  adc_init();
  // Activates the EXT driver 1.
  extStart(&EXTD1, &extcfg);
  // Init touch subsystem
  touch_init();
}

void wait_user(void)
{
  touch_wait_released();
#if 0
  operation_requested = OP_NONE;
  while (true) {
    if (operation_requested & OP_TOUCH)
      break;
    if (operation_requested & OP_LEVER)
      break;
  }
#endif
}

int check_touched(void)
{
  int touched = false;
  if (touch_check() == EVT_TOUCH_RELEASED)
    touched = true;
  return touched;
}



#pragma GCC pop_options
