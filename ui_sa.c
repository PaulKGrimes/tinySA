/* Copyright (c) 2020, Erik Kaashoek erik@kaashoek.com
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


#define FORM_ICON_WIDTH      16
#define FORM_ICON_HEIGHT     16

static const uint8_t left_icons [] =
{
#define I_EMPTY   (0*16)
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),

#define I_HIGH_INPUT (1*16)
                              // +----------------+
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000001100000), // |         **     |
  _BMP16(0b0000000000111001), // |          ***  *|
  _BMP16(0b0000111111111111), // |   *************|
  _BMP16(0b0000000000111001), // |          ***  *|
  _BMP16(0b0000000001100000), // |         **     |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
                              // +----------------+
#define I_LOW_INPUT (2*16)
                              // +----------------+
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000001), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000001100000), // |         **     |
  _BMP16(0b0000000000111001), // |         ****  *|
  _BMP16(0b0000111111111111), // |   *************|
  _BMP16(0b0000000000111001), // |         ****  *|
  _BMP16(0b0000000001100000), // |         **     |
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0000000000000000), // |                |
                              // +----------------+
#define I_LOW_OUTPUT (3*16)
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000110000000),
  _BMP16(0b0000011100000001),
  _BMP16(0b0000111111111111),
  _BMP16(0b0000011100000001),
  _BMP16(0b0000000110000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),

#define I_HIGH_OUTPUT (4*16)
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000110000000),
  _BMP16(0b0000011100000001),
  _BMP16(0b0000111111111111),
  _BMP16(0b0000011100000001),
  _BMP16(0b0000000110000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000001),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),

#define I_CONNECT (5*16)
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000110000),
  _BMP16(0b0000000000111101),
  _BMP16(0b0000001111111111),
  _BMP16(0b0000010000111101),
  _BMP16(0b0000100000110000),
  _BMP16(0b0001000000000000),
  _BMP16(0b0001000000000000),
  _BMP16(0b0000100000110000),
  _BMP16(0b0000010000111101),
  _BMP16(0b0000001111111111),
  _BMP16(0b0000000000111101),
  _BMP16(0b0000000000110000),
  _BMP16(0b0000000000000000),
  _BMP16(0b0000000000000000),
};

const uint8_t right_icons [] =
{
#define I_SA    0
                              // +----------------+
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0111111111111111), // | ***************|
  _BMP16(0b0100000000000001), // | *             *|
  _BMP16(0b1100000000000001), // |**             *|
  _BMP16(0b1100000000000001), // |**  *          *|
  _BMP16(0b1100000000000001), // |**  *          *|
  _BMP16(0b0100100000000001), // | *  *   *      *|
  _BMP16(0b0100100000000001), // | *  *   *      *|
  _BMP16(0b0100101010001001), // | *  *   *   *  *|
  _BMP16(0b0100101010101001), // | *  * * *   *  *|
  _BMP16(0b1100101010101001), // |**  * * * * *  *|
  _BMP16(0b1101111111111101), // |**  * * * * *  *|
  _BMP16(0b1100000000000001), // |** *********** *|
  _BMP16(0b0100000000000001), // | *             *|
  _BMP16(0b0111111111111111), // | ***************|
  _BMP16(0b0000000000000000), // |                |
                              // +----------------+

#define I_GEN   1
                              // +----------------+
  _BMP16(0b0000000000000000), // |                |
  _BMP16(0b0111111111111111), // | ***************|
  _BMP16(0b0100000000000001), // | *             *|
  _BMP16(0b1100000000000001), // |**             *|
  _BMP16(0b1100111110001101), // |**  *****   ** *|
  _BMP16(0b1100100010001001), // |**  *   *   *  *|
  _BMP16(0b0100100010001001), // | *  *   *   *  *|
  _BMP16(0b0100100010001001), // | *  *   *   *  *|
  _BMP16(0b0100100010001001), // | *  *   *   *  *|
  _BMP16(0b0100100010001001), // | *  *   *   *  *|
  _BMP16(0b1100100010001001), // |**  *   *   *  *|
  _BMP16(0b1101100011111001), // |** **   *****  *|
  _BMP16(0b1100000000000001), // |**             *|
  _BMP16(0b0100000000000001), // | *             *|
  _BMP16(0b0111111111111111), // | ***************|
  _BMP16(0b0000000000000000), // |                |
                              // +----------------+
#define I_CONFIG 2
  _BMP16(0b0000000000000000),
  _BMP16(0b0111111111111111),
  _BMP16(0b0100000000000001),
  _BMP16(0b1100000010000001),
  _BMP16(0b1100001111000001),
  _BMP16(0b1100011110001001),
  _BMP16(0b0100011100011101),
  _BMP16(0b0100011110111001),
  _BMP16(0b0100001111111001),
  _BMP16(0b0100011111110001),
  _BMP16(0b1100111110000001),
  _BMP16(0b1101111100000001),
  _BMP16(0b1100111000000001),
  _BMP16(0b0100000000000001),
  _BMP16(0b0111111111111111),
  _BMP16(0b0000000000000000),

#define I_SINUS 3
  _BMP16(0b0000000000000000),
  _BMP16(0b0111111111111111),  // 1
  _BMP16(0b0100000000000001),  // 2
  _BMP16(0b1100000000000001),  // 3
  _BMP16(0b1100000000110001),  // 4
  _BMP16(0b1100000001001001),  // 5
  _BMP16(0b0100000010000101),  // 6
  _BMP16(0b0101000010000101),  // 7
  _BMP16(0b0101000010000101),  // 8
  _BMP16(0b0101000010000001),  // 9
  _BMP16(0b1100100100000001),  //10
  _BMP16(0b1100011000000001),  //11
  _BMP16(0b1100000000000001),  //12
  _BMP16(0b0100000000000001),  //13
  _BMP16(0b0111111111111111),  //14
  _BMP16(0b0000000000000000),
};

#define KP_X(x) (48*(x) + 2 + (LCD_WIDTH-BUTTON_WIDTH-192))
#define KP_Y(y) (48*(y) + 2)


#define KP_PERIOD 10
#define KP_MINUS 11
#define KP_X1 12
#define KP_K 13
#define KP_M 14
#define KP_G 15
#define KP_BS 16
#define KP_INF 17
#define KP_DB 18
#define KP_PLUSMINUS 19
#define KP_KEYPAD 20
#define KP_m 21
#define KP_u 22
#define KP_n 23
#define KP_p 24

#define KP_0    31
#define KP_1    32
#define KP_2    33
#define KP_5    34
#define KP_10   35
#define KP_20   36
#define KP_50   37
#define KP_100  38
#define KP_200  39
#define KP_500  40


typedef struct {
  uint8_t x:4;
  uint8_t y:4;
  int8_t  c;
} keypads_t;

static const keypads_t *keypads;

// 7 8 9 G
// 4 5 6 M
// 1 2 3 k
// 0 . < x

static const keypads_t keypads_freq[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 3 },
  { 0, 1, 4 },
  { 1, 1, 5 },
  { 2, 1, 6 },
  { 0, 0, 7 },
  { 1, 0, 8 },
  { 2, 0, 9 },
  { 3, 0, KP_G },
  { 3, 1, KP_M },
  { 3, 2, KP_K },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};

// 7 8 9
// 4 5 6
// 1 2 3
// 0 . < x

static const keypads_t keypads_positive[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 3 },
  { 0, 1, 4 },
  { 1, 1, 5 },
  { 2, 1, 6 },
  { 0, 0, 7 },
  { 1, 0, 8 },
  { 2, 0, 9 },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};

// 100 200 500 n
// 10  20  50  u
// 1   2   5   m
// 0   .   <   x

static const keypads_t keypads_pos_unit[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 5 },
  { 0, 1, KP_10 },
  { 1, 1, KP_20 },
  { 2, 1, KP_50 },
  { 0, 0, KP_100 },
  { 1, 0, KP_200 },
  { 2, 0, KP_500 },
  { 3, 0, KP_n },
  { 3, 1, KP_u },
  { 3, 2, KP_m },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};

// 7 8 9 m
// 4 5 6 u
// 1 2 3 -
// 0 . < x

static const keypads_t keypads_plusmin_unit[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 3 },
  { 0, 1, 4 },
  { 1, 1, 5 },
  { 2, 1, 6 },
  { 0, 0, 7 },
  { 1, 0, 8 },
  { 2, 0, 9 },
  { 3, 0, KP_u},
  { 3, 1, KP_m},
  { 3, 2, KP_MINUS },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};
// 7 8 9
// 4 5 6
// 1 2 3 -
// 0 . < x

static const keypads_t keypads_plusmin[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 3 },
  { 0, 1, 4 },
  { 1, 1, 5 },
  { 2, 1, 6 },
  { 0, 0, 7 },
  { 1, 0, 8 },
  { 2, 0, 9 },
  { 3, 0, KP_u},
  { 3, 1, KP_m},
  { 3, 2, KP_MINUS },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};

// 7 8 9
// 4 5 6
// 1 2 3 m
// 0 . < x
static const keypads_t keypads_time[] = {
  { 1, 3, KP_PERIOD },
  { 0, 3, 0 },
  { 0, 2, 1 },
  { 1, 2, 2 },
  { 2, 2, 3 },
  { 0, 1, 4 },
  { 1, 1, 5 },
  { 2, 1, 6 },
  { 0, 0, 7 },
  { 1, 0, 8 },
  { 2, 0, 9 },
//  { 3, 0, KP_n},
//  { 3, 1, KP_u},
  { 3, 2, KP_m },
  { 3, 3, KP_X1 },
  { 2, 3, KP_BS },
  { 0, 0, -1 }
};

enum {
  KM_START, KM_STOP, KM_CENTER, KM_SPAN, KM_CW, // These must be first to share common help text
  //#5
  KM_REFLEVEL, KM_SCALE, KM_ATTENUATION, KM_ACTUALPOWER, KM_IF,
  // #10
  KM_SAMPLETIME, KM_LOWOUTLEVEL, KM_DECAY, KM_NOISE,
#ifdef TINYSA4
  KM_30MHZ, 
#else
  KM_10MHZ, 
#endif
  // #15
  KM_REPEAT, KM_EXT_GAIN, KM_TRIGGER, KM_LEVELSWEEP, KM_SWEEP_TIME,
  // #20
  KM_OFFSET_DELAY, KM_FAST_SPEEDUP, KM_GRIDLINES, KM_MARKER, KM_MODULATION,
  // #25
  KM_HIGHOUTLEVEL,
#ifdef TINYSA4
  KM_COR_AM,  KM_COR_WFM, KM_COR_NFM,
  KM_IF2,
  // #30
  KM_R,KM_MOD,KM_CP,
#endif
  KM_ATTACK,
#ifdef __ULTRA__
  KM_ULTRA_START,
#endif
#ifdef TINYSA4
  KM_EXP_AVER,
#endif
  KM_LEVEL,
#ifdef __LIMITS__
  KM_LIMIT_FREQ, KM_LIMIT_LEVEL,
#endif
  KM_MARKER_TIME,
  // #35
  KM_VAR,
#ifdef __NOISE_FIGURE__
  KM_NF,
#endif
  KM_LINEAR_SCALE,
#ifdef TINYSA4
  KM_DIRECT_START,
  KM_DIRECT_STOP,
#ifdef __USE_RTC__
  KM_RTC_DATE,
  KM_RTC_TIME,
#endif
#endif
  KM_NONE // always at enum end
};

static const struct {
  const keypads_t *keypad_type;
  char * name;
} keypads_mode_tbl[KM_NONE] = {
[KM_START]        = {keypads_freq        , "START"}, // start
[KM_STOP]         = {keypads_freq        , "STOP"}, // stop
[KM_CENTER]       = {keypads_freq        , "CENTER"}, // center
[KM_SPAN]         = {keypads_freq        , "SPAN"}, // span
[KM_CW]           = {keypads_freq        , "FREQ"}, // cw freq
[KM_REFLEVEL]     = {keypads_plusmin_unit, "REF\nLEVEL"}, // reflevel #5
[KM_SCALE]        = {keypads_pos_unit    , "SCALE"}, // scale
[KM_ATTENUATION]  = {keypads_positive    , "ATTENUATE"}, // attenuation
[KM_ACTUALPOWER]  = {keypads_plusmin_unit, "ACTUAL\nPOWER"}, // actual power
[KM_IF]           = {keypads_freq        , "IF"}, // IF
[KM_SAMPLETIME]   = {keypads_positive    , "SAMPLE\nDELAY"}, // sample delay #10
[KM_LOWOUTLEVEL]  = {keypads_plusmin     , "OUTPUT\nLEVEL"},    // KM_LOWOUTLEVEL
[KM_DECAY]        = {keypads_positive    , "DECAY"},    // KM_DECAY
[KM_NOISE]        = {keypads_positive    , "NOISE\nLEVEL"},    // KM_NOISE
#ifdef TINYSA4
[KM_30MHZ]        = {keypads_freq        , "FREQ"},    // KM_30MHz
#else
[KM_10MHZ]        = {keypads_freq        , "FREQ"},    // KM_10MHz
#endif
[KM_REPEAT]       = {keypads_positive    , "SAMPLE\nREPEAT"},    // KM_REPEA #15
[KM_EXT_GAIN]     = {keypads_plusmin     , "EXT\nGAIN"},    // KM_EXT_GAIN
[KM_TRIGGER]      = {keypads_plusmin_unit, "TRIGGER\nLEVEL"},    // KM_TRIGGER
[KM_LEVELSWEEP]   = {keypads_plusmin     , "LEVEL\nSWEEP"},    // KM_LEVELSWEEP
[KM_SWEEP_TIME]   = {keypads_time        , "SWEEP\nSECONDS"},    // KM_SWEEP_TIME
[KM_OFFSET_DELAY] = {keypads_positive    , "OFFSET\nDELAY"}, // KM_OFFSET_DELAY #20
[KM_FAST_SPEEDUP] = {keypads_positive    , "FAST\nSPEEDUP"}, // KM_FAST_SPEEDUP
[KM_GRIDLINES]    = {keypads_positive    , "MINIMUM\nGRIDLINES"}, // KM_GRIDLINES
[KM_MARKER]       = {keypads_freq        , "MARKER\nFREQ"}, // KM_MARKER
[KM_MODULATION]   = {keypads_freq        , "MODULATION\nFREQ"}, // KM_MODULATION
[KM_HIGHOUTLEVEL] = {keypads_plusmin     , "OUTPUT\nLEVEL"},    // KM_HIGHOUTLEVEL #25
#ifdef TINYSA4
[KM_COR_AM]       = {keypads_plusmin     , "COR\nAM"},    // KM_COR_AM
[KM_COR_WFM]      = {keypads_plusmin     , "COR\nWFM"},    // KM_COR_WFM
[KM_COR_NFM]      = {keypads_plusmin     , "COR\nNFM"},    // KM_COR_NFM
[KM_IF2]          = {keypads_freq        , "IF2"}, // KM_IF2
[KM_R]            = {keypads_plusmin     , "R"}, // KM_R    #30
[KM_MOD]          = {keypads_positive    , "MODULO"}, // KM_MOD
[KM_CP]           = {keypads_positive    , "CP"}, // KM_CP
#endif
[KM_ATTACK]       = {keypads_positive    , "ATTACK"},    // KM_ATTACK
#ifdef __ULTRA__
[KM_ULTRA_START]  = {keypads_freq        , "ULTRA\nSTART"}, // KM_ULTRA_START
#endif
#ifdef TINYSA4
[KM_EXP_AVER]     = {keypads_positive    , "EXPONENTIAL\nAVERAGING"}, //KM_EXP_AVER
#endif
[KM_LEVEL]        = {keypads_plusmin     , "LEVEL"}, // KM_LEVEL
#ifdef __LIMITS__
[KM_LIMIT_FREQ]   = {keypads_freq         , "FREQ"},  // KM_LIMIT_FREQ
[KM_LIMIT_LEVEL]  = {keypads_plusmin_unit , "LEVEL"},  // KM_LIMIT_LEVEL
#endif
[KM_MARKER_TIME]  = {keypads_time        , "MARKER\nTIME"}, // KM_MARKER_TIME
[KM_VAR]          = {keypads_freq        , "JOG\nSTEP"}, // jog step
#ifdef __NOISE_FIGURE__
[KM_NF]           = {keypads_plusmin     , "NOISE\nFIGURE"}, // noise figure of tinySA
#endif
[KM_LINEAR_SCALE] = {keypads_plusmin     , "SCALE"}, // scale for linear units
#ifdef TINYSA4
[KM_DIRECT_START] = {keypads_freq        , "DIRECT\nSTART"}, // KM_DIRECT_START
[KM_DIRECT_STOP]  = {keypads_freq        , "DIRECT\nSTOP"}, // KM_DIRECT_STOP
#ifdef __USE_RTC__
[KM_RTC_DATE]     = {keypads_positive    , "SET DATE\n YYMMDD"}, // Date
[KM_RTC_TIME]     = {keypads_positive    , "SET TIME\n HHMMSS"}, // Time
#endif
#endif
};

#if 0 // Not used
enum { SL_GENLOW_FREQ, SL_GENHIGH_FREQ, SL_GENLOW_LEVEL, SL_GENHIGH_LEVEL };
ui_slider_t ui_sliders [] =
{
 { KM_CENTER,       true, 0, 1000000,   0,          350000000,  M_GENLOW},
 { KM_CENTER,       true, 0, 1000000,   240000000,  960000000,  M_GENHIGH},
 { KM_LOWOUTLEVEL,  false,0, 1,         -76,        -6,         M_GENLOW},
 { KM_HIGHOUTLEVEL, false,0, 1,         -38,        +6,         M_GENHIGH},
};
#endif


// ===[MENU CALLBACKS]=========================================================
const menuitem_t  menu_lowoutputmode[];
const menuitem_t  menu_highoutputmode[];
const menuitem_t  menu_mode[];
static const menuitem_t  menu_modulation[];
static const menuitem_t  menu_top[];
static const menuitem_t  menu_trace[];
static const menuitem_t  menu_marker_trace[];
static const menuitem_t  menu_subtract_trace[];
#ifdef __LIMITS__
static const menuitem_t  menu_limit_modify[];
static const menuitem_t  menu_limit_select[];
#endif
static const menuitem_t  menu_average[];
static const menuitem_t  menu_reffer[];
static const menuitem_t  menu_sweep_points[];
static const menuitem_t  menu_sweep_points_form[];
static const menuitem_t  menu_modulation[];
static const menuitem_t  menu_marker_ref_select[];
#ifdef __USE_SERIAL_CONSOLE__
static const menuitem_t  menu_connection[];
#endif
//static const menuitem_t  menu_drive_wide[];
#ifdef TINYSA4
static const menuitem_t  menu_settings3[];
static const menuitem_t  menu_curve[];
static const menuitem_t  menu_curve_confirm[];
static const menuitem_t  menu_measure_noise_figure[];
#endif
static const menuitem_t  menu_sweep[];
static const menuitem_t  menu_settings[];
extern bool dirty;
char range_text[20];

static UI_FUNCTION_ADV_CALLBACK(menu_sweep_acb)
{
  (void)data;
  (void)item;
  if (b){
    if (setting.level_sweep != 0 || get_sweep_frequency(ST_SPAN) != 0)  {
      plot_printf(b->text, sizeof b->text, "SW:%3.2fMHz %+ddB %.3Fs",
                  get_sweep_frequency(ST_SPAN) / 1000000.0,
                  (int)setting.level_sweep,
                  setting.sweep_time_us/(float)ONE_SECOND_TIME);
    }
    else
      plot_printf(b->text, sizeof b->text, "SWEEP: OFF");
    return;
  }
  menu_push_submenu(menu_sweep);
}

#ifdef __SWEEP_RESTART__
static UI_FUNCTION_ADV_CALLBACK(menu_restart_acb){
  (void)item;
  (void)data;
  if(b){
    if (setting.sweep) {
      if (current_index >= 0) {
        float current_level = setting.level + ((float)current_index)* setting.level_sweep / (float)sweep_points;
        plot_printf(b->text, sizeof b->text, "STOP %5.3QHz %+.1fdBm", getFrequency(current_index), current_level);
      } else
        plot_printf(b->text, sizeof b->text, "STOP SWEEP");
    }
    else
      plot_printf(b->text, sizeof b->text, "START SWEEP");
    return;
  }
  setting.sweep = !setting.sweep;
  dirty = true;
}
#endif

#ifdef TINYSA4

float local_actual_level;
int current_curve;
int current_curve_index;

static UI_FUNCTION_ADV_CALLBACK(menu_curve_acb)
        {
  (void)item;
  int old_m;
  if (b){
    plot_printf(b->text, sizeof b->text, "%8.3QHz %+4.1fdB",
                config.correction_frequency[current_curve][data],
                config.correction_value[current_curve][data]);
    return;
  }
  switch(current_curve) {
  case CORRECTION_LOW_OUT:
    old_m = setting.mode;
    reset_settings(M_GENLOW);
    set_level(-35);
    set_sweep_frequency(ST_CW, config.correction_frequency[current_curve][data]);
    setting.mute = false;
    perform(false, 0, config.correction_frequency[current_curve][data], false);
    perform(false, 1, config.correction_frequency[current_curve][data], false);
    plot_printf(uistat.text, sizeof uistat.text, "Level of %.3QHz output",
                config.correction_frequency[current_curve][data]);
    kp_help_text = uistat.text;
    kp_buf[0]=0;
    ui_mode_keypad(KM_LEVEL);

    if (kp_buf[0] != 0) {
      float new_offset = (-35.0) - uistat.value + config.correction_value[current_curve][data];        // calculate offset based on difference between measured peak level and known peak level
      if (new_offset > -25 && new_offset < 25) {
        config.correction_value[current_curve][data] = new_offset;
        config_save();
      }
    }
    reset_settings(old_m);
    break;
  case CORRECTION_LNA:
    reset_settings(M_LOW);
    setting.extra_lna = true;
    goto common;
  case CORRECTION_LOW:
    reset_settings(M_LOW);
    common:
    set_sweep_frequency(ST_SPAN,   1000000);
    set_sweep_frequency(ST_CENTER, config.correction_frequency[current_curve][data]);
    setting.step_delay_mode = SD_PRECISE;
    current_curve_index = data;
    menu_push_submenu(menu_curve_confirm);
    break;
  }
}

extern float peakLevel;

UI_FUNCTION_CALLBACK(menu_curve_confirm_cb)
{
  (void)item;
  if (data) {
    float new_offset = local_actual_level - peakLevel + config.correction_value[current_curve][current_curve_index];        // calculate offset based on difference between measured peak level and known peak level
    if (new_offset > -30 && new_offset < 30) {
      config.correction_value[current_curve][current_curve_index] = new_offset;
      config_save();
    }
  }
  menu_move_back(false);
}

float measured_noise_figure;

UI_FUNCTION_CALLBACK(menu_noise_figure_confirm_cb)
{
  (void)item;
  if (data) {
    if (measured_noise_figure > 3 && measured_noise_figure < 15) {
      config.noise_figure = measured_noise_figure;
      config_save();
      nf_gain = 0.00001;                            // almost zero
      set_measurement(M_NF_VALIDATE);               // Continue to validate
      return;
    }
  }
  menu_move_back(false);
}

static UI_FUNCTION_CALLBACK(menu_input_curve_prepare_cb)
{
  (void)item;
  (void)data;
  if (config.low_level_offset == 100)
    return;
  kp_help_text = "Enter actual input level";
  kp_buf[0]=0;
  ui_mode_keypad(KM_LEVEL);
  if (kp_buf[0] != 0) {
    local_actual_level = uistat.value;
    current_curve = CORRECTION_LOW;
    menu_push_submenu(menu_curve);
  }
}

static UI_FUNCTION_CALLBACK(menu_lna_curve_prepare_cb)
{
  (void)item;
  (void)data;
  if (config.low_level_offset == 100)
    return;
  kp_help_text = "Enter actual input level";
  kp_buf[0]=0;
  ui_mode_keypad(KM_LEVEL);
  if (kp_buf[0] != 0) {
    local_actual_level = uistat.value;
    current_curve = CORRECTION_LNA;
    menu_push_submenu(menu_curve);
  }
}

static UI_FUNCTION_CALLBACK(menu_lna_u_curve_prepare_cb)
{
  (void)item;
  (void)data;
  if (config.low_level_offset == 100)
    return;
  kp_help_text = "Enter actual input level";
  kp_buf[0]=0;
  ui_mode_keypad(KM_LEVEL);
  if (kp_buf[0] != 0) {
    local_actual_level = uistat.value;
    current_curve = CORRECTION_LNA_ULTRA;
    menu_push_submenu(menu_curve);
  }
}

static UI_FUNCTION_CALLBACK(menu_ultra_curve_prepare_cb)
{
  (void)item;
  (void)data;
  if (config.low_level_offset == 100)
    return;
  kp_help_text = "Enter actual input level";
  kp_buf[0]=0;
  ui_mode_keypad(KM_LEVEL);
  if (kp_buf[0] != 0) {
    local_actual_level = uistat.value;
    current_curve = CORRECTION_LOW_ULTRA;
    menu_push_submenu(menu_curve);
  }
}

static UI_FUNCTION_CALLBACK(menu_output_curve_prepare_cb)
{
  (void)item;
  (void)data;
  if (config.low_level_output_offset == 100)
    return;
  current_curve = CORRECTION_LOW_OUT;
  menu_push_submenu(menu_curve);
}

#endif

static UI_FUNCTION_ADV_CALLBACK(menu_output_level_acb)
{
  (void)item;
  (void)data;
  if (b){
    return;
  }
  int old_m = setting.mode;
  reset_settings(M_GENLOW);
  set_level(-25);
  set_sweep_frequency(ST_CW, 30000000);
  setting.mute = false;
  perform(false, 0, 30000000, false);
  perform(false, 1, 30000000, false);
  kp_buf[0]=0;
  ui_mode_keypad(KM_LEVEL);
  if (kp_buf[0] != 0) {
    float old_offset = config.low_level_output_offset;
    if (old_offset == 100) old_offset = 0;
    float new_offset = uistat.value - (-25.0) + old_offset;        // calculate offset based on difference between measured peak level and known peak level
    if (uistat.value == 100) new_offset = 100;
    if ((new_offset > -10 && new_offset < 10) || new_offset == 100) {
      config.low_level_output_offset = new_offset;
      config_save();
    }
  }
  reset_settings(old_m);
}



static UI_FUNCTION_ADV_CALLBACK(menu_mode_acb)
{
  (void)data;
  if (b){
    if (item == setting.mode)  {
      b->param_1.text = "Return";
      b->bg = LCD_MENU_ACTIVE_COLOR;
      b->border = BUTTON_BORDER_FALLING | MENU_BUTTON_BORDER;
    }
    else
      b->param_1.text = "Switch";
    return;
  }
  set_mode(item);
//  draw_cal_status();
  switch (item) {
  case 0:
//    if (setting.mode != M_LOW)
//      set_mode(M_LOW);
    menu_move_back(true);
    break;
  case 1:
//    if (setting.mode != M_HIGH)
//      set_mode(M_HIGH);
    menu_move_back(true);
    break;
  case 2:
    menu_push_submenu(menu_lowoutputmode);
    break;
  case 3:
    menu_push_submenu(menu_highoutputmode);
    break;
  }
  redraw_request |= REDRAW_CAL_STATUS;
}

static UI_FUNCTION_ADV_CALLBACK(menu_load_preset_acb)
{
  (void)item;
  if(b){
    setting_t *p = caldata_pointer(data);
    if (p)
      plot_printf(b->text, sizeof(b->text), "%.6FHz\n%.6FHz", (float)p->frequency0, (float)p->frequency1);
    else
      plot_printf(b->text, sizeof(b->text), "EMPTY %d", (int)data);
    return;
  }
  if (caldata_recall(data) == -1) {
    if (data == 0)
      reset_settings(setting.mode);  // Restore factory defaults
  }
  menu_move_back(true);
}

static UI_FUNCTION_ADV_CALLBACK(menu_store_preset_acb)
{
  (void)item;
  if(b){
    b->param_1.u = data;
    return;
  }
  if (data == 100) {
    reset_settings(M_LOW);  // Restore all defaults in Low mode
    set_refer_output(-1);
 //   setting.mode = -1;
    data = 0;
  }
  caldata_save(data);
  menu_move_back(true);
}

#ifdef __SD_CARD_LOAD__
UI_FUNCTION_CALLBACK(menu_load_config_cb)
{
  (void)item;
  (void)data;
  sd_card_load_config("config.ini");
  ui_mode_normal();
}
#endif

UI_FUNCTION_CALLBACK(menu_autosettings_cb)
{
  (void)item;
  (void)data;
  reset_settings(setting.mode);

  markers_reset();
  //  set_refer_output(1);

  //  SetPowerLevel(100); // Reset
//  set_clear_storage();
  dirty = true;
  //  menu_move_back(true);   // stay in input menu
  ui_mode_normal();
//  draw_cal_status();
}

static UI_FUNCTION_CALLBACK(menu_scale_cb)
{
  (void)item;
  (void)data;
  kp_help_text = "Enter scale";
  kp_buf[0]=0;

  if (UNIT_IS_LINEAR(setting.unit))
    ui_mode_keypad(KM_LINEAR_SCALE);
  else
    ui_mode_keypad(KM_SCALE);
  ui_mode_normal();
}

#ifdef __CALIBRATE__
static UI_FUNCTION_CALLBACK(menu_calibrate_cb)
{
  (void)data;
  switch (item) {
  case 1:
    sweep_mode = SWEEP_CALIBRATE;
    menu_move_back(true);
    break;
  case 2:
    reset_calibration();
    break;
  }
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_scanning_speed_acb)
{
  (void)item;
  if(b){
    b->icon = data == setting.step_delay_mode ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_step_delay(data);
//    menu_move_back(false);
  ui_mode_normal();
}

#define CONFIG_MENUITEM_TOUCH_CAL   0
#define CONFIG_MENUITEM_TOUCH_TEST  1
#define CONFIG_MENUITEM_SELFTEST    2
#define CONFIG_MENUITEM_VERSION     3
static UI_FUNCTION_CALLBACK(menu_config_cb)
{
  (void)item;
  switch (data) {
  case CONFIG_MENUITEM_TOUCH_CAL:
    touch_cal_exec();
    break;
  case CONFIG_MENUITEM_TOUCH_TEST:
    touch_draw_test();
    break;
  case CONFIG_MENUITEM_SELFTEST:
    sweep_mode = 0;         // Suspend sweep to save time
    menu_move_back(true);
    setting.test = 0;
    setting.test_argument = 0;
    sweep_mode = SWEEP_SELFTEST;
    return;
  case CONFIG_MENUITEM_VERSION:
    show_version();
    break;
  }
  ui_mode_normal();
  redraw_frame();
  request_to_redraw_grid();
}
#ifndef TINYSA4
static UI_FUNCTION_CALLBACK(menu_dfu_cb)
{
  (void)data;
  (void)item;
  enter_dfu();
}
#endif

#ifdef __LISTEN__
static UI_FUNCTION_ADV_CALLBACK(menu_listen_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = (sweep_mode & SWEEP_LISTEN) ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  if (sweep_mode & SWEEP_LISTEN) {
    sweep_mode = SWEEP_ENABLE;
  } else {
    sweep_mode = SWEEP_LISTEN;
  }
  ui_mode_normal();
  redraw_frame();
  request_to_redraw_grid();

#if 0
  if (markers[active_marker].enabled == M_ENABLED) {
    do {
      perform(false,0,frequencies[markers[active_marker].index], false);
      SI4432_Listen(MODE_SELECT(setting.mode));
    } while (ui_process_listen_lever());
  }
#endif
}
#endif

// const int menu_modulation_value[]={MO_NONE,MO_AM, MO_NFM, MO_WFM, MO_EXTERNAL};
const char *menu_modulation_text[]={"None", "AM", "NFM", "WFM", "External"};

static UI_FUNCTION_ADV_CALLBACK(menu_modulation_acb)
{
  (void)item;
  if (b){
    b->icon = data == setting.modulation ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
//Serial.println(item);
  if (data) {
    set_sweep_frequency(ST_SPAN, 0);      // No other scanning allowed when modulation is on!!!!!
    set_level_sweep(0);
  }
  set_modulation(data);
  menu_move_back(false);
//  ui_mode_normal();   // Stay in menu mode
//  draw_cal_status();
}

static UI_FUNCTION_ADV_CALLBACK(menu_smodulation_acb){
  (void)item;
  (void)data;
  if(b){
    if (setting.modulation == MO_NONE || setting.modulation == MO_EXTERNAL)
      plot_printf(b->text, sizeof b->text, "MOD: %s", menu_modulation_text[setting.modulation]);
    else {
      plot_printf(b->text, sizeof b->text, "MOD: %5.3fkHz %s", setting.modulation_frequency / 1000.0, menu_modulation_text[setting.modulation]);
    }
    return;
  }
  menu_push_submenu(menu_modulation);
}

//                               0      1       2       3      4      5      6      7
const char *menu_reffer_text[]={"OFF","30MHz","15MHz","10MHz","4MHz","3MHz","2MHz","1MHz"};
static UI_FUNCTION_ADV_CALLBACK(menu_reffer_acb)
{
  (void)item;
  if (b){
    b->icon = setting.refer == ((int)data-1) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    b->param_1.text = menu_reffer_text[data];
    return;
  }
//Serial.println(item);
  set_refer_output((int)data - 1);
  menu_move_back(false);
//  ui_mode_normal();   // Stay in menu mode
//  draw_cal_status();
}

static UI_FUNCTION_ADV_CALLBACK(menu_sreffer_acb){
  (void)item;
  (void)data;
  if(b){
    b->param_1.text = menu_reffer_text[setting.refer+1];
    return;
  }
  menu_push_submenu(menu_reffer);
}

#ifdef TINYSA3
static UI_FUNCTION_ADV_CALLBACK(menu_lo_drive_acb)
{
  (void)item;
  if(b){
    b->param_1.i = drive_dBm[data];
    b->icon = data == setting.lo_drive ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
//Serial.println(item);
  set_lo_drive(data);
  menu_move_back(false);
//  ui_mode_normal();
//  draw_cal_status();
}
#else
static UI_FUNCTION_ADV_CALLBACK(menu_mixer_drive_acb)
{
  (void)item;
  if(b){
    b->param_1.i = data;
    b->icon = data == setting.lo_drive ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
//Serial.println(item);
  set_lo_drive(data);
  menu_move_back(false);
//  ui_mode_normal();
//  draw_cal_status();
}
#endif

#if 0
static UI_FUNCTION_ADV_CALLBACK(menu_sdrive_acb){
  (void)item;
  (void)data;
  if(b){
#ifdef TINYSA4
    b->param_1.i = setting.lo_drive;
#else
    b->param_1.i = drive_dBm[setting.lo_drive] + (setting.mode==M_GENHIGH ? setting.external_gain : 0);
#endif
    return;
  }
  menu_push_submenu(menu_drive_wide);
}
#endif

#ifdef __SPUR__
static UI_FUNCTION_ADV_CALLBACK(menu_spur_acb)
{
  (void)data;
  (void)item;
  if (b){
    if (setting.mode == M_LOW) {
      b->param_1.text = "SPUR\nREMOVAL";
      b->icon = AUTO_ICON(setting.spur_removal);
    } else {
      b->param_1.text = "MIRROR\nMASKING";
#ifdef TINYSA4
      b->icon = AUTO_ICON(setting.mirror_masking ? 1 : 0);  // mirror_masking does not yet have an auto mode so this is never an auto icon
#else
      b->icon = setting.mirror_masking == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
#endif
    }
    return;
  }
  if (setting.mode == M_LOW) {
    toggle_spur();
  } else
    toggle_mirror_masking();
  //  menu_move_back(false);
  ui_mode_normal();
}
#if 0
#ifdef __HARMONIC__
static UI_FUNCTION_ADV_CALLBACK(menu_harmonic_spur_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = AUTO_ICON(setting.spur_removal);
    return;
  }
  toggle_spur();
  ui_mode_normal();
}#endif
#endif
#endif
#endif

#ifdef __ULTRA__
static UI_FUNCTION_ADV_CALLBACK(menu_debug_spur_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = debug_spur == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  toggle_debug_spur();
  //  menu_move_back();
  ui_mode_normal();
}
#endif

#ifdef TINYSA4
static UI_FUNCTION_ADV_CALLBACK(menu_progress_bar_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = progress_bar == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  progress_bar = !progress_bar;
  //  menu_move_back();
  ui_mode_normal();
}

static UI_FUNCTION_ADV_CALLBACK(menu_extra_lna_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = setting.extra_lna == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  toggle_extra_lna();
  //  menu_move_back(false);
  ui_mode_normal();
}

static UI_FUNCTION_ADV_CALLBACK(menu_adf_out_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = config.high_out_adf4350 == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  toggle_high_out_adf4350();
  //  menu_move_back(false);
  ui_mode_normal();
}

#ifdef __WAIT_CTS_WHILE_SLEEPING__
volatile int sleep = 0;
static UI_FUNCTION_ADV_CALLBACK(menu_sleep_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = sleep == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  sleep = !sleep;
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_debug_avoid_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = debug_avoid == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  toggle_debug_avoid();
  //  menu_move_back();
  ui_mode_normal();
}

static UI_FUNCTION_ADV_CALLBACK(menu_debug_freq_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = debug_frequencies == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  debug_frequencies = ! debug_frequencies;
  //  menu_move_back();
  ui_mode_normal();
}


static UI_FUNCTION_ADV_CALLBACK(menu_linear_averaging_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = linear_averaging == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  linear_averaging = ! linear_averaging;
  dirty = true;
  //  menu_move_back();
  ui_mode_normal();
}


#endif

#ifdef __ULTRA__
static UI_FUNCTION_ADV_CALLBACK(menu_ultra_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = config.ultra == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  if (!config.ultra) {
    kp_help_text = "Ultra unlock code";
    ui_mode_keypad(KM_CENTER);
    if (uistat.value != 4321)
      return;
  }
  config.ultra = !config.ultra;
  config_save();
  reset_settings(M_LOW);
  if (config.ultra){
    set_sweep_frequency(ST_START, 0);
    set_sweep_frequency(ST_STOP, 3000000000ULL);
  }
  //  menu_move_back(false);
  ui_mode_normal();
}

static UI_FUNCTION_ADV_CALLBACK(menu_direct_acb)
{
  (void)data;
  (void)item;
  if (b){
    b->icon = config.direct == 0 ? BUTTON_ICON_NOCHECK : BUTTON_ICON_CHECK;
    return;
  }
  config.direct = !config.direct;
  config_save();
  //  menu_move_back(false);
  ui_mode_normal();
}

#endif

static UI_FUNCTION_CALLBACK(menu_clearconfig_cb)
{
  (void)data;
  (void)item;
  kp_help_text = "Clear unlock code";
  ui_mode_keypad(KM_CENTER);
  if (uistat.value != 1234)
    return;
  clear_all_config_prop_data();
#ifdef TINYSA4
#define SCB_AIRCR_VECTKEYSTAT_LSB 16
#define SCB_AIRCR_VECTKEYSTAT       (0xFFFF << SCB_AIRCR_VECTKEYSTAT_LSB)
#define SCB_AIRCR_VECTKEY       (0x05FA << SCB_AIRCR_VECTKEYSTAT_LSB)
#define SCB_AIRCR_SYSRESETREQ         (1 << 2)
  SCB->AIRCR = SCB_AIRCR_VECTKEY | SCB_AIRCR_SYSRESETREQ;
          while(true);
#else
  reset_settings(M_LOW);
  ui_mode_normal();
#endif
}

float nf_gain;
const char * const averageText[] = { "OFF", "MIN", "MAX", "MAXD", " A 4", "A 16", "AVER", "QUASI", "TABLE", "DECONV"};

static UI_FUNCTION_ADV_CALLBACK(menu_measure_acb)
{
  (void)item;
  if (b){
    b->icon = data == setting.measurement ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  menu_move_back(false);
  markers_reset();

  if ((data != M_OFF && setting.measurement != M_OFF) || data == M_OFF )
  {
    //      reset_settings(setting.mode);
    if (0) {
    no_measurement:
      data = M_OFF;
    }
    if (setting.measurement == M_LINEARITY) {
      TRACE_DISABLE(TRACE_STORED_FLAG);
    }
    set_average(0,AV_OFF);
    set_external_gain(0.0);
#ifdef TINYSA4
    set_extra_lna(false);
#endif
  }
  switch(data) {
    case M_OFF:                                     // Off
//      set_measurement(M_OFF);
      break;
    case M_IMD:                                     // IMD
      reset_settings(setting.mode);
      for (int i = 1; i< MARKERS_MAX; i++) {
        markers[i].mtype = M_DELTA;
      }
      kp_help_text = "Frequency of fundamental";
      ui_mode_keypad(KM_CENTER);
      set_sweep_frequency(ST_START, 0);
      set_sweep_frequency(ST_STOP, uistat.value*(MARKERS_MAX+1));
      set_average(0,AV_4);
//      set_measurement(M_IMD);
      break;
    case M_OIP3:                                     // OIP3
      reset_settings(setting.mode);
      for (int i = 0; i< 4; i++) {
        markers[i].enabled = M_ENABLED;
      }
      markers[1].mtype = M_TRACKING;
      kp_help_text = "Frequency of left signal";
      ui_mode_keypad(KM_CENTER);
      freq_t left =  uistat.value;
      kp_help_text = "Right signal";
      ui_mode_keypad(KM_CENTER);
      freq_t right =  uistat.value;
      set_sweep_frequency(ST_CENTER, (left+right)/2);
      freq_t local_span = (right - left)*10;
      if (local_span < 30000)
        local_span = 30000;
      set_sweep_frequency(ST_SPAN, local_span);
      set_average(0,AV_4);
//      set_measurement(M_OIP3);
      break;
    case M_PHASE_NOISE:                             // Phase noise
      reset_settings(setting.mode);
      markers[1].enabled = M_ENABLED;
      markers[1].mtype = M_DELTA | M_NOISE;
      kp_help_text = "Frequency of signal";
      ui_mode_keypad(KM_CENTER);
      kp_help_text = "Frequency offset";
      ui_mode_keypad(KM_SPAN);
      set_sweep_frequency(ST_SPAN, uistat.value*4);
//      set_measurement(M_PHASE_NOISE);
      set_average(0,AV_4);

      break;
    case M_SNR:                             // STop band measurement
      reset_settings(setting.mode);
      kp_help_text = "Frequency of signal";
      ui_mode_keypad(KM_CENTER);
      kp_help_text = "Bandwidth";
      ui_mode_keypad(KM_SPAN);
      set_sweep_frequency(ST_SPAN, uistat.value*3);
//      set_measurement(M_SNR);
      set_average(0,AV_4);

      break;
    case M_PASS_BAND:                             // pass band measurement
//      reset_settings(setting.mode);
      markers[1].enabled = M_ENABLED;
      markers[2].enabled = M_ENABLED;
//      kp_help_text = "Frequency of signal";
//      ui_mode_keypad(KM_CENTER);
//      kp_help_text = "Width of signal";
//      ui_mode_keypad(KM_SPAN);
//      set_sweep_frequency(ST_SPAN, uistat.value*2);
      set_measurement(M_PASS_BAND);
//      SetAverage(4);

      break;
#ifdef __LINEARITY__
    case M_LINEARITY:
      TRACE_ENABLE(TRACE_STORED_FLAG);
//      set_measurement(M_LINEARITY);
      break;
#endif
    case M_AM:                                     // AM
      reset_settings(setting.mode);
      for (int i = 1; i< 3; i++) {
        markers[i].enabled = M_ENABLED;
#ifdef TINYSA4
        markers[i].mtype = M_DELTA| M_TRACKING;
#else
//        markers[i].mtype = M_DELTA;// | M_TRACKING;
#endif
      }
#ifdef TINYSA4
      freq_t span;
#else
      freq_t center, span;
      markers[0].mtype = M_NORMAL; // M_REFERENCE | M_TRACKING;           // Not M_TRACKING!!!!
#endif
      kp_help_text = "Frequency of signal";
      ui_mode_keypad(KM_CENTER);
#ifdef TINYSA3
      center = uistat.value;
#endif
      kp_help_text = "Modulation frequency: 3 .. 10kHz";
      ui_mode_keypad(KM_SPAN);
//      if (uistat.value < 3000)
//        break;
      span = uistat.value;
#ifdef TINYSA4
      set_RBW((span * 5 / 50) / 100);
#endif
      set_sweep_frequency(ST_SPAN, span * 5);
//      update_frequencies();                     // To ensure markers are positioned right!!!!!!
//      set_measurement(M_AM);
#ifndef TINYSA4
      set_marker_frequency(0, center);
      set_marker_frequency(1, center-span);
      set_marker_frequency(2, center+span);
#endif
      set_average(0,AV_4);
      break;
    case M_FM:                                     // FM
      reset_settings(setting.mode);
      for (int i = 1; i< 3; i++) {
        markers[i].enabled = M_ENABLED;
      }
      markers[0].mtype = M_NORMAL;              /// Not M_TRACKING !!!!
      kp_help_text = "Frequency of signal";
      ui_mode_keypad(KM_CENTER);
      set_marker_frequency(0, uistat.value);
#ifdef TINYSA4
      kp_help_text = "Modulation frequency: 1 .. 10kHz";
      ui_mode_keypad(KM_SPAN);
      if (uistat.value < 1000 || uistat.value > 10000)
        goto no_measurement;
      set_RBW(uistat.value/300);
#else
      kp_help_text = "Modulation frequency: 1 .. 2.5kHz";
      ui_mode_keypad(KM_SPAN);
      if (uistat.value < 1000 || uistat.value > 2500)
        goto no_measurement;
      set_RBW(uistat.value/100);
#endif
      // actual_rbw_x10
      kp_help_text = "Frequency deviation: 3 .. 500kHz";
      ui_mode_keypad(KM_SPAN);
      if (uistat.value < 12000)
        uistat.value = 12000;   // minimum span
      set_sweep_frequency(ST_SPAN, uistat.value*4);
//      set_measurement(M_FM);
      break;
    case M_THD:
      set_average(0,AV_4);
//      set_measurement(M_THD);
      break;
#ifdef __CHANNEL_POWER__
    case M_CP:                             // channel power
      reset_settings(setting.mode);
      markers[0].enabled = M_DISABLED;
      kp_help_text = "Channel frequency";
      ui_mode_keypad(KM_CENTER);
      kp_help_text = "Channel width";
      ui_mode_keypad(KM_SPAN);
      set_sweep_frequency(ST_SPAN, uistat.value*3);
//      set_measurement(M_CP);
      break;
#endif
#ifdef __NOISE_FIGURE__
    case M_NF_TINYSA:
      reset_settings(setting.mode);
      set_refer_output(-1);
      nf_gain = 0;
      goto noise_figure;
    case M_NF_STORE:
      if (measured_noise_figure > 3 && measured_noise_figure < 15) {
        config.noise_figure = measured_noise_figure;
        config_save();
        data = M_NF_VALIDATE;               // Continue to validate
        goto validate;
      } else
        data = M_NF_TINYSA;               // Continue to measure
      break;
    case M_NF_VALIDATE:
validate:
      nf_gain = 0.00001;                            // almost zero
      goto noise_figure;
    case M_NF_AMPLIFIER:                             // noise figure
//      reset_settings(setting.mode);
      reset_settings(setting.mode);
      set_refer_output(-1);
      kp_help_text = "Amplifier Gain ";
      float old_gain = setting.external_gain;
      ui_mode_keypad(KM_EXT_GAIN);
      nf_gain = setting.external_gain;
      setting.external_gain = old_gain;
  noise_figure:
      markers[0].enabled = M_ENABLED;
      markers[0].mtype = M_NOISE | M_AVER;          // Not tracking
      set_extra_lna(true);
      set_attenuation(0);
      if (data != M_NF_VALIDATE) {
        kp_help_text = "Noise center frequency";
        ui_mode_keypad(KM_CENTER);
        set_marker_frequency(0, uistat.value);
#if 0
        kp_help_text = "Noise span";
        ui_mode_keypad(KM_SPAN);
#else
        set_sweep_frequency(ST_SPAN, 100000);
#endif
        set_RBW(get_sweep_frequency(ST_SPAN)/100 / 100);
      }

//      set_sweep_frequency(ST_SPAN, 0);
      set_average(0,AV_100);
      if (data == M_NF_TINYSA || data == M_NF_VALIDATE ) {
        menu_push_submenu(menu_measure_noise_figure);
        goto leave;
      }
      break;
#endif
#ifdef __FFT_DECONV__
    case M_DECONV:
      set_average(0,AV_DECONV);
      break;
#endif
  }

//  selection = -1;
  ui_mode_normal();
  goto leave;       // to get rid of warning
leave:
  set_measurement(data);
//  draw_cal_status();
}

static UI_FUNCTION_ADV_CALLBACK(menu_atten_acb)
{
  (void)item;
  (void)data;
  if(b){
    b->icon = setting.auto_attenuation ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_auto_attenuation();
  menu_move_back(true);
}

static UI_FUNCTION_ADV_CALLBACK(menu_atten_high_acb)
{
  (void)item;
  if(b){
    b->icon = (setting.atten_step*30 == data) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  setting.auto_attenuation = false;
  set_attenuation(data);
  menu_move_back(true);
}

static UI_FUNCTION_ADV_CALLBACK(menu_reflevel_acb)
{
  (void)item;
  (void)data;
  if(b){
    b->icon = setting.auto_reflevel ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_auto_reflevel(true);
  menu_move_back(true);
}

static uint8_t current_trace = TRACE_ACTUAL;

static UI_FUNCTION_ADV_CALLBACK(menu_average_acb)
{
  (void)item;
  if (b){
    b->icon = setting.average[current_trace] == data ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_average(current_trace,data);
  if (data == AV_TABLE)
    menu_push_submenu(menu_limit_select);
  else {
    ui_mode_normal();
  }
//  menu_move_back(true);
}

static UI_FUNCTION_ADV_CALLBACK(menu_trace_acb)
{
  (void)item;
  if(b){
    b->param_1.i = data+1;
    b->icon = (data == current_trace) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    b->bg = LCD_TRACE_1_COLOR+data;
    return;
  }
  if (setting.normalized_trace != -1 && data == TRACE_TEMP) {
    drawMessageBox("Error", "Disable normalization first", 2000);
    redraw_request|= REDRAW_AREA;
  } else
    current_trace = data;
  menu_move_back(false);
}

static UI_FUNCTION_ADV_CALLBACK(menu_marker_trace_acb)
{
  (void)item;
  if(b){
    b->param_1.i = data+1;
    b->icon = (data == markers[active_marker].trace) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    b->bg = LCD_TRACE_1_COLOR+data;
    return;
  }
  markers[active_marker].trace = data;
  menu_move_back(false);
}

static UI_FUNCTION_ADV_CALLBACK(menu_store_trace_acb)
{
  (void)item;
  if(b){
    plot_printf(b->text, sizeof(b->text), S_RARROW"TRACE %d", data+1);
    b->bg = LCD_TRACE_1_COLOR+data;
    if (current_trace == data)
      b->fg = LCD_DARK_GREY;
    return;
  }
  store_trace(current_trace,data);
  menu_move_back(false);
}


static UI_FUNCTION_ADV_CALLBACK(menu_subtract_trace_acb)
{
  (void)item;
  if(b){
    if (data) {
      plot_printf(b->text, sizeof(b->text), "SUBTRACT\nTRACE %d", data);
      b->bg = LCD_TRACE_1_COLOR+data-1;
    }
    else
      plot_printf(b->text, sizeof(b->text), "SUBTRACT\nOFF");
    b->icon = (data == setting.subtract[current_trace]) ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    if (data - 1 == current_trace)
      b->fg = LCD_DARK_GREY;
    return;
  }
  subtract_trace(current_trace,data-1);
  menu_move_back(false);
}

static UI_FUNCTION_ADV_CALLBACK(menu_traces_acb)
{
  (void)item;
  int count = 0;
  if(b){
    if (data == 0) {                // Select trace
      b->param_1.i = current_trace+1;
      b->bg = LCD_TRACE_1_COLOR+current_trace;
    } else if (data == 1) {           // View
      b->icon = IS_TRACE_ENABLE(current_trace) ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    }
    else if (data == 2)               // freeze
      b->icon = setting.stored[current_trace] ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    else if (data == 5) {
      if (setting.subtract[current_trace])
        plot_printf(b->text, sizeof(b->text), "SUBTRACT\nTRACE %d", setting.subtract[current_trace]);
      else
        plot_printf(b->text, sizeof(b->text), "SUBTRACT\nOFF");
      b->icon = setting.subtract[current_trace] ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    } else if (data == 4) {
      if (current_trace == TRACES_MAX-1)
        b->fg = LCD_DARK_GREY;
      else
        b->icon = setting.normalized[current_trace] ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    } else if (data == 3) {
      plot_printf(b->text, sizeof(b->text), "CALC\n%s", averageText[setting.average[current_trace]]);
      b->icon = setting.average[current_trace] ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    }
    return;
  }
  switch(data) {
  case 0:
    menu_push_submenu(menu_trace);
    return;
  case 1:
    for (int i=0;i<TRACES_MAX;i++)
      if (IS_TRACE_ENABLE(i))
        count++;
    if (IS_TRACE_ENABLE(current_trace)) {
      if (count > 1)  {                   // Always 1 trace enabled
        TRACE_DISABLE(1<<current_trace);
      } else {
        drawMessageBox("Trace", "Enable at least one trace", 2000);
        redraw_request|= REDRAW_AREA;
      }
    } else {
      TRACE_ENABLE(1<<current_trace);
    }
    break;
  case 2:                               // Freeze
    setting.stored[current_trace] = !setting.stored[current_trace];
    break;
  case 5:
    menu_push_submenu(menu_subtract_trace);
    return;
    break;
  case 4:
    if (current_trace < TRACES_MAX-1) {
      toggle_normalize(current_trace);
      if (setting.normalized[current_trace]) {
//        kp_help_text = "Ref level";
//        ui_mode_keypad(KM_REFLEVEL);
//        setting.normalize_level = 20.0; // uistat.value;
      } else {
//        set_auto_reflevel(true);
      }
    }
    break;
  case 3:
    menu_push_submenu(menu_average);
    return;
    break;

#ifdef TINYSA4
    case 6:
      save_to_sd(1+(2<<current_trace));      // frequencies + trace
      break;
#endif
  }
//  ui_mode_normal();
//  draw_cal_status();
}
#if 0
static UI_FUNCTION_ADV_CALLBACK(menu_storage_acb)
{
  (void)item;
  if(b){
    if (data == 0 && setting.show_stored)
      b->icon = BUTTON_ICON_CHECK;
    if (setting.subtract[0]){
      if (data == 2 && setting.show_stored)
        b->icon = BUTTON_ICON_CHECK;
      if (data == 3 && !setting.show_stored)
        b->icon = BUTTON_ICON_CHECK;
    }
    return;
  }
  switch(data) {
    case 0:
      store_trace(0,2);
      break;
    case 1:
      set_clear_storage();
      break;
    case 2:
      set_subtract_storage();
      break;
    case 3:
      toggle_normalize();
      if (setting.subtract[0]) {
        kp_help_text = "Ref level";
        ui_mode_keypad(KM_REFLEVEL);
//        setting.normalize_level = uistat.value;
      } else
        set_auto_reflevel(true);
      break;
#ifdef TINYSA4
    case 4:
      save_to_sd(1+2);      // frequencies + actual
      break;
    case 5:
      save_to_sd(1+4);      // frequencies + stored
      break;
#endif
  }
  ui_mode_normal();
//  draw_cal_status();
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_waterfall_acb){
  (void)item;
  (void)data;
  if (b){
    b->icon = setting.waterfall ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  setting.waterfall++; if (setting.waterfall>W_BIG)setting.waterfall = W_OFF;
  if (setting.waterfall != W_OFF)
    setting.level_meter = false;
  set_waterfall();
  ui_mode_normal();
}

#ifdef __LEVEL_METER__
static UI_FUNCTION_ADV_CALLBACK(menu_level_meter_acb){
  (void)item;
  (void)data;
  if (b){
    b->icon = setting.level_meter ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  setting.level_meter = !setting.level_meter;
  if (setting.level_meter)
    setting.waterfall = W_OFF;
  set_level_meter();
  ui_mode_normal();
}
#endif


#ifdef __LIMITS__
uint8_t active_limit = 0;
static UI_FUNCTION_ADV_CALLBACK(menu_limit_select_acb)
{
  (void)item;
  if(b){
    plot_printf(b->text, sizeof(b->text), "%.6FHz\n%.2F%s", (float)setting.limits[current_trace][data].frequency, value(setting.limits[current_trace][data].level),unit_string[setting.unit]);
    b->icon = (setting.limits[current_trace][data].enabled?BUTTON_ICON_CHECK:BUTTON_ICON_NOCHECK) ;
    return;
  }
  active_limit = data;
  setting.limits[current_trace][active_limit].enabled = true;
  dirty = true;
  limits_update();
  menu_push_submenu(menu_limit_modify);
}

#endif

extern const menuitem_t menu_marker_select[];

static UI_FUNCTION_ADV_CALLBACK(menu_marker_modify_acb)
{
  (void)item;
  if (active_marker == MARKER_INVALID) return;
  if(b){
    uistat.text[0] = 0;
    uistat.text[1] = 0;
    switch(data) {
    case 0:
      uistat.text[0] = active_marker+'1';
      break;
    case M_DELTA:
      uistat.text[0] = markers[active_marker].ref+'1';
      /* fall through */
    case M_NOISE:
    case M_TRACKING:
    case M_AVER:
      b->icon = BUTTON_ICON_NOCHECK;
      if (markers[active_marker].mtype & data)
        b->icon = BUTTON_ICON_CHECK;
      break;
    case M_STORED:
      uistat.text[0] = markers[active_marker].trace+'1';
      b->bg = LCD_TRACE_1_COLOR+markers[active_marker].trace;
      break;
    }
    b->param_1.text = uistat.text;
    return;
  }
  if (data == M_DELTA && !(markers[active_marker].mtype & M_DELTA)) {   // Not yet set
    menu_push_submenu(menu_marker_ref_select);
    goto set_delta;
    return;
  } else if (data == M_STORED) {
    current_trace = 0;
    menu_push_submenu(menu_marker_trace);
    return;
  } else if (markers[active_marker].mtype & data)
    markers[active_marker].mtype &= ~data;
  else if (data) {
    set_delta:
    markers[active_marker].mtype |= data;
  } else
    menu_push_submenu(menu_marker_select);
  markmap_all_markers();
//  redraw_marker(active_marker, TRUE);
//  menu_move_back(false);
}

static UI_FUNCTION_ADV_CALLBACK(menu_marker_ref_select_acb)
{
  (void)item;
  if(b){
//    b->icon = markers[data-1].enabled ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    b->icon = (markers[active_marker].ref == data-1 ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP );
    b->param_1.i = data;
    return;
  }
  markers[data-1].enabled = true;
//  interpolate_maximum(data-1);        // possibly not a maximum
  set_marker_index(data-1, markers[data-1].index);
  markers[active_marker].ref = data-1;
  redraw_marker(active_marker);
  menu_move_back(false);
}

extern const menuitem_t menu_marker_modify[];
static UI_FUNCTION_ADV_CALLBACK(menu_marker_select_acb)
{
  (void)item;
  if(b){
    b->icon = markers[data-1].enabled ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    b->param_1.i = data;
    return;
  }
  markers[data-1].enabled = true;
//  interpolate_maximum(data-1);        // possibly not a maximum
  set_marker_index(data-1, markers[data-1].index);
  active_marker_select(data-1);
//  menu_push_submenu(menu_marker_modify);
  redraw_marker(active_marker);
  menu_move_back(false);
}

static UI_FUNCTION_CALLBACK(menu_marker_delete_cb)
{
  (void)item;
  (void)data;
  if (active_marker>=0){
    for (int i = 0; i<MARKER_COUNT; i++ ) {
      if (markers[i].enabled && (markers[i].mtype & M_DELTA) && markers[i].ref == active_marker)
        markers[i].enabled = false;
    }
    markers[active_marker].enabled = false;
    markmap_all_markers();
    menu_move_back(false);
  }
}

#ifdef __LIMITS__
static UI_FUNCTION_CALLBACK(menu_limit_disable_cb)
{
  (void)item;
  (void)data;
  if (active_limit<LIMITS_MAX){
    setting.limits[current_trace][active_limit].enabled = false;
    dirty = true;
    limits_update();
    menu_move_back(false);
  }
}
#endif

#ifdef TINYSA4
static const uint16_t rbwsel_x10[]={0,3,10,30,100,300,1000,3000,6000,8500};
static const char* rbwsel_text[]={"AUTO","300","1k","3k","10k","30k","100k","300k","600k","850k"};
#else
static const uint16_t rbwsel_x10[]={0,30,100,300,1000,3000,6000};
#endif
#ifdef __VBW__
static const uint16_t vbwsel_x100[]={0,100,30,10,3,1};
//static const char* vbwsel_text[]={"AUTO","0.01","0.03", "0.1", "0.3","   "};
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_rbw_acb)
{
  (void)item;
  if (b){
#ifdef TINYSA4
  b->param_1.text = rbwsel_text[data];
#else
    b->param_1.u = rbwsel_x10[data]/10;
#endif
  b->icon = setting.rbw_x10 == rbwsel_x10[data] ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_RBW(rbwsel_x10[data]);
  menu_move_back(true);
}

#ifdef __VBW__
static UI_FUNCTION_ADV_CALLBACK(menu_vbw_acb)
{
  (void)item;
  if (b){
  b->param_1.f = vbwsel_x100[data] > 0 ? 1.0f/vbwsel_x100[data] : 0;
  b->icon = setting.vbw_x100 == vbwsel_x100[data] ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_VBW(vbwsel_x100[data]);
  menu_move_back(true);
}

#endif

static UI_FUNCTION_ADV_CALLBACK(menu_unit_acb)
{
  (void)item;
  if (b){
    b->icon = data == setting.unit ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  set_unit(data);
  menu_move_back(true);
}

#if 0
enum {
  S_20,S_10,S_5,S_2,S_1,S_P5,S_P2,S_P1,S_P05,S_P02,S_P01
};
static const float menu_scale_per_value[11]={20,10,5,2,1,0.5,0.2,0.1,0.05,0.02,0.01};

static UI_FUNCTION_ADV_CALLBACK(menu_scale_per_acb)
{
  (void)item;
  if(b){
    return;
  }
  set_scale(menu_scale_per_value[data]);
  menu_move_back(true);
}
#endif

const char *mode_text[] = {"PRE","POST","MID"};

static UI_FUNCTION_ADV_CALLBACK(menu_trigger_acb)
{
  (void)item;
  if(b){
    if (data == T_MODE) {
      b->param_1.text = mode_text[setting.trigger_mode - T_PRE];
    } else if (data == T_UP || data == T_DOWN)
      b->icon = setting.trigger_direction == data ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    else
      b->icon = setting.trigger == data ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  if (data == T_MODE) {
    setting.trigger_mode += 1;
    if (setting.trigger_mode > T_MID)
      setting.trigger_mode = T_PRE;
    set_trigger(setting.trigger_mode);
  } else if (data != T_DONE) {
    set_trigger(data);
//  menu_move_back(false);
    ui_mode_normal();
  }
  completed = true;
}

#if 0
static void choose_active_trace(void)
{
  int i;
  if (trace[uistat.current_trace].enabled)
    // do nothing
    return;
  for (i = 0; i < TRACE_COUNT ; i++)
    if (trace[i].enabled) {
      uistat.current_trace = i;
      return;
    }
}
#endif
static void choose_active_marker(void)
{
  int i;
  for (i = 0; i < MARKER_COUNT; i++)
    if (markers[i].enabled == M_ENABLED) {
      active_marker = i;
      return;
    }
  active_marker = MARKER_INVALID;
}

#ifdef __HARMONIC__
static UI_FUNCTION_ADV_CALLBACK(menu_harmonic_acb)
{
  (void)item;
  if(b){
    b->icon = setting.harmonic == data ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  set_harmonic(data);
}
#endif


static UI_FUNCTION_ADV_CALLBACK(menu_settings_agc_acb){
  (void)item;
  (void)data;
  if(b){
    b->icon = AUTO_ICON(setting.agc);
    return;
  }
  toggle_AGC();
}

static UI_FUNCTION_ADV_CALLBACK(menu_settings_lna_acb){
  (void)item;
  (void)data;
  if(b){
    if (S_IS_AUTO(setting.lna))
      b->icon = BUTTON_ICON_CHECK_AUTO;
    else
      b->icon = setting.lna ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_LNA();
}

static UI_FUNCTION_ADV_CALLBACK(menu_settings_bpf_acb){
  (void)item;
  (void)data;
  if(b){
    b->icon = setting.tracking ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_tracking();
}

#ifdef __LCD_BRIGHTNESS__
static UI_FUNCTION_CALLBACK(menu_brightness_cb)
{
  (void)item;
  (void)data;
  int16_t value = config._brightness;
  ili9341_set_foreground(LCD_MENU_TEXT_COLOR);
  ili9341_set_background(LCD_MENU_COLOR);
  ili9341_fill(LCD_WIDTH/2-80, LCD_HEIGHT/2-20, 160, 40);
  ili9341_drawstring_7x13("BRIGHTNESS", LCD_WIDTH/2-35, LCD_HEIGHT/2-13);
  ili9341_drawstring_7x13(S_LARROW" USE LEVELER BUTTON "S_RARROW, LCD_WIDTH/2-72, LCD_HEIGHT/2+2);
  while (TRUE) {
    int status = btn_check();
    if (status & (EVT_UP|EVT_DOWN)) {
      do {
        if (status & EVT_UP  ) value+=5;
        if (status & EVT_DOWN) value-=5;
        if (value <   0) value =   0;
        if (value > 100) value = 100;
        lcd_setBrightness(value);
        status = btn_wait_release();
      } while (status != 0);
    }
    if (status == EVT_BUTTON_SINGLE_CLICK)
      break;
  }
  config._brightness = (uint8_t)value;
  lcd_setBrightness(value);
  redraw_request|= REDRAW_AREA;
  ui_mode_normal();
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_settings_pulse_acb){
  (void)item;
  (void)data;
  if(b){
    b->icon = setting.pulse ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_pulse();
}

#ifdef __HAM_BAND__
static UI_FUNCTION_ADV_CALLBACK(menu_settings_ham_bands){
  (void)item;
  (void)data;
  if(b){
    b->icon = config.hambands ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_hambands();
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_settings_below_if_acb){
  (void)item;
  (void)data;
  if(b){
    if (S_IS_AUTO(setting.below_IF))
      b->icon = BUTTON_ICON_CHECK_AUTO;
    else
      b->icon = setting.below_IF ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_below_IF();
}

#if 0
static UI_FUNCTION_ADV_CALLBACK(menu_settings_ultra_acb){
  (void)item;
  (void)data;
  if(b){
    if (S_IS_AUTO(setting.ultra))
      b->icon = BUTTON_ICON_CHECK_AUTO;
    else
      b->icon = setting.ultra ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_ultra();
}
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_lo_output_acb){
  (void)item;
  (void)data;
  if(b){
    b->icon = setting.tracking_output ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_tracking_output();
}

static UI_FUNCTION_ADV_CALLBACK(menu_pause_acb)
{
  (void) data;
  (void) item;
  if (b){
    b->icon = is_paused() ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  toggle_sweep();
//  menu_move_back(true);
//  draw_cal_status();
}

static UI_FUNCTION_ADV_CALLBACK(menu_flip_acb)
{
  (void) data;
  (void) item;
  if (b){
    return;
  }
  config.flip = ! config.flip;
  ili9341_flip(config.flip);
  config_save();
  redraw_request|= REDRAW_AREA | REDRAW_FREQUENCY | REDRAW_CAL_STATUS;
}

static UI_FUNCTION_ADV_CALLBACK(menu_shift_acb)
{
  (void) data;
  (void) item;
  if (b){
    b->icon = setting.frequency_offset != FREQUENCY_SHIFT ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  if (setting.frequency_offset != FREQUENCY_SHIFT) {
    setting.frequency_offset = FREQUENCY_SHIFT;
  } else {
    if (FREQ_IS_STARTSTOP()) {
      freq_t old_start = get_sweep_frequency(ST_START);
      freq_t old_stop = get_sweep_frequency(ST_STOP);
      kp_help_text = "Actual start frequency";
      ui_mode_keypad(KM_START);
      setting.frequency_offset = uistat.value - old_start + FREQUENCY_SHIFT;
      set_sweep_frequency(ST_START, old_start);
      set_sweep_frequency(ST_STOP, old_stop);
    } else {
      freq_t old_center = get_sweep_frequency(ST_CENTER);
      freq_t old_span = get_sweep_frequency(ST_SPAN);
      kp_help_text = "Actual center frequency";
      ui_mode_keypad(KM_CENTER);
      setting.frequency_offset = uistat.value - old_center + FREQUENCY_SHIFT;
      set_sweep_frequency(ST_CENTER, old_center);
      set_sweep_frequency(ST_SPAN, old_span);
    }
  }
  ui_mode_normal();
//  menu_move_back(true);
//  draw_cal_status();
}

#ifdef __REMOTE_DESKTOP__
#if 0   // Not used in UI
static UI_FUNCTION_ADV_CALLBACK(menu_send_display_acb)
{
  (void) data;
  (void) item;
  if (b){
    b->icon = auto_capture ? BUTTON_ICON_CHECK : BUTTON_ICON_NOCHECK;
    return;
  }
  auto_capture = ! auto_capture;
  // Update all screen to CPU
  if (auto_capture)
    redraw_request|=REDRAW_AREA|REDRAW_BATTERY|REDRAW_FREQUENCY|REDRAW_CAL_STATUS;
}
#endif
#endif

static UI_FUNCTION_ADV_CALLBACK(menu_outputmode_acb)
{
  (void) data;
  (void) item;
  if(b){
    b->param_1.text = setting.mute ? "OFF" : "ON";
    return;
  }
  toggle_mute();
}

static UI_FUNCTION_ADV_CALLBACK(menu_enter_marker_acb)
{
  (void) data;
  (void) item;
  if(b){
    b->param_1.text = FREQ_IS_CW() ? "TIME" : "FREQUENCY";
    return;
  }
  if (FREQ_IS_CW())
    ui_mode_keypad(KM_MARKER_TIME);
  else
    ui_mode_keypad(KM_MARKER);
}

#ifdef TINYSA4
static const uint16_t points_setting[] = {51, 101, 201, 256, 290, 450};
#else
static const uint16_t points_setting[] = {51, 101, 145, 290};
#endif
static UI_FUNCTION_ADV_CALLBACK(menu_points_acb){
  (void)item;
  if(b){
    b->icon = points_setting[data] == sweep_points ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    b->param_1.i = points_setting[data];
    return;
  }
  set_sweep_points(points_setting[data]);
}

#ifdef __USE_SERIAL_CONSOLE__
static UI_FUNCTION_ADV_CALLBACK(menu_serial_speed_acb)
{
  static const uint32_t usart_speed[] = {19200, 38400, 57600, 115200, 230400, 460800, 921600, 1843200, 2000000, 3000000};
  (void)item;
  uint32_t speed = usart_speed[data];
  if (b){
    b->icon = config._serial_speed == speed ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    b->param_1.u = speed;
    return;
  }
  config._serial_speed = speed;
  shell_update_speed();
}

static UI_FUNCTION_ADV_CALLBACK(menu_connection_acb)
{
  (void)item;
  if (b){
    b->icon = (config._mode&_MODE_CONNECTION_MASK) == data ? BUTTON_ICON_GROUP_CHECKED : BUTTON_ICON_GROUP;
    return;
  }
  config._mode&=~_MODE_CONNECTION_MASK;
  config._mode|=data;
  config_save();
  shell_reset_console();
}
#endif
// ===[MENU DEFINITION]=========================================================
// Back button submenu list

static const menuitem_t menu_back[] = {
  { MT_CANCEL,   0, S_LARROW" BACK", NULL },
  { MT_NONE,     0, NULL, NULL } // sentinel
};

static const menuitem_t menu_store_preset[] =
{
  { MT_ADV_CALLBACK, 0,  "STORE AS\nSTARTUP",menu_store_preset_acb},
  { MT_ADV_CALLBACK |MT_REPEATS,  DATA_STARTS_REPEATS(1,4),  "STORE %d",         menu_store_preset_acb},
  { MT_ADV_CALLBACK, 100,"FACTORY\nDEFAULTS",menu_store_preset_acb},
  { MT_NONE,     0,     NULL,menu_back} // next-> menu_back
};

static const menuitem_t menu_load_preset[] =
{
  { MT_ADV_CALLBACK,            0,                          "LOAD\nSTARTUP", menu_load_preset_acb},
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(1,4),   MT_CUSTOM_LABEL, menu_load_preset_acb},
  { MT_SUBMENU,                 0,                          "STORE"  ,       menu_store_preset},
  { MT_NONE,     0,     NULL, menu_back} // next-> menu_back
};
#ifdef TINYSA4
static const menuitem_t menu_mixer_drive[] = {
  { MT_ADV_CALLBACK, 5, "Auto",     menu_mixer_drive_acb},
  { MT_ADV_CALLBACK, 3, "%+ddBm",   menu_mixer_drive_acb},
  { MT_ADV_CALLBACK, 2, "%+ddBm",   menu_mixer_drive_acb},
  { MT_ADV_CALLBACK, 1, "%+ddBm",   menu_mixer_drive_acb},
  { MT_ADV_CALLBACK, 0, "%+ddBm",   menu_mixer_drive_acb},
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};
#else
static const menuitem_t menu_lo_drive[] = {
  { MT_ADV_CALLBACK, 15, "%+ddBm",   menu_lo_drive_acb},
  { MT_ADV_CALLBACK, 14, "%+ddBm",   menu_lo_drive_acb},
  { MT_ADV_CALLBACK, 13, "%+ddBm",   menu_lo_drive_acb},
  { MT_ADV_CALLBACK, 12, "%+ddBm",   menu_lo_drive_acb},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#endif

static const menuitem_t  menu_modulation[] = {
  { MT_FORM | MT_TITLE,    0,  "MODULATION",NULL},
  { MT_FORM | MT_ADV_CALLBACK, MO_NONE,              "None",      menu_modulation_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_LOW, MO_AM,  "AM",   menu_modulation_acb},
  { MT_FORM | MT_ADV_CALLBACK, MO_NFM,               "Narrow FM", menu_modulation_acb},
  { MT_FORM | MT_ADV_CALLBACK, MO_WFM,               "Wide FM",   menu_modulation_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_LOW, MO_EXTERNAL, "External",  menu_modulation_acb},
  { MT_FORM | MT_KEYPAD,   KM_MODULATION,           "FREQ: %s",         "50Hz..6kHz"},
  { MT_FORM | MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t  menu_sweep[] = {
  { MT_FORM | MT_KEYPAD,   KM_SPAN,             "SPAN: %s",         VARIANT("0..350MHz", range_text)},
  { MT_FORM | MT_KEYPAD | MT_LOW, KM_LEVELSWEEP,"LEVEL CHANGE: %s", VARIANT("-70..70","-90..90")},
  { MT_FORM | MT_KEYPAD,   KM_SWEEP_TIME,       "SWEEP TIME: %s",   "0..600 seconds"},
  { MT_FORM | MT_SUBMENU,  0,                   "SWEEP POINTS",   menu_sweep_points_form},
  { MT_FORM | MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

char low_level_help_text[12] = "-76..-6";
char center_text[18] = "FREQ: %s";

const menuitem_t  menu_lowoutputmode[] = {
  { MT_FORM | MT_ADV_CALLBACK, 0,               "LOW OUTPUT            %s", menu_outputmode_acb},
//  { MT_FORM | MT_ADV_CALLBACK,  0,              "MOD: %s",   menu_smodulation_acb},
  { MT_FORM | MT_KEYPAD,   KM_CENTER,           center_text,         VARIANT("10kHz..350MHz","10kHz..850MHz")},
  { MT_FORM | MT_KEYPAD,   KM_LOWOUTLEVEL,      "LEVEL: %s",        low_level_help_text},
  { MT_FORM | MT_ADV_CALLBACK,  0,              MT_CUSTOM_LABEL,   menu_smodulation_acb},
  { MT_FORM | MT_ADV_CALLBACK,  0,              MT_CUSTOM_LABEL,   menu_sweep_acb},
#ifdef __SWEEP_RESTART__
  { MT_FORM | MT_ADV_CALLBACK,  0,              MT_CUSTOM_LABEL,   menu_restart_acb},
#endif
  { MT_FORM | MT_KEYPAD,  KM_EXT_GAIN,            "EXTERNAL GAIN: %s",   "-100..+100"},
#ifdef TINYSA4
  { MT_FORM | MT_SUBMENU,  255, S_RARROW" Settings", menu_settings},
#endif
  { MT_FORM | MT_CANCEL,   0,                   "MODE",             NULL },
  { MT_FORM | MT_NONE, 0, NULL, NULL } // sentinel
};

const menuitem_t  menu_highoutputmode[] = {
  { MT_FORM | MT_ADV_CALLBACK,  0,      "HIGH OUTPUT           %s", menu_outputmode_acb},
  { MT_FORM | MT_KEYPAD,    KM_CENTER,  center_text,         VARIANT("240MHz..959MHz",range_text)},
  { MT_FORM | MT_KEYPAD,   KM_HIGHOUTLEVEL,      "LEVEL: %s",        low_level_help_text /* "-76..-6" */},
  { MT_FORM | MT_ADV_CALLBACK,   0,     MT_CUSTOM_LABEL,   menu_smodulation_acb},
  { MT_FORM | MT_ADV_CALLBACK,  0,      MT_CUSTOM_LABEL,   menu_sweep_acb},
#ifdef __SWEEP_RESTART__
  { MT_FORM | MT_ADV_CALLBACK,  0,      MT_CUSTOM_LABEL,   menu_restart_acb},
#endif
  { MT_FORM | MT_KEYPAD,  KM_EXT_GAIN,            "EXTERNAL GAIN: %s",          "-100..+100"},
#ifdef TINYSA4
  { MT_FORM | MT_SUBMENU,  255, S_RARROW" Settings", menu_settings3},
#endif
  { MT_FORM | MT_CANCEL,    0,          "MODE",             NULL },
  { MT_FORM | MT_NONE, 0, NULL, NULL } // sentinel
};

static const menuitem_t  menu_average[] = {
  { MT_ADV_CALLBACK, AV_OFF,        "OFF",          menu_average_acb},
  { MT_ADV_CALLBACK, AV_MIN,        "MIN\nHOLD",    menu_average_acb},
  { MT_ADV_CALLBACK, AV_MAX_HOLD,   "MAX\nHOLD",    menu_average_acb},
  { MT_ADV_CALLBACK, AV_MAX_DECAY,  "MAX\nDECAY",   menu_average_acb},
  { MT_ADV_CALLBACK, AV_4,          "AVER 4",       menu_average_acb},
  { MT_ADV_CALLBACK, AV_16,         "AVER 16",      menu_average_acb},
#ifdef TINYSA4
  { MT_ADV_CALLBACK, AV_100,        "AVER",     menu_average_acb},
#endif
#ifdef __QUASI_PEAK__
  { MT_ADV_CALLBACK, AV_QUASI,      "QUASI\nPEAK",  menu_average_acb},
#endif
#ifdef __LIMITS__
  { MT_ADV_CALLBACK, AV_TABLE,      "TABLE"S_RARROW"\nTRACE",  menu_average_acb},
#endif
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_rbw[] = {
  { MT_ADV_CALLBACK, 0, "  AUTO",   menu_rbw_acb},
#ifdef TINYSA4
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(1,9), "%sHz",   menu_rbw_acb},
#else
  { MT_ADV_CALLBACK | MT_REPEATS, DATA_STARTS_REPEATS(1,6), "%4dkHz",   menu_rbw_acb},
#endif  
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back

};

#ifdef __VBW__
static const menuitem_t menu_vbw[] = {
  { MT_ADV_CALLBACK, 0, "     AUTO",   menu_vbw_acb},
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(1,5), "%b.2f RBW",   menu_vbw_acb},
  { MT_NONE,      0, NULL, menu_back} // next-> menu_back
};
#endif

static const menuitem_t menu_reffer[] = {
  { MT_FORM | MT_ADV_CALLBACK|MT_REPEATS,  DATA_STARTS_REPEATS(0,8), "%s", menu_reffer_acb},
  { MT_FORM | MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_atten[] = {
  { MT_ADV_CALLBACK | MT_LOW, 0,           "AUTO",    menu_atten_acb},
  { MT_KEYPAD | MT_LOW,   KM_ATTENUATION,  "MANUAL\n\b%s",  "0 - 30dB"},
  { MT_ADV_CALLBACK | MT_HIGH,0,           "0dB",     menu_atten_high_acb},
  { MT_ADV_CALLBACK | MT_HIGH,30,          "22.5 - 40dB",    menu_atten_high_acb},
  { MT_FORM | MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_reflevel[] = {
  { MT_ADV_CALLBACK,0,        "AUTO",    menu_reflevel_acb},
  { MT_KEYPAD,  KM_REFLEVEL,  "MANUAL\n\b%s",  NULL},
  { MT_CANCEL, 0,      S_LARROW" BACK", NULL },
  { MT_NONE,   0, NULL, NULL } // sentinel
};

const menuitem_t menu_marker_search[] = {
  { MT_CALLBACK, 4, "PEAK\n SEARCH",          menu_marker_search_cb },
  { MT_CALLBACK, 0, "MIN\n" S_LARROW" LEFT",  menu_marker_search_cb },
  { MT_CALLBACK, 1, "MIN\n" S_RARROW" RIGHT", menu_marker_search_cb },
  { MT_CALLBACK, 2, "MAX\n" S_LARROW" LEFT",  menu_marker_search_cb },
  { MT_CALLBACK, 3, "MAX\n" S_RARROW" RIGHT", menu_marker_search_cb },
  { MT_ADV_CALLBACK, 0, "ENTER\n%s",          menu_enter_marker_acb},
  { MT_ADV_CALLBACK, M_TRACKING,    "TRACKING",menu_marker_modify_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

const menuitem_t menu_marker_modify[] = {
  { MT_ADV_CALLBACK, 0,            "MARKER %s",      menu_marker_modify_acb},
  { MT_ADV_CALLBACK, M_DELTA,       "DELTA %s",    menu_marker_modify_acb},
  { MT_ADV_CALLBACK, M_NOISE,       "NOISE",    menu_marker_modify_acb},
  { MT_ADV_CALLBACK, M_TRACKING,    "TRACKING", menu_marker_modify_acb},
  { MT_ADV_CALLBACK, M_STORED,      "TRACE %s",   menu_marker_modify_acb},
  { MT_ADV_CALLBACK, M_AVER,      "TRACE\nAVERAGE",   menu_marker_modify_acb},
  { MT_SUBMENU,  0,                 "SEARCH",   menu_marker_search},
  { MT_CALLBACK, M_DELETE,          "DELETE",   menu_marker_delete_cb},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

#ifdef __LIMITS__
static const menuitem_t menu_limit_modify[] =
{
  { MT_KEYPAD,   KM_LIMIT_FREQ,   "FREQUENCY\n\b%s",          "Frequency"},
  { MT_KEYPAD,   KM_LIMIT_LEVEL,  "LEVEL\n\b%s",              "Level"},
  { MT_CALLBACK, 0,               "DISABLE",            menu_limit_disable_cb},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_limit_select[] = {
  { MT_ADV_CALLBACK | MT_REPEATS,   DATA_STARTS_REPEATS(0,LIMITS_MAX), MT_CUSTOM_LABEL, menu_limit_select_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};
#endif

#if 0
const menuitem_t menu_marker_sel[] = {
  { MT_CALLBACK, 1, "MARKER %d", menu_marker_sel_cb },
  { MT_CALLBACK, 2, "MARKER %d", menu_marker_sel_cb },
  { MT_CALLBACK, 3, "MARKER %d", menu_marker_sel_cb },
  { MT_CALLBACK, 4, "MARKER %d", menu_marker_sel_cb },
//  { MT_CALLBACK, 0, "ALL OFF", menu_marker_sel_cb },
  { MT_CALLBACK, 0, "DELTA", menu_marker_sel_cb },
  { MT_CALLBACK, 0, "NOISE", menu_marker_sel_cb },
  { MT_CALLBACK, 0, "TRACKING", menu_marker_sel_cb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};
#endif

const menuitem_t menu_marker_select[] = {
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(1,MARKER_COUNT), "MARKER %d", menu_marker_select_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_marker_ref_select[] = {
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(1,MARKER_COUNT), S_RARROW"MARKER %d", menu_marker_ref_select_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

const menuitem_t menu_marker_ops[] = {
  { MT_CALLBACK, ST_START,  S_RARROW" START",    menu_marker_op_cb },
  { MT_CALLBACK, ST_STOP,   S_RARROW" STOP",     menu_marker_op_cb },
  { MT_CALLBACK, ST_CENTER, S_RARROW" CENTER",   menu_marker_op_cb },
  { MT_CALLBACK, ST_SPAN,   S_RARROW" SPAN",     menu_marker_op_cb },
  { MT_CALLBACK, 4,         S_RARROW" REF LEVEL",menu_marker_op_cb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_marker[] = {
//  { MT_SUBMENU,  0, "SELECT\nMARKER",     menu_marker_sel},
  { MT_SUBMENU,  0, "MODIFY\nMARKERS",    menu_marker_modify},
  { MT_SUBMENU,  0, "MARKER\nOPS", menu_marker_ops},
  { MT_SUBMENU,  0, "SEARCH\nMARKER",     menu_marker_search},
  { MT_CALLBACK, 0, "RESET\nMARKERS",     menu_markers_reset_cb},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

#ifndef TINYSA4
static const menuitem_t menu_dfu[] = {
  { MT_FORM | MT_CALLBACK, 0, "ENTER DFU",      menu_dfu_cb},
  { MT_FORM | MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#endif

#ifdef __HARMONIC__
static const menuitem_t menu_harmonic[] =
{
  { MT_ADV_CALLBACK, 0,   "OFF",              menu_harmonic_acb},
  { MT_ADV_CALLBACK, 2,     "2",              menu_harmonic_acb},
  { MT_ADV_CALLBACK, 3,     "3",              menu_harmonic_acb},
  { MT_ADV_CALLBACK, 4,     "4",              menu_harmonic_acb},
  { MT_ADV_CALLBACK, 5,     "5",              menu_harmonic_acb},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#endif


static const menuitem_t menu_scanning_speed[] =
{
// { MT_ADV_CALLBACK,     SD_NORMAL,  "NORMAL",           menu_scanning_speed_acb},    // order must match definition of enum
// { MT_ADV_CALLBACK,     SD_PRECISE, PRECISE",           menu_scanning_speed_acb},
// { MT_ADV_CALLBACK | MT_LOW,SD_FAST,  "FAST",           menu_scanning_speed_acb},
// { MT_KEYPAD   | MT_LOW,KM_FAST_SPEEDUP,    "FAST\nSPEEDUP",   "2..20"},
  { MT_KEYPAD, KM_SAMPLETIME,        "SDELAY\n\b%s",   "250..10000, 0=auto"},              // This must be item 4 to match highlighting
  { MT_KEYPAD, KM_OFFSET_DELAY,      "ODELAY\n\b%s",   "250..10000, 0=auto"},              // This must be item 5 to match highlighting
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_sweep_points[] = {
#ifdef TINYSA4  
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(0,6), "%3d point", menu_points_acb },
#else
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(0,4), "%3d point", menu_points_acb },
#endif
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_sweep_points_form[] = {
#ifdef TINYSA4
  { MT_FORM|MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(0,6), "%3d point", menu_points_acb },
#else
  { MT_FORM|MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(0,4), "%3d point", menu_points_acb },
#endif
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_sweep_speed[] =
{
 { MT_ADV_CALLBACK,     SD_NORMAL,     "NORMAL",          menu_scanning_speed_acb},    // order must match definition of enum
 { MT_ADV_CALLBACK,     SD_PRECISE,    "PRECISE",         menu_scanning_speed_acb},
#ifdef TINYSA4
 { MT_ADV_CALLBACK,     SD_FAST,        "FAST",            menu_scanning_speed_acb},
#else
 { MT_ADV_CALLBACK | MT_LOW,SD_FAST,   "FAST",            menu_scanning_speed_acb},
#endif
 { MT_ADV_CALLBACK,     SD_NOISE_SOURCE,      "NOISE\nSOURCE",   menu_scanning_speed_acb},
#ifdef TINYSA4
 { MT_KEYPAD,           KM_FAST_SPEEDUP,"SPEEDUP\n\b%s",  "2..20, 0=disable"},
#else
 { MT_KEYPAD   | MT_LOW,KM_FAST_SPEEDUP,"SPEEDUP\n\b%s",  "2..20, 0=disable"},
#endif
 { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

#ifdef TINYSA4
static const menuitem_t menu_settings4[];
#endif

static const menuitem_t menu_settings3[] =
{
#ifdef TINYSA4
//  { MT_KEYPAD,   KM_GRIDLINES,  "MINIMUM\nGRIDLINES", "Enter minimum horizontal grid divisions"},
  { MT_ADV_CALLBACK,     0,     "ADF OUT",          menu_adf_out_acb},
  { MT_ADV_CALLBACK,     0,     "ENABLE\nULTRA",    menu_ultra_acb},
  { MT_KEYPAD,   KM_ULTRA_START,"ULTRASTART\n\b%s",   "10G=auto"},
  { MT_ADV_CALLBACK,     0,     "ENABLE\nDIRECT",    menu_direct_acb},
//  { MT_KEYPAD | MT_LOW, KM_IF2,  "IF2 FREQ",           "Set to zero for no IF2"},
  { MT_KEYPAD,  KM_R,  "R\n\b%s",           "Set R"},
  { MT_KEYPAD,  KM_MOD,  "MODULO\n\b%s",           "Set MODULO"},
  { MT_KEYPAD,  KM_CP,  "CP",             "Set CP"},
#ifdef __WAIT_CTS_WHILE_SLEEPING__
  { MT_ADV_CALLBACK,     0,     "SLEEP\nWAIT",    menu_sleep_acb},
#endif
//  { MT_ADV_CALLBACK | MT_LOW, 0,    "ULTRA\nMODE",      menu_settings_ultra_acb},
#ifdef __HAM_BAND__
  { MT_ADV_CALLBACK, 0,         "HAM\nBANDS",         menu_settings_ham_bands},
#endif
  { MT_SUBMENU,  0,             S_RARROW" MORE",     menu_settings4},
#else
#ifdef __ULTRA__
  { MT_ADV_CALLBACK,     0,     "ENABLE\nULTRA",    menu_ultra_acb},
  { MT_KEYPAD,   KM_ULTRA_START,        "ULTRA\nSTART",   "Enter ULTRA mode start freq"},
  { MT_ADV_CALLBACK,     0,     "DEBUG\nSPUR",        menu_debug_spur_acb},
#endif
  { MT_KEYPAD,   KM_10MHZ,      "CORRECT\nFREQUENCY", "Enter actual l0MHz frequency"},
  { MT_KEYPAD,   KM_GRIDLINES,  "MINIMUM\nGRIDLINES", "Enter minimum horizontal grid divisions"},
  { MT_CALLBACK,        0 ,     "CLEAR\nCONFIG",    menu_clearconfig_cb},
  { MT_ADV_CALLBACK,     0,     "PULSE\nHIGH",            menu_settings_pulse_acb},
#ifdef __HARMONIC__
  { MT_SUBMENU | MT_HIGH,0,               "HARMONIC",         menu_harmonic},
//  { MT_ADV_CALLBACK,0,          "SPUR\nREMOVAL",          menu_harmonic_spur_acb},
#endif
#ifdef __USE_SERIAL_CONSOLE__
  { MT_SUBMENU,  0, "CONNECTION", menu_connection},
#endif
#ifdef __HAM_BAND__
  { MT_ADV_CALLBACK, 0,         "HAM\nBANDS",         menu_settings_ham_bands},
#endif
#endif  // TINYSA4
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#ifdef TINYSA4
static const menuitem_t menu_settings4[] =
{
 { MT_ADV_CALLBACK,     0,     "DEBUG\nFREQ",          menu_debug_freq_acb},
 { MT_ADV_CALLBACK,     0,     "DEBUG\nAVOID",          menu_debug_avoid_acb},
 { MT_ADV_CALLBACK,     0,     "DEBUG\nSPUR",        menu_debug_spur_acb},
 { MT_ADV_CALLBACK,     0,     "PROGRESS\nBAR",        menu_progress_bar_acb},
#if 0                                                                           // only used during development
  { MT_KEYPAD,   KM_COR_AM,     "COR\nAM", "Enter AM modulation correction"},
  { MT_KEYPAD,   KM_COR_WFM,     "COR\nWFM", "Enter WFM modulation correction"},
  { MT_KEYPAD,   KM_COR_NFM,     "COR\nNFM", "Enter NFM modulation correction"},
#endif
#ifdef __NOISE_FIGURE__
  { MT_KEYPAD,   KM_NF,        "NF\n\b%s", "Enter tinySA noise figure"},
#endif
#ifdef __SD_CARD_LOAD__
  { MT_CALLBACK,        0 ,     "LOAD\nCONFIG.INI",    menu_load_config_cb},
//  { MT_CALLBACK,        1 ,     "LOAD\nSETTING.INI",    menu_load_config_cb},
#endif
  { MT_CALLBACK,        0 ,     "CLEAR\nCONFIG",    menu_clearconfig_cb},
#ifdef __HARMONIC__
  { MT_SUBMENU,0,               "HARMONIC",         menu_harmonic},
#endif
  { MT_ADV_CALLBACK,     0,     "LINEAR\nAVERAGING",          menu_linear_averaging_acb},
//  { MT_SUBMENU,  0,             S_RARROW" MORE",     menu_settings3},
  { MT_KEYPAD,   KM_DIRECT_START,     "DSTART\n\b%s", ""},
  { MT_KEYPAD,   KM_DIRECT_STOP,     "DSTOP\n\b%s", ""},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#endif

static const menuitem_t menu_settings2[] =
{
  { MT_ADV_CALLBACK, 0,             "AGC",           menu_settings_agc_acb},
  { MT_ADV_CALLBACK, 0,             "LNA",           menu_settings_lna_acb},
  { MT_ADV_CALLBACK | MT_LOW, 0,    "BPF",           menu_settings_bpf_acb},
  { MT_ADV_CALLBACK | MT_LOW, 0,    "BELOW IF",      menu_settings_below_if_acb},
  { MT_KEYPAD,   KM_DECAY,      "DECAY\n\b%s",   "0..1000000ms or sweeps"},
#ifdef __QUASI_PEAK__
  { MT_KEYPAD,   KM_ATTACK,      "ATTACK\n\b%s",   "0..100000ms"},
#endif
#ifdef TINYSA4
  { MT_KEYPAD,   KM_NOISE,      "NOISE LEV\n\b%s",   "2..20 dB"},
  { MT_KEYPAD,   KM_30MHZ,      "30MHz*100\n\b%s", "Enter actual 30MHz * 100"},
#endif
  { MT_SUBMENU,  0,             S_RARROW" MORE",     menu_settings3},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

#ifdef TINYSA4
static const menuitem_t menu_curve3[] = {
  { MT_FORM | MT_ADV_CALLBACK | MT_REPEATS, DATA_STARTS_REPEATS(14,6), MT_CUSTOM_LABEL, menu_curve_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_curve2[] = {
  { MT_FORM | MT_ADV_CALLBACK | MT_REPEATS, DATA_STARTS_REPEATS(7,7), MT_CUSTOM_LABEL, menu_curve_acb },
  { MT_FORM | MT_SUBMENU,      0,  S_RARROW" MORE",     menu_curve3},
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_curve[] = {
  { MT_FORM | MT_ADV_CALLBACK | MT_REPEATS, DATA_STARTS_REPEATS(0,7), MT_CUSTOM_LABEL, menu_curve_acb },
  { MT_FORM | MT_SUBMENU,      0,  S_RARROW" MORE",     menu_curve2},
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_curve_confirm[] = {
  { MT_CALLBACK, 1,               "OK",       menu_curve_confirm_cb },
  { MT_CALLBACK, 0,               "CANCEL",   menu_curve_confirm_cb },
  { MT_NONE, 0, NULL, NULL } // sentinel
};

static const menuitem_t menu_noise_figure_confirm[] = {
  { MT_CALLBACK, 1,               "STORE\nTINYSA NF",       menu_noise_figure_confirm_cb },
  { MT_CALLBACK, 0,               "CANCEL",   menu_noise_figure_confirm_cb },
  { MT_NONE, 0, NULL, NULL } // sentinel
};

#endif

static const menuitem_t menu_actual_power[] =
{
 { MT_KEYPAD,           KM_ACTUALPOWER, "INPUT\nLEVEL",  "dBm"},
 { MT_ADV_CALLBACK,     0,              "OUTPUT\nLEVEL", menu_output_level_acb},
#ifdef TINYSA4
 { MT_CALLBACK,     0,                  "INPUT\nCURVE",  menu_input_curve_prepare_cb},
 { MT_CALLBACK,     0,                  "LNA\nCURVE",    menu_lna_curve_prepare_cb},
 { MT_CALLBACK,     0,                  "ULTRA\nCURVE",  menu_ultra_curve_prepare_cb},
 { MT_CALLBACK,     0,                  "LNA_U\nCURVE",    menu_lna_u_curve_prepare_cb},
 { MT_CALLBACK,     0,                  "OUTPUT\nCURVE", menu_output_curve_prepare_cb},
#endif
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};


static const menuitem_t menu_settings[] =
{
  { MT_SUBMENU, 0,              "LEVEL\nCORRECTION",  menu_actual_power},
  { MT_ADV_CALLBACK | MT_LOW, 0,"LO OUTPUT", menu_lo_output_acb},
  { MT_KEYPAD | MT_LOW, KM_IF,  "IF FREQ\n\b%s",           "0=auto IF"},
  { MT_SUBMENU,0,               "SCAN\nSPEED",        menu_scanning_speed},
#ifndef TINYSA4
  { MT_KEYPAD, KM_REPEAT,       "SAMPLE REP\n\b%s",    "1..100"},
#endif
#ifdef TINYSA4
  { MT_SUBMENU | MT_LOW,0,      "MIXER\nDRIVE",      menu_mixer_drive},
  { MT_ADV_CALLBACK,     0,     "PULSE\nHIGH",            menu_settings_pulse_acb},
#ifdef __USE_SERIAL_CONSOLE__
  { MT_SUBMENU,  0, "CONNECTION", menu_connection},
#endif
#else
  { MT_SUBMENU | MT_LOW,0,      "MIXER\nDRIVE",      menu_lo_drive},
#endif
  { MT_SUBMENU,  0,             S_RARROW" MORE",     menu_settings2},
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};

#ifdef __NOISE_FIGURE__
static const menuitem_t menu_measure_noise_figure[] =
{
 { MT_ADV_CALLBACK,            M_NF_TINYSA,         "MEASURE\nTINYSA NF",menu_measure_acb},
 { MT_ADV_CALLBACK,            M_NF_STORE,          "STORE\nTINYSA NF",menu_measure_acb},
 { MT_ADV_CALLBACK,            M_NF_VALIDATE,       "VALIDATE\nTINYSA NF",menu_measure_acb},
 { MT_ADV_CALLBACK,            M_NF_AMPLIFIER,      "MEASURE\nAMP NF",menu_measure_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};
#endif

static const menuitem_t menu_measure2[] = {
  { MT_ADV_CALLBACK,            M_AM,           "AM",           menu_measure_acb},
  { MT_ADV_CALLBACK,            M_FM,           "FM",           menu_measure_acb},
  { MT_ADV_CALLBACK,            M_THD,          "THD",           menu_measure_acb},
#ifdef __CHANNEL_POWER__
  { MT_ADV_CALLBACK,            M_CP,           "CHANNEL\nPOWER",menu_measure_acb},
#endif
#ifdef __LINEARITY__
{ MT_ADV_CALLBACK | MT_LOW,   M_LINEARITY,  "LINEAR",         menu_measure_acb},
#endif
#ifdef __NOISE_FIGURE__
{ MT_SUBMENU | MT_LOW,          0,            "NOISE\nFIGURE",    menu_measure_noise_figure},
#endif
#ifdef __FFT_DECONV__
  { MT_ADV_CALLBACK,            M_DECONV,  "DECONV",         menu_measure_acb},
#endif
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_measure[] = {
  { MT_ADV_CALLBACK,            M_OFF,        "OFF",            menu_measure_acb},
  { MT_ADV_CALLBACK,            M_IMD,        "HARMONIC",       menu_measure_acb},
  { MT_ADV_CALLBACK,            M_OIP3,       "OIP3",           menu_measure_acb},
  { MT_ADV_CALLBACK,            M_PHASE_NOISE,"PHASE\nNOISE",   menu_measure_acb},
  { MT_ADV_CALLBACK,            M_SNR,        "SNR",            menu_measure_acb},
  { MT_ADV_CALLBACK,            M_PASS_BAND,  "-3dB\nWIDTH",     menu_measure_acb},
  { MT_SUBMENU,  0,             S_RARROW" MORE",                menu_measure2},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

#ifdef __CALIBRATE__
static const menuitem_t menu_calibrate[] =
{
  { MT_FORM | MT_TITLE,      0, "Connect HIGH and LOW",  NULL},
  { MT_FORM | MT_CALLBACK,   0, "CALIBRATE",                 menu_calibrate_cb},
  { MT_FORM | MT_CALLBACK,   0, "RESET CALIBRATION",          menu_calibrate_cb},
  { MT_FORM | MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#endif

#ifdef __USE_SERIAL_CONSOLE__
const menuitem_t menu_serial_speed[] = {
  { MT_ADV_CALLBACK|MT_REPEATS, DATA_STARTS_REPEATS(0,10), "%u", menu_serial_speed_acb },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_connection[] = {
  { MT_ADV_CALLBACK, _MODE_USB,    "USB",    menu_connection_acb },
  { MT_ADV_CALLBACK, _MODE_SERIAL, "SERIAL", menu_connection_acb },
  { MT_SUBMENU,  0, "SERIAL\nSPEED", menu_serial_speed },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};
#endif

const menuitem_t menu_touch[] = {
  { MT_CALLBACK, CONFIG_MENUITEM_TOUCH_CAL,  "TOUCH CAL",  menu_config_cb},
  { MT_CALLBACK, CONFIG_MENUITEM_TOUCH_TEST, "TOUCH TEST", menu_config_cb},
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};

#ifdef __USE_RTC__
const menuitem_t menu_date_time[] = {
  { MT_KEYPAD, KM_RTC_TIME,  "SET TIME\n\b%s",                          0 },        // MUST BE BEFORE DATE
  { MT_KEYPAD, KM_RTC_DATE,  "SET DATE\n\b%s",                          0 },
  { MT_NONE, 0, NULL, menu_back} // next-> menu_back
};
#endif


static const menuitem_t menu_config[] = {
  { MT_SUBMENU,  0,                        "TOUCH",     menu_touch},
  { MT_CALLBACK, CONFIG_MENUITEM_SELFTEST, "SELF TEST", menu_config_cb},
#ifdef __CALIBRATE__
  { MT_SUBMENU,  0,                        "LEVEL CAL", menu_calibrate},
#endif
  { MT_CALLBACK, CONFIG_MENUITEM_VERSION,  "VERSION",   menu_config_cb},
#ifdef __SPUR__
  { MT_ADV_CALLBACK,0,          "%s",          menu_spur_acb},
#endif
#ifdef TINYSA4
  { MT_KEYPAD, KM_REPEAT,       "SAMPLE REP\n\b%s",    "1..100"},
#endif
#ifdef __LCD_BRIGHTNESS__
  { MT_CALLBACK, 0, "BRIGHTNESS", menu_brightness_cb},
#endif
#ifdef __USE_RTC__
  { MT_SUBMENU,  0, "DATE\nTIME", menu_date_time},
#endif
  { MT_SUBMENU,  0, "EXPERT\nCONFIG", menu_settings},
#ifndef TINYSA4
  { MT_SUBMENU,  0, S_RARROW" DFU",  menu_dfu},
#endif
  { MT_NONE,     0, NULL, menu_back} // next-> menu_back
};
#if 0
static const menuitem_t menu_storage[] =
{
 { MT_ADV_CALLBACK,0,          "TRACE %d",        menu_storage_acb},
 { MT_ADV_CALLBACK,1,          "%s",              menu_storage_acb},
 { MT_ADV_CALLBACK,1,          "DISPLAY",         menu_storage_acb},
 { MT_ADV_CALLBACK,2,          "COPY\nFROM",      menu_storage_acb},
 { MT_ADV_CALLBACK,3,          "SUBTRACT",        menu_storage_acb},
 { MT_ADV_CALLBACK,4,          "NORMALIZE",       menu_storage_acb},
 { MT_ADV_CALLBACK,5,          "WRITE\n"S_RARROW"SD",menu_storage_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};
#endif
static const menuitem_t menu_trace[] =
{
 { MT_ADV_CALLBACK|MT_REPEATS,DATA_STARTS_REPEATS(0,TRACES_MAX),          "TRACE %d",        menu_trace_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_marker_trace[] =
{
 { MT_ADV_CALLBACK|MT_REPEATS,DATA_STARTS_REPEATS(0,TRACES_MAX),          "TRACE %d",        menu_marker_trace_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_store_trace[] =
{
 { MT_ADV_CALLBACK|MT_REPEATS,DATA_STARTS_REPEATS(0,TRACES_MAX),          MT_CUSTOM_LABEL,        menu_store_trace_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_subtract_trace[] =
{
 { MT_ADV_CALLBACK|MT_REPEATS,DATA_STARTS_REPEATS(0,TRACES_MAX+1),          MT_CUSTOM_LABEL,        menu_subtract_trace_acb},
 { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_traces[] =
{
 { MT_ADV_CALLBACK,0,          "TRACE %d",                  menu_traces_acb},
 { MT_ADV_CALLBACK,1,          "ENABLE",                    menu_traces_acb},
 { MT_ADV_CALLBACK,2,          "FREEZE",                    menu_traces_acb},
 { MT_ADV_CALLBACK,3,          MT_CUSTOM_LABEL,             menu_traces_acb},       // Calc
 { MT_ADV_CALLBACK,4,          "NORMALIZE",                 menu_traces_acb},
 { MT_ADV_CALLBACK,5,          MT_CUSTOM_LABEL,             menu_traces_acb},       // Trace Math
 { MT_SUBMENU,     0,          "COPY\n"S_RARROW"TRACE",     menu_store_trace},
#ifdef TINYSA4
 { MT_ADV_CALLBACK,6,          "WRITE\n"S_RARROW"SD",       menu_traces_acb},
#endif
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_display[] = {
  { MT_ADV_CALLBACK,0,             "PAUSE\nSWEEP",    menu_pause_acb},
  { MT_ADV_CALLBACK,1,             "WATER\nFALL",     menu_waterfall_acb},
#ifdef __LEVEL_METER__
  { MT_ADV_CALLBACK,1,             "BIG\nNUMBER",     menu_level_meter_acb},
#endif
#ifdef __VBW__
  { MT_SUBMENU,     0,             "VBW",             menu_vbw},
#endif
  { MT_KEYPAD,      KM_SWEEP_TIME, "SWEEP\nTIME",     "0..600s, 0=disable"},       // This must be item 3 to match highlighting
  { MT_SUBMENU,     0,             "SWEEP\nPOINTS",   menu_sweep_points},
  { MT_SUBMENU,     0,             "SWEEP\nACCURACY",  menu_sweep_speed},
  { MT_ADV_CALLBACK,0,             "ROTATE\nDISPLAY",    menu_flip_acb},
//#ifdef __REMOTE_DESKTOP__
//  { MT_ADV_CALLBACK,0,          "SEND\nDISPLAY",    menu_send_display_acb},
//#endif
//  { MT_KEYPAD,  KM_SWEEP_TIME,  "SWEEP\nTIME",    NULL},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_unit[] =
{
  { MT_ADV_CALLBACK,U_DBM,   "dBm",             menu_unit_acb},
  { MT_ADV_CALLBACK,U_DBMV,  "dBmV",            menu_unit_acb},
  { MT_ADV_CALLBACK,U_DBUV,  "dB"S_MICRO"V",    menu_unit_acb},
  { MT_ADV_CALLBACK,U_VOLT,  "Volt",            menu_unit_acb},
//{ MT_ADV_CALLBACK,U_UVOLT, S_MICRO"Volt",     menu_unit_acb},
  { MT_ADV_CALLBACK,U_WATT,  "Watt",            menu_unit_acb},
//{ MT_ADV_CALLBACK,U_UWATT, S_MICRO"Watt",     menu_unit_acb},
#ifdef TINYSA4
  { MT_ADV_CALLBACK,U_RAW,   "RAW",             menu_unit_acb},
#endif
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_trigger[] = {
  { MT_ADV_CALLBACK, T_AUTO,     "AUTO",           menu_trigger_acb},
  { MT_ADV_CALLBACK, T_NORMAL,   "NORMAL",         menu_trigger_acb},
  { MT_ADV_CALLBACK, T_SINGLE,   "SINGLE",         menu_trigger_acb},
//  { MT_ADV_CALLBACK, T_DONE,     "READY",          menu_trigger_acb},
  { MT_KEYPAD,       KM_TRIGGER, "TRIGGER LEV\n\b%s", NULL},
  { MT_ADV_CALLBACK, T_UP,       "UP\nEDGE",       menu_trigger_acb},
  { MT_ADV_CALLBACK, T_DOWN,     "DOWN\nEDGE",     menu_trigger_acb},
  { MT_ADV_CALLBACK, T_MODE,     "%s\nTRIGGER",     menu_trigger_acb},
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_level[] = {
  { MT_SUBMENU, 0,              "REF LEVEL",    menu_reflevel},
//{ MT_SUBMENU, 0,              "SCALE/DIV",    menu_scale_per},
  { MT_CALLBACK,0,              "SCALE/DIV",    menu_scale_cb},
  { MT_SUBMENU, 0,              "ATTENUATE",    menu_atten},
//  { MT_SUBMENU,0,             "CALC",         menu_average},
  { MT_SUBMENU, 0,              "UNIT",         menu_unit},
  { MT_KEYPAD,  KM_EXT_GAIN,      "EXT GAIN\n\b%s",NULL},
#ifdef TINYSA4
  { MT_ADV_CALLBACK | MT_LOW ,0,"LNA",          menu_extra_lna_acb},
 #endif
  { MT_SUBMENU,  0,             "TRIGGER",      menu_trigger},
#ifdef __LISTEN__
  { MT_ADV_CALLBACK, 0,             "LISTEN",       menu_listen_acb},
#endif
  { MT_NONE,   0, NULL, menu_back} // next-> menu_back
};

static const menuitem_t menu_stimulus[] = {
  { MT_KEYPAD,  KM_START,       "START\n\b%s",       NULL},
  { MT_KEYPAD,  KM_STOP,        "STOP\n\b%s",        NULL},
  { MT_KEYPAD,  KM_CENTER,      "CENTER\n\b%s",      NULL},
  { MT_KEYPAD,  KM_SPAN,        "SPAN\n\b%s",        NULL},
  { MT_KEYPAD,  KM_CW,          "ZERO SPAN",   NULL},
  { MT_KEYPAD,  KM_VAR,         "JOG STEP\n\b%s","0 = AUTO"},
  { MT_SUBMENU,0,               "RBW",         menu_rbw},
  { MT_ADV_CALLBACK,0,          "SHIFT\nFREQ", menu_shift_acb},
  { MT_NONE,    0, NULL, menu_back} // next-> menu_back
};

const menuitem_t menu_mode[] = {
//  { MT_FORM | MT_TITLE,                 0,                      "tinySA MODE",           NULL},
  { MT_FORM | MT_ADV_CALLBACK | MT_ICON,    I_LOW_INPUT+I_SA,       "%s to LOW in",      menu_mode_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_ICON,    I_HIGH_INPUT+I_SA,      "%s to HIGH in",     menu_mode_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_ICON,    I_LOW_OUTPUT+I_SINUS,   "%s to LOW out",     menu_mode_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_ICON,    I_HIGH_OUTPUT+I_GEN,    "%s to HIGH out",    menu_mode_acb},
  { MT_FORM | MT_ADV_CALLBACK | MT_ICON,    I_CONNECT+I_GEN,        "Cal. output: %s",   menu_sreffer_acb},
//  { MT_SUBMENU,  0, "EXPERT\nCONFIG", menu_settings3},

//  { MT_FORM | MT_CANCEL,   0, S_RARROW" BACK", NULL },
  { MT_FORM | MT_NONE,     0, NULL, NULL } // sentinel
};

static const menuitem_t menu_top[] = {
  { MT_SUBMENU,  0, "PRESET",       menu_load_preset},
  { MT_SUBMENU,  0, "FREQUENCY",    menu_stimulus},
  { MT_SUBMENU,  0, "LEVEL",        menu_level},
  { MT_SUBMENU,  0, "TRACE",        menu_traces},
  { MT_SUBMENU,  0, "DISPLAY",      menu_display},
  { MT_SUBMENU,  0, "MARKER",       menu_marker},
  { MT_SUBMENU,  0, "MEASURE",      menu_measure},
  { MT_SUBMENU,  0, "CONFIG",       menu_config},
  { MT_SUBMENU,  0, "MODE",         menu_mode},
  { MT_NONE,     0, NULL, NULL } // sentinel,
 // MENUITEM_CLOSE,
};

// ===[MENU DEFINITION END]======================================================

#define ACTIVE_COLOR RGBHEX(0x007FFF)

static void menu_item_modify_attribute(                     // To modify menu buttons with keypad modes
    const menuitem_t *menu, int item, ui_button_t *button)
{
  if (menu == menu_settings) {
//    if (item == 2)
//      button->icon = setting.auto_IF ? BUTTON_ICON_CHECK_AUTO : BUTTON_ICON_CHECK_MANUAL;
  } else if (menu == menu_scanning_speed) {
//    if (item == 0)
//      button->icon = setting.step_delay > 0 ? BUTTON_ICON_CHECK_MANUAL : BUTTON_ICON_CHECK_AUTO;
//    else if (item == 1)
//      button->icon =setting.offset_delay > 0 ? BUTTON_ICON_CHECK_MANUAL : BUTTON_ICON_CHECK_AUTO;
  } else if (menu == menu_display) {
    if (item == 4)
      button->icon = setting.sweep_time_us != 0 ? BUTTON_ICON_CHECK_MANUAL : BUTTON_ICON_CHECK_AUTO;
  } else if (menu == menu_sweep_speed) {
    if (item == 3)
    button->icon = setting.fast_speedup != 0 ? BUTTON_ICON_CHECK_MANUAL : BUTTON_ICON_CHECK_AUTO;
  } else if (menu == menu_reflevel) {
    if (item == 1)
      button->icon = setting.auto_reflevel ? BUTTON_ICON_GROUP: BUTTON_ICON_GROUP_CHECKED;
  } else if (menu == menu_atten) {
    if (item == 1)
      button->icon = setting.auto_attenuation ? BUTTON_ICON_GROUP: BUTTON_ICON_GROUP_CHECKED;
  }
}

static void fetch_numeric_target(uint8_t mode)
{
  switch (mode) {
  case KM_START:
    uistat.freq_value = get_sweep_frequency(ST_START) + (setting.frequency_offset - FREQUENCY_SHIFT);
    plot_printf(uistat.text, sizeof uistat.text, "%.3QMHz", uistat.freq_value);
    break;
  case KM_STOP:
    uistat.freq_value = get_sweep_frequency(ST_STOP) + (setting.frequency_offset - FREQUENCY_SHIFT);
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_CENTER:
    uistat.freq_value = get_sweep_frequency(ST_CENTER) + (setting.frequency_offset - FREQUENCY_SHIFT);
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_SPAN:
    uistat.freq_value = get_sweep_frequency(ST_SPAN);
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_CW:
    uistat.freq_value = get_sweep_frequency(ST_CW) + (setting.frequency_offset - FREQUENCY_SHIFT);
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_SCALE:
  case KM_LINEAR_SCALE:
    uistat.value = setting.scale;
    plot_printf(uistat.text, sizeof uistat.text, "%f/", uistat.value);
    break;
  case KM_REFLEVEL:
    uistat.value = setting.reflevel;
    plot_printf(uistat.text, sizeof uistat.text, "%f", uistat.value);
    break;
  case KM_ATTENUATION:
    uistat.value = get_attenuation();
    plot_printf(uistat.text, sizeof uistat.text, "%ddB", ((int32_t)uistat.value));
     break;
  case KM_ACTUALPOWER:
    uistat.value = get_level_offset();
    plot_printf(uistat.text, sizeof uistat.text, "%ddB", ((int32_t)uistat.value));
    break;
  case KM_IF:
    uistat.freq_value = setting.frequency_IF;
    if (!setting.auto_IF)
      plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    else
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");
    break;
#ifdef TINYSA4
  case KM_IF2:
    uistat.freq_value = config.frequency_IF2;
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_R:
    uistat.value = setting.R;
    if (setting.R)
      plot_printf(uistat.text, sizeof uistat.text, "%d", setting.R);
    else
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");
    break;
  case KM_MOD:
    if (local_modulo)
      plot_printf(uistat.text, sizeof uistat.text, "%d", local_modulo);
    else
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");
   break;
  case KM_CP:
    uistat.value = ADF4350_modulo;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ADF4350_modulo);
    break;
#endif
  case KM_SAMPLETIME:
    uistat.value = setting.step_delay;
    if (uistat.value)
      plot_printf(uistat.text, sizeof uistat.text, "%dus", ((int32_t)uistat.value));
    else
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");
    break;
  case KM_FAST_SPEEDUP:
    uistat.value = setting.fast_speedup;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ((int32_t)uistat.value));
    break;
  case KM_REPEAT:
    uistat.value = setting.repeat;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ((int32_t)uistat.value));
    break;
  case KM_LOWOUTLEVEL:
    uistat.value = get_level();           // compensation for dB offset during low output mode
    float end_level =  ((int32_t)uistat.value)+setting.level_sweep;
    if (end_level < level_min())
      end_level = level_min();
    if (end_level > level_max())
      end_level = level_max();
    uistat.value += setting.external_gain;
    end_level += setting.external_gain;
    if (setting.level_sweep != 0)
      plot_printf(uistat.text, sizeof uistat.text, "%.1f to %.1fdBm", uistat.value, end_level);
    else
      plot_printf(uistat.text, sizeof uistat.text, "%+.1fdBm", uistat.value);
    break;
  case KM_HIGHOUTLEVEL:
    uistat.value = get_level();           // compensation for dB offset during low output mode
    uistat.value += setting.external_gain;
    plot_printf(uistat.text, sizeof uistat.text, "%+.1fdBm", uistat.value);
    break;
  case KM_DECAY:
    uistat.value = setting.decay;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ((int32_t)uistat.value));
    break;
#ifdef __QUASI_PEAK__
  case KM_ATTACK:
    uistat.value = setting.attack;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ((int32_t)uistat.value));
    break;
#endif
#ifdef __ULTRA__
  case KM_ULTRA_START:
    uistat.freq_value = config.ultra_threshold;
    if (config.ultra_threshold == ULTRA_AUTO)
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");
    else
      plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value );
    break;
  case KM_DIRECT_START:
    uistat.freq_value = config.direct_start;
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_DIRECT_STOP:
    uistat.freq_value = config.direct_stop;
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
#endif
#ifdef __LIMITS__
  case KM_LIMIT_FREQ:
    uistat.freq_value = setting.limits[current_trace][active_limit].frequency;
    plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    break;
  case KM_LIMIT_LEVEL:
    uistat.value = value(setting.limits[current_trace][active_limit].level);
    plot_printf(uistat.text, sizeof uistat.text, "%.1f", uistat.value);
    break;
#endif
  case KM_NOISE:
    uistat.value = setting.noise;
    plot_printf(uistat.text, sizeof uistat.text, "%d", ((int32_t)uistat.value));
    break;
#ifdef TINYSA4
  case KM_30MHZ:
    uistat.freq_value = config.setting_frequency_30mhz;
    plot_printf(uistat.text, sizeof uistat.text, "%.5QHz", uistat.freq_value);
    break;
#else
  case KM_10MHZ:
    uistat.freq_value = config.setting_frequency_10mhz;
    plot_printf(uistat.text, sizeof uistat.text, "%3.6fMHz", uistat.freq_value);
    break;
#endif
  case KM_EXT_GAIN:
    uistat.value = setting.external_gain;
    plot_printf(uistat.text, sizeof uistat.text, "%.1fdB", uistat.value);
    break;
  case KM_LEVELSWEEP:
    uistat.value = setting.level_sweep;
    plot_printf(uistat.text, sizeof uistat.text, "%.1fdB", uistat.value);
    break;
  case KM_SWEEP_TIME:
//    if (setting.sweep_time_us < calc_min_sweep_time_us())
//      uistat.value = calc_min_sweep_time_us();
//    else
      uistat.value = setting.sweep_time_us;
    uistat.value /= (float)ONE_SECOND_TIME;
    plot_printf(uistat.text, sizeof uistat.text, "%.3Fs", uistat.value);
    break;
  case KM_TRIGGER:
    uistat.value = setting.trigger_level;
    plot_printf(uistat.text, sizeof uistat.text, "%.1fdB", uistat.value);
    break;
  case KM_MARKER:
    if (active_marker >=0) {
      uistat.freq_value = markers[active_marker].frequency;
      plot_printf(uistat.text, sizeof uistat.text, "%.3QHz", uistat.freq_value);
    }
    break;
  case KM_MODULATION:
    if (active_marker >=0) {
      uistat.value = setting.modulation_frequency;
      plot_printf(uistat.text, sizeof uistat.text, "%7.0fHz", uistat.value);
    }
    break;
  case KM_VAR:
    uistat.freq_value = setting.frequency_var;
    if ( setting.frequency_var)
      plot_printf(uistat.text, sizeof uistat.text, "%.4QHz", setting.frequency_var);
    else
      plot_printf(uistat.text, sizeof uistat.text, "AUTO");

    break;
#ifdef __NOISE_FIGURE__
  case KM_NF:
    uistat.value = config.noise_figure;
    plot_printf(uistat.text, sizeof uistat.text, "%.1fdB", uistat.value);
    break;
#endif
#ifdef __USE_RTC__
  case KM_RTC_TIME:
  {
    uint32_t tr = rtc_get_tr_bin(); // TR read first
    plot_printf(uistat.text, sizeof uistat.text, "%02d:%02d:%02d",
      RTC_TR_HOUR(dr),
      RTC_TR_MIN(dr),
      RTC_TR_SEC(dr));
  }
  break;
  case KM_RTC_DATE:
  {
    uint32_t dr = rtc_get_dr_bin(); // DR read second
    plot_printf(uistat.text, sizeof uistat.text, "20%02d/%02d/%02d",
      RTC_DR_YEAR(dr),
      RTC_DR_MONTH(dr),
      RTC_DR_DAY(dr));
  }
    break;
#endif
  }
}

static void
set_numeric_value(void)
{
  switch (keypad_mode) {
  case KM_START:
    set_sweep_frequency(ST_START, uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT));
    break;
  case KM_STOP:
    set_sweep_frequency(ST_STOP, (freq_t)(uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT)));
    break;
  case KM_CENTER:
    set_sweep_frequency(ST_CENTER, uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT));
    break;
  case KM_SPAN:
    setting.modulation = MO_NONE;
    set_sweep_frequency(ST_SPAN, uistat.freq_value);
    break;
  case KM_CW:
    set_sweep_frequency(ST_CW, uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT));
    break;
  case KM_LINEAR_SCALE:
  case KM_SCALE:
    user_set_scale(uistat.value);
    break;
  case KM_REFLEVEL:
    user_set_reflevel(uistat.value);
    break;
  case KM_ATTENUATION:
    setting.auto_attenuation = false;
    set_attenuation(uistat.value);
    break;
  case KM_ACTUALPOWER:
    set_actual_power(uistat.value);
    config_save();
    break;
  case KM_IF:
    set_IF(uistat.freq_value);
//    config_save();
    break;
#ifdef TINYSA4
  case KM_IF2:
    set_IF2(uistat.freq_value);
//    config_save();
    break;
  case KM_R:
    set_R(uistat.value);
//    config_save();
    break;
  case KM_MOD:
    set_modulo(uistat.value);
    break;
  case KM_CP:
    ADF4351_CP((int)uistat.value);
//    config_save();
    break;
#endif
  case KM_SAMPLETIME:
    set_step_delay(uistat.value);
    break;
  case KM_OFFSET_DELAY:
    set_offset_delay(uistat.value);
    break;
  case KM_FAST_SPEEDUP:
    set_fast_speedup(uistat.value);
    break;
  case KM_REPEAT:
    set_repeat(uistat.value);
    break;
  case KM_LOWOUTLEVEL:
    set_level(uistat.value - setting.external_gain);
    break;
  case KM_HIGHOUTLEVEL:
    set_level(uistat.value - setting.external_gain);
    break;
  case KM_DECAY:
    set_decay(uistat.value);
    break;
#ifdef __QUASI_PEAK__
  case KM_ATTACK:
    set_attack(uistat.value);
    break;
#endif
#ifdef __ULTRA__
  case KM_ULTRA_START:
    config.ultra_threshold = uistat.freq_value;
    reset_settings(setting.mode);
//    config_save(); // TODO not now
    //ultra_threshold = config.ultra_threshold;
    break;
  case KM_DIRECT_START:
    config.direct_start = uistat.freq_value;
    config_save();
    break;
  case KM_DIRECT_STOP:
    config.direct_stop = uistat.freq_value;
    config_save();
    break;
#endif
#ifdef TINYSA4
  case KM_EXP_AVER:
    setting.exp_aver = uistat.value;
    dirty = true;
#endif
  case KM_LEVEL:
    break;
#ifdef __LIMITS__
  case KM_LIMIT_FREQ:
    setting.limits[current_trace][active_limit].frequency = uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT);
    dirty = true;
    limits_update();
    break;
  case KM_LIMIT_LEVEL:
    setting.limits[current_trace][active_limit].level = to_dBm(uistat.value);
    dirty = true;
    limits_update();
    break;
#endif
  case KM_NOISE:
    set_noise(uistat.value);
    break;
#ifdef TINYSA4
  case KM_30MHZ:
    set_30mhz(uistat.freq_value);
    break;
#else
  case KM_10MHZ:
    set_10mhz(uistat.freq_value);
    break;
#endif
  case KM_EXT_GAIN:
    set_external_gain(uistat.value);
    break;
  case KM_LEVELSWEEP:
    setting.modulation = MO_NONE;
    set_level_sweep(uistat.value);
    break;
  case KM_SWEEP_TIME:
    set_sweep_time_us(uistat.value*ONE_SECOND_TIME);
    update_grid();
    break;
  case KM_TRIGGER:
    if (setting.trigger == T_AUTO )
      set_trigger(T_NORMAL);
    set_trigger_level(to_dBm(uistat.value));
    completed = true;

    break;
  case KM_GRIDLINES:
    set_gridlines(uistat.value);
    break;
  case KM_MARKER:
    set_marker_frequency(active_marker, uistat.freq_value - (setting.frequency_offset - FREQUENCY_SHIFT));
    break;
  case KM_MARKER_TIME:
    set_marker_time(active_marker, uistat.value);
    break;
  case KM_MODULATION:
    set_modulation_frequency((int)uistat.value);
    break;
#ifdef TINYSA4
  case KM_COR_AM:
    config.cor_am = -(int)uistat.value;
    config_save();
    dirty = true;
    break;
  case KM_COR_WFM:
    config.cor_wfm = -(int)uistat.value;
    config_save();
    dirty = true;
    break;
  case KM_COR_NFM:
    config.cor_nfm = -(int)uistat.value;
    config_save();
    dirty = true;
    break;
#endif
  case KM_VAR:
    setting.frequency_var = uistat.freq_value;
    break;
#ifdef __NOISE_FIGURE__
  case KM_NF:
    config.noise_figure = uistat.value;
    config_save();
    dirty = true;
    break;
#endif
#ifdef __USE_RTC__
  case KM_RTC_DATE:
  case KM_RTC_TIME:
    {
      int i = 0;
      uint32_t  dt_buf[2];
      dt_buf[0] = rtc_get_tr_bcd(); // TR should be read first for sync
      dt_buf[1] = rtc_get_dr_bcd(); // DR should be read second
      //            0    1   2       4      5     6
      // time[] ={sec, min, hr, 0, day, month, year, 0}
      uint8_t   *time = (uint8_t*)dt_buf;
      for (; i < 6 && kp_buf[i]!=0; i++) kp_buf[i]-= '0';
      for (; i < 6                ; i++) kp_buf[i] =   0;
      for (i = 0; i < 3; i++) kp_buf[i] = (kp_buf[2*i]<<4) | kp_buf[2*i+1]; // BCD format
      if (keypad_mode == KM_RTC_DATE) {
        // Month limit 1 - 12 (in BCD)
             if (kp_buf[1] <    1) kp_buf[1] =    1;
        else if (kp_buf[1] > 0x12) kp_buf[1] = 0x12;
        // Day limit (depend from month):
        uint8_t day_max = 28 + ((0b11101100000000000010111110111011001100>>(kp_buf[1]<<1))&3);
        day_max = ((day_max/10)<<4)|(day_max%10); // to BCD
             if (kp_buf[2] <  1)      kp_buf[2] = 1;
        else if (kp_buf[2] > day_max) kp_buf[2] = day_max;
        time[6] = kp_buf[0]; // year
        time[5] = kp_buf[1]; // month
        time[4] = kp_buf[2]; // day
      }
      else {
        // Hour limit 0 - 23, min limit 0 - 59, sec limit 0 - 59 (in BCD)
        if (kp_buf[0] > 0x23) kp_buf[0] = 0x23;
        if (kp_buf[1] > 0x59) kp_buf[1] = 0x59;
        if (kp_buf[2] > 0x59) kp_buf[2] = 0x59;
        time[2] = kp_buf[0]; // hour
        time[1] = kp_buf[1]; // min
        time[0] = kp_buf[2]; // sec
      }
      rtc_set_time(dt_buf[1], dt_buf[0]);
    }
    break;
#endif

  }
}

void
menu_move_top(void)
{
  while (menu_current_level > 0)
    menu_move_back(false);
}


// -------------------------- CAL STATUS ---------------------------------------------
const char * const dBText[] = { "1dB/", "2dB/", "5dB/", "10dB/", "20dB/"};
const int refMHz[] = { 30, 15, 10, 4, 3, 2, 1 };

float my_round(float v)
{
  float m = 1;
  int sign = 1;
  if (v < 0) {
    sign = -1;
    v = -v;
  }
  while (v < 100) {
    v = v * 10;
    m = m / 10;
  }
  while (v > 1000) {
    v = v / 10;
    m = m * 10;
  }
  v = (int)(v+0.5);
  v = v * m;
  if (sign == -1) {
    v = -v;
  }
  return v;
}
const char * const unit_string[MAX_UNIT_TYPE*2] = { "dBm", "dBmV", "dB"S_MICRO"V", "RAW", "V", "W", "dB", "dB", "dB", "RAW", "V", "W" }; // unit + 6 is delta unit

static const float scale_value[]={50000, 20000, 10000, 5000, 2000, 1000, 500, 200, 100, 50, 20,10,5,2,1,0.5,0.2,0.1,0.05,0.02,0.01,0.005,0.002, 0.001,0.0005,0.0002, 0.0001};
static const char * const scale_vtext[]= {"50000", "20000", "10000", "5000", "2000", "1000", "500", "200", "100", "50", "20","10","5","2","1","0.5","0.2","0.1","0.05","0.02","0.01", "0.005","0.002","0.001", "0.0005","0.0002","0.0001"};

// Quick menu
#define MAX_QUICK_MENU  20
#define MAX_ITEM_SPACE   2
static uint16_t    quick_menu_y[MAX_QUICK_MENU];
static menuitem_t  *quick_menu[MAX_QUICK_MENU];
static uint8_t max_quick_menu = 0;
static uint8_t item_space = 0; //

int invoke_quick_menu(int y)
{
  int i;
  for (i = 0; i < max_quick_menu;i++) {
    if (y < quick_menu_y[i]) {
      if ((uint32_t)quick_menu[i] < KM_NONE) {
        ui_mode_keypad((int)quick_menu[i]);
      } else {
        selection = -1;
        menu_current_level = 0;
        menu_push_submenu(quick_menu[i]);
      }
      return TRUE;
    }
  }
  return FALSE;
}
#define YSTEP   8

int add_quick_menu(int y, menuitem_t *menu)
{
  y += YSTEP*item_space/2 + YSTEP;
  if (max_quick_menu<MAX_QUICK_MENU-1) {
    quick_menu_y[max_quick_menu] = y;
    quick_menu[max_quick_menu++] = menu;
  }
  return y;
}

void draw_cal_status(void)
{
#define BLEN    7
  char buf[BLEN+1];
  buf[6]=0;
  int x = 0;
  int y = OFFSETY;
  unsigned int color;
  const bool rounding = !UNIT_IS_LINEAR(setting.unit);
  const char * const unit = unit_string[setting.unit];
redraw_cal_status:
  buf[6]=0;
  x = 0;
  y = OFFSETY;
  ili9341_set_background(LCD_BG_COLOR);
  ili9341_fill(0, 0, OFFSETX, LCD_HEIGHT);
  max_quick_menu = 0;
  if (MODE_OUTPUT(setting.mode)) {     // No cal status during output
#ifdef TINYSA4
    if (level_error) {
      ili9341_set_foreground(LCD_BRIGHT_COLOR_RED);
      ili9341_drawstring("LEVEL\nERROR", 0 , 80);
    }
#endif
    return;
  }

    //  if (current_menu_is_form() && !in_selftest)
//    return;

//ili9341_set_background(LCD_BG_COLOR);

  float yMax = setting.reflevel;

  if (level_is_calibrated())
    color = setting.auto_reflevel ? LCD_FG_COLOR : LCD_BRIGHT_COLOR_GREEN;
  else
    color = LCD_BRIGHT_COLOR_RED;
  ili9341_set_foreground(color);
  // Ref level
  if (rounding)
    lcd_printf(x, y, "%+4d", (int)yMax);
  else
    lcd_printf(x, y, "%+4.3F", (yMax/setting.unit_scale));
  y = add_quick_menu(y, (menuitem_t *)menu_reflevel);

  // Unit
#if 0
  color = LCD_FG_COLOR;
  ili9341_set_foreground(color);
  if (setting.auto_reflevel){
    y += YSTEP + YSTEP/2 ;
    ili9341_drawstring("AUTO", x, y);
  }
#endif
  lcd_printf(x, y, "%c%s", unit_scale_text[setting.unit_scale_index], unit);
  y = add_quick_menu(y, (menuitem_t *)menu_unit);

  // Scale
  ili9341_set_foreground(LCD_FG_COLOR);
#if 0
  unsigned int i = 0;
  while (i < ARRAY_COUNT(scale_value)) {
    float t = (setting.scale/setting.unit_scale) / scale_value[i];
    if (t > 0.9 && t < 1.1){
      lcd_printf(x, y, "%s%c/",scale_vtext[i],unit_scale_text[setting.unit_scale_index]);
      break;
    }
    i++;
  }
#else
  lcd_printf(x, y, "%.2F/",setting.scale);
#endif
  y = add_quick_menu(y, (menuitem_t *)KM_SCALE);

  // Trigger status
  if (is_paused()) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    ili9341_drawstring("PAUSED", x, y);
    y += YSTEP + YSTEP/2 ;
  }
  if (setting.trigger == T_SINGLE || setting.trigger == T_NORMAL ) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    ili9341_drawstring("ARMED", x, y);
    y += YSTEP + YSTEP/2 ;
  }
  // AM warning
  if (signal_is_AM) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_RED);
    ili9341_drawstring("AM", x, y);
    y += YSTEP + YSTEP/2 ;
  }
  quick_menu_y[max_quick_menu] = y;
  quick_menu[max_quick_menu++] = (menuitem_t *)NULL;

//  if (setting.mode == M_LOW) {
    // Attenuation
    ili9341_set_foreground(setting.auto_attenuation ? LCD_FG_COLOR : LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "Atten:\n%4.2FdB", get_attenuation());
    y = add_quick_menu(y+= YSTEP, (menuitem_t *)menu_atten);
//  }

  // Calc
  if (setting.average[0]>0) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "Calc:\n%s", averageText[setting.average[0]]);
    y = add_quick_menu(y+= YSTEP, (menuitem_t *)menu_average);
  }
#ifdef __SPUR__  // Spur
#ifdef TINYSA3
  if (setting.spur_removal != S_OFF) {
#endif
    ili9341_set_foreground(setting.spur_removal == S_ON ? LCD_BRIGHT_COLOR_GREEN : LCD_FG_COLOR);
    lcd_printf(x, y, "Spur:\n%s", S_IS_AUTO(setting.spur_removal) ? "AUTO" : (setting.spur_removal == S_OFF ?"OFF" : "ON"));
    y = add_quick_menu(y += YSTEP, (menuitem_t *)menu_config);
#ifdef TINYSA3
  }
#endif
  if (setting.mirror_masking) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    ili9341_drawstring("Mask:\nON", x, y);
    y = add_quick_menu(y+=YSTEP, (menuitem_t *)menu_stimulus);
  }
#endif

  if (setting.subtract[0]) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    ili9341_drawstring("Norm.", x, y);
    y = add_quick_menu(y, (menuitem_t *)menu_display);
  }

  // RBW
  ili9341_set_foreground(setting.rbw_x10 ? LCD_BRIGHT_COLOR_GREEN : LCD_FG_COLOR);
  if (dirty) update_rbw();
  lcd_printf(x, y, "RBW:\n%.1FHz", actual_rbw_x10*100.0);
  y = add_quick_menu(y+=YSTEP, (menuitem_t *)menu_rbw);

#ifdef __VBW__
  // VBW
  if (setting.frequency_step > 0) {
    int vbw = setting.vbw_x100;
    if (vbw != 0)
      color = LCD_BRIGHT_COLOR_GREEN;
    else {
      color = LCD_FG_COLOR;
      vbw = 1;
    }
    ili9341_set_foreground(color);
    lcd_printf(x, y, "VBW:\n%.1FHz", actual_rbw_x10*100.0 / vbw);
    y = add_quick_menu(y+=YSTEP, (menuitem_t *)menu_vbw);
  }
#endif
  // Sweep time: SD_NORMAL, SD_PRECISE, SD_FAST, SD_MANUAL
  static const char fscan[]={0, 'P', 'F', 'N', 'M'};
  if (dirty) {
    calculate_step_delay();
    setting.actual_sweep_time_us = calc_min_sweep_time_us();
  }
#if 0                   // Activate for sweep time debugging
  lcd_printf(x, y, "%cScan:\n%5.3Fs", fscan[setting.step_delay_mode&7], (float)setting.sweep_time_us/ONE_SECOND_TIME);
#endif
  ili9341_set_foreground((setting.step_delay_mode&7) != 0 ? LCD_BRIGHT_COLOR_GREEN : LCD_FG_COLOR);
  lcd_printf(x, y, "%cScan:", fscan[setting.step_delay_mode&7]);
  ili9341_set_foreground((setting.step_delay || setting.sweep_time_us ) ? LCD_BRIGHT_COLOR_GREEN : LCD_FG_COLOR);
  lcd_printf(x, y+YSTEP, "%5.3Fs",(float)setting.actual_sweep_time_us/ONE_SECOND_TIME);
  y = add_quick_menu(y+=YSTEP, (menuitem_t *)menu_sweep_speed);


  #if 0                   // Activate for sweep time debugging
  y += YSTEP;
  update_rbw();             // To ensure the calc_min_sweep time shown takes the latest delay into account
  calculate_step_delay();
  uint32_t t = calc_min_sweep_time_us();
  lcd_printf(x, y, "%5.3Fs", (float)t/ONE_SECOND_TIME);
  y += YSTEP;
  lcd_printf(x, y, "%5.3Fs", (float)setting.additional_step_delay_us/ONE_SECOND_TIME);
  y += YSTEP + YSTEP/2 ;
#endif
#ifdef TINYSA4
  if (setting.extra_lna){
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "LNA:ON");
    y = add_quick_menu(y, (menuitem_t *)menu_level);
    y += YSTEP;
  }
#endif
  // Cal output
  if (setting.refer >= 0) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "Ref:\n%dMHz",reffer_freq[setting.refer]/1000000);
    y = add_quick_menu(y+=YSTEP, (menuitem_t *)menu_reffer);
  }

  // Offset
  if (setting.external_gain != 0.0) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "Gain:\n%4.1fdB",setting.external_gain);
    y = add_quick_menu(y+=YSTEP, (menuitem_t *)KM_EXT_GAIN);
  }

  // Repeat
  if (setting.repeat != 1) {
    ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    lcd_printf(x, y, "Repeat\n x%d", setting.repeat);
    y = add_quick_menu(y+=YSTEP,( menuitem_t *)KM_REPEAT);
  }

  // Trigger
  if (setting.trigger != T_AUTO) {
    if (is_paused() || setting.trigger == T_NORMAL) {
      ili9341_set_foreground(LCD_BRIGHT_COLOR_GREEN);
    } else {
      ili9341_set_foreground(LCD_BRIGHT_COLOR_RED);
    }
    ili9341_drawstring("TRIG:", x, y);

    y += YSTEP;
    if (rounding)
      lcd_printf(x, y, "%6.3f", value(setting.trigger_level));
    else
      lcd_printf(x, y, "%6.4F", value(setting.trigger_level));
//    lcd_printf(x, y, "%4f", value(setting.trigger_level)/setting.unit_scale);
    y = add_quick_menu(y,(menuitem_t *)menu_trigger);
  }

  // Mode
  ili9341_set_foreground(level_is_calibrated() ? LCD_BRIGHT_COLOR_GREEN : LCD_BRIGHT_COLOR_RED);
  ili9341_drawstring_7x13(MODE_LOW(setting.mode) ? "LOW" : "HIGH", x, y);

  // Compact status string
//  ili9341_set_background(LCD_FG_COLOR);
  ili9341_set_foreground(LCD_FG_COLOR);
  y += YSTEP + YSTEP/2 ;
  strncpy(buf,"     ",BLEN-1);
  if (setting.auto_attenuation)
    buf[0] = 'a';
  else
    buf[0] = 'A';
  if (setting.auto_IF)
    buf[1] = 'f';
  else
    buf[1] = 'F';
  if (setting.auto_reflevel)
    buf[2] = 'r';
  else
    buf[2] = 'R';
  if (S_IS_AUTO(setting.agc))
    buf[3] = 'g';
  else if (S_STATE(setting.agc))
    buf[3] = 'G';
  if (S_IS_AUTO(setting.lna))
    buf[4] = 'n';
  else if (S_STATE(setting.lna))
    buf[4] = 'N';
  if (S_IS_AUTO(setting.below_IF))
    buf[5] = 'b';
  else if (S_STATE(setting.below_IF))
    buf[5] = 'B';
  ili9341_drawstring(buf, x, y);

  // Version
  y += YSTEP + YSTEP/2 ;
#ifdef TINYSA4
  strncpy(buf,&TINYSA_VERSION[9], BLEN+1);
#else
  strncpy(buf,&TINYSA_VERSION[8], BLEN+1);
#endif
  if (buf[7]=='-') {
    buf[3] = buf[4];
    buf[4] = buf[5];
    buf[5] = buf[6];
  }
  buf[6] = 0;
  ili9341_drawstring(buf, x, y);

  if (y >= BATTERY_START && item_space > 0) {
    item_space--;                       // Reduce item spacing
    goto redraw_cal_status;
  }
  if ((y + (max_quick_menu+1) * YSTEP/2) < BATTERY_START && item_space < MAX_ITEM_SPACE) {
    item_space++;                       // Increase item spacing
    goto redraw_cal_status;
  }

//  ili9341_set_background(LCD_BG_COLOR);
  if (!setting.waterfall) {               // Do not draw bottom level if in waterfall mode
    // Bottom level
    y = area_height + OFFSETY;
    if (level_is_calibrated())
      if (setting.auto_reflevel)
        color = LCD_FG_COLOR;
      else
        color = LCD_BRIGHT_COLOR_GREEN;
    else
      color = LCD_BRIGHT_COLOR_RED;
    ili9341_set_foreground(color);
    if (rounding)
      lcd_printf(x, y, "%4d", (int)(yMax - setting.scale * NGRIDY));
    else
      lcd_printf(x, y, "%+4.3F", ((yMax - setting.scale * NGRIDY)/setting.unit_scale));
    y = add_quick_menu(y,(menuitem_t *)menu_average);
  }
}

