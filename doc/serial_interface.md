# TinySA v1.3 USB/Serial Interface
==================================
Defined in main.c

pause
pause sweep

resume
resume sweep

freq {frequency(Hz)}
set frequency (of output?)

dac {value(0-4095)}
gets and sets dac output value

saveconfig

clearconfig {protection key}

data [0..2]
returns the data in trace 1 to 3

refresh off|on

touch x y
simulate screen touch at position x, y

release x y
simulate screen release at position x, y

capture
dump a screenshot as a 320x240 16bit R(5), G(6), B(5) image

sd_list, sd_read, sd_delete

scan {start(Hz)} {stop(Hz)} [points] [outmask]

tinySA 4 only:
hop {start(Hz)} {stop(Hz)} {step(Hz) | points} [outmask]

sweep {start(Hz)} [stop(Hz)] [points]
sweep {go|abort}
sweep {start|stop|center|span|cw} {freq(Hz)}
get current sweep or set sweep

save {id}
recall {id}
Save and recall state of SA

trace {dBm|dBmV|dBuV|RAW|V|W}
trace {scale|reflevel} auto|{value}
trace [{trace#}] value
trace [{trace#}] {copy|freeze|subtract|view|value} {trace#}|off|on|[{index} {value}]
set or return the trace information

marker [n] [on|off|peak|delta|noise|tracking|trace|trace_aver|{freq}|{index}] [{n}|off|on]
set or return marker information

touchcal
calibrate touchscreen

touchtest
test touchscreen

frequencies
return the frequencies in the current sweep

version
return the version

vbat
return battery voltage

vbat_offset
return the battery voltage offset

info
return information about the tinySA

color {id} {rgb24}
set of return the color palette members

threads
display program thread information

usart_cfg
display USB serial interface config information

help
display list of commands

Defined in sa_cmd.c

mode {low|high} {input|output}

modulation {off|am|nfm|wfm|extern|freq} [rate]

calc [trace] {off|minh|maxh|maxd|aver4|aver16|aver|quasip}

tinySA 4 only:
calc [trace] {off|minh|maxh|maxd|aver4|aver16|aver|quasip|log|lin}

spur {off|on}

tinySA 4 only:
spur {off|on|auto}

tinySA 4 only:
lna {off|on}

ULTRA only:
ultra {off|on|auto|start} {freq}

output {on|off}

load [0..4]

tinySA 4 only:
lna2 {0..7|auto}

tinySA 4 only
agc {0..7|auto}

attenuate {0..31|auto}

low mode:
level {-76..-6}

high mode
level {-38..13}

sweeptime {0.003..60}

ext_gain {-100.0..100.0}

levelchange {-70..+70}

leveloffset <low|high|switch|receive_switch> {output} <-20.0..20.0>
returns current leveloffset values

tinySA 4 only:
leveloffset <low|high|switch|receive_switch|lna|harmonic|shift|drive1|drive2> <offset>

deviceid [<number>]
sets or gets device id

sweep_voltage {value(0-3.3)}
sets or gets sweep voltage

__NOISE_FIGURE__ only:
nf {value}

rbw {2..600|auto}

v4 only:
rbw {0.3..850|auto}

if {433M..435M}
sets or gets IF frequency

zero {level}
sets or gets zero level in -dBm?

tinySA 4 only:
direct {start|stop|on|off} {freq(Hz)}

tinySA 4 only:
if1 {975M..979M}

actual_freq {freq(Hz)}
30 MHz

tinySA 3 only
actual_freq {freq(Hz)}
10 MHz

sets or gets actual_freq(?)

trigger {value|auto|normal|single}

selftest (1-3) [arg]

correction {low|high} <freq> <value>

tinySA 4 only:
correction {low|ultra|ultra_lna|out|high} <freq> <value>

scanraw {start(Hz)} {stop(Hz)} [points]

caloutput {off|30|15|10|4|3|2|1}

x {value(0...FFFFFFFF)}

i
initializes

o
sets IF frequency

d a d
sets  output level, drive or aux drive to d

g

a {freq(Hz)}
sets start frequency

b {freq(Hz)}
sets stop frequency

e {bool}
sets tracking

s <pts>
sets or gets number of points

v <vfo>
sets or gets VFO

y {addr(0-95)} [value(0-0xFF)]
sets or gets value in register addr

tinySA 4 Only
z <0..30000>
sets or gets step delay

tinySA 4 only
n



tinySA 4 only:
q s|d 0..18|a 0..63

k
returns temperature

m
restarts sweep?

p p a
sets analog output pin p to a

g p a
sets GPIO pin p to a

w p
sets the RBW to p*10

u
toggles debug

f
