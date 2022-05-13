#!/usr/bin/env python3
import serial
import numpy as np
import pylab as pl
import struct
from serial.tools import list_ports

VID = 0x0483
PID = 0x5740

# Get nanovna device automatically
def getport() -> str:
    device_list = list_ports.comports()
    for device in device_list:
        if device.vid == VID and device.pid == PID:
            return device.device
    raise OSError("device not found")

REF_LEVEL = (1<<9)

class TinySA:
    def __init__(self, dev = None):
        self.dev = dev or getport()
        self.serial = None
        self._frequencies = None
        self.points = 101
        self._unit = None
        self._reflevel = None
        self._scale = None
        # list of booleans for the status of the three possible traces
        self._traces = [False, False, False]

    @property
    def frequencies(self):
        return self._frequencies

    def set_frequencies(self, start = 1e6, stop = 900e6, points = None):
        if points:
            self.points = points
        self._frequencies = np.linspace(start, stop, self.points)

    @property
    def unit(self):
        return self._unit

    def open(self):
        if self.serial is None:
            self.serial = serial.Serial(self.dev)

    def close(self):
        if self.serial:
            self.serial.close()
        self.serial = None

    def send_command(self, cmd):
        self.open()
        self.serial.write(cmd.encode())
        self.serial.readline() # discard empty line

    def set_sweep(self, start, stop):
        if start is not None:
            self.send_command("sweep start %d\r" % start)
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_frequency(self, freq):
        if freq is not None:
            self.send_command("freq %d\r" % freq)

    def set_gain(self, gain):
        if gain is not None:
            self.send_command("gain %d\r" % gain)

    def set_output_level(self, level):
        if level is not None:
            self.send_command("level %d\r" % level)

    def get_trace_info(self):
        """Get information about the current traces from the tinySA.
        Most importantly, this tells us the units for the data returned by
        `.data()`.

        parameters:
            trace (int): the trace number to set."""
        self.send_command("trace\r")
        resp = self.fetch_data()
        lines = resp.split("\n")
        self._traces = [False, False, False]

        for line in lines:
            if line:
                lline = line.split()
                trace = int(lline[0][0])
                unit = lline[1]
                reflevel = float(lline[2])
                scale = float(lline[3])
                self._traces[trace] = True

        self._unit = unit
        self._reflevel = reflevel
        self._scale = scale


    def fetch_data(self):
        """Receive data returned by a command sent to the TinySA

        returns:
            str: string of lines separated by `\r\n`.
        """
        result = ''
        line = ''
        while True:
            c = self.serial.read().decode('utf-8')
            if c == chr(13):
                next # ignore CR
            line += c
            if c == chr(10):
                result += line
                line = ''
                next
            if line.endswith('ch>'):
                # stop on prompt
                break
        return result

    def fetch_buffer(self, freq = None, buffer = 0):
        self.send_command("dump %d\r" % buffer)
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x, dtype=np.int16)

    def fetch_rawwave(self, freq = None):
        if freq:
            self.set_frequency(freq)
            time.sleep(0.05)
        self.send_command("dump 0\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.extend([int(d, 16) for d in line.strip().split(' ')])
        return np.array(x[0::2], dtype=np.int16), np.array(x[1::2], dtype=np.int16)

    def resume(self):
        self.send_command("resume\r")

    def pause(self):
        self.send_command("pause\r")

    def data(self, array = 0):
        """Fetch data in array from the TinySA, and convert to a Numpy Array.

        parameters:
            array (int): Number of array to fetch.
                0 = temp value
                1 = stored trace
                2 = measurement

        returns:
            np.array: Array of data values in current data format
        """
        self.get_trace_info()
        self.send_command("data %d\r" % array)
        data = self.fetch_data()
        x = []
        for line in data.split('\r\n'):
            if line:
                d = line.strip()
                x.append(float(d))
        return np.array(x)

    def fetch_frequencies(self):
        self.send_command("frequencies\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.append(float(line))
        self._frequencies = np.array(x)

    def send_scan(self, start = 1e6, stop = 900e6, points = None):
        if points:
            self.send_command("scan %d %d %d\r"%(start, stop, points))
        else:
            self.send_command("scan %d %d\r"%(start, stop))

    def scan(self):
        segment_length = 101
        array0 = []
        array1 = []
        if self._frequencies is None:
            self.fetch_frequencies()
        freqs = self._frequencies
        while len(freqs) > 0:
            seg_start = freqs[0]
            seg_stop = freqs[segment_length-1] if len(freqs) >= segment_length else freqs[-1]
            length = segment_length if len(freqs) >= segment_length else len(freqs)
            #print((seg_start, seg_stop, length))
            self.send_scan(seg_start, seg_stop, length)
            array0.extend(self.data(0))
            array1.extend(self.data(1))
            freqs = freqs[segment_length:]
        self.resume()
        return (array0, array1)

    def capture(self):
        from PIL import Image
        self.send_command("capture\r")
        b = self.serial.read(320 * 240 * 2)
        x = struct.unpack(">76800H", b)
        # convert pixel format from 565(RGB) to 8888(RGBA)
        arr = np.array(x, dtype=np.uint32)
        arr = 0xFF000000 + ((arr & 0xF800) >> 8) + ((arr & 0x07E0) << 5) + ((arr & 0x001F) << 19)
        return Image.frombuffer('RGBA', (320, 240), arr, 'raw', 'RGBA', 0, 1)

    def logmag(self, x):
        """Plot the supplied amplitude data in dB"""
        pl.grid(True)
        pl.xlim(self.frequencies[0], self.frequencies[-1])
        pl.plot(self.frequencies, 20*np.log10(x))

    def linmag(self, x):
        """Plot the supplied amplitude data"""
        pl.grid(True)
        pl.xlim(self.frequencies[0], self.frequencies[-1])
        pl.plot(self.frequencies, x)

def plot_sample0(samp):
    N = min(len(samp), 256)
    fs = 48000
    pl.subplot(211)
    pl.grid()
    pl.plot(samp)
    pl.subplot(212)
    pl.grid()
    #pl.ylim((-50, 50))
    pl.psd(samp, N, window = pl.blackman(N), Fs=fs)

def plot_sample(ref, samp):
    N = min(len(samp), 256)
    fs = 48000
    pl.subplot(211)
    pl.grid()
    pl.plot(ref)
    pl.plot(samp)
    pl.subplot(212)
    pl.grid()
    #pl.ylim((-50, 50))
    pl.psd(ref, N, window = pl.blackman(N), Fs=fs)
    pl.psd(samp, N, window = pl.blackman(N), Fs=fs)

if __name__ == '__main__':
    from optparse import OptionParser
    parser = OptionParser(usage="%prog: [options]")
    parser.add_option("-r", "--raw", dest="rawwave",
                      type="int", default=None,
                      help="plot raw waveform", metavar="RAWWAVE")
    parser.add_option("-p", "--plot", dest="plot",
                      action="store_true", default=False,
                      help="plot rectanglar", metavar="PLOT")
    parser.add_option("-c", "--scan", dest="scan",
                      action="store_true", default=False,
                      help="scan by script", metavar="SCAN")
    parser.add_option("-S", "--start", dest="start",
                      type="float", default=1e6,
                      help="start frequency", metavar="START")
    parser.add_option("-E", "--stop", dest="stop",
                      type="float", default=900e6,
                      help="stop frequency", metavar="STOP")
    parser.add_option("-N", "--points", dest="points",
                      type="int", default=101,
                      help="scan points", metavar="POINTS")
    parser.add_option("-d", "--dev", dest="device",
                      help="device node", metavar="DEV")
    parser.add_option("-v", "--verbose",
                      action="store_true", dest="verbose", default=False,
                      help="verbose output")
    parser.add_option("-C", "--capture", dest="capture",
                      help="capture current display to FILE", metavar="FILE")
    parser.add_option("-e", dest="command", action="append",
                      help="send raw command", metavar="COMMAND")
    parser.add_option("-o", dest="save",
                      help="write touch stone file", metavar="SAVE")
    (opt, args) = parser.parse_args()

    nv = NanoVNA(opt.device or getport())

    if opt.command:
        for c in opt.command:
            nv.send_command(c + "\r")

    if opt.capture:
        print("capturing...")
        img = nv.capture()
        img.save(opt.capture)
        exit(0)

    nv.set_port(opt.port)
    if opt.rawwave is not None:
        samp = nv.fetch_buffer(buffer = opt.rawwave)
        print(len(samp))
        if opt.rawwave == 1 or opt.rawwave == 2:
            plot_sample0(samp)
            print(np.average(samp))
        else:
            plot_sample(samp[0::2], samp[1::2])
            print(np.average(samp[0::2]))
            print(np.average(samp[1::2]))
            print(np.average(samp[0::2] * samp[1::2]))
        pl.show()
        exit(0)
    if opt.start or opt.stop or opt.points:
        nv.set_frequencies(opt.start, opt.stop, opt.points)
    plot = opt.phase or opt.plot or opt.vswr or opt.delay or opt.groupdelay or opt.smith or opt.unwrapphase or opt.polar or opt.tdr
    if plot or opt.save:
        p = int(opt.port) if opt.port else 0
        if opt.scan or opt.points > 101:
            s = nv.scan()
            s = s[p]
        else:
            if opt.start or opt.stop:
                nv.set_sweep(opt.start, opt.stop)
            nv.fetch_frequencies()
            s = nv.data(p)
            nv.fetch_frequencies()
    if opt.save:
        n = nv.skrf_network(s)
        n.write_touchstone(opt.save)
    if opt.smith:
        nv.smith(s)
    if opt.polar:
        nv.polar(s)
    if opt.plot:
        nv.logmag(s)
    if opt.phase:
        nv.phase(s)
    if opt.unwrapphase:
        nv.phase(s, unwrap=True)
    if opt.delay:
        nv.delay(s)
    if opt.groupdelay:
        nv.groupdelay(s)
    if opt.vswr:
        nv.vswr(s)
    if opt.tdr:
        nv.tdr(s)
    if plot:
        pl.show()
