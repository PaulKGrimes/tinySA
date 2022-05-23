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
    _valid_units = [
        "dBm",
        "dBmV",
        "dBuV",
        "RAW",
        "V",
        "W"
    ]

    def __init__(self, dev = None):
        self.dev = dev or getport()
        self.serial = None
        self._frequencies = None
        self.points = 101
        self._unit = None
        self._reflevel = None
        self._scale = None

    def open(self):
        """Open the serial connection to the tinySA."""
        if self.serial is None:
            self.serial = serial.Serial(self.dev)

    def close(self):
        """Close the serial connection to the tinySA."""
        if self.serial:
            self.serial.close()
        self.serial = None

    def send_command(self, cmd):
        if cmd[-1] != "\r":
            cmd = cmd + "\r"
        self.open()
        self.serial.write(cmd.encode())
        self.serial.readline() # discard empty line

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

    def query(self, cmd):
        """Send a command and listen for the response.

        parameters:
            cmd (str): command to set to the TinySA.

        returns:
            str: string of lines separated by `\r\n`
        """
        self.send_command(cmd)
        return self.fetch_data()

    def capture(self):
        """Return a screen capture of the tinySA"""
        from PIL import Image
        self.send_command("capture\r")
        b = self.serial.read(320 * 240 * 2)
        x = struct.unpack(">76800H", b)
        # convert pixel format from 565(RGB) to 8888(RGBA)
        arr = np.array(x, dtype=np.uint32)
        arr = 0xFF000000 + ((arr & 0xF800) >> 8) + ((arr & 0x07E0) << 5) + ((arr & 0x001F) << 19)
        return Image.frombuffer('RGBA', (320, 240), arr, 'raw', 'RGBA', 0, 1)

    def get_sweep(self):
        """Get the sweep start and stop frequencies in Hz, and the number of
        points.

        returns:
            tuple of (float, float, int): start and stop frequencies in Hz and number
                                          of points in sweep.
        """
        lines = self.query("sweep")
        slines = lines.split()
        self._sweep_start = float(slines[0])
        self._sweep_stop = float(slines[1])
        self._points = int(slines[2])

        return self._sweep_start, self._sweep_stop, self._points

    @property
    def sweep_start(self):
        """Return the starting frequency of the sweep in Hz. This is a stored value
        from the last time that `.get_sweep()` was called.

        return:
            float: sweep start frequency in Hz
        """
        return self._sweep_start

    @property
    def sweep_stop(self):
        """Return the stoping frequency of the sweep in Hz. This is a stored value
        from the last time that `.get_sweep()` was called.

        return:
            float: sweep stop frequency in Hz
        """
        return self._sweep_stop

    @property
    def sweep_center(self):
        """Return the center frequency of the sweep in Hz. This is a stored value
        from the last time that `.get_sweep()` was called.

        return:
            float: sweep center frequency in Hz
        """
        return (self._sweep_start + self._sweep_stop) / 2

    @property
    def sweep_span(self):
        """Return the span frequency of the sweep in Hz. This is a stored value
        from the last time that `.get_sweep()` was called.

        return:
            float: sweep span frequency in Hz
        """
        return (self._sweep_stop - self._sweep_start)

    @property
    def sweep_points(self):
        """Return the number of points in the sweep. This is a stored value
        from the last time that `.get_sweep()` was called

        returns:
            int: number of points in the sweep.
        """
        return self._points


    def set_sweep(self, start, stop):
        """Set the sweep with a start and stop frequency in Hz

        parameters:
            start (float): start frequency in Hz
            stop  (float): stop frequency in Hz
        """
        if start is not None:
            self.send_command("sweep start %d\r" % start)
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_sweep_start(self, start):
        """Set the sweep with start frequency in Hz

        parameters:
            start (float): start frequency in Hz
        """
        if start is not None:
            self.send_command("sweep start %d\r" % start)

    def set_sweep_stop(self, stop):
        """Set the sweep with stop frequency in Hz

        parameters:
            stop (float): stop frequency in Hz
        """
        if stop is not None:
            self.send_command("sweep stop %d\r" % stop)

    def set_sweep_center(self, center):
        """Set the sweep with center frequency in Hz

        parameters:
            center (float): center frequency in Hz
        """
        if center is not None:
            self.send_command("sweep center %d\r" % center)

    def set_sweep_span(self, span):
        """Set the sweep with span frequency in Hz

        parameters:
            span (float): span frequency in Hz
        """
        if span is not None:
            self.send_command("sweep span %d\r" % span)

    def set_sweep_cw(self, cw):
        """Set the sweep to a fixed cw frequency in Hz

        parameters:
            cw (float): cw frequency in Hz
        """
        if span is not None:
            self.send_command("sweep cw %d\r" % cw)

    def resume(self):
        """Resume the sweep"""
        self.send_command("resume\r")

    def pause(self):
        """Pause the sweep"""
        self.send_command("pause\r")

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
                units = lline[1]
                reflevel = float(lline[2])
                scale = float(lline[3])
                self._traces[trace] = True

        self._units = units
        self._reflevel = reflevel
        self._scale = scale

    @property
    def units(self):
        """A string representing the units of the trace.  This value is a stored value
        from the last time `.get_trace_info()` was called.

        returns:
            str: units
        """
        return self._units

    @property
    def ref_level(self):
        """The reference level of the trace in `.units`.  This value is a stored value
        from the last time `.get_trace_info()` was called.

        returns:
            float: ref_level in `.units`
        """
        return self._ref_level

    @property
    def scale(self):
        """The scale of the trace in `.units`/div. This value is a stored value from
        the last time `.get_trace_info()` was called.

        returns:
            float: scale in `.units`/div
        """
        return self._scale

    def set_units(self, units="dBm"):
        """Set the units of the traces to one of: dBm, dBmV, dBuV, RAW, V, W

        parameters:
            units (str): string representing the units, one of `._valid_units`"""
        if units not in self._valid_units:
            raise runtimeError(f"{units} is not a valid option.")
        else
            self.send_command(f"trace {units}")
            self._units = units

    def set_ref_level(self, ref_level):
        """Set the reference level in `.units`.

        parameters:
            ref_level (float): ref_level in `units`.
        """
        self.send_command(f"trace reflevel {ref_level:.2f}")
        self._ref_level = ref_level

    def set_scale(self, scale):
        """Set the scale on the trace in `.units`/div.

        parameters:
            scale (float): scale in `.units`/div
        """
        self.send_command(f"trace scale {scale:.1f}")
        self._scale = scale

    def copy_trace(self, source, destination):
        """Copy the source trace to the destination trace.

        parameters:
            source (int): 1..3 number of trace to copy from
            destination (int): 1..3 number of trace to copy to
        """
        if source <= 0 or source > 3:
            raise ValueError("Source trace number must be between 1 and 3")
        if destination <= 0 or destination > 3:
            raise ValueError("Destination trace number must be between 1 and 3")
        if source == destination:
            # silently do nothing
            return
        else:
            self.send_command(f"trace copy {source:d} {destination:d}")
        

    def set_gain(self, gain):
        """Sets the extenal gain value, which is used to correct the input or output
        levels.

        parameters:
            gain (float): external gain value in dB
        """
        if gain is not None:
            self.send_command("ext_gain %d\r" % gain)

    def set_output_level(self, level):
        if level is not None:
            self.send_command("level %d\r" % level)


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

    @property
    def frequencies(self):
        """Frequencies in sweep. This is a stored set of frequencies from the last
        time `.get_frequencies()` was called.

        returns:
            numpy.Array: array of frequencies in Hz
        """
        return self._frequencies

    def get_frequencies(self):
        """Get the frequencies in the current sweep.

        returns:
            numpy.Array: array of frequencies in Hz."""
        self.send_command("frequencies\r")
        data = self.fetch_data()
        x = []
        for line in data.split('\n'):
            if line:
                x.append(float(line))
        self._frequencies = np.array(x)

        return self._frequencies

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
