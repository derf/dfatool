"""
Utilities for running benchmarks.

Classes:
    SerialMonitor -- captures serial output for a specific amount of time
    ShellMonitor -- captures UNIX program output for a specific amount of time

Functions:
    get_monitor -- return Monitor class suitable for the selected multipass arch
    get_counter_limits -- return arch-specific multipass counter limits (max value, max overflow)
"""

import re
import serial
import serial.threaded
import subprocess
import sys
import time

class SerialReader(serial.threaded.Protocol):
    """
    Character- to line-wise data buffer for serial interfaces.

    Reads in new data whenever it becomes available and exposes a line-based
    interface to applications.
    """
    def __init__(self, callback = None):
        """Create a new SerialReader object."""
        self.callback = callback
        self.recv_buf = ''
        self.lines = []

    def __call__(self):
        return self

    def data_received(self, data):
        """Append newly received serial data to the line buffer."""
        try:
            str_data = data.decode('UTF-8')
            self.recv_buf += str_data

            if '\n\r' in self.recv_buf:
                lines = self.recv_buf.split('\n\r')
            elif '\r\n' in self.recv_buf:
                lines = self.recv_buf.split('\r\n')
            else:
                lines = []
            if len(lines) > 1:
                self.lines.extend(lines[:-1])
                self.recv_buf = lines[-1]
                if self.callback:
                    self.callback(lines[:-1])

        except UnicodeDecodeError:
            pass
            #sys.stderr.write('UART output contains garbage: {data}\n'.format(data = data))

    def get_lines(self) -> list:
        """
        Return the latest batch of complete lines.

        The return value is a list and may be empty.

        Empties the internal line buffer to ensure that no line is returned twice.
        """
        ret = self.lines
        self.lines = []
        return ret

    def get_line(self) -> str:
        """
        Return the latest complete line, or None.

        Empties the entire internal line buffer to ensure that no line is returned twice.
        """
        if len(self.lines):
            ret = self.lines[-1]
            self.lines = []
            return ret
        return None

class SerialMonitor:
    """SerialMonitor captures serial output for a specific amount of time."""

    def __init__(self, port: str, baud: int, callback = None):
        """
        Create a new SerialMonitor connected to port at the specified baud rate.

        Communication uses no parity, no flow control, and one stop bit.
        Data collection starts immediately.
        """
        self.ser = serial.serial_for_url(port, do_not_open=True)
        self.ser.baudrate = baud
        self.ser.parity = 'N'
        self.ser.rtscts = False
        self.ser.xonxoff = False

        try:
            self.ser.open()
        except serial.SerialException as e:
            sys.stderr.write('Could not open serial port {}: {}\n'.format(self.ser.name, e))
            sys.exit(1)

        self.reader = SerialReader(callback = callback)
        self.worker = serial.threaded.ReaderThread(self.ser, self.reader)
        self.worker.start()

    def run(self, timeout: int = 10) -> list:
        """
        Collect serial output for timeout seconds and return a list of all output lines.

        Blocks until data collection is complete.
        """
        time.sleep(timeout)
        return self.reader.get_lines()

    def get_lines(self) ->list:
        return self.reader.get_lines()

    def close(self):
        """Close serial connection."""
        self.worker.stop()
        self.ser.close()

class ShellMonitor:
    """SerialMonitor runs a program and captures its output for a specific amount of time."""
    def __init__(self, script: str, callback = None):
        """
        Create a new ShellMonitor object.

        Does not start execution and monitoring yet.
        """
        self.script = script
        self.callback = callback

    def run(self, timeout: int = 4) -> list:
        """
        Run program for timeout seconds and return a list of its stdout lines.

        stderr and return status are discarded at the moment.
        """
        if type(timeout) != int:
            raise ValueError('timeout argument must be int')
        res = subprocess.run(['timeout', '{:d}s'.format(timeout), self.script],
            stdout = subprocess.PIPE, stderr = subprocess.PIPE,
            universal_newlines = True)
        if self.callback:
            self.callback(res.stdout.split('\n'))
        return res.stdout.split('\n')

    def monitor(self):
        raise NotImplementedError

    def close(self):
        """
        Do nothing, successfully.

        Intended for compatibility with SerialMonitor.
        """
        pass

def build(arch, app, opts = []):
    command = ['make', '-B', 'arch={}'.format(arch), 'app={}'.format(app)]
    command.extend(opts)
    res = subprocess.run(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        universal_newlines = True)
    if res.returncode != 0:
        raise RuntimeError('Build failure: ' + res.stderr)
    return command

def flash(arch, app, opts = []):
    command = ['make', 'arch={}'.format(arch), 'app={}'.format(app), 'program']
    command.extend(opts)
    res = subprocess.run(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        universal_newlines = True)
    if res.returncode != 0:
        raise RuntimeError('Flash failure')
    return command

def get_info(arch, opts: list = []) -> list:
    """
    Return multipass "make info" output.

    Returns a list.
    """
    command = ['make', 'arch={}'.format(arch), 'info']
    command.extend(opts)
    res = subprocess.run(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        universal_newlines = True)
    if res.returncode != 0:
        raise RuntimeError('make info Failure')
    return res.stdout.split('\n')

def get_monitor(arch: str, **kwargs) -> object:
    """Return a SerialMonitor or ShellMonitor."""
    for line in get_info(arch):
        if 'Monitor:' in line:
            _, port, arg = line.split(' ')
            if port == 'run':
                return ShellMonitor(arg, **kwargs)
            else:
                return SerialMonitor(port, arg, **kwargs)
    raise RuntimeError('Monitor failure')

def get_counter_limits(arch: str) -> tuple:
    """Return multipass max counter and max overflow value for arch."""
    for line in get_info(arch):
        match = re.match('Counter Overflow: ([^/]*)/(.*)', line)
        if match:
            overflow_value = int(match.group(1))
            max_overflow = int(match.group(2))
            return overflow_value, max_overflow
    raise RuntimeError('Did not find Counter Overflow limits')
