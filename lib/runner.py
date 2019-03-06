"""
Utilities for running benchmarks.

Foo.
"""

import re
import serial
import serial.threaded
import subprocess
import time

class SerialReader(serial.threaded.Protocol):
    def __init__(self):
        self.recv_buf = ''
        self.lines = []

    def expect(self, num_chars):
        self.recv_buf = ''

    def __call__(self):
        return self

    def data_received(self, data):
        try:
            str_data = data.decode('UTF-8')
            self.recv_buf += str_data

            lines = self.recv_buf.split("\n\r")
            if len(lines) > 1:
                self.lines.extend(lines[:-1])
                self.recv_buf = lines[-1]

        except UnicodeDecodeError:
            pass
            #sys.stderr.write('UART output contains garbage: {data}\n'.format(data = data))

    def get_lines(self):
        ret = self.lines
        self.lines = []
        return ret

    def get_line(self):
        if len(self.lines):
            ret = self.lines[-1]
            self.lines = []
            return ret
        return None

class SerialMonitor:
    def __init__(self, port, baud):
        self.ser = serial.serial_for_url(port, do_not_open=True)
        self.ser.baudrate = baud
        self.ser.parity = 'N'
        self.ser.rtscts = False
        self.ser.xonxoff = False
        self.check_command = None

        try:
            self.ser.open()
        except serial.SerialException as e:
            sys.stderr.write('Could not open serial port {}: {}\n'.format(self.ser.name, e))
            sys.exit(1)

        self.reader = SerialReader()
        self.worker = serial.threaded.ReaderThread(self.ser, self.reader)
        self.worker.start()

    def run(self, timeout = 10):
        time.sleep(timeout)
        return self.reader.get_lines()

    def close(self):
        self.worker.stop()
        self.ser.close()

class ShellMonitor:
    def __init__(self, script):
        self.script = script

    def run(self, timeout = 4):
        res = subprocess.run(['timeout', '{:d}s'.format(timeout), self.script],
            stdout = subprocess.PIPE, stderr = subprocess.PIPE,
            universal_newlines = True)
        return res.stdout.split('\n')

    def close(self):
        pass

def get_info(arch, opts = []):
    command = ['make', 'arch={}'.format(arch), 'info']
    command.extend(opts)
    res = subprocess.run(command, stdout = subprocess.PIPE, stderr = subprocess.PIPE,
        universal_newlines = True)
    if res.returncode != 0:
        raise RuntimeError('make info Failure')
    return res.stdout.split('\n')

def get_monitor(arch):
    for line in get_info(arch):
        if 'Monitor:' in line:
            _, port, arg = line.split(' ')
            if port == 'run':
                return ShellMonitor(arg)
            else:
                return SerialMonitor(port, arg)
    raise RuntimeError('Monitor failure')

def get_counter_limits(arch):
    for line in get_info(arch):
        match = re.match('Counter Overflow: ([^/]*)/(.*)', line)
        if match:
            overflow_value = int(match.group(1))
            max_overflow = int(match.group(2))
            return overflow_value, max_overflow
    raise RuntimeError('Did not find Counter Overflow limits')
