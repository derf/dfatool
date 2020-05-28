"""
Utilities for running benchmarks.

Classes:
    SerialMonitor -- captures serial output for a specific amount of time
    ShellMonitor -- captures UNIX program output for a specific amount of time

Functions:
    get_monitor -- return Monitor class suitable for the selected multipass arch
    get_counter_limits -- return arch-specific multipass counter limits (max value, max overflow)
"""

import os
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

    def __init__(self, callback=None):
        """Create a new SerialReader object."""
        self.callback = callback
        self.recv_buf = ""
        self.lines = []

    def __call__(self):
        return self

    def data_received(self, data):
        """Append newly received serial data to the line buffer."""
        try:
            str_data = data.decode("UTF-8")
            self.recv_buf += str_data

            # We may get anything between \r\n, \n\r and simple \n newlines.
            # We assume that \n is always present and use str.strip to remove leading/trailing \r symbols
            # Note: Do not call str.strip on lines[-1]! Otherwise, lines may be mangled
            lines = self.recv_buf.split("\n")
            if len(lines) > 1:
                self.lines.extend(map(str.strip, lines[:-1]))
                self.recv_buf = lines[-1]
                if self.callback:
                    for line in lines[:-1]:
                        self.callback(str.strip(line))

        except UnicodeDecodeError:
            pass
            # sys.stderr.write('UART output contains garbage: {data}\n'.format(data = data))

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

    def __init__(self, port: str, baud: int, callback=None):
        """
        Create a new SerialMonitor connected to port at the specified baud rate.

        Communication uses no parity, no flow control, and one stop bit.
        Data collection starts immediately.
        """
        self.ser = serial.serial_for_url(port, do_not_open=True)
        self.ser.baudrate = baud
        self.ser.parity = "N"
        self.ser.rtscts = False
        self.ser.xonxoff = False

        try:
            self.ser.open()
        except serial.SerialException as e:
            sys.stderr.write(
                "Could not open serial port {}: {}\n".format(self.ser.name, e)
            )
            sys.exit(1)

        self.reader = SerialReader(callback=callback)
        self.worker = serial.threaded.ReaderThread(self.ser, self.reader)
        self.worker.start()

    def run(self, timeout: int = 10) -> list:
        """
        Collect serial output for timeout seconds and return a list of all output lines.

        Blocks until data collection is complete.
        """
        time.sleep(timeout)
        return self.reader.get_lines()

    def get_lines(self) -> list:
        return self.reader.get_lines()

    def get_files(self) -> list:
        return list()

    def get_config(self) -> dict:
        return dict()

    def close(self):
        """Close serial connection."""
        self.worker.stop()
        self.ser.close()


# TODO Optionale Kalibrierung mit bekannten Widerständen an GPIOs am Anfang
# TODO Sync per LED? -> Vor und ggf nach jeder Transition kurz pulsen
# TODO Für Verbraucher mit wenig Energiebedarf: Versorgung direkt per GPIO
#      -> Zu Beginn der Messung ganz ausknipsen


class EnergyTraceMonitor(SerialMonitor):
    """EnergyTraceMonitor captures serial timing output and EnergyTrace energy data."""

    def __init__(self, port: str, baud: int, callback=None, voltage=3.3):
        super().__init__(port=port, baud=baud, callback=callback)
        self._voltage = voltage
        self._output = time.strftime("%Y%m%d-%H%M%S.etlog")
        self._start_energytrace()

    def _start_energytrace(self):
        cmd = ["msp430-etv", "--save", self._output, "0"]
        self._logger = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

    def close(self):
        super().close()
        self._logger.send_signal(subprocess.signal.SIGINT)
        stdout, stderr = self._logger.communicate(timeout=15)

    def get_files(self) -> list:
        return [self._output]

    def get_config(self) -> dict:
        return {
            "voltage": self._voltage,
        }


class MIMOSAMonitor(SerialMonitor):
    """MIMOSAMonitor captures serial output and MIMOSA energy data for a specific amount of time."""

    def __init__(
        self, port: str, baud: int, callback=None, offset=130, shunt=330, voltage=3.3
    ):
        super().__init__(port=port, baud=baud, callback=callback)
        self._offset = offset
        self._shunt = shunt
        self._voltage = voltage
        self._start_mimosa()

    def _mimosactl(self, subcommand):
        cmd = ["mimosactl"]
        cmd.append(subcommand)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise RuntimeError(
                    "{} returned {}".format(" ".join(cmd), res.returncode)
                )

    def _mimosacmd(self, opts):
        cmd = ["MimosaCMD"]
        cmd.extend(opts)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError("{} returned {}".format(" ".join(cmd), res.returncode))

    def _start_mimosa(self):
        self._mimosactl("disconnect")
        self._mimosacmd(["--start"])
        self._mimosacmd(["--parameter", "offset", str(self._offset)])
        self._mimosacmd(["--parameter", "shunt", str(self._shunt)])
        self._mimosacmd(["--parameter", "voltage", str(self._voltage)])
        self._mimosacmd(["--mimosa-start"])
        time.sleep(2)
        self._mimosactl("1k")  # 987 ohm
        time.sleep(2)
        self._mimosactl("100k")  # 99.3 kohm
        time.sleep(2)
        self._mimosactl("connect")

    def _stop_mimosa(self):
        # Make sure the MIMOSA daemon has gathered all needed data
        time.sleep(2)
        self._mimosacmd(["--mimosa-stop"])
        mtime_changed = True
        mim_file = None
        time.sleep(1)
        # reverse sort ensures that we will get the latest file, which must
        # belong to the current measurements. This ensures that older .mim
        # files lying around in the directory will not confuse our
        # heuristic.
        for filename in sorted(os.listdir(), reverse=True):
            if re.search(r"[.]mim$", filename):
                mim_file = filename
                break
        while mtime_changed:
            mtime_changed = False
            if time.time() - os.stat(mim_file).st_mtime < 3:
                mtime_changed = True
            time.sleep(1)
        self._mimosacmd(["--stop"])
        return mim_file

    def close(self):
        super().close()
        self.mim_file = self._stop_mimosa()

    def get_files(self) -> list:
        return [self.mim_file]

    def get_config(self) -> dict:
        return {
            "offset": self._offset,
            "shunt": self._shunt,
            "voltage": self._voltage,
        }


class ShellMonitor:
    """SerialMonitor runs a program and captures its output for a specific amount of time."""

    def __init__(self, script: str, callback=None):
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
            raise ValueError("timeout argument must be int")
        res = subprocess.run(
            ["timeout", "{:d}s".format(timeout), self.script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if self.callback:
            for line in res.stdout.split("\n"):
                self.callback(line)
        return res.stdout.split("\n")

    def monitor(self):
        raise NotImplementedError

    def close(self):
        """
        Do nothing, successfully.

        Intended for compatibility with SerialMonitor.
        """
        pass


def build(arch, app, opts=[]):
    command = ["make", "arch={}".format(arch), "app={}".format(app), "clean"]
    command.extend(opts)
    res = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            "Build failure, executing {}:\n".format(command) + res.stderr
        )
    command = ["make", "-B", "arch={}".format(arch), "app={}".format(app)]
    command.extend(opts)
    res = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if res.returncode != 0:
        raise RuntimeError(
            "Build failure, executing {}:\n ".format(command) + res.stderr
        )
    return command


def flash(arch, app, opts=[]):
    command = ["make", "arch={}".format(arch), "app={}".format(app), "program"]
    command.extend(opts)
    res = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if res.returncode != 0:
        raise RuntimeError("Flash failure")
    return command


def get_info(arch, opts: list = []) -> list:
    """
    Return multipass "make info" output.

    Returns a list.
    """
    command = ["make", "arch={}".format(arch), "info"]
    command.extend(opts)
    res = subprocess.run(
        command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
    )
    if res.returncode != 0:
        raise RuntimeError("make info Failure")
    return res.stdout.split("\n")


def get_monitor(arch: str, **kwargs) -> object:
    """
    Return an appropriate monitor for arch, depending on "make info" output.

    Port and Baud rate are taken from "make info".

    :param arch: architecture name, e.g. 'msp430fr5994lp' or 'posix'
    :param energytrace: `EnergyTraceMonitor` options. Returns an EnergyTrace monitor if not None.
    :param mimosa: `MIMOSAMonitor` options. Returns a MIMOSA monitor if not None.
    """
    for line in get_info(arch):
        if "Monitor:" in line:
            _, port, arg = line.split(" ")
            if port == "run":
                return ShellMonitor(arg, **kwargs)
            elif "mimosa" in kwargs and kwargs["mimosa"] is not None:
                mimosa_kwargs = kwargs.pop("mimosa")
                return MIMOSAMonitor(port, arg, **mimosa_kwargs, **kwargs)
            elif "energytrace" in kwargs and kwargs["energytrace"] is not None:
                energytrace_kwargs = kwargs.pop("energytrace")
                return EnergyTraceMonitor(port, arg, **energytrace_kwargs, **kwargs)
            else:
                kwargs.pop("energytrace", None)
                kwargs.pop("mimosa", None)
                return SerialMonitor(port, arg, **kwargs)
    raise RuntimeError("Monitor failure")


def get_counter_limits(arch: str) -> tuple:
    """Return multipass max counter and max overflow value for arch."""
    for line in get_info(arch):
        match = re.match("Counter Overflow: ([^/]*)/(.*)", line)
        if match:
            overflow_value = int(match.group(1))
            max_overflow = int(match.group(2))
            return overflow_value, max_overflow
    raise RuntimeError("Did not find Counter Overflow limits")


def get_counter_limits_us(arch: str) -> tuple:
    """Return duration of one counter step and one counter overflow in us."""
    cpu_freq = 0
    overflow_value = 0
    max_overflow = 0
    for line in get_info(arch):
        match = re.match(r"CPU\s+Freq:\s+(.*)\s+Hz", line)
        if match:
            cpu_freq = int(match.group(1))
        match = re.match(r"Counter Overflow:\s+([^/]*)/(.*)", line)
        if match:
            overflow_value = int(match.group(1))
            max_overflow = int(match.group(2))
    if cpu_freq and overflow_value:
        return 1000000 / cpu_freq, overflow_value * 1000000 / cpu_freq, max_overflow
    raise RuntimeError("Did not find Counter Overflow limits")
