"""
Utilities for running benchmarks.

Classes:
    SerialMonitor -- captures serial output for a specific amount of time
    ShellMonitor -- captures UNIX program output for a specific amount of time

Functions:
    get_monitor -- return Monitor class suitable for the selected multipass arch
    get_counter_limits -- return arch-specific multipass counter limits (max value, max overflow)
"""

import json
import logging
import os
import re
import serial
import serial.threaded
import subprocess
import sys
import time
from dfatool.lennart.SigrokCLIInterface import SigrokCLIInterface

logger = logging.getLogger(__name__)


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
        self.lines = list()
        self.all_lines = list()

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
                new_lines = list(map(str.strip, lines[:-1]))
                self.lines.extend(new_lines)
                self.all_lines.extend(new_lines)
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
        logger.debug(f"Opening serial port {port} with {baud}N1")

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
        return self.reader.all_lines

    def get_files(self) -> list:
        return list()

    def get_config(self) -> dict:
        return dict()

    def close(self):
        """Close serial connection."""
        self.worker.stop()
        self.ser.close()


# TODO Optionale Kalibrierung mit bekannten Widerständen an GPIOs am Anfang (EnergyTrace selbst macht nur bis 1,5mA)
# TODO Sync per LED? -> Vor und ggf nach jeder Transition kurz pulsen
# TODO Für Verbraucher mit wenig Energiebedarf: Versorgung direkt per GPIO
#      -> Zu Beginn der Messung ganz ausknipsen


class EnergyTraceMonitor(SerialMonitor):
    """EnergyTraceMonitor captures serial timing output and EnergyTrace energy data."""

    # Zusätzliche key-value-Argumente von generate-dfa-benchmark.py --energytrace=... landen hier
    # (z.B. --energytrace=var1=bar,somecount=2 => EnerygTraceMonitor(..., var1="bar", somecount="2")).
    # Soald das EnergyTraceMonitor-Objekt erzeugt wird, beginnt die Messung (d.h. hier: msp430-etv wird gestartet)
    def __init__(
        self, port: str, baud: int, callback=None, voltage=3.3, plusplus=False
    ):
        super().__init__(port=port, baud=baud, callback=callback)
        self._voltage = voltage
        self._plusplus = plusplus
        self._output = time.strftime("%Y%m%d-%H%M%S.etlog")
        self._start_energytrace()

    def _start_energytrace(self):
        print("[%s] Starting Measurement" % type(self).__name__)
        if self._plusplus:
            cmd = ["msp430-etv", "--with-hardware-states", "--save", self._output, "0"]
        else:
            cmd = ["msp430-etv", "--save", self._output, "0"]
        self._logger = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True
        )

    # Benchmark fertig -> externe Hilfsprogramme beenden
    def close(self):
        # wait for tail sync
        time.sleep(2)
        super().close()
        self._logger.send_signal(subprocess.signal.SIGINT)
        stdout, stderr = self._logger.communicate(timeout=15)
        print("[%s] Stopped Measurement" % type(self).__name__)

    # Zusätzliche Dateien, die mit dem Benchmark-Log und -Plan abgespeichert werden sollen
    # (hier: Die von msp430-etv generierten Logfiles)
    def get_files(self) -> list:
        print("[%s] Getting files" % type(self).__name__)
        return [self._output]

    # Benchmark-Konfiguration. Hier: Die (konstante) Spannung.
    # MSP430FR5969: 3,6V (wird aktuell nicht unterstützt)
    # MSP430FR5994: 3,3V (default)
    def get_config(self) -> dict:
        return {"voltage": self._voltage}


class EnergyTraceLogicAnalyzerMonitor(EnergyTraceMonitor):
    """EnergyTraceLogicAnalyzerMonitor captures EnergyTrace energy data and LogicAnalyzer timing output."""

    def __init__(self, port: str, baud: int, callback=None, voltage=3.3):
        super().__init__(port=port, baud=baud, callback=callback, voltage=voltage)

        options = {"fake": False, "sample_rate": 1_000_000}
        self.log_file = "logic_output_log_%s.json" % (time.strftime("%Y%m%d-%H%M%S"))

        # Initialization of Interfaces
        self.sig = SigrokCLIInterface(
            sample_rate=options["sample_rate"], fake=options["fake"]
        )

        # Start Measurements
        self.sig.runMeasureAsynchronous()

    def close(self):
        super().close()
        # Read measured data
        # self.sig.waitForAsynchronousMeasure()
        self.sig.forceStopMeasure()
        time.sleep(0.2)
        sync_data = self.sig.getData()
        # TODO ensure that sync_data.getDict()["timestamps"] is not empty
        # (if it is, communication with the LA was broken the entire time)
        with open(self.log_file, "w") as fp:
            json.dump(sync_data.getDict(), fp)

    def get_files(self) -> list:
        files = [self.log_file]
        files.extend(super().get_files())
        return files


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
        logger.debug(f"Executing {cmd}")
        res = subprocess.run(cmd)
        if res.returncode != 0:
            res = subprocess.run(cmd)
            if res.returncode != 0:
                raise RuntimeError(
                    f"'{' '.join(cmd)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
                )

    def _mimosacmd(self, opts):
        cmd = ["MimosaCMD"]
        cmd.extend(opts)
        res = subprocess.run(cmd)
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(cmd)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )

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
        return {"offset": self._offset, "shunt": self._shunt, "voltage": self._voltage}


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


class KRATOS:
    def __init__(self, opts=list()):
        self.opts = opts

    def app_header(self):
        ret = (
            '#include "AEMR.h"\n'
            '#include "syscall/guarded_buzzer.h"\n'
            '#include "drivers/counter.h"\n'
            '#include "drivers/gpio.h"\n'
        )
        return ret

    def app_function_start(self):
        ret = (
            "ImplementThread(EnergyBenchmarkApp, energyBenchmarkApp, 512);\n"
            "void EnergyBenchmarkApp::action()\n"
            "{\n"
            "Guarded_Buzzer buz(500);\n"
        )

        return ret

    def app_function_end(self):
        return "}\n"

    def app_delay(self, ms, comment=""):
        return self.app_sleep(ms, comment)

    def app_sleep(self, ms, comment=""):
        if comment:
            comment = f" // {comment}"
        return f"buz.sleep({ms});{comment}\n"

    def app_newlines(self):
        return "uart << endl << endl;\n"

    def app_lasync_call(self):
        return "runLASync(buz);\n"

    def app_lasync_function(self):
        ret = """void runLASync(Guarded_Buzzer &buz){
    setOutput(1, 0);
    setOutput(1, 1);
    pinHigh(1, 0);
    pinHigh(1, 1);

    buz.sleep(1000);

    pinLow(1, 0);
    pinLow(1, 1);
}
"""
        return ret

    def sanity_check_config(self):
        with open(".config", "r") as f:
            config_lines = f.readlines()
        for line in config_lines:
            line = line.strip()
            if (
                "CONFIG_eUSCI_A_UART_BAUDRATE" in line
                and line != "CONFIG_eUSCI_A_UART_BAUDRATE=9600"
            ):
                raise RuntimeError(
                    f"Unsupported UART baud rate in .config, must be 9600: {line}"
                )
            if (
                "CONFIG_Architecture_MSP430FR_CycleCounter" in line
                and line != "CONFIG_Architecture_MSP430FR_CycleCounter=y"
            ):
                raise RuntimeError(f"Kratos Cycle Counter is disabled")

    def build(self, app, opts=list()):
        self.sanity_check_config()
        command = ["make", "clean"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Building: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        command = ["make"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Building: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return command

    def flash(self, app, opts=list()):
        command = ["make", "flash"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Flashing: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return command

    def get_monitor(self, **kwargs) -> object:
        """
        Return an appropriate monitor for arch, depending on "make info" output.

        Port and Baud rate are taken from "make info".

        :param energytrace: `EnergyTraceMonitor` options. Returns an EnergyTrace monitor if not None.
        :param mimosa: `MIMOSAMonitor` options. Returns a MIMOSA monitor if not None.
        """
        if "mimosa" in kwargs and kwargs["mimosa"] is not None:
            mimosa_kwargs = kwargs.pop("mimosa")
            return MIMOSAMonitor("/dev/ttyACM1", "9600", **mimosa_kwargs, **kwargs)
        elif "energytrace" in kwargs and kwargs["energytrace"] is not None:
            energytrace_kwargs = kwargs.pop("energytrace").copy()
            sync_mode = energytrace_kwargs.pop("sync")
            if sync_mode == "la":
                return EnergyTraceLogicAnalyzerMonitor(
                    "/dev/ttyACM1", "9600", **energytrace_kwargs, **kwargs
                )
            else:
                return EnergyTraceMonitor(
                    "/dev/ttyACM1", "9600", **energytrace_kwargs, **kwargs
                )
        else:
            kwargs.pop("energytrace", None)
            kwargs.pop("mimosa", None)
            return SerialMonitor("/dev/ttyACM1", "9600", **kwargs)

    def sleep_ms(self, duration: int, opts=list()) -> str:
        max_sleep = 250
        if max_sleep is not None and duration > max_sleep:
            sub_sleep_count = duration // max_sleep
            tail_sleep = duration % max_sleep
            ret = f"for (unsigned char i = 0; i < {sub_sleep_count}; i++) {{ buz.sleep({max_sleep}); }}\n"
            if tail_sleep > 0:
                ret += f"buz.sleep({tail_sleep});\n"
            return ret
        return f"buz.sleep({duration});\n"

    def get_counter_limits_us(self, opts=list()) -> tuple:
        """Return duration of one counter step and one counter overflow in us."""
        cpu_freq = 16_000_000
        overflow_value = 65536
        max_overflow = 65535
        return (
            1_000_000 / cpu_freq,
            overflow_value * 1_000_000 / cpu_freq,
            max_overflow,
        )

    def get_nfpvalues(self) -> list:
        """
        Return Kratos "make nfpvalues" output.

        Returns a dict.
        """
        command = ["make", "nfpvalues"]
        logger.debug(f"Getting nfpvalues: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return json.loads(res.stdout)


class Multipass:
    def __init__(self, name, opts=list()):
        self.name = name
        self.opts = opts
        self.info = self.get_info()

    def app_header(self):
        ret = '#include "arch.h"\n' '#include "driver/gpio.h"\n'
        return ret

    def app_function_start(self):
        ret = "int main(void)\n{\n"

        for driver in ("arch", "gpio", "kout"):
            ret += f"{driver}.setup();\n"

        return ret

    def app_function_end(self):
        return "while (1) { }\nreturn 0;\n}\n"

    def app_delay(self, ms, comment=""):
        if comment:
            comment = f" // {comment}"
        return f"arch.delay_ms({ms});{comment}\n"

    def app_sleep(self, ms, comment=""):
        if ms > 250:
            # TODO build multiple sleep_ms calls
            raise ValueError("msp430 does not support sleep longer than 250ms")
        if comment:
            comment = f" // {comment}"
        return f"arch.sleep_ms({ms});{comment}\n"

    def app_newlines(self):
        return "kout << endl << endl;\n"

    def app_lasync_call(self):
        return "runLASync();\n"

    def app_lasync_function(self):
        ret = """void runLASync(){
    #ifdef PTALOG_GPIO
    gpio.write(PTALOG_GPIO, 1);
    #endif
    gpio.led_on(0);
    gpio.led_on(1);
    #ifdef PTALOG_GPIO
    gpio.write(PTALOG_GPIO, 0);
    #endif

    for (unsigned char i = 0; i < 4; i++) {
        arch.sleep_ms(250);
    }

    #ifdef PTALOG_GPIO
    gpio.write(PTALOG_GPIO, 1);
    #endif
    gpio.led_off(0);
    gpio.led_off(1);
    #ifdef PTALOG_GPIO
    gpio.write(PTALOG_GPIO, 0);
    #endif
}\n\n"""
        return ret

    def build(self, app, opts=list()):
        command = ["make", "arch={}".format(self.name), "app={}".format(app), "clean"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Building: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        command = ["make", "-B", "arch={}".format(self.name), "app={}".format(app)]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Building: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return command

    def flash(self, app, opts=list()):
        command = ["make", "arch={}".format(self.name), "app={}".format(app), "program"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Flashing: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return command

    def get_info(self, opts=list()) -> list:
        """
        Return multipass "make info" output.

        Returns a list.
        """
        command = ["make", "arch={}".format(self.name), "app=donothing", "info"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Getting Info: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return res.stdout.split("\n")

    def get_nfpvalues(self, app, opts=list()) -> list:
        """
        Return multipass "make nfpvalues" output.

        Returns a dict.
        """
        command = ["make", f"arch={self.name}", f"app={app}", "nfpvalues"]
        command.extend(self.opts)
        command.extend(opts)
        logger.debug(f"Getting nfpvalues: {' '.join(command)}")
        res = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
        )
        if res.returncode != 0:
            raise RuntimeError(
                f"'{' '.join(command)}' exited with a non-zero status of {res.returncode}. stderr = '{res.stderr}', stdout = '{res.stdout}'"
            )
        return json.loads(res.stdout)

    def _cached_info(self, opts=list()) -> list:
        if len(opts):
            return self.get_info(opts)
        return self.info

    def get_monitor(self, **kwargs) -> object:
        """
        Return an appropriate monitor for arch, depending on "make info" output.

        Port and Baud rate are taken from "make info".

        :param energytrace: `EnergyTraceMonitor` options. Returns an EnergyTrace monitor if not None.
        :param mimosa: `MIMOSAMonitor` options. Returns a MIMOSA monitor if not None.
        """
        for line in self.info:
            if "Monitor:" in line:
                _, port, arg = line.split(" ")
                if port == "run":
                    return ShellMonitor(arg, **kwargs)
                elif "mimosa" in kwargs and kwargs["mimosa"] is not None:
                    mimosa_kwargs = kwargs.pop("mimosa")
                    return MIMOSAMonitor(port, arg, **mimosa_kwargs, **kwargs)
                elif "energytrace" in kwargs and kwargs["energytrace"] is not None:
                    energytrace_kwargs = kwargs.pop("energytrace").copy()
                    sync_mode = energytrace_kwargs.pop("sync")
                    if sync_mode == "la":
                        return EnergyTraceLogicAnalyzerMonitor(
                            port, arg, **energytrace_kwargs, **kwargs
                        )
                    else:
                        return EnergyTraceMonitor(
                            port, arg, **energytrace_kwargs, **kwargs
                        )
                else:
                    kwargs.pop("energytrace", None)
                    kwargs.pop("mimosa", None)
                    return SerialMonitor(port, arg, **kwargs)
        raise RuntimeError("Monitor failure")

    def get_counter_limits(self, opts=list()) -> tuple:
        """Return multipass max counter and max overflow value for arch."""
        for line in self._cached_info(opts):
            match = re.match("Counter Overflow: ([^/]*)/(.*)", line)
            if match:
                overflow_value = int(match.group(1))
                max_overflow = int(match.group(2))
                return overflow_value, max_overflow
        raise RuntimeError("Did not find Counter Overflow limits")

    def sleep_ms(self, duration: int, opts=list()) -> str:
        max_sleep = None
        if "msp430fr" in self.name:
            cpu_freq = None
            for line in self._cached_info(opts):
                match = re.match(r"CPU\s+Freq:\s+(.*)\s+Hz", line)
                if match:
                    cpu_freq = int(match.group(1))
            if cpu_freq is not None and cpu_freq > 8_000_000:
                max_sleep = 250
            else:
                max_sleep = 500
        if max_sleep is not None and duration > max_sleep:
            sub_sleep_count = duration // max_sleep
            tail_sleep = duration % max_sleep
            ret = f"for (unsigned char i = 0; i < {sub_sleep_count}; i++) {{ arch.sleep_ms({max_sleep}); }}\n"
            if tail_sleep > 0:
                ret += f"arch.sleep_ms({tail_sleep});\n"
            return ret
        return f"arch.sleep_ms({duration});\n"

    def get_counter_limits_us(self, opts=list()) -> tuple:
        """Return duration of one counter step and one counter overflow in us."""
        cpu_freq = 0
        counter_freq = 0
        overflow_value = 0
        max_overflow = 0
        for line in self._cached_info(opts):
            match = re.match(r"CPU\s+Freq:\s+(.*)\s+Hz", line)
            if match:
                cpu_freq = int(match.group(1))
            match = re.match(r"Count\s+Freq:\s+(.*)\s+Hz", line)
            if match:
                counter_freq = int(match.group(1))
            match = re.match(r"Counter Overflow:\s+([^/]*)/(.*)", line)
            if match:
                overflow_value = int(match.group(1))
                max_overflow = int(match.group(2))
        if cpu_freq and not counter_freq:
            counter_freq = cpu_freq
        if counter_freq and overflow_value:
            return (
                1_000_000 / counter_freq,
                overflow_value * 1_000_000 / counter_freq,
                max_overflow,
            )
        raise RuntimeError("Did not find Counter Overflow limits")
