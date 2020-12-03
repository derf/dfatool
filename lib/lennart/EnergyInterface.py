import re
import subprocess

from dfatool.lennart.DataInterface import DataInterface
import logging

logger = logging.getLogger(__name__)


class EnergyInterface(DataInterface):
    def __init__(
        self,
        duration_seconds=10,
        console_output=False,
        temp_file="temp/energytrace.log",
        fake=False,
    ):
        """
        class is not used in embedded into dfatool.

        :param duration_seconds: seconds the EnergyTrace should be running
        :param console_output: if EnergyTrace output should be printed to the user
        :param temp_file: file path for temporary file
        :param fake: if already existing file should be used
        """
        self.energytrace = None
        self.duration_seconds = duration_seconds
        self.console_output = console_output
        self.temp_file = temp_file
        self.fake = fake

    def runMeasure(self):
        """
        starts the measurement, with waiting for done
        """
        if self.fake:
            return
        self.runMeasureAsynchronously()
        self.waitForAsynchronousMeasure()

    def runMeasureAsynchronously(self):
        """
        starts the measurement, not waiting for done
        """
        if self.fake:
            return
        self.energytrace = subprocess.Popen(
            "msp430-etv --save %s %s %s"
            % (
                self.temp_file,
                self.duration_seconds,
                "" if self.console_output else "> /dev/null",
            ),
            shell=True,
        )
        print(
            "msp430-etv --save %s %s %s"
            % (
                self.temp_file,
                self.duration_seconds,
                "" if self.console_output else "> /dev/null",
            )
        )

    def waitForAsynchronousMeasure(self):
        """
        Wait until is command call is done
        """
        if self.fake:
            return
        self.energytrace.wait()

    def setFile(self, path):
        """
        changeing the temporary file

        :param path: file path of new temp file
        :return: None
        """
        self.temp_file = path
        pass

    def forceStopMeasure(self):
        """
        force stops the Measurement, with signals
        :return: None
        """
        self.energytrace.send_signal(subprocess.signal.SIGINT)
        stdout, stderr = self.energytrace.communicate(timeout=15)
