import subprocess
import time

from dfatool.lennart.SigrokInterface import SigrokInterface


class SigrokCLIInterface(SigrokInterface):
    def __init__(
        self,
        bin_temp_file="temp/out.bin",
        sample_rate=100_000,
        sample_count=1_000_000,
        fake=False,
    ):
        """
        creates SigrokCLIInterface object. Uses the CLI Interface (Command: sigrok-cli)

        :param bin_temp_file: temporary file for binary output
        :param sample_rate: The sample rate of the Logic analyzer
        :param sample_count: The sample count of the Logic analyzer
        :param fake: if it should use existing data
        """
        super(SigrokCLIInterface, self).__init__(sample_rate, sample_count)
        self.fake = fake
        self.bin_temp_file = bin_temp_file
        self.sigrok_cli_thread = None

    def forceStopMeasure(self):
        """
        Force stopping measure, sometimes needs pkill for killing definitly
        :return: None
        """
        self.sigrok_cli_thread.send_signal(subprocess.signal.SIGKILL)
        stdout, stderr = self.sigrok_cli_thread.communicate(timeout=15)
        time.sleep(5)
        # TODO not nice solution, make better
        import os

        os.system("pkill -f sigrok")
        self.runOpenAnalyze()

    def runMeasure(self):
        """
        starts the measurement, with waiting for done
        """
        if not self.fake:
            self.runMeasureAsynchronous()
        self.waitForAsynchronousMeasure()

    def runMeasureAsynchronous(self):
        """
        starts the measurement, not waiting for done
        """
        shellcommand = (
            'sigrok-cli --output-file %s --output-format binary --samples %s -d %s --config "samplerate=%s Hz"'
            % (self.bin_temp_file, self.sample_count, self.driver, self.sample_rate)
        )
        self.sigrok_cli_thread = subprocess.Popen(shellcommand, shell=True)

    def waitForAsynchronousMeasure(self):
        """
        Wait until is command call is done
        """
        if not self.fake:
            self.sigrok_cli_thread.wait()
        self.runOpenAnalyze()

    def runOpenAnalyze(self):
        """
        Opens the generated binary file and parses it byte by byte

        """
        in_file = open(self.bin_temp_file, "rb")  # opening for [r]eading as [b]inary
        data = in_file.read()  # if you only wanted to read 512 bytes, do .read(512)
        in_file.close()

        for x in data:
            self.analyzeData(x)
