import time

from dfatool.lennart.SigrokInterface import SigrokInterface

import sigrok.core as sr
from sigrok.core.classes import *

from util.ByteHelper import ByteHelper
import logging

logger = logging.getLogger(__name__)


class SigrokAPIInterface(SigrokInterface):
    def datafeed_changes(self, device, packet):
        """
        Callback type with changes analysis
        :param device: device object
        :param packet: data (String with binary data)
        """
        data = ByteHelper.rawbytes(self.output.receive(packet))
        if data:
            # only using every second byte,
            # because only every second contains the useful information.
            for x in data[1::2]:
                self.analyzeData(x)

    def datafeed_in_all(self, device, packet):
        """
        Callback type which writes all data into the array
        :param device: device object
        :param packet: data (String with binary data)
        """
        data = ByteHelper.rawbytes(self.output.receive(packet))
        if data:
            # only using every second byte,
            # because only every second contains the useful information.
            self.all_data += data[1::2]

    def datafeed_file(self, device, packet):
        """
        Callback type which writes all data into a file
        :param device: device object
        :param packet: data (String with binary data)
        """
        data = ByteHelper.rawbytes(self.output.receive(packet))
        if data:
            # only using every second byte,
            # because only every second contains the useful information.
            for x in data[1::2]:
                self.file.write(str(x) + "\n")

    def __init__(
        self,
        driver="fx2lafw",
        sample_rate=100_000,
        debug_output=False,
        used_datafeed=datafeed_changes,
        fake=False,
    ):
        """

        :param driver: Driver that should be used
        :param sample_rate: The sample rate of the Logic analyzer
        :param debug_output: Should be true if output should be displayed to user
        :param used_datafeed: one of the datafeeds above, user later as callback.
        :param fake:
        """
        super(SigrokAPIInterface, self).__init__(sample_rate)
        if fake:
            raise NotImplementedError("Not implemented!")
        self.used_datafeed = used_datafeed

        self.debug_output = debug_output
        self.session = None

    def forceStopMeasure(self):
        """
        Force stopping the measurement
        :return: None
        """
        self.session.stop()

    def runMeasure(self):
        """
        Start the Measurement and set all settings
        """
        context = sr.Context_create()

        devs = context.drivers[self.driver].scan()
        # print(devs)
        if len(devs) == 0:
            raise RuntimeError("No device with that driver found!")
        sigrokDevice = devs[0]
        if len(devs) > 1:
            raise Warning(
                "Attention! Multiple devices with that driver found! Using ",
                sigrokDevice.connection_id(),
            )

        sigrokDevice.open()
        sigrokDevice.config_set(ConfigKey.SAMPLERATE, self.sample_rate)

        enabled_channels = ["D1"]
        for channel in sigrokDevice.channels:
            channel.enabled = channel.name in enabled_channels

        self.session = context.create_session()
        self.session.add_device(sigrokDevice)
        self.session.start()

        self.output = context.output_formats["binary"].create_output(sigrokDevice)

        print(context.output_formats)
        self.all_data = b""

        def datafeed(device, packet):
            self.used_datafeed(self, device, packet)

        self.session.add_datafeed_callback(datafeed)
        time_running = time.time()
        self.session.run()
        total_time = time.time() - time_running
        print(
            "Used time: ",
            total_time * 1_000_000,
            "µs",
        )
        self.session.stop()

        if self.debug_output:
            # if self.used_datafeed == self.datafeed_in_change:
            if True:
                changes = [x / self.sample_rate for x in self.changes]
                print(changes)
                is_on = self.start == 0xFF
                print("0", " - ", changes[0], " # Pin ", "HIGH" if is_on else "LOW")
                for x in range(len(changes) - 1):
                    is_on = not is_on
                    print(
                        changes[x],
                        " - ",
                        changes[x + 1],
                        " / ",
                        changes[x + 1] - changes[x],
                        " # Pin ",
                        "HIGH" if is_on else "LOW",
                    )
            elif self.used_datafeed == self.datafeed_in_all:
                print(self.all_data)
