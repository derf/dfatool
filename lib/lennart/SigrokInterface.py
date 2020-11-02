import json
import numpy as np

from dfatool.lennart.DataInterface import DataInterface
import logging

logger = logging.getLogger(__name__)


# Adding additional parsing functionality
class SigrokResult:
    def __init__(self, timestamps, onbeforefirstchange):
        """
        Creates SigrokResult object, struct for timestamps and onBeforeFirstChange.

        :param timestamps: list of changing timestamps
        :param onbeforefirstchange: if the state before the first change is already on / should always be off, just to be data correct
        """
        self.timestamps = timestamps
        self.onBeforeFirstChange = onbeforefirstchange

    def __str__(self):
        """
        :return: string representation of object
        """
        return "<Sigrok Result onBeforeFirstChange=%s timestamps=%s>" % (
            self.onBeforeFirstChange,
            self.timestamps,
        )

    def getDict(self):
        """
        :return: dict representation of object
        """
        data = {
            "onBeforeFirstChange": self.onBeforeFirstChange,
            "timestamps": self.timestamps,
        }
        return data

    @classmethod
    def fromFile(cls, path):
        """
        Generates SigrokResult from json_file

        :param path: file path
        :return: SigrokResult object
        """
        with open(path) as json_file:
            data = json.load(json_file)
            return SigrokResult(data["timestamps"], data["onBeforeFirstChange"])
        pass

    @classmethod
    def fromString(cls, string):
        """
        Generates SigrokResult from string

        :param string: string
        :return: SigrokResult object
        """
        data = json.loads(string)
        return SigrokResult(data["timestamps"], data["onBeforeFirstChange"])
        pass


class SigrokInterface(DataInterface):
    def __init__(self, sample_rate, driver="fx2lafw", filename="temp/sigrok.log"):
        """

        :param sample_rate: Samplerate of the Logic Analyzer
        :param driver: for many Logic Analyzer from Saleae the "fx2lafw" should be working
        :param filename: temporary file name
        """
        # options
        self.sample_rate = sample_rate
        self.file = open(filename, "w+")
        self.driver = driver

        # internal data
        self.changes = []
        self.start = None
        self.last_val = None
        self.index = 0

    def runMeasure(self):
        """
        Not implemented because implemented in subclasses
        :return: None
        """
        raise NotImplementedError("The method not implemented")

    def forceStopMeasure(self):
        """
        Not implemented because implemented in subclasses
        :return: None
        """
        raise NotImplementedError("The method not implemented")

    def getData(self):
        """

        :return:
        """
        # return sigrok_energy_api_result(self.changes, True if self.start == 0xff else False)
        return SigrokResult(
            [x / self.sample_rate for x in self.changes],
            True if self.start == 0xFF else False,
        )

    def analyzeData(self, byte):
        """
        analyze one byte if it differs from the last byte, it will be appended to changes.

        :param byte: one byte to analyze
        """
        if self.start is None:
            self.start = byte
            self.last_val = byte
        if byte != self.last_val:
            self.changes.append(self.index)
            self.last_val = byte
        self.index += 1
