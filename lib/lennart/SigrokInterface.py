import json

from dfatool.lennart.DataInterface import DataInterface


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

    @classmethod
    def fromTraces(cls, traces):
        """
        Generates SigrokResult from ptalog.json traces

        :param traces: traces from dfatool ptalog.json
        :return: SigrokResult object
        """
        timestamps = [0]
        for tr in traces:
            for t in tr["trace"]:
                # print(t['online_aggregates']['duration'][0])
                timestamps.append(
                    timestamps[-1] + (t["online_aggregates"]["duration"][0] * 10 ** -6)
                )

        # print(timestamps)
        # prepend FAKE Sync point
        t_neu = [0.0, 0.0000001, 1.0, 1.00000001]
        for i, x in enumerate(timestamps):
            t_neu.append(
                round(float(x) + t_neu[3] + 0.20, 6)
            )  # list(map(float, t_ist.split(",")[:i+1]))

        # append FAKE Sync point / eine überschneidung
        # [30.403632, 30.403639, 31.407265, 31.407271]
        # appendData = [29.144855,30.148495,30.148502,30.403632,30.403639,31.407265,31.407271,]
        appendData = [0, 1.000001, 1.000002, 1.25, 1.2500001]

        # TODO future work here, why does the sync not work completely
        t_neu[-1] = (
            t_neu[-2] + (t_neu[-1] - t_neu[-2]) * 0.9
        )  # Weird offset failure with UART stuff

        offset = t_neu[-1] - appendData[0]
        for x in appendData:
            t_neu.append(x + offset)

        # print(t_neu)
        print(len(t_neu))
        return SigrokResult(t_neu, False)


class SigrokInterface(DataInterface):
    def __init__(
        self, sample_rate, sample_count, driver="fx2lafw", filename="temp/sigrok.log"
    ):
        """

        :param sample_rate: Samplerate of the Logic Analyzer
        :param sample_count: Count of samples
        :param driver: for many Logic Analyzer from Saleae the "fx2lafw" should be working
        :param filename: temporary file name
        """
        # options
        self.sample_count = sample_count
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
