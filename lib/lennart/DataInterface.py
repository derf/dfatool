class DataInterface:
    def runMeasure(self):
        """
        Implemented in subclasses.

        Starts the measurement
        """
        raise NotImplementedError("The method not implemented")

    def getData(self):
        """
        Implemented in subclasses

        :returns: gathered data
        """
        raise NotImplementedError("The method not implemented")

    def forceStopMeasure(self):
        """
        Implemented in subclasses

        Force stops the measurement
        """
        raise NotImplementedError("The method not implemented")
