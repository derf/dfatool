#!/usr/bin/env python3

import csv
import logging
import numpy as np

logger = logging.getLogger(__name__)


class KeysightCSV:
    """Simple loader for Keysight CSV data, as exported by the windows software."""

    def __init__(self):
        """Create a new KeysightCSV object."""
        pass

    def load_data(self, filename: str):
        """
        Load log data from filename, return timestamps and currents.

        Returns two one-dimensional NumPy arrays: timestamps and corresponding currents.
        """
        with open(filename) as f:
            for i, _ in enumerate(f):
                pass
            timestamps = np.ndarray((i - 3), dtype=float)
            currents = np.ndarray((i - 3), dtype=float)
        # basically seek back to start
        with open(filename) as f:
            for _ in range(4):
                next(f)
            reader = csv.reader(f, delimiter=",")
            for i, row in enumerate(reader):
                timestamps[i] = float(row[0])
                currents[i] = float(row[2]) * -1
        return timestamps, currents
