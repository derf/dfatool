#!/usr/bin/env python3

import json
import numpy as np
import os
from scipy.cluster.vq import kmeans2
import struct
import sys
import tarfile
from dfatool import AEMRAnalyzer

if __name__ == '__main__':
    filename = sys.argv[1]
    analyzer = AEMRAnalyzer(filename)

    analyzer.preprocess()
    sys.exit(0)
