#!/usr/bin/env python3

import sys
import tensorflow as tf


def main(model_file):
    tf.lite.experimental.Analyzer.analyze(model_path=model_file)


if __name__ == "__main__":
    main(*sys.argv[1:])
