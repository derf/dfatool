#!/usr/bin/env python3

import numpy as np
import sys
import tensorflow as tf


def main(infile, outfile):
    keras_model = tf.keras.models.load_model(infile)
    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)

    rng = np.random.default_rng()
    rng_i = 0

    # not representative, but identical with the random data used by
    # benchmark-tflite.py
    def repr_gen():
        if rng_i < 1000:
            yield {
                keras_model.inputs[0].name: (
                    rng.random((1, *keras_model.input_shape[1:]), dtype=np.float32)
                    - 0.5
                )
                * 128
            }

    if outfile.endswith(".f32.tflite"):
        # 32-bit float weights (i.e., no quantization / optimization at all)
        converter.optimizations = []
    elif outfile.endswith(".f16.tflite"):
        # 16-bit float weights
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif outfile.endswith(".i8d.tflite"):
        # Default optimization: 8-bit integer weights with 32-bit float runtime operations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif outfile.endswith(".i8.tflite"):
        # 8-bit integer weights with 8-bit integer runtime operations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = repr_gen
    else:
        print("Output file suffix must match .{f32,f16,i8,i8d}.tflite", file=sys.stderr)
        sys.exit(1)

    tflite_model = converter.convert()
    with open(outfile, "wb") as f:
        f.write(tflite_model)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(
            f"Usage: {sys.argv[0]} <infile>.keras outfile.f32.f16|i8|i8d.<tflite>",
            file=sys.stderr,
        )
        sys.exit(1)

    main(*sys.argv[1:])
