#!/usr/bin/env python3
# vim:tabstop=4 softtabstop=4 shiftwidth=4 textwidth=160 smarttab expandtab colorcolumn=160

import argparse
import dfatool.codegen.tree as cg
import dfatool.functions as df
from dfatool.utils import NpEncoder
import json
import numpy as np
import sklearn.datasets
import sys


class SerializedTree:
    model_type = "tree"

    def __init__(self, data):
        self.tree = data
        self.max_id = self._set_max_id(self.tree, 0)

    def _set_max_id(self, node, n):
        node["id"] = n
        if "left" in node:
            n = self._set_max_id(node["left"], n + 1)
            n = self._set_max_id(node["right"], n + 1)
        # TODO deal with RMT (node["children"])

        return n

    def get_n_features(self, node=None):
        if node is None:
            return self.get_n_features(self.tree)

        if node["type"] == "scalarSplit":
            feat_id = node["paramIndex"]
            return max(
                (
                    feat_id,
                    self.get_n_features(node["left"]),
                    self.get_n_features(node["right"]),
                )
            )

        return 0


class SerializedForest:
    model_type = "forest"

    def __init__(self, data):
        assert data["type"] == "ensemble"
        self.aggregate = data["aggregate"]
        self.intercept = data["intercept"]
        self.trees = list(map(SerializedTree, data["models"]))
        self.max_id = max(map(lambda tree: tree.max_id, self.trees))

    def get_n_features(self):
        return max(map(lambda tree: tree.get_n_features(), self.trees))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=("diabetes", "synthetic"),
        default="synthetic",
    )
    parser.add_argument("--dataset-n-samples", type=int, default=1000)
    parser.add_argument("--dataset-n-features", type=int, default=10)
    parser.add_argument("--dataset-load", type=str, metavar="data.json[.xz]")
    parser.add_argument("--dataset-save", type=str, metavar="data.json[.xz]")
    parser.add_argument(
        "--implementation", choices=("plain", "const", "template"), default="plain"
    )
    parser.add_argument("--model", choices=("CART", "XGB"), default="CART")
    parser.add_argument("--model-load", metavar="model.json[.xz]", type=str)
    parser.add_argument("--model-save", metavar="model.json[.xz]", type=str)
    parser.add_argument(
        "--multipass-base", type=str, default="../../../projects/multipass"
    )
    parser.add_argument("--multipass-app", type=str, default="treebench")
    parser.add_argument(
        "--type", choices="int8_t int16_t int32_t float double".split(), default="float"
    )

    args = parser.parse_args()

    if args.dataset_load:
        if args.dataset_load.endswith(".xz"):
            import lzma

            with lzma.open(args.dataset_load, "rt") as f:
                dataset = json.load(f)
        else:
            with open(args.dataset_load, "r") as f:
                dataset = json.load(f)
        X = np.array(dataset["X"])
        y = np.array(dataset["y"])
        param_names = list(map(lambda x: f"feat{x+1:02d}", range(len(X[0]))))
    else:
        X, y = sklearn.datasets.make_regression(
            n_samples=args.dataset_n_samples,
            n_features=args.dataset_n_features,
            n_informative=args.dataset_n_features,
        )
        if "int" in args.type:
            X_min, X_max = np.min(X), np.max(X)
            y_min, y_max = np.min(y), np.max(y)
            if args.type == "int8_t":
                range_min, range_max = -128, 127
            elif args.type == "int16_t":
                range_min, range_max = -32768, 32767
            elif args.type == "int32_t":
                range_min, range_max = -2147483648, 2147483647

            if X_min < 0:
                X_add = 0
                X_mul = range_max / max(X_max, -X_min)
            else:
                X_add = -X_min
                X_mul = (range_max - range_min) / abs(X_max - X_min)
            X = ((X + X_add) * X_mul).astype(int)

            if y_min < 0:
                y_add = 0
                y_mul = range_max / max(y_max, -y_min)
            else:
                y_add = -y_min
                y_mul = (range_max - range_min) / abs(y_max - y_min)
            y = ((y + y_add) * y_mul).astype(int)

        param_names = list(
            map(lambda x: f"feat{x+1:02d}", range(args.dataset_n_features))
        )

    if args.dataset_save:
        if args.dataset_save.endswith(".xz"):
            import lzma

            with lzma.open(args.dataset_save, "wt") as f:
                json.dump({"X": X, "y": y}, f, cls=NpEncoder)
        else:
            with open(args.dataset_save, "w") as f:
                json.dump({"X": X, "y": y}, f, cls=NpEncoder)
        sys.exit(0)

    if args.model_load:
        if args.model_load.endswith(".xz"):
            import lzma

            with lzma.open(args.model_load, "rt") as f:
                data = json.load(f)
        else:
            with open(args.model_load, "r") as f:
                data = json.load(f)
        if data["type"] == "ensemble":
            ser = SerializedForest(data)
        else:
            ser = SerializedTree(data)
        n_features = ser.get_n_features()
    elif args.model == "CART":
        cart = df.CARTFunction(np.mean(y), param_names=param_names, arg_count=0)
        cart.fit(X, y)
        data = cart.to_json()
        ser = SerializedTree(data)
    elif args.model == "XGB":
        cart = df.XGBoostFunction(np.mean(y), param_names=param_names, arg_count=0)
        cart.fit(X, y)
        data = cart.to_json()
        ser = SerializedForest(data)

    if args.model_save:
        if args.model_save.endswith(".xz"):
            import lzma

            with lzma.open(args.model_save, "wt") as f:
                json.dump(data, f, cls=NpEncoder)
        else:
            with open(args.model_save, "w") as f:
                json.dump(data, f, cls=NpEncoder)
        sys.exit(0)

    del data

    if args.implementation == "plain":
        impl = cg.PlainTree(model=ser)
    elif args.implementation == "const":
        impl = cg.ConstTree(model=ser)
    elif args.implementation == "template":
        impl = cg.TemplateTree(model=ser)
    else:
        raise RuntimeError(f"Invalid argument: --implementation={impl}")

    impl.id_type = (
        "uint8_t"
        if ser.max_id < 256
        else ("uint16_t" if ser.max_id < 65536 else "uint32_t")
    )
    impl.set_feature_type(args.type)
    impl.set_leaf_type(args.type)
    impl.set_feature_index_type("uint8_t")
    impl.set_num_features(len(X[0]))

    with open(f"{args.multipass_base}/src/app/{args.multipass_app}/tree.cc", "w") as f:
        f.write("\n".join(impl.to_c()) + "\n")

    with open(f"{args.multipass_base}/src/app/{args.multipass_app}/main.cc", "w") as f:
        f.write("\n".join(impl.get_benchmark(X, y)) + "\n")
