#!/usr/bin/env python3
# vim:tabstop=4 softtabstop=4 shiftwidth=4 textwidth=160 smarttab expandtab colorcolumn=160

import argparse
import dfatool.codegen.tree as cg
import dfatool.functions as df
import numpy as np
import sklearn.datasets


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


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset-source",
        type=str,
        choices=("diabetes", "synthetic"),
        default="synthetic",
    )
    parser.add_argument("--dataset-n-samples", type=int, default=1000)
    parser.add_argument("--dataset-n-features", type=int, default=20)
    parser.add_argument("--dataset-n-informative", type=int, default=10)
    parser.add_argument(
        "--implementation", choices=("plain", "const", "template"), default="plain"
    )
    parser.add_argument(
        "--multipass-base", type=str, default="../../../projects/multipass"
    )
    parser.add_argument("--multipass-app", type=str, default="treebench")

    args = parser.parse_args()

    X, y = sklearn.datasets.make_regression(
        n_samples=args.dataset_n_samples,
        n_features=args.dataset_n_features,
        n_informative=args.dataset_n_informative,
    )

    param_names = list(map(lambda x: f"feat{x+1:02d}", range(args.dataset_n_features)))

    cart = df.CARTFunction(np.mean(y), param_names=param_names, arg_count=0)
    cart.fit(X, y)

    data = cart.to_json()
    ser = SerializedTree(data)

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
    impl.set_feature_type("float")
    impl.set_leaf_type("float")
    impl.set_feature_index_type("uint8_t")
    impl.set_num_features(len(X[0]))

    with open(f"{args.multipass_base}/src/app/{args.multipass_app}/tree.cc", "w") as f:
        f.write("\n".join(impl.to_c()) + "\n")

    with open(f"{args.multipass_base}/src/app/{args.multipass_app}/main.cc", "w") as f:
        f.write("\n".join(impl.get_benchmark(X, y)) + "\n")
