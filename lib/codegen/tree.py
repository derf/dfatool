#!/usr/bin/env python3
# vim:tabstop=4 softtabstop=4 shiftwidth=4 textwidth=160 smarttab expandtab colorcolumn=160

import numpy as np


class TreeImplementation:
    name = None
    id_type = "uint16_t"
    feature_type = "float"
    leaf_type = "float"
    feature_index_type = "uint8_t"
    section_prefix = str()
    is_forest = False
    num_trees = 1
    n_features = None
    split_cond = None
    split_le = False
    split_lt = False

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        assert self.feature_type == self.leaf_type
        if self.model.model_type == "forest":
            self.is_forest = True
            self.num_trees = len(self.model.trees)

    def __repr__(self):
        return f"{type(self).__name__}<id={self.id_type}, feat={self.feature_type}, ret={self.leaf_type}, K={self.num_trees}, n={self.n_features}>"

    def set_feature_type(self, feature_type):
        self.feature_type = feature_type
        if "int" in feature_type:
            self.feature_format = "5.0f"
        else:
            self.feature_format = "20.12f"

    def set_leaf_type(self, leaf_type):
        self.leaf_type = leaf_type
        assert self.leaf_type == self.feature_type
        if "int" in leaf_type:
            self.leaf_format = "5.0f"
        else:
            self.leaf_format = "20.12f"

    def set_feature_index_type(self, index_type):
        self.feature_index_type = index_type

    def set_num_features(self, n_features):
        self.n_features = n_features

    def get_attr(self):
        return {
            "implementation": self.name,
            "input_type": self.feature_type,
            "output_type": self.leaf_type,
            "n_features": self.n_features,
        }

    def struct_node(self):
        raise NotImplementedError

    def traversal_function(self):
        raise NotImplementedError

    def feature_vector(self):
        raise NotImplementedError

    def get_benchmark(self, X, y, debug=False, steps=5):
        ret = [
            '#include "arch.h"',
            '#include "driver/gpio.h"',
            '#include "driver/stdout.h"',
            '#include "driver/uptime.h"',
            '#include "driver/counter.h"',
            "#include <stdlib.h>",
            f"{self.leaf_type} traverse({self.feature_type} *param_vec);",
            f"{self.feature_type} param_vec[{self.n_features}];",
        ]

        if len(X):
            assert len(X[0]) == self.n_features

        param_values = list()
        for i in range(self.n_features):
            sorted_values = sorted(X[:, i])
            this_values = list()
            for j in range(steps):
                this_values.append(
                    sorted_values[int(((j + 0.5) / steps) * len(sorted_values))]
                )
            param_values.append(this_values)

        ret.append(
            f"const {self.feature_type} param_values[{len(param_values)}][{steps}] = {{"
        )
        for param_value in param_values:
            ret.append("{" + ",".join(map(str, param_value)) + "},")
        ret.append("};")

        ret += [
            f"volatile {self.leaf_type} result;",
            "void run_benchmark() {",
            "arch.delay_ms(4000);",
            "counter.start();",
            "counter.stop();",
            """kout << "nop=" << counter.value << "/" << counter.overflow << endl;""",
        ]
        for i in range(self.n_features):
            ret.append(f"for (uint8_t pv{i} = 0; pv{i} < {steps}; pv{i}++) {{")
            ret.append(f"param_vec[{i}] = param_values[{i}][pv{i}];")
        ret.append("counter.start();")
        ret.append("result = traverse(param_vec);")
        ret.append("counter.stop();")
        if debug:
            for i in range(self.n_features):
                ret.append(f"""kout << param_vec[{i}] << ";";""")
            ret.append("""kout << " = " << result << " @ ";""")
        ret.append(
            """kout << "cycles=" << counter.value << "/" << counter.overflow << endl;"""
        )
        ret.append("gpio.led_toggle();")
        for i in range(self.n_features):
            ret.append("}")
        ret.append("""kout << "done" << endl;""")
        ret.append("""kout << "done" << endl;""")
        ret.append("}")

        ret += [
            "int main(void)",
            "{",
            "    arch.setup();",
            "    gpio.setup();",
            "    kout.setup();",
            "        run_benchmark();",
            "    return 0;",
            "}",
        ]

        return ret

    def to_c(self):
        header = ["#include <stdint.h>"]
        lines = []
        if self.is_forest:
            for i, tree in enumerate(self.model.trees):
                lines += [
                    f"{self.section_prefix} struct node tree{i:03d}[{tree.max_id + 1}] = "
                    + "{"
                ]
                lines += self._node_to_c(tree.tree)
                lines += ["};"]
            lines.append(
                self.section_prefix
                + " struct node * const forest[] = {\n"
                + ",\n".join(
                    map(lambda i: f"tree{i:03d}", range(len(self.model.trees)))
                )
                + "};"
            )
        else:
            lines += [
                f"{self.section_prefix} struct node tree[{self.model.max_id + 1}] = "
                + "{"
            ]
            lines += self._node_to_c(self.model.tree)
            lines += ["};"]
        lines += self.traversal_function()
        header += self.struct_node()
        return header + lines

    def _node_to_c(self, node):
        if node["type"] == "static":
            node_id = node["id"]
            value = node["value"]
            return [
                f"{{.threshold = {value:{self.leaf_format}}, .rightChild =     0, .feat = 255}}, // {node_id:5d}"
            ]
        elif node["type"] == "scalarSplit":
            node_id = node["id"]
            feat = node["paramIndex"]
            if not self.split_le and not self.split_lt:
                if node["condition"] == "<":
                    self.split_lt = True
                    self.split_cond = "<"
                elif node["condition"] == "≤":
                    self.split_le = True
                    self.split_cond = "<="
            if self.split_le:
                assert node["condition"] == "≤"
            elif self.split_lt:
                assert node["condition"] == "<"
            threshold = node["threshold"]
            right_child = node["right"]["id"]
            lines = (
                [
                    f"{{.threshold = {threshold:{self.feature_format}}, .rightChild = {right_child:5d}, .feat = {feat:3d}}}, // {node_id:5d}"
                ]
                + self._node_to_c(node["left"])
                + self._node_to_c(node["right"])
            )
            return lines
        else:
            raise NotImplementedError(f"""node type {node["type"]} not supported yet""")


class PlainTree(TreeImplementation):
    name = "plain"
    section_prefix = "const"

    def struct_node(self):
        return [
            "struct node {",
            f"{self.feature_type} threshold;",
            f"{self.id_type} rightChild;",
            f"{self.feature_index_type} feat;",
            "};",
        ]

    def traversal_function(self):
        if self.is_forest:
            if self.num_trees > 255:
                tt = "uint16_t"
            else:
                tt = "uint8_t"
            return [
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                f"    {self.leaf_type} ret = 0;",
                f"    for ({tt} i = 0; i < {self.num_trees}; i++) {{",
                f"        const struct node *tree = forest[i];",
                f"        {self.id_type} index = 0;",
                "        while (tree[index].feat != 255) {",
                f"            bool cmp = features[tree[index].feat] {self.split_cond} tree[index].threshold;",
                "            index = cmp * (index + 1) + !cmp * tree[index].rightChild;",
                "        }",
                "        ret += tree[index].threshold;",
                "    }",
                "    return ret;",
                "}",
            ]
        else:
            return [
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                f"    {self.id_type} index = 0;",
                "    while (tree[index].feat != 255) {",
                f"        bool cmp = features[tree[index].feat] {self.split_cond} tree[index].threshold;",
                "        index = cmp * (index + 1) + !cmp * tree[index].rightChild;",
                "    }",
                "    return tree[index].threshold;",
                "}",
            ]

    def feature_vector(self):
        return [f"{self.feature_type} param_vec[{self.n_features:d}];"]


class ConstTree(PlainTree):
    name = "const"
    section_prefix = "const"

    def struct_node(self):
        return [
            "struct node {",
            f"{self.feature_type} threshold;",
            f"{self.id_type} rightChild;",
            f"{self.feature_index_type} feat;",
            f"{self.leaf_type} traverse(const node *tree, {self.feature_type} *features) const",
            "{",
            "    if (feat == 255) {",
            "        return threshold;",
            "    }",
            f"    if (features[feat] {self.split_cond} threshold) " + "{",
            "        return (this+1)->traverse(tree, features);",
            "    }",
            "    return tree[rightChild].traverse(tree, features);",
            "}",
            "};",
        ]

    def traversal_function(self):
        if self.is_forest:
            if self.num_trees > 255:
                tt = "uint16_t"
            else:
                tt = "uint8_t"
            return [
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                f"    {self.leaf_type} ret = 0;",
                f"    for ({tt} i = 0; i < {self.num_trees}; i++) {{",
                f"        ret += forest[i][0].traverse(forest[i], features);",
                "    }",
                "    return ret;",
                "}",
            ]
        else:
            return [
                "",
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                "    return tree[0].traverse(tree, features);",
                "}",
            ]


class TemplateTree(PlainTree):
    name = "template"
    section_prefix = "constexpr const"

    def traversal_function(self):
        if self.is_forest:
            ret = [
                f"template <int findex, int index> {self.leaf_type} traverseTree({self.feature_type} *features)",
                "{",
                "    if (forest[findex][index].feat == 255) {",
                "        return forest[findex][index].threshold;",
                "    }",
                f"    if (features[forest[findex][index].feat] {self.split_cond} forest[findex][index].threshold) "
                + "{",
                "        return traverseTree<findex, index+1>(features);",
                "    }",
                "    return traverseTree<findex, forest[findex][index].rightChild>(features);",
                "}",
                "",
                f"template <int findex> {self.leaf_type} traverseForest({self.feature_type} *features)",
                "{",
                "    return traverseTree<findex, 0>(features) + traverseForest<findex + 1>(features);",
                "}",
                "",
                f"template <> {self.leaf_type} traverseForest<{self.num_trees}> ({self.feature_type} *features)",
                "{",
                "    (void)features;",
                "    return 0;",
                "}",
            ]
            for i in range(self.num_trees):
                ret += [
                    "",
                    f"template <> {self.leaf_type} traverseTree<{i}, sizeof(tree{i:03d})/sizeof(node)> ({self.feature_type} *features)",
                    "{",
                    "    (void)features;",
                    "    return 0;",
                    "}",
                ]
            ret += [
                "",
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                "    return traverseForest<0>(features);",
                "}",
            ]
            return ret
        else:
            return [
                f"template <int index> {self.leaf_type} traverseTree({self.feature_type} *features)",
                "{",
                "    if (tree[index].feat == 255) {",
                "        return tree[index].threshold;",
                "    }",
                f"    if (features[tree[index].feat] {self.split_cond} tree[index].threshold) "
                + "{",
                "        return traverseTree<index+1>(features);",
                "    }",
                "    return traverseTree<tree[index].rightChild>(features);",
                "}",
                "",
                f"template <> {self.leaf_type} traverseTree<sizeof(tree)/sizeof(node)> ({self.feature_type} *features)",
                "{",
                "    (void)features;",
                "    return 0;",
                "}",
                "",
                f"{self.leaf_type} traverse({self.feature_type} *features)",
                "{",
                "    return traverseTree<0>(features);",
                "}",
            ]


class TreeData:
    algorithm = None

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

    def _set_id(self, node, node_id=0):
        """
        :param node_id: next free node id
        :returns: highest set node id (for leaves: identical to node_id)
        """
        node["id"] = node_id
        if node.get("left"):
            node_id = self._set_id(node["left"], node_id + 1)
        if node.get("right"):
            node_id = self._set_id(node["right"], node_id + 1)
        return node_id

    def _tree_to_c(self, impl, node):
        feature_type = impl.feature_type
        result_type = impl.leaf_type
        if "int" in feature_type:
            feature_format = ".0f"
        else:
            feature_format = "f"
        if "int" in result_type:
            result_format = ".0f"
        else:
            result_format = "f"
        if node["type"] == "split":
            ret = [
                "{"
                + f""".threshold = {node["threshold"]:{feature_format}}, .rightChild = {node["right"]["id"]:d}, .feat = {node["index"]:d}"""
                + "}"
            ]
            ret += self._tree_to_c(impl, node["left"])
            ret += self._tree_to_c(impl, node["right"])
            return ret
        else:
            return [
                "{"
                + f""".threshold = {node["value"]:{result_format}}, .rightChild = 0, .feat = 255"""
                + "}"
            ]
