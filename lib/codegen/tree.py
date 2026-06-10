#!/usr/bin/env python3
# vim:tabstop=4 softtabstop=4 shiftwidth=4 textwidth=160 smarttab expandtab colorcolumn=160

import numpy as np


class TreeImplementation:
    name = None
    id_type = "uint16_t"
    feature_type = "float"
    leaf_type = "float"
    feature_index_type = "uint8_t"
    intercept = 0
    section_prefix = str()
    is_forest = False
    num_trees = 1
    n_features = None
    split_cond = None
    split_le = False
    split_lt = False
    header = ["#include <stdint.h>"]

    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        if self.model.model_type == "forest":
            self.is_forest = True
            self.num_trees = len(self.model.trees)
            self.aggregate = self.model.aggregate
            self.intercept = self.model.intercept

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
        return [f"{self.feature_type} param_vec[{self.n_features:d}];"]

    def get_benchmark(self, X, y, verify=False, steps=5, counter_key="cycles"):
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

        self.param_values = list()
        for i in range(self.n_features):
            sorted_values = sorted(X[:, i])
            this_values = list()
            for j in range(steps):
                this_values.append(
                    sorted_values[int(((j + 0.5) / steps) * len(sorted_values))]
                )
            self.param_values.append(this_values)

        ret.append(
            f"const {self.feature_type} param_values[{len(self.param_values)}][{steps}] = {{"
        )
        for param_value in self.param_values:
            ret.append(
                "{"
                + ",".join(map(lambda v: f"{v:{self.feature_format}}", param_value))
                + "},"
            )
        ret.append("};")

        ret += [
            f"volatile {self.leaf_type} result;",
            "void run_benchmark() {",
            "arch.delay_ms(1000);",
            "counter.start();",
            "counter.stop();",
            """kout << "nop=" << counter.value << "/" << counter.overflow << endl;""",
        ]
        if verify:
            ret.append("kout.setDigits(6);")
        for i in range(self.n_features):
            ret.append(f"for (uint8_t pv{i} = 0; pv{i} < {steps}; pv{i}++) {{")
            ret.append(f"param_vec[{i}] = param_values[{i}][pv{i}];")
        ret.append("counter.start();")
        ret.append("result = traverse(param_vec);")
        ret.append("counter.stop();")
        ret.append("gpio.led_on(0);")
        if verify:
            ret.append("""kout << "prediction=";""")
            for i in range(self.n_features):
                ret.append(f"""kout << param_vec[{i}] << ";";""")
            ret.append("""kout << result << endl;""")
        ret.append(
            f"""kout << "{counter_key}=" << counter.value << "/" << counter.overflow << endl;"""
        )
        ret.append("gpio.led_off(0);")
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
            "    gpio.led_on(1);",
            "    run_benchmark();",
            "    gpio.led_off(1);",
            "    gpio.led_on(2);",
            "    return 0;",
            "}",
        ]

        return ret

    def to_c(self):
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
        return self.header + self.struct_node() + lines

    def _node_to_c(self, node):
        if node["type"] == "static":
            node_id = node["id"]
            value = node["value"]
            return [
                f"{{.rightChild =     0, .feat = 255, .threshold = {value:{self.leaf_format}}}}, // {node_id:5d}"
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
                    f"{{.rightChild = {right_child:5d}, .feat = {feat:3d}, .threshold = {threshold:{self.feature_format}}}}, // {node_id:5d}"
                ]
                + self._node_to_c(node["left"])
                + self._node_to_c(node["right"])
            )
            return lines
        else:
            raise NotImplementedError(f"""node type {node["type"]} not supported yet""")


class PlainRMT(TreeImplementation):
    name = "plain"
    section_prefix = "const"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.header += [
            "#include <stdio.h>",
            "#include <stdlib.h>",
            "#include <cstddef>",
        ]

    def set_num_categorical(self, n_categorical, n_categories):
        self.n_categorical = n_categorical
        self.n_categories = n_categories

        self.empty_child = "{" + ", ".join(["  0" for i in range(n_categories)]) + "}"

    def struct_node(self):
        return [
            "struct params {",
            f"    uint8_t categorical[{self.n_categorical}];",
            f"    {self.feature_type} numeric[{self.n_features}];",
            "};",
            "struct node {",
            f"    {self.feature_index_type} const feat;",
            "    uint8_t n_keys;",
            f"    {self.id_type} const children[{self.n_categories}];",
            f"    {self.leaf_type} (* const leaf)({self.feature_type} *);",
            "};",
        ]

    def traversal_function(self):
        return [
            f"{self.leaf_type} traverse(struct params *features)",
            "{",
            f"    {self.id_type} i = 0;",
            "    while (tree[i].leaf == NULL) {",
            "        bool found = false;",
            "        for (uint8_t j = 0; j < tree[i].n_keys; j++) {",
            "            if (features->categorical[tree[i].feat] == j) {",
            "                i = tree[i].children[j];",
            "                found = true;",
            "                break;",
            "            }",
            "        }",
            "        if (!i || !found) {",
            # TODO calculate mean instead
            """            printf("tree[%u]: did not find a child for features.categorical[%u] == %u\\n", i, tree[i].feat, features->categorical[tree[i].feat]);""",
            "            return 0;",
            "        }",
            "    }",
            "    return tree[i].leaf(features->numeric);",
            "}",
        ]

    def to_c(self):
        lines = []
        self.leaves = list()
        tree_lines = self._node_to_c(self.model.tree)
        lines += self.leaves
        lines += [
            f"{self.section_prefix} struct node tree[{self.model.max_id + 1}] = " + "{"
        ]
        lines += tree_lines
        lines += ["};"]
        lines += self.traversal_function()
        return self.header + self.struct_node() + lines

    def _node_to_c(self, node):
        # {.feat = 0, .n_keys = 3, .children = children000, .leaf =      NULL}, // 000
        if node["type"] == "split":
            node_id = node["id"]
            feature = node["paramIndex"]
            n_children = len(node["child"])
            keys = list(range(0, max(map(int, node["child"].keys())) + 1))
            child_ids = list()
            for child_key in keys:
                if child_key in node["child"]:
                    child_ids.append(node["child"][child_key]["id"])
                else:
                    child_ids.append(0)

            child_array = (
                "{" + ", ".join(map(lambda child_id: f"{child_id:3d}", child_ids)) + "}"
            )

            # We only support splits on categorical features
            ret = [
                f"{{.feat = {feature - self.n_features:2d}, .n_keys = {n_children:2d}, .children = {child_array}, .leaf =      NULL}}, // {node_id:5d}"
            ]
            for child_key in sorted(node["child"].keys()):
                ret += self._node_to_c(node["child"][child_key])
            return ret
        elif node["type"] == "static":
            node_id = node["id"]
            value = node["value"]
            self.leaves.extend(
                [
                    f"{self.leaf_type} leaf{node_id:05d}({self.feature_type} * feat)",
                    "{",
                    "    (void)feat;",
                    f"    return {value};",
                    "}",
                ]
            )
            return [
                f"{{.feat =   0, .n_keys =  0, .children =          NULL, .leaf = leaf{node_id:05d}}}, // {node_id:5d}"
            ]
        elif node["type"] == "analytic":
            node_id = node["id"]
            function = node["functionStr"]
            for i in range(self.n_features):
                function = function.replace(f"parameter(feat{i+1:02d})", f"feat[{i:d}]")
            self.leaves.extend(
                [
                    f"{self.leaf_type} leaf{node_id:05d}({self.feature_type} * feat)",
                    "{",
                    f"    return {function};",
                    "}",
                ]
            )
            return [
                f"{{.feat =  0, .n_keys =  0, .children = {self.empty_child}, .leaf = leaf{node_id:05d}}}, // {node_id:5d}"
            ]
        else:
            raise NotImplementedError(f"""node type {node["type"]} not supported yet""")

    def get_benchmark(self, X, y, verify=False, steps=5, counter_key="cycles"):
        ret = [
            '#include "arch.h"',
            '#include "driver/gpio.h"',
            '#include "driver/stdout.h"',
            '#include "driver/uptime.h"',
            '#include "driver/counter.h"',
            "#include <stdlib.h>",
            "struct params {",
            f"    uint8_t categorical[{self.n_categorical}];",
            f"    {self.feature_type} numeric[{self.n_features}];",
            "};",
            f"{self.leaf_type} traverse(struct params *param_vec);",
            f"struct params param_vec;",
        ]

        if len(X):
            if len(X[0]) != self.n_features + self.n_categorical:
                raise ValueError(
                    f"Got {len(X[0])} features, expected {self.n_features + self.n_categorical} == {self.n_features} + {self.n_categorical}"
                )

        self.param_values = list()
        for i in range(self.n_features):
            sorted_values = sorted(X[:, i])
            this_values = list()
            for j in range(steps):
                this_values.append(
                    sorted_values[int(((j + 0.5) / steps) * len(sorted_values))]
                )
            self.param_values.append(this_values)

        ret.append(
            f"const {self.feature_type} param_values[{self.n_features}][{steps}] = {{"
        )
        for param_value in self.param_values:
            ret.append(
                "{"
                + ",".join(map(lambda v: f"{v:{self.feature_format}}", param_value))
                + "},"
            )
        ret.append("};")

        # needed for Python3 latency benchmark in eval-model-codegen.py
        for i in range(self.n_categorical):
            self.param_values.append(list(range(self.n_categories)))

        ret += [
            f"volatile {self.leaf_type} result;",
            "void run_benchmark() {",
            "arch.delay_ms(1000);",
            "counter.start();",
            "counter.stop();",
            """kout << "nop=" << counter.value << "/" << counter.overflow << endl;""",
        ]
        if verify:
            ret.append("kout.setDigits(6);")
        for i in range(self.n_features):
            ret.append(f"for (uint8_t pv{i} = 0; pv{i} < {steps}; pv{i}++) {{")
            ret.append(f"param_vec.numeric[{i}] = param_values[{i}][pv{i}];")
        for i in range(self.n_categorical):
            ret.append(
                f"for (uint8_t cat{i} = 0; cat{i} < {self.n_categories}; cat{i}++) {{"
            )
            ret.append(f"param_vec.categorical[{i}] = cat{i};")
        ret.append("counter.start();")
        ret.append("result = traverse(&param_vec);")
        ret.append("counter.stop();")
        if verify:
            ret.append("""kout << "prediction=";""")
            for i in range(self.n_features):
                ret.append(f"""kout << param_vec.numeric[{i}] << ";";""")
            for i in range(self.n_categorical):
                ret.append(f"""kout << param_vec.categorical[{i}] << ";";""")
            ret.append("""kout << result << endl;""")
        ret.append(
            f"""kout << "{counter_key}=" << counter.value << "/" << counter.overflow << endl;"""
        )
        ret.append("gpio.led_toggle();")
        for i in range(self.n_features + self.n_categorical):
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
            "    gpio.led_on(1);",
            "    run_benchmark();",
            "    gpio.led_off(1);",
            "    gpio.led_on(2);",
            "    return 0;",
            "}",
        ]

        return ret


class ConstRMT(PlainRMT):
    name = "const"

    def struct_node(self):
        return [
            "struct params {",
            f"    uint8_t categorical[{self.n_categorical}];",
            f"    {self.feature_type} numeric[{self.n_features}];",
            "};",
            "struct node {",
            f"    {self.feature_index_type} const feat;",
            "    uint8_t n_keys;",
            f"    {self.id_type} const children[{self.n_categories}];",
            f"    {self.leaf_type} (* const leaf)({self.feature_type} *);",
            f"    {self.leaf_type} traverse(const node *tree, struct params *features) const",
            "    {",
            "        if (this->leaf == NULL) {",
            "            for (uint8_t j = 0; j < this->n_keys; j++) {",
            "                if (features->categorical[this->feat] == j) {",
            "                    return tree[this->children[j]].traverse(tree, features);",
            "                }",
            "            }",
            # TODO calculate mean instead
            """           printf("tree: did not find a child for features.categorical[%u] == %u\\n", this->feat, features->categorical[this->feat]);""",
            "            return 0;",
            "        }",
            "        return this->leaf(features->numeric);",
            "    }",
            "};",
        ]

    def traversal_function(self):
        return [
            "",
            f"{self.leaf_type} traverse(struct params *features)",
            "{",
            "    return tree[0].traverse(tree, features);",
            "}",
        ]


class TemplateRMT(PlainRMT):
    name = "template"
    section_prefix = "constexpr const"

    def traversal_function(self):
        ret = [
            f"template <int index> {self.leaf_type} traverseTree(struct params *features)",
            "{",
            "    if (tree[index].leaf == NULL) {",
        ]
        for category in range(self.n_categories):
            ret += [
                f"        if (features->categorical[tree[index].feat] == {category}) "
                + "{",
                f"            return traverseTree<tree[index].children[{category}]>(features);",
                "        }",
            ]
        ret += [
            "        return 0;",
            "    }",
            "    return tree[index].leaf(features->numeric);",
            "}",
            f"template <> {self.leaf_type} traverseTree<sizeof(tree)/sizeof(node)> (struct params *features)",
            "{",
            "    (void)features;",
            "    return 0;",
            "}",
            "",
            f"{self.leaf_type} traverse(struct params *features)",
            "{",
            "    return traverseTree<0>(features);",
            "}",
        ]

        return ret


class PlainTree(TreeImplementation):
    name = "plain"
    section_prefix = "const"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        assert self.feature_type == self.leaf_type

    def set_leaf_type(self, leaf_type):
        super().set_leaf_type(leaf_type)
        assert self.leaf_type == self.feature_type

    def struct_node(self):
        return [
            "struct node {",
            f"const {self.id_type} rightChild;",
            f"const {self.feature_index_type} feat;",
            f"const {self.feature_type} threshold;",
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
                f"    return {self.intercept:{self.leaf_format}} + ret;",
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


class ConstTree(PlainTree):
    name = "const"

    # hier ginge auch "constexpr const", das erzeugt aber identischen Maschinencode. Also lassen wir es.
    section_prefix = "const"

    def struct_node(self):
        return [
            "struct node {",
            f"const {self.id_type} rightChild;",
            f"const {self.feature_index_type} feat;",
            f"const {self.feature_type} threshold;",
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
                f"    return {self.intercept:{self.leaf_format}} + ret;",
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
                f"    return {self.intercept:{self.leaf_format}} + traverseForest<0>(features);",
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
                # Hier kann man nicht den "bool cmp"-Trick fahren, denn features ist keine constexpr
                f"    if (features[tree[index].feat] {self.split_cond} tree[index].threshold) "
                + "{",
                "        return traverseTree<index+1>(features);",
                "    }",
                "    return traverseTree<tree[index].rightChild>(features);",
                "}",
                "",
                "/*",
                " * This function will never be called as the corresponding template instance is unreachable.",
                " * However, the compiler does not know that (yet), so we must provided it to terminate template instantiation.",
                " */",
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
