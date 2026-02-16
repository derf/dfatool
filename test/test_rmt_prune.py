#!/usr/bin/env python3

from dfatool.model import ModelAttribute
import unittest

import numpy as np


class TestRMTPruning(unittest.TestCase):

    def test_rmt_flatten(self):
        X = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
        y = (1, 2, 3, 4, 5, 6)
        ma = ModelAttribute("test", "test", y, X, ("p1", "p2"))
        ma.build_rmt(with_function_leaves=False, threshold=0)

        output = list(
            map(
                lambda kkv: (
                    [("p1", "==", kkv[0][0]), ("p2", "==", kkv[0][1])],
                    kkv[1],
                ),
                zip(X, y),
            )
        )

        self.assertEqual(ma.model_function.flatten(), output)

    def test_rmt_no_prune(self):
        X = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
        y = (1, 2, 2, 4, 5, 5)
        ma = ModelAttribute("test", "test", y, X, ("p1", "p2"))
        ma.build_rmt(with_function_leaves=False, threshold=0)

        output = list(
            map(
                lambda kkv: (
                    [("p1", "==", kkv[0][0]), ("p2", "==", kkv[0][1])],
                    kkv[1],
                ),
                zip(X, y),
            )
        )

        self.assertEqual(ma.model_function.flatten(), output)

    def test_rmt_prune_categorical(self):
        X = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
        y = (1, 2, 2, 4, 5, 5)
        ma = ModelAttribute("test", "test", y, X, ("p1", "p2"))
        ma.build_rmt(with_function_leaves=False, threshold=0, prune=True)

        output = [
            ([("p1", "==", 1), ("p2", "==", 1)], 1),
            ([("p1", "==", 1), ("p2", "==", (2, 3))], 2),
            ([("p1", "==", 2), ("p2", "==", 1)], 4),
            ([("p1", "==", 2), ("p2", "==", (2, 3))], 5),
        ]

        self.assertEqual(ma.model_function.flatten(), output)

    def test_rmt_prune_scalar(self):
        X = ((1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (2, 3))
        y = (1, 2, 2, 4, 5, 5)
        ma = ModelAttribute("test", "test", y, X, ("p1", "p2"))
        ma.build_rmt(
            with_function_leaves=False, threshold=0, prune=True, prune_scalar=True
        )

        output = [
            ([("p1", "<=", 1), ("p2", "<=", 1)], 1),
            ([("p1", "<=", 1), ("p2", ">", 1)], 2),
            ([("p1", ">", 1), ("p2", "<=", 1)], 4),
            ([("p1", ">", 1), ("p2", ">", 1)], 5),
        ]

        self.assertEqual(ma.model_function.flatten(), output)
