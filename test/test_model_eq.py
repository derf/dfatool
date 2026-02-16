#!/usr/bin/env python3

import dfatool.functions as df
import unittest

import numpy as np


class TestModelEquality(unittest.TestCase):
    def test_static(self):
        s1a = df.StaticFunction(23.0, n_samples=50)
        s1b = df.StaticFunction(23.0, n_samples=100)
        s2 = df.StaticFunction(24, n_samples=50)

        self.assertEqual(s1a, s1b, "StaticFunction equality")
        self.assertNotEqual(s1a, s2, "StaticFunction inequality")

    def test_split_categarical(self):
        s1a = df.StaticFunction(23.0, n_samples=50)
        s1b = df.StaticFunction(24.0, n_samples=150)
        s2a = df.StaticFunction(23.0, n_samples=600)
        s2b = df.StaticFunction(24.0, n_samples=200)
        s3a = df.StaticFunction(23.0, n_samples=600)
        s3b = df.StaticFunction(24.0, n_samples=200)
        s4a = df.StaticFunction(25.0, n_samples=600)
        s4b = df.StaticFunction(26.0, n_samples=200)

        c1 = {"a": s1a, "b": s1b}
        c2 = {"a": s2a, "b": s2b}
        c2e = {"a": s2b, "b": s2a}
        c3 = {"a": s3a, "b": s3b}
        c4 = {"a": s4a, "b": s4b}

        t1 = df.SplitFunction(23.25, 1, "foo", c1)
        t2 = df.SplitFunction(23.75, 1, "foo", c2)
        t2e = df.SplitFunction(23.75, 1, "foo", c2e)
        t3 = df.SplitFunction(23.25, 2, "bar", c3)
        t4 = df.SplitFunction(23.25, 1, "foo", c4)

        self.assertEqual(t1, t2, "SplitFunction equality")
        self.assertNotEqual(t1, t2e, "SplitFunction child key equality")
        self.assertNotEqual(t1, t3, "SplitFunction index equality")
        self.assertNotEqual(t1, t4, "SplitFunction child value equality")

    def test_split_scalar(self):
        s1a = df.StaticFunction(23.0, n_samples=50)
        s1b = df.StaticFunction(24.0, n_samples=150)
        s2a = df.StaticFunction(23.0, n_samples=600)
        s2b = df.StaticFunction(24.0, n_samples=200)
        s3a = df.StaticFunction(23.0, n_samples=600)
        s3b = df.StaticFunction(24.0, n_samples=200)
        s4a = df.StaticFunction(25.0, n_samples=600)
        s4b = df.StaticFunction(26.0, n_samples=200)

        t1 = df.ScalarSplitFunction(23.25, 1, "foo", 10, s1a, s1b)
        t2 = df.ScalarSplitFunction(23.75, 1, "foo", 10, s2a, s2b)
        t2e = df.ScalarSplitFunction(23.75, 1, "foo", 10, s2b, s2a)
        t2n = df.ScalarSplitFunction(23.75, 1, "foo", 11, s2a, s2b)
        t3 = df.ScalarSplitFunction(23.25, 2, "bar", 10, s3a, s3b)
        t4 = df.ScalarSplitFunction(23.25, 1, "foo", 10, s4a, s4b)

        self.assertEqual(t1, t2, "ScalarSplitFunction equality")
        self.assertNotEqual(t1, t2e, "ScalarSplitFunction child value equality")
        self.assertNotEqual(t1, t2n, "ScalarSplitFunction threshold equality")
        self.assertNotEqual(t1, t3, "ScalarSplitFunction index equality")
        self.assertNotEqual(t1, t4, "ScalarSplitFunction child value equality")
