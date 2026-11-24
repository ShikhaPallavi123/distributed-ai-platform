import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(importlib.util.find_spec("sklearn"), "research dependency")
class TestPublicSentiment(unittest.TestCase):
    def test_cascade_and_threshold(self):
        from inferbench.public_sentiment import cascade, choose_threshold, unique_rows
        cheap = np.array([[.9, .1], [.51, .49]])
        strong = np.array([[.9, .1], [.1, .9]])
        labels = np.array([0, 1])
        (_, threshold, accuracy), _ = choose_threshold(cheap, strong, labels, 0)
        predictions, _, escalation = cascade(cheap, strong, np.ones(2), np.ones(2), threshold)
        self.assertEqual(accuracy, 1)
        self.assertEqual(predictions.tolist(), [0, 1])
        self.assertEqual(escalation.tolist(), [False, True])
        self.assertEqual(unique_rows([("Same", 0), ("same", 1), ("new", 0)]), [("Same", 0), ("new", 0)])

    def test_multiline_record(self):
        from inferbench.public_sentiment import parse_labeled_text
        self.assertEqual(parse_labeled_text("first part\nsecond part\t1\nbad\t0\n"),
                         [("first part second part", 1), ("bad", 0)])
        with self.assertRaises(ValueError):
            parse_labeled_text("unfinished")
