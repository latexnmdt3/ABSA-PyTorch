"""Smoke tests for the unified evaluator.

Run with::

    python -m unittest eval.tests.test_score

These tests use only the Python standard library so they can run without any
additional dependencies installed.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from eval.schema import validate_prediction, validate_predictions_file
from eval.score import score_predictions, _macro_f1, _accuracy

REPO_ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = REPO_ROOT / "eval" / "examples"


def _good_record(**overrides):
    rec = {
        "id": "x1",
        "dataset": "UIT-VSFC",
        "language": "vi",
        "task": "ACSA",
        "text": "abc",
        "gold": {"aspect": "lecturer", "sentiment": "positive"},
        "pred": {"aspect": "lecturer", "sentiment": "positive"},
        "raw_output": "lecturer: positive",
        "parse_ok": True,
        "method": "LCF-BERT",
        "paradigm": "Discriminative",
        "backbone": "vinai/phobert-base",
        "latency_ms": 1.0,
    }
    rec.update(overrides)
    return rec


class SchemaTests(unittest.TestCase):
    def test_good_record_passes(self):
        self.assertEqual(validate_prediction(_good_record()), [])

    def test_missing_field(self):
        rec = _good_record()
        del rec["latency_ms"]
        self.assertIn("missing field: latency_ms", validate_prediction(rec))

    def test_invalid_sentiment(self):
        rec = _good_record(gold={"aspect": "lecturer", "sentiment": "happy"})
        errs = validate_prediction(rec)
        self.assertTrue(any("gold.sentiment invalid" in e for e in errs), errs)

    def test_parse_error_consistency(self):
        rec = _good_record(
            parse_ok=False,
            pred={"aspect": "lecturer", "sentiment": "positive"},
        )
        errs = validate_prediction(rec)
        self.assertTrue(
            any("parse_ok=false" in e for e in errs),
            errs,
        )

    def test_parse_error_with_correct_tokens(self):
        rec = _good_record(
            parse_ok=False,
            pred={"aspect": "__PARSE_ERROR__", "sentiment": "__PARSE_ERROR__"},
        )
        self.assertEqual(validate_prediction(rec), [])

    def test_task_dataset_mismatch(self):
        rec = _good_record(task="ATSC")  # still UIT-VSFC dataset
        errs = validate_prediction(rec)
        self.assertTrue(any("ATSC requires SemEval" in e for e in errs), errs)

    def test_invalid_vsfc_aspect(self):
        rec = _good_record(gold={"aspect": "professor", "sentiment": "positive"})
        errs = validate_prediction(rec)
        self.assertTrue(
            any("aspect invalid for UIT-VSFC" in e for e in errs),
            errs,
        )

    def test_negative_latency(self):
        rec = _good_record(latency_ms=-1)
        errs = validate_prediction(rec)
        self.assertTrue(any("latency_ms must be >= 0" in e for e in errs), errs)

    def test_non_dict_pred_does_not_crash(self):
        # Regression: a non-dict truthy ``pred`` (e.g. a string) used to crash
        # with AttributeError because ``rec.get('pred', {}) or {}`` short-
        # circuits on truthy values, then ``.get()`` was called on the str.
        rec = _good_record(pred="positive")
        errs = validate_prediction(rec)
        self.assertTrue(
            any("pred: must be an object" in e for e in errs),
            errs,
        )
        # And ditto when parse_ok is False — must not crash.
        rec_false = _good_record(pred="__PARSE_ERROR__", parse_ok=False)
        errs_false = validate_prediction(rec_false)
        self.assertTrue(
            any("pred: must be an object" in e for e in errs_false),
            errs_false,
        )

    def test_non_dict_pred_list_does_not_crash(self):
        rec = _good_record(pred=["lecturer", "positive"], parse_ok=True)
        errs = validate_prediction(rec)
        self.assertTrue(
            any("pred: must be an object" in e for e in errs),
            errs,
        )

    def test_gold_cannot_be_parse_error_sentiment(self):
        # __PARSE_ERROR__ is only valid for ``pred`` (when parse_ok=false).
        # A gold record with __PARSE_ERROR__ would silently distort metrics,
        # so the validator must reject it.
        rec = _good_record(
            gold={"aspect": "lecturer", "sentiment": "__PARSE_ERROR__"}
        )
        errs = validate_prediction(rec)
        self.assertTrue(
            any("gold.sentiment invalid" in e for e in errs),
            errs,
        )

    def test_gold_cannot_be_parse_error_aspect(self):
        rec = _good_record(
            gold={"aspect": "__PARSE_ERROR__", "sentiment": "positive"}
        )
        errs = validate_prediction(rec)
        self.assertTrue(
            any("gold.aspect invalid for UIT-VSFC" in e for e in errs),
            errs,
        )

    def test_pred_can_be_parse_error_when_parse_ok_false(self):
        # The opposite side: pred *is* allowed to use __PARSE_ERROR__ when
        # parse_ok=false. Already covered above, but explicit here for clarity.
        rec = _good_record(
            parse_ok=False,
            pred={"aspect": "__PARSE_ERROR__", "sentiment": "__PARSE_ERROR__"},
        )
        self.assertEqual(validate_prediction(rec), [])


class MetricUtilTests(unittest.TestCase):
    def test_accuracy(self):
        self.assertAlmostEqual(_accuracy(["a", "b", "c"], ["a", "b", "x"]), 2 / 3)
        self.assertEqual(_accuracy([], []), 0.0)

    def test_macro_f1_perfect(self):
        labels = ["pos", "neg"]
        self.assertAlmostEqual(_macro_f1(["pos", "neg"], ["pos", "neg"], labels), 1.0)

    def test_macro_f1_zero(self):
        labels = ["pos", "neg"]
        # All wrong both ways → F1 should be 0 for both classes.
        self.assertAlmostEqual(_macro_f1(["pos", "neg"], ["neg", "pos"], labels), 0.0)


class ScoreVSFCDemoTests(unittest.TestCase):
    def test_score_vsfc_demo(self):
        path = EXAMPLES / "predictions_lcf_bert_vsfc_demo.jsonl"
        self.assertEqual(validate_predictions_file(path), [])
        m = score_predictions(path)
        self.assertEqual(m["dataset"], "UIT-VSFC")
        self.assertEqual(m["n_samples"], 5)
        # 1 sample has parse_ok=false out of 5
        self.assertAlmostEqual(m["parse_failure_rate"], 0.2)
        # Sentiment accuracy: 5 samples, gold = [pos, neg, neutral, neg, pos]
        # pred = [pos, neg, pos, neg, __PARSE_ERROR__] → 3 correct
        self.assertAlmostEqual(m["sentiment"]["accuracy"], 3 / 5)
        # As of SPEC v1.1 the topic is given as input — no aspect block
        # and no joint metric.
        self.assertNotIn("aspect", m)
        self.assertNotIn("joint_aspect_sentiment_acc", m)

    def test_score_semeval_demo_no_aspect_block(self):
        path = EXAMPLES / "predictions_lcf_bert_semeval_demo.jsonl"
        self.assertEqual(validate_predictions_file(path), [])
        m = score_predictions(path)
        self.assertEqual(m["task"], "ATSC")
        self.assertNotIn("aspect", m)
        self.assertNotIn("joint_aspect_sentiment_acc", m)
        self.assertEqual(m["n_samples"], 3)


class FailureModeTests(unittest.TestCase):
    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            empty = fh.name
        try:
            with self.assertRaises(ValueError):
                score_predictions(empty)
        finally:
            Path(empty).unlink(missing_ok=True)

    def test_mixed_method_raises(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            for method in ("LCF-BERT", "DOT"):
                rec = _good_record(method=method, id=f"x_{method}")
                fh.write(json.dumps(rec) + "\n")
            tmp = fh.name
        try:
            with self.assertRaises(ValueError):
                score_predictions(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
