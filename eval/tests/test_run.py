"""End-to-end tests for the standalone runner ``evaluate.py``.

We use a tiny fake Predictor so the test suite stays dependency-free
(no torch, no transformers, no GPU). Run with::

    python -m unittest eval.tests.test_run
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from evaluate import (  # noqa: E402  - sys.path tweak above
    Predictor,
    RunConfig,
    _import_predictor,
    _percentile,
    load_test_set,
    run_evaluation,
)


class _PerfectVSFCPredictor:
    """Always returns the gold label. Used to verify the runner end-to-end."""

    method = "Fake-Perfect"
    paradigm = "Discriminative"
    backbone = "fake/backbone"

    def __init__(self, gold_lookup: dict[str, tuple[str, str]]) -> None:
        self._gold = gold_lookup

    def predict(self, text, aspect=None):
        # Deliberately ignore ``aspect`` and look up by text instead.
        return (*self._gold[text], "ok")


class _BrokenPredictor:
    """Predicts random nonsense and occasionally raises."""

    method = "Fake-Broken"
    paradigm = "Discriminative"
    backbone = "fake/backbone"

    def __init__(self) -> None:
        self.calls = 0

    def predict(self, text, aspect=None):
        self.calls += 1
        if self.calls % 3 == 0:
            raise RuntimeError("simulated model crash")
        # Wrong sentiment (always positive) so accuracy != 1.
        if aspect is not None:
            return (aspect, "positive", "raw_atsc")
        return ("lecturer", "positive", "raw_acsa")


class PredictorProtocolTests(unittest.TestCase):
    def test_perfect_predictor_satisfies_protocol(self):
        # Build with empty lookup just to instantiate.
        p = _PerfectVSFCPredictor({})
        self.assertIsInstance(p, Predictor)

    def test_class_without_required_attributes_fails(self):
        class Bad:
            def predict(self, text, aspect=None):
                return ("lecturer", "positive", "raw")

        self.assertNotIsInstance(Bad(), Predictor)

    def test_import_predictor_rejects_bad_spec(self):
        with self.assertRaises(ValueError):
            _import_predictor("module_only_no_colon", None)


class LoadTestSetTests(unittest.TestCase):
    DEMO = REPO_ROOT / "datasets" / "unified" / "uit_vsfc_test_demo.jsonl"

    def test_load_demo_vsfc(self):
        samples = load_test_set(self.DEMO)
        self.assertEqual(len(samples), 8)
        self.assertEqual(samples[0].dataset, "UIT-VSFC")
        self.assertEqual(samples[0].task, "ACSA")
        self.assertEqual(samples[0].language, "vi")

    def test_empty_file_raises(self):
        with tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False) as fh:
            empty = fh.name
        try:
            with self.assertRaises(ValueError):
                load_test_set(empty)
        finally:
            Path(empty).unlink(missing_ok=True)

    def test_mixed_dataset_raises(self):
        with tempfile.NamedTemporaryFile(
            "w", suffix=".jsonl", delete=False, encoding="utf-8"
        ) as fh:
            fh.write(json.dumps({
                "id": "1", "dataset": "UIT-VSFC", "language": "vi", "task": "ACSA",
                "text": "a", "gold": {"aspect": "lecturer", "sentiment": "positive"},
            }) + "\n")
            fh.write(json.dumps({
                "id": "2", "dataset": "SemEval-2014-Restaurant", "language": "en",
                "task": "ATSC", "text": "b",
                "gold": {"aspect": "food", "sentiment": "positive"},
            }) + "\n")
            tmp = fh.name
        try:
            with self.assertRaises(ValueError):
                load_test_set(tmp)
        finally:
            Path(tmp).unlink(missing_ok=True)


class PercentileTests(unittest.TestCase):
    def test_p95_on_short_list(self):
        self.assertAlmostEqual(_percentile([10, 20, 30, 40, 50], 95), 48.0)

    def test_empty(self):
        self.assertEqual(_percentile([], 95), 0.0)


class RunEvaluationVSFCTests(unittest.TestCase):
    DEMO = REPO_ROOT / "datasets" / "unified" / "uit_vsfc_test_demo.jsonl"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outdir = Path(self.tmp.name)
        self.gold = {
            s.text: (s.gold_aspect, s.gold_sentiment) for s in load_test_set(self.DEMO)
        }

    def test_perfect_predictor_yields_perfect_metrics(self):
        config = RunConfig(
            test_set=self.DEMO,
            output_dir=self.outdir,
            warmup=2,
            params_million=1.23,
            training_hours=0.0,
        )
        metrics = run_evaluation(
            _PerfectVSFCPredictor(self.gold), config, progress_every=0
        )

        self.assertEqual(metrics["dataset"], "UIT-VSFC")
        self.assertEqual(metrics["method"], "Fake-Perfect")
        self.assertEqual(metrics["n_samples"], 8)
        self.assertAlmostEqual(metrics["sentiment"]["accuracy"], 1.0)
        # SPEC v1.2: model predicts both aspect & sentiment; perfect
        # predictor → perfect aspect + joint accuracy.
        self.assertAlmostEqual(metrics["aspect"]["accuracy"], 1.0)
        self.assertAlmostEqual(metrics["aspect"]["macro_f1"], 1.0)
        self.assertAlmostEqual(metrics["joint_aspect_sentiment_acc"], 1.0)
        self.assertEqual(metrics["parse_failure_rate"], 0.0)

        # Efficiency block populated by the runner.
        eff = metrics["efficiency"]
        self.assertEqual(eff["params_million"], 1.23)
        self.assertEqual(eff["batch_size_inference"], 1)
        self.assertEqual(eff["hardware"], "1x T4 16GB")
        self.assertEqual(eff["precision"], "fp16")
        self.assertEqual(eff["warmup_samples_run"], 2)
        for k in ("avg_latency_ms", "median_latency_ms", "p95_latency_ms"):
            self.assertGreaterEqual(eff[k], 0.0)
        # Files written.
        pred_path = self.outdir / "predictions" / "fake_perfect_vsfc.jsonl"
        metric_path = self.outdir / "metrics" / "fake_perfect_vsfc.json"
        self.assertTrue(pred_path.exists())
        self.assertTrue(metric_path.exists())
        # Each line of the predictions file contains exactly the schema fields.
        first = json.loads(pred_path.read_text().splitlines()[0])
        self.assertIn("latency_ms", first)
        self.assertTrue(first["parse_ok"])
        self.assertEqual(first["pred"], first["gold"])

    def test_broken_predictor_marks_failures_as_wrong(self):
        config = RunConfig(
            test_set=self.DEMO, output_dir=self.outdir, warmup=0
        )
        metrics = run_evaluation(_BrokenPredictor(), config, progress_every=0)
        # Some predictions raised → counted as parse_ok=false → wrong.
        self.assertGreater(metrics["parse_failure_rate"], 0.0)
        # Sentiment accuracy is < 1 because the broken predictor also gives
        # wrong labels even when it doesn't crash.
        self.assertLess(metrics["sentiment"]["accuracy"], 1.0)

    def test_partial_parse_error_is_normalised(self):
        """A predictor that returns one PARSE_ERROR and one valid label must
        not crash the runner — both fields must be normalised to
        PARSE_ERROR before the record is written, otherwise the post-write
        schema validation rejects the file.
        """

        class _PartialErrorPredictor:
            method = "Fake-Partial"
            paradigm = "Discriminative"
            backbone = "fake/partial"

            def predict(self_inner, text, aspect=None):
                # Return aspect cleanly but PARSE_ERROR for sentiment — i.e.
                # the kind of partial failure a generative-model wrapper can
                # emit when it parses out the topic but not the sentiment.
                return ("lecturer", "__PARSE_ERROR__", "raw")

        metrics = run_evaluation(
            _PartialErrorPredictor(),
            RunConfig(test_set=self.DEMO, output_dir=self.outdir, warmup=0),
            progress_every=0,
        )
        # Every prediction is a partial PARSE_ERROR → all 8 must be counted
        # as parse failures.
        self.assertAlmostEqual(metrics["parse_failure_rate"], 1.0)
        self.assertAlmostEqual(metrics["sentiment"]["accuracy"], 0.0)

        # All written records must have BOTH pred fields normalised.
        pred_path = self.outdir / "predictions" / "fake_partial_vsfc.jsonl"
        for line in pred_path.read_text().splitlines():
            rec = json.loads(line)
            self.assertFalse(rec["parse_ok"])
            self.assertEqual(rec["pred"]["aspect"], "__PARSE_ERROR__")
            self.assertEqual(rec["pred"]["sentiment"], "__PARSE_ERROR__")


class RunEvaluationSemEvalTests(unittest.TestCase):
    DEMO = REPO_ROOT / "datasets" / "unified" / "semeval14_restaurant_test_demo.jsonl"

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.outdir = Path(self.tmp.name)

    def test_atsc_aspect_and_joint_metrics(self):
        # SPEC v1.2: the model must predict the aspect term too. The
        # runner does not provide ``aspect`` (it's always None), so
        # this oracle predictor looks up the gold pair by ``text``
        # alone — we just want to verify that the aspect block and
        # joint accuracy are computed for ATSC.
        gold_by_text: dict[str, tuple[str, str]] = {}
        for s in load_test_set(self.DEMO):
            # If the same text has multiple aspects, the last one wins
            # for this fake — metrics will be < 1.0 in that case.
            gold_by_text[s.text] = (s.gold_aspect, s.gold_sentiment)

        class OraclePredictor:
            method = "Fake-Oracle"
            paradigm = "Discriminative"
            backbone = "fake/oracle"

            def predict(self_inner, text, aspect=None):
                # SPEC v1.2: the runner does not pass ``aspect``.
                self.assertIsNone(aspect)
                a, s_ = gold_by_text[text]
                return (a, s_, "raw")

        metrics = run_evaluation(
            OraclePredictor(),
            RunConfig(test_set=self.DEMO, output_dir=self.outdir, warmup=0),
            progress_every=0,
        )
        self.assertEqual(metrics["task"], "ATSC")
        # ATSC aspect vocabulary is open → accuracy only, no macro_f1.
        self.assertIn("aspect", metrics)
        self.assertIn("accuracy", metrics["aspect"])
        self.assertNotIn("macro_f1", metrics["aspect"])
        self.assertIn("joint_aspect_sentiment_acc", metrics)
        self.assertGreaterEqual(metrics["joint_aspect_sentiment_acc"], 0.0)
        self.assertLessEqual(metrics["joint_aspect_sentiment_acc"], 1.0)


if __name__ == "__main__":
    unittest.main()
