"""Single evaluator for the ABSA survey.

Reads a JSONL prediction file produced by any method, validates it against
the schema in :mod:`eval.schema`, computes the canonical metrics
(sentiment accuracy / macro-F1 / per-class F1, aspect accuracy — plus
macro-F1 for UIT-VSFC's closed topic vocabulary, joint
``(aspect, sentiment)`` accuracy, and parse-failure rate), optionally
merges an external ``efficiency.json`` block, and writes the resulting
metrics JSON to disk.

SPEC v1.2: aspect/topic is predicted by the model for both datasets.
SemEval aspect strings are open-vocabulary so only exact-match accuracy
is reported there; UIT-VSFC topics are closed-vocabulary so we also
report macro-F1.

Use this script (and only this script) to score every method's predictions —
see ``docs/SURVEY_SPEC.md`` §6.

Example::

    python eval/score.py \\
        --predictions results/predictions/dot_vsfc.jsonl \\
        --output      results/metrics/dot_vsfc.json \\
        --efficiency  results/efficiency/dot_vsfc.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Sequence

from eval.schema import (
    ALLOWED_VSFC_ASPECTS,
    PARSE_ERROR_TOKEN,
    iter_records,
    validate_predictions_file,
)

SENTIMENT_LABELS: tuple[str, ...] = ("positive", "neutral", "negative")
VSFC_ASPECT_LABELS: tuple[str, ...] = (
    "lecturer",
    "training_program",
    "facility",
    "others",
)


def _accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def _f1_per_class(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> dict[str, float]:
    """Per-class F1, computed without sklearn to keep this module dependency-free."""
    f1: dict[str, float] = {}
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        denom = precision + recall
        f1[label] = (2 * precision * recall / denom) if denom else 0.0
    return f1


def _macro_f1(
    y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]
) -> float:
    per_class = _f1_per_class(y_true, y_pred, labels)
    return sum(per_class.values()) / len(labels)


def score_predictions(predictions_path: str | Path) -> dict[str, Any]:
    """Compute the canonical metrics dict for one prediction file.

    Raises ``ValueError`` if the file fails schema validation or is empty.
    The returned dict matches the schema described in ``SURVEY_SPEC.md`` §5
    *minus* the ``efficiency`` block (which is merged in separately).
    """
    errors = validate_predictions_file(predictions_path)
    if errors:
        raise ValueError(
            "Schema validation failed for "
            f"{predictions_path}:\n  " + "\n  ".join(errors)
        )

    samples = [rec for _, rec in iter_records(predictions_path)]
    if not samples:
        raise ValueError(f"No prediction records found in {predictions_path}")

    # All samples in a file must share method/dataset/task — sanity check.
    method = samples[0]["method"]
    dataset = samples[0]["dataset"]
    task = samples[0]["task"]
    for rec in samples:
        if rec["method"] != method or rec["dataset"] != dataset or rec["task"] != task:
            raise ValueError(
                "Mixed method/dataset/task within a single predictions file is not allowed"
            )

    n = len(samples)
    parse_fail = sum(1 for s in samples if not s.get("parse_ok", True))

    sent_true = [s["gold"]["sentiment"] for s in samples]
    sent_pred = [s["pred"]["sentiment"] for s in samples]

    metrics: dict[str, Any] = {
        "method": method,
        "dataset": dataset,
        "task": task,
        "n_samples": n,
        "sentiment": {
            "accuracy": _accuracy(sent_true, sent_pred),
            "macro_f1": _macro_f1(sent_true, sent_pred, SENTIMENT_LABELS),
            "f1_per_class": _f1_per_class(sent_true, sent_pred, SENTIMENT_LABELS),
        },
        "parse_failure_rate": parse_fail / n,
    }

    # SPEC v1.2: aspect / topic is predicted by the model on both datasets.
    # For UIT-VSFC the topic is closed-vocab → accuracy + macro-F1.
    # For SemEval the aspect term is open-vocab → accuracy only.
    asp_true = [s["gold"]["aspect"] for s in samples]
    asp_pred = [s["pred"]["aspect"] for s in samples]

    aspect_block: dict[str, Any] = {"accuracy": _accuracy(asp_true, asp_pred)}

    if task == "ACSA":
        bad_gold = {a for a in asp_true if a not in ALLOWED_VSFC_ASPECTS} - {
            PARSE_ERROR_TOKEN
        }
        if bad_gold:
            raise ValueError(
                f"gold aspect labels outside the UIT-VSFC vocabulary: {sorted(bad_gold)}"
            )
        aspect_block["macro_f1"] = _macro_f1(
            asp_true, asp_pred, VSFC_ASPECT_LABELS
        )
        aspect_block["f1_per_class"] = _f1_per_class(
            asp_true, asp_pred, VSFC_ASPECT_LABELS
        )

    metrics["aspect"] = aspect_block
    metrics["joint_aspect_sentiment_acc"] = sum(
        1
        for s in samples
        if s["gold"]["aspect"] == s["pred"]["aspect"]
        and s["gold"]["sentiment"] == s["pred"]["sentiment"]
    ) / n

    return metrics


def merge_efficiency(metrics: dict[str, Any], efficiency_path: str | Path) -> None:
    """Merge an efficiency block (in-place) onto a metrics dict."""
    eff = json.loads(Path(efficiency_path).read_text(encoding="utf-8"))
    metrics["efficiency"] = eff


def write_metrics(metrics: dict[str, Any], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Score an ABSA survey prediction file (JSONL) against the unified spec."
        )
    )
    parser.add_argument(
        "--predictions",
        required=True,
        help="Path to the JSONL prediction file produced by a method.",
    )
    parser.add_argument(
        "--output",
        help="Where to write the metrics JSON. If omitted, prints to stdout only.",
    )
    parser.add_argument(
        "--efficiency",
        help=(
            "Optional JSON file with the efficiency block (params/mem/latency/...) "
            "to merge into the final metrics."
        ),
    )
    args = parser.parse_args(argv)

    try:
        metrics = score_predictions(args.predictions)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 2

    if args.efficiency:
        merge_efficiency(metrics, args.efficiency)

    serialised = json.dumps(metrics, ensure_ascii=False, indent=2)
    print(serialised)

    if args.output:
        write_metrics(metrics, args.output)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
