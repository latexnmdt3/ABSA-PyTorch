"""Schema validation for ABSA survey prediction records.

The full specification lives in ``docs/SURVEY_SPEC.md``. This module is the
authoritative validator: every method's prediction file must pass
``validate_predictions_file`` before metrics are computed.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

PARSE_ERROR_TOKEN = "__PARSE_ERROR__"

ALLOWED_DATASETS = {
    "SemEval-2014-Restaurant",
    "SemEval-2014-Laptop",
    "UIT-VSFC",
}
ALLOWED_TASKS = {"ATSC", "ACSA"}
ALLOWED_LANGUAGES = {"en", "vi"}
ALLOWED_SENTIMENTS = {"positive", "neutral", "negative", PARSE_ERROR_TOKEN}
ALLOWED_VSFC_ASPECTS = {
    "lecturer",
    "training_program",
    "facility",
    "others",
    PARSE_ERROR_TOKEN,
}

REQUIRED_TOP_FIELDS = (
    "id",
    "dataset",
    "language",
    "task",
    "text",
    "gold",
    "pred",
    "method",
    "paradigm",
    "backbone",
    "latency_ms",
    "parse_ok",
)
REQUIRED_PAIR_FIELDS = ("aspect", "sentiment")


def validate_prediction(rec: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors for a single record.

    An empty list means the record is valid.
    """
    errors: list[str] = []

    for field in REQUIRED_TOP_FIELDS:
        if field not in rec:
            errors.append(f"missing field: {field}")

    dataset = rec.get("dataset")
    task = rec.get("task")
    language = rec.get("language")

    if dataset is not None and dataset not in ALLOWED_DATASETS:
        errors.append(
            f"invalid dataset: {dataset!r} (allowed: {sorted(ALLOWED_DATASETS)})"
        )
    if task is not None and task not in ALLOWED_TASKS:
        errors.append(f"invalid task: {task!r} (allowed: {sorted(ALLOWED_TASKS)})")
    if language is not None and language not in ALLOWED_LANGUAGES:
        errors.append(
            f"invalid language: {language!r} (allowed: {sorted(ALLOWED_LANGUAGES)})"
        )

    # Cross-check: ATSC -> SemEval, ACSA -> UIT-VSFC.
    if task == "ATSC" and dataset is not None and not str(dataset).startswith(
        "SemEval"
    ):
        errors.append(f"task=ATSC requires SemEval-* dataset, got {dataset!r}")
    if task == "ACSA" and dataset is not None and dataset != "UIT-VSFC":
        errors.append(f"task=ACSA requires UIT-VSFC dataset, got {dataset!r}")

    for side in ("gold", "pred"):
        pair = rec.get(side)
        if not isinstance(pair, dict):
            errors.append(f"{side}: must be an object with aspect+sentiment")
            continue
        for f in REQUIRED_PAIR_FIELDS:
            if f not in pair:
                errors.append(f"{side} missing field: {f}")

        # The PARSE_ERROR_TOKEN is only ever a valid value on the *pred* side
        # (when parse_ok=false). Gold labels must come from the normalised
        # label space; otherwise corrupted gold data would silently distort
        # metric computation. See SURVEY_SPEC.md §4.
        allowed_sentiments = (
            ALLOWED_SENTIMENTS if side == "pred"
            else ALLOWED_SENTIMENTS - {PARSE_ERROR_TOKEN}
        )
        allowed_aspects = (
            ALLOWED_VSFC_ASPECTS if side == "pred"
            else ALLOWED_VSFC_ASPECTS - {PARSE_ERROR_TOKEN}
        )

        sentiment = pair.get("sentiment")
        if sentiment is not None and sentiment not in allowed_sentiments:
            errors.append(
                f"{side}.sentiment invalid: {sentiment!r} "
                f"(allowed: {sorted(allowed_sentiments)})"
            )
        # Aspect-vocabulary constraint for UIT-VSFC.
        if dataset == "UIT-VSFC":
            aspect = pair.get("aspect")
            if aspect is not None and aspect not in allowed_aspects:
                errors.append(
                    f"{side}.aspect invalid for UIT-VSFC: {aspect!r} "
                    f"(allowed: {sorted(allowed_aspects)})"
                )

    parse_ok = rec.get("parse_ok")
    pred = rec.get("pred")
    if parse_ok is False and isinstance(pred, dict):
        if pred.get("aspect") != PARSE_ERROR_TOKEN:
            errors.append(
                "parse_ok=false but pred.aspect must be '__PARSE_ERROR__'"
            )
        if pred.get("sentiment") != PARSE_ERROR_TOKEN:
            errors.append(
                "parse_ok=false but pred.sentiment must be '__PARSE_ERROR__'"
            )
    elif parse_ok is True and isinstance(pred, dict):
        if (
            pred.get("aspect") == PARSE_ERROR_TOKEN
            or pred.get("sentiment") == PARSE_ERROR_TOKEN
        ):
            errors.append(
                "parse_ok=true but pred contains '__PARSE_ERROR__' token"
            )

    latency = rec.get("latency_ms")
    if latency is not None and not isinstance(latency, (int, float)):
        errors.append(f"latency_ms must be a number, got {type(latency).__name__}")
    if isinstance(latency, (int, float)) and latency < 0:
        errors.append(f"latency_ms must be >= 0, got {latency}")

    return errors


def iter_records(path: str | Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield ``(line_number, record)`` for every non-empty line in a JSONL file."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            yield lineno, json.loads(line)


def validate_predictions_file(path: str | Path) -> list[str]:
    """Validate every record in a JSONL prediction file.

    Returns a list of error strings (empty if the file is valid). JSON parse
    errors and schema errors are both reported, prefixed with ``line N:``.
    """
    path = Path(path)
    if not path.exists():
        return [f"file not found: {path}"]

    all_errors: list[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError as exc:
                all_errors.append(f"line {lineno}: invalid JSON: {exc}")
                continue
            for err in validate_prediction(rec):
                all_errors.append(f"line {lineno}: {err}")
    return all_errors
