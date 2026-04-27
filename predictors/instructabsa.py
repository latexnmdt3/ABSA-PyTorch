"""InstructABSA (Scaria et al., 2023) predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``allenai/tk-instruct-base-def-pos``
Backbone VI: ``google/mt5-base``

The model is instruction-tuned, so its raw output is free-form text. The
``predict`` method must parse it back into the ``(aspect, sentiment)`` pair
defined in SPEC §3. If parsing fails, return
``("__PARSE_ERROR__", "__PARSE_ERROR__", raw)`` — the runner counts this as
wrong.
"""
from __future__ import annotations

import re

from eval.schema import PARSE_ERROR_TOKEN

# Map common surface forms back to the canonical labels in the SPEC.
_SENTIMENT_ALIASES = {
    "positive": "positive", "pos": "positive", "tích cực": "positive",
    "negative": "negative", "neg": "negative", "tiêu cực": "negative",
    "neutral": "neutral", "neu": "neutral", "trung tính": "neutral",
}
_VSFC_ASPECT_ALIASES = {
    "lecturer": "lecturer", "giảng viên": "lecturer",
    "training_program": "training_program", "training program": "training_program",
    "chương trình": "training_program", "chương trình đào tạo": "training_program",
    "facility": "facility", "cơ sở vật chất": "facility",
    "others": "others", "khác": "others",
}


class InstructABSAPredictor:
    method = "InstructABSA"
    paradigm = "Instruction-Tuning"
    backbone = "allenai/tk-instruct-base-def-pos"

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "allenai/tk-instruct-base-def-pos",
        device: str = "cuda",
        max_new_tokens: int = 32,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.device = device
        self.max_new_tokens = max_new_tokens
        # TODO: load the seq2seq model & tokenizer (e.g. AutoModelForSeq2SeqLM).

    def _build_prompt(self, text: str, aspect: str | None) -> str:
        # TODO: replace with the exact instruction prompt format used during
        # fine-tuning. The SPEC does not constrain the prompt — only the
        # output parsing.
        if aspect is not None:
            return f"What is the sentiment toward '{aspect}' in: {text}"
        return f"Identify the topic and sentiment of: {text}"

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: run model.generate(...) and capture the decoded string.
        raw = "TODO: model output"  # noqa: F841

        return self._parse(raw, gold_aspect=aspect)

    @staticmethod
    def _parse(raw: str, gold_aspect: str | None) -> tuple[str, str, str]:
        text = raw.lower().strip()
        sentiment = next(
            (canon for alias, canon in _SENTIMENT_ALIASES.items() if alias in text),
            None,
        )
        if gold_aspect is not None:
            return (gold_aspect, sentiment or PARSE_ERROR_TOKEN, raw) if sentiment else (
                PARSE_ERROR_TOKEN,
                PARSE_ERROR_TOKEN,
                raw,
            )
        aspect = next(
            (canon for alias, canon in _VSFC_ASPECT_ALIASES.items() if alias in text),
            None,
        )
        if aspect and sentiment:
            return aspect, sentiment, raw
        return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
