"""InstructABSA (Scaria et al., 2023) predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``allenai/tk-instruct-base-def-pos``
Backbone VI: ``google/mt5-base``

The model is instruction-tuned, so its raw output is free-form text. As of
SPEC v1.2 the model must emit both an aspect and a sentiment — the runner
does not provide ``aspect`` as input. Parse the raw output into a
``(aspect, sentiment)`` pair; if parsing fails return
``("__PARSE_ERROR__", "__PARSE_ERROR__", raw)`` and the runner counts
this as wrong.
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
_VSFC_TOPIC_ALIASES = {
    "lecturer": "lecturer", "giảng viên": "lecturer",
    "training_program": "training_program", "training program": "training_program",
    "chương trình đào tạo": "training_program", "chương trình": "training_program",
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

    def _build_prompt(self, text: str) -> str:
        # TODO: replace with the exact instruction prompt format used during
        # fine-tuning. The SPEC does not constrain the prompt — only the
        # output parsing.
        return (
            "Identify the aspect/topic and its sentiment in the following "
            f"text. Output as 'aspect: sentiment'.\nText: {text}"
        )

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: run model.generate(...) and capture the decoded string.
        raw = "TODO: model output"  # noqa: F841
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> tuple[str, str, str]:
        # Try the structured ``aspect: sentiment`` format first.
        m = re.match(r"\s*([^:\n]+?)\s*:\s*([a-zA-ZÀ-ỹ_]+)\s*$", raw.strip())
        if m:
            aspect_raw = m.group(1).strip().lower()
            sentiment_raw = m.group(2).strip().lower()
            sentiment = _SENTIMENT_ALIASES.get(sentiment_raw)
            aspect = _VSFC_TOPIC_ALIASES.get(aspect_raw, aspect_raw)
            if sentiment and aspect:
                return aspect, sentiment, raw
        # Fallback: scan for any sentiment alias and any topic alias.
        text_lc = raw.lower()
        sentiment = next(
            (canon for alias, canon in _SENTIMENT_ALIASES.items() if alias in text_lc),
            None,
        )
        topic = next(
            (canon for alias, canon in _VSFC_TOPIC_ALIASES.items() if alias in text_lc),
            None,
        )
        if sentiment and topic:
            return topic, sentiment, raw
        return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
