"""InstructABSA (Scaria et al., 2023) predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``allenai/tk-instruct-base-def-pos``
Backbone VI: ``google/mt5-base``

The model is instruction-tuned, so its raw output is free-form text. As of
SPEC v1.1 the aspect/topic is given to the model as input, so ``predict``
only needs to parse the sentiment back from the raw output. If parsing
fails, return ``("__PARSE_ERROR__", "__PARSE_ERROR__", raw)`` — the
runner counts this as wrong.
"""
from __future__ import annotations

from eval.schema import PARSE_ERROR_TOKEN

# Map common surface forms back to the canonical labels in the SPEC.
_SENTIMENT_ALIASES = {
    "positive": "positive", "pos": "positive", "tích cực": "positive",
    "negative": "negative", "neg": "negative", "tiêu cực": "negative",
    "neutral": "neutral", "neu": "neutral", "trung tính": "neutral",
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

    def _build_prompt(self, text: str, aspect: str) -> str:
        # TODO: replace with the exact instruction prompt format used during
        # fine-tuning. The SPEC does not constrain the prompt — only the
        # output parsing.
        return f"What is the sentiment toward '{aspect}' in: {text}"

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: run model.generate(...) and capture the decoded string.
        raw = "TODO: model output"  # noqa: F841
        return self._parse(raw, given_aspect=aspect or "")

    @staticmethod
    def _parse(raw: str, given_aspect: str) -> tuple[str, str, str]:
        # SPEC v1.1: only sentiment needs to be parsed. The runner overrides
        # ``aspect`` with the gold value, so the first slot is just echoed.
        text = raw.lower().strip()
        sentiment = next(
            (canon for alias, canon in _SENTIMENT_ALIASES.items() if alias in text),
            None,
        )
        if sentiment is None:
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        return given_aspect, sentiment, raw
