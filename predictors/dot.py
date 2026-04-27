"""DOT (2025) — Generative Seq2Seq predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``t5-base``
Backbone VI: ``VietAI/vit5-base``

DOT casts ABSA as a sequence-to-sequence problem; the target string contains
both aspect and sentiment in a fixed format like ``"<aspect>: <sentiment>"``.
As of SPEC v1.1 the aspect is given as input, so this wrapper only needs to
parse the sentiment back to the canonical labels in SPEC §3.
"""
from __future__ import annotations

import re

from eval.schema import ALLOWED_SENTIMENTS, PARSE_ERROR_TOKEN


class DOTPredictor:
    method = "DOT"
    paradigm = "Generative-Seq2Seq"
    backbone = "VietAI/vit5-base"

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "VietAI/vit5-base",
        device: str = "cuda",
        max_new_tokens: int = 24,
        num_beams: int = 4,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams
        # TODO: load the seq2seq model & tokenizer (T5/ViT5).

    def _build_input(self, text: str, aspect: str) -> str:
        # TODO: match the exact prompt format used at training time.
        return f"absa | aspect: {aspect} | text: {text}"

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: encode prompt, generate, decode → raw_output.
        raw = "TODO: model output"
        return self._parse(raw, given_aspect=aspect or "")

    @staticmethod
    def _parse(raw: str, given_aspect: str) -> tuple[str, str, str]:
        # SPEC v1.1: only the sentiment after the colon needs to validate
        # against the canonical vocabulary; the aspect slot is echoed back
        # and overwritten by the runner.
        m = re.match(r"\s*([^:]+?)\s*:\s*([a-zA-ZÀ-ỹ_]+)\s*$", raw)
        if not m:
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        sentiment_str = m.group(2).strip().lower()
        if sentiment_str not in ALLOWED_SENTIMENTS:
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        return given_aspect, sentiment_str, raw
