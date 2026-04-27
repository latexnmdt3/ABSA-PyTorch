"""DOT (2025) — Generative Seq2Seq predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``t5-base``
Backbone VI: ``VietAI/vit5-base``

DOT casts ABSA as a sequence-to-sequence problem; the target string contains
both aspect and sentiment in a fixed format like ``"<aspect>: <sentiment>"``.
As of SPEC v1.2 the model must emit both fields — the runner does not
provide ``aspect`` as input. Sentiment must be one of the 3 canonical
labels; aspect is an open string for SemEval but must be one of the 4
canonical topics for UIT-VSFC.
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

    def _build_input(self, text: str) -> str:
        # TODO: match the exact prompt format used at training time.
        return f"absa | text: {text}"

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: encode prompt, generate, decode → raw_output.
        raw = "TODO: model output"
        return self._parse(raw)

    @staticmethod
    def _parse(raw: str) -> tuple[str, str, str]:
        # Expected format: ``<aspect>: <sentiment>``.
        m = re.match(r"\s*([^:]+?)\s*:\s*([a-zA-ZÀ-ỹ_]+)\s*$", raw)
        if not m:
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        aspect_str = m.group(1).strip().lower().replace(" ", "_")
        sentiment_str = m.group(2).strip().lower()
        if sentiment_str not in ALLOWED_SENTIMENTS or aspect_str == "":
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        return aspect_str, sentiment_str, raw
