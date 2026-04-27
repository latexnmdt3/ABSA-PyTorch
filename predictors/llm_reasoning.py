"""LLM-Reasoning (2025) — LLM + QLoRA + Chain-of-Thought predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``meta-llama/Llama-3-8B``
Backbone VI: ``SeaLLMs/SeaLLM-7B-v2``

The model is fine-tuned with QLoRA and prompted to emit a short reasoning
chain followed by a final ``Answer: <aspect>: <sentiment>`` line. We extract
the answer line and discard the reasoning. Per SPEC decision #4 the model
must be fine-tuned (no zero/few-shot rows in the comparison).
"""
from __future__ import annotations

import re

from eval.schema import (
    ALLOWED_SENTIMENTS,
    ALLOWED_VSFC_ASPECTS,
    PARSE_ERROR_TOKEN,
)


_ANSWER_RE = re.compile(
    r"answer\s*:\s*([^:\n]+?)\s*:\s*([a-zA-ZÀ-ỹ_]+)",
    flags=re.IGNORECASE,
)


class LLMReasoningPredictor:
    method = "LLM-Reasoning"
    paradigm = "LLM-Reasoning"
    backbone = "SeaLLMs/SeaLLM-7B-v2"

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "SeaLLMs/SeaLLM-7B-v2",
        device: str = "cuda",
        max_new_tokens: int = 256,
        temperature: float = 0.0,
    ) -> None:
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.device = device
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        # TODO: load the base model + apply QLoRA adapter from ckpt_path.

    def _build_prompt(self, text: str, aspect: str | None) -> str:
        # TODO: match the exact CoT prompt template used during fine-tuning.
        if aspect is not None:
            return (
                "Reason step-by-step about the sentiment expressed toward the "
                f"aspect '{aspect}' in this text, then answer.\n"
                f"Text: {text}\nAnswer:"
            )
        return (
            "Identify the single topic and sentiment of this Vietnamese "
            "feedback sentence. Reason step-by-step then output one line:\n"
            f"Text: {text}\nAnswer:"
        )

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: model.generate(...) → raw_output (full CoT + answer line).
        raw = "TODO: model output"
        return self._parse(raw, gold_aspect=aspect)

    @staticmethod
    def _parse(raw: str, gold_aspect: str | None) -> tuple[str, str, str]:
        m = _ANSWER_RE.search(raw)
        if not m:
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        aspect_str = m.group(1).strip().lower().replace(" ", "_")
        sentiment_str = m.group(2).strip().lower()
        if gold_aspect is not None:
            if sentiment_str not in ALLOWED_SENTIMENTS:
                return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
            return gold_aspect, sentiment_str, raw
        if (
            aspect_str not in ALLOWED_VSFC_ASPECTS
            or sentiment_str not in ALLOWED_SENTIMENTS
        ):
            return PARSE_ERROR_TOKEN, PARSE_ERROR_TOKEN, raw
        return aspect_str, sentiment_str, raw
