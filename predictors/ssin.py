"""SSIN (2024) — Syntactic + Semantic Interaction Network predictor wrapper.

Owner: TODO assign teammate.
Backbone EN: ``bert-base-uncased`` + spaCy dependency parser
Backbone VI: ``vinai/phobert-base`` + VnCoreNLP dependency parser

This is a graph-based discriminative model: a syntactic dependency graph and
a semantic-similarity graph are fused over a BERT encoder. Since it's
discriminative, parse failures should not occur — but the wrapper still
follows the same protocol so output handling is consistent with the
generative methods.
"""
from __future__ import annotations


class SSINPredictor:
    method = "SSIN"
    paradigm = "Graph"
    backbone = "vinai/phobert-base"

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "vinai/phobert-base",
        parser: str = "vncorenlp",  # "spacy" for English
        device: str = "cuda",
    ) -> None:
        self.ckpt_path = ckpt_path
        self.backbone = backbone
        self.parser = parser
        self.device = device
        # TODO: load BERT/PhoBERT encoder, dep parser, and the SSIN head.

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        # TODO: build the syntactic + semantic graphs, run the model, take
        # argmax over the sentiment head. As of SPEC v1.1 the aspect/topic
        # is given as input for both ATSC and ACSA, so no topic head is
        # needed — return (aspect, sentiment, raw).
        raise NotImplementedError("Wire SSIN inference here.")
