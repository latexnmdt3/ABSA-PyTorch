"""LCF-BERT predictor wrapper.

Owner: Latex (this repo's main author).
Status: TODO — wire this up to the existing ``models/lcf_bert.py`` and
``infer_example.py`` once the actual checkpoint paths are decided.
"""
from __future__ import annotations


class LCFBertPredictor:
    method = "LCF-BERT"
    paradigm = "Discriminative"
    # Set to "bert-base-uncased" when running SemEval, "vinai/phobert-base"
    # when running UIT-VSFC. Decide via constructor argument so this single
    # class can serve both datasets.
    backbone = "vinai/phobert-base"

    def __init__(
        self,
        ckpt_path: str,
        backbone: str = "vinai/phobert-base",
        device: str = "cuda",
        max_seq_len: int = 80,
        local_context_focus: str = "cdm",
        srd: int = 3,
    ) -> None:
        self.backbone = backbone
        self.ckpt_path = ckpt_path
        self.device = device
        self.max_seq_len = max_seq_len
        self.local_context_focus = local_context_focus
        self.srd = srd
        # TODO(latex): load tokenizer + LCF_BERT model from checkpoint.
        #   from transformers import BertModel, BertTokenizer
        #   from models.lcf_bert import LCF_BERT
        #   ...

    def predict(
        self, text: str, aspect: str | None = None
    ) -> tuple[str, str, str]:
        """Discriminative model: aspect/topic is always given as input
        (SPEC v1.1) for both ATSC and ACSA. The model only predicts
        sentiment.
        """
        # TODO(latex): tokenize (text, aspect), run LCF-BERT, argmax over
        # the 3-class sentiment head, map int → string per SPEC §3, and
        # return (aspect, sentiment, raw). The runner overwrites the
        # aspect slot with the gold value, so echoing back ``aspect`` is
        # fine.
        raise NotImplementedError("Wire LCF-BERT inference here.")
