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
        """Discriminative model: must predict BOTH aspect/topic and
        sentiment (SPEC v1.2).

        - For SemEval (ATSC): predict the aspect term as a free-form
          string (open vocabulary; ideally a substring of ``text``).
        - For UIT-VSFC (ACSA): predict one of the 4 closed topic
          categories ``lecturer`` / ``training_program`` / ``facility`` /
          ``others``.

        The original LCF-BERT only has a sentiment head, so wiring this
        into v1.2 requires either (a) adding a topic-classification head
        on top of the same encoder for UIT-VSFC, or (b) running a small
        upstream aspect-extraction module for SemEval. Pick whichever
        matches the experimental setup; this skeleton just documents
        the contract.
        """
        # TODO(latex): tokenize the text, run LCF-BERT (with the extra
        # aspect/topic head) → predicted aspect string + sentiment int,
        # map int → string per SPEC §3, and return
        # (pred_aspect, pred_sentiment, raw).
        raise NotImplementedError("Wire LCF-BERT inference here.")
