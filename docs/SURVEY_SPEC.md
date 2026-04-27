# ABSA Survey — Unified Output Specification

> Version: 1.2 (updated 2026-04-27)
>
> This document is the **single source of truth** for how every method in the
> survey reports predictions, metrics, and efficiency numbers. Every method
> implementation **must** emit files conforming to the schemas below so that a
> single evaluator can produce the comparison tables.
>
> Changelog:
> - **v1.2**: Both datasets now require the model to predict **both**
>   `aspect` and `sentiment`. SemEval becomes "aspect-term + sentiment"
>   prediction (open-vocabulary aspect strings); UIT-VSFC becomes full
>   ACSA (closed-set 4-way topic + 3-way sentiment). Joint (aspect,
>   sentiment) accuracy is reported for both datasets, alongside aspect
>   and sentiment metrics. Reverts the v1.1 simplification.
> - **v1.1** (superseded): UIT-VSFC was sentiment-only; topic was given
>   as input.
> - **v1.0**: Initial spec.

## 1. Methods and datasets in scope

| # | Method         | Year | Paradigm                | Backbone (EN)                              | Backbone (VI)             |
|---|----------------|------|-------------------------|--------------------------------------------|---------------------------|
| 1 | LCF-BERT       | 2019 | Discriminative          | `bert-base-uncased`                        | `vinai/phobert-base`      |
| 2 | InstructABSA   | 2023 | Instruction Tuning      | `allenai/tk-instruct-base-def-pos`         | `google/mt5-base`         |
| 3 | SSIN           | 2024 | Graph (Syn + Sem)       | `bert-base` + spaCy                        | `phobert-base` + VnCoreNLP|
| 4 | DOT            | 2025 | Generative Seq2Seq      | `t5-base`                                  | `VietAI/vit5-base`        |
| 5 | LLM-Reasoning  | 2025 | LLM + QLoRA + CoT       | `meta-llama/Llama-3-8B`                    | `SeaLLMs/SeaLLM-7B-v2`    |

Datasets:

- **SemEval-2014 (English)** — task **ATSC** (Aspect Term Sentiment
  Classification) on Restaurant + Laptop reviews. ~6,000 sentences after the
  `Conflict` label is removed. The model **predicts the aspect term as a
  free-form string** (open vocabulary) and one of 3 sentiment classes for
  that aspect.
- **UIT-VSFC (Vietnamese)** — task **ACSA single-label** (Aspect Category
  Sentiment Analysis) on student feedback. ~16,175 sentences. Each sentence
  has exactly one `(topic, sentiment)` pair. The model **predicts both**
  the topic (closed-vocabulary, 4 classes) and the sentiment.
  - topic vocabulary: `lecturer` (0), `training_program` (1),
    `facility` (2), `others` (3)
  - sentiments: `negative` (0), `neutral` (1), `positive` (2)

## 2. Locked decisions

| # | Item                              | Decision                                                                                  |
|---|-----------------------------------|-------------------------------------------------------------------------------------------|
| 1 | Headline metric                   | For each dataset, report **Aspect metric**, **Sentiment Accuracy + Macro-F1**, and **Joint (aspect, sentiment) Accuracy**. No combined ranking. |
| 2 | Generative parse failures         | `parse_ok=false` → predicted aspect/sentiment set to `__PARSE_ERROR__` and counted as **wrong**. No retries, no skips. |
| 3 | Hardware for efficiency reporting | **1× T4 16GB**, batch size **1**, **fp16**. (Closer to a real deployment than A100.)      |
| 4 | Setting for every method          | Every method **must fine-tune on the corresponding training set**. LLM-Reasoning is fine-tuned with QLoRA; **no zero-shot or few-shot rows** in the comparison tables. |

## 3. Common task contract

Both datasets are normalised to a single contract:

> A sample is `{text, [{aspect, sentiment}]}` where each sample contains
> exactly one `(aspect, sentiment)` pair.

| Dataset            | `aspect` field         | Aspect comes from   | Sentiment label space                                |
|--------------------|------------------------|---------------------|------------------------------------------------------|
| SemEval-2014 (EN)  | aspect term string     | **predicted by model** (open vocabulary; substring of `text`) | `positive` / `neutral` / `negative`        |
| UIT-VSFC (VI)      | one of 4 topic strings | **predicted by model** (closed vocabulary)                    | `positive` / `neutral` / `negative`        |

Sentiment normalisation:

- SemEval: drop the `Conflict` class entirely.
- UIT-VSFC: integer → string mapping `0→negative`, `1→neutral`, `2→positive`.

Aspect normalisation for UIT-VSFC: integer → string mapping
`0→lecturer`, `1→training_program`, `2→facility`, `3→others`.

## 4. Per-sample prediction schema

Each `(method, dataset)` run writes one JSONL file:

```
results/predictions/{method}_{dataset}.jsonl
```

with one record per line:

```json
{
  "id": "vsfc_test_00042",
  "dataset": "UIT-VSFC",
  "language": "vi",
  "task": "ACSA",
  "text": "Giảng viên dạy rất nhiệt tình.",
  "gold":   {"aspect": "lecturer", "sentiment": "positive"},
  "pred":   {"aspect": "lecturer", "sentiment": "positive"},
  "raw_output": "lecturer: positive",
  "parse_ok": true,
  "method": "DOT",
  "paradigm": "Generative-Seq2Seq",
  "backbone": "VietAI/vit5-base",
  "latency_ms": 38.7
}
```

Field rules:

- `id`: stable string id per sample (e.g. `semeval14_restaurant_test_00031`).
- `dataset`: one of `SemEval-2014-Restaurant`, `SemEval-2014-Laptop`, `UIT-VSFC`.
- `language`: `en` or `vi`.
- `task`: `ATSC` or `ACSA`.
- `gold.aspect`, `gold.sentiment`: ground truth, normalised as in §3.
- `pred.aspect`, `pred.sentiment`: model prediction, same vocabulary as gold.
- `raw_output`: the raw text emitted by the model (required for generative /
  LLM methods to allow audit of parse failures, optional for discriminative
  classifiers — they may store the predicted label).
- `parse_ok`: `true` if the raw output could be parsed into a valid
  `(aspect, sentiment)` pair. When `false`, both `pred.aspect` and
  `pred.sentiment` **must** be set to the literal string `__PARSE_ERROR__`.
- `latency_ms`: per-sample inference latency at batch size 1, measured on the
  hardware of §2.

## 5. Per-run metrics schema

Each run also writes one JSON file:

```
results/metrics/{method}_{dataset}.json
```

```json
{
  "method": "DOT",
  "dataset": "UIT-VSFC",
  "n_samples": 3166,
  "sentiment": {
    "accuracy": 0.842,
    "macro_f1": 0.811,
    "f1_per_class": {"positive": 0.88, "neutral": 0.71, "negative": 0.84}
  },
  "aspect": {
    "accuracy": 0.913,
    "macro_f1": 0.892
  },
  "joint_aspect_sentiment_acc": 0.795,
  "parse_failure_rate": 0.012,
  "efficiency": {
    "params_million": 247,
    "gpu_mem_peak_gb": 6.4,
    "avg_latency_ms": 38.7,
    "throughput_qps": 25.8,
    "training_hours": 2.1,
    "hardware": "1x T4 16GB",
    "precision": "fp16",
    "batch_size_inference": 1
  },
  "config": {
    "seed": 42,
    "epochs": 5,
    "lr": 2e-5,
    "batch_size_train": 16,
    "setting": "fine-tuned"
  }
}
```

Notes:

- The `aspect` block always contains `accuracy` (exact-string match
  against `gold.aspect`). For UIT-VSFC (closed 4-way vocabulary) it also
  contains `macro_f1` and `f1_per_class`. For SemEval ATSC the aspect
  vocabulary is open, so only `accuracy` is reported.
- `joint_aspect_sentiment_acc` is the fraction of samples where **both**
  the predicted aspect and the predicted sentiment match the gold pair.
- `parse_failure_rate` is `(# samples with parse_ok=false) / n_samples`.
- `efficiency.*` is filled in by the runner (not the evaluator) and merged
  into the metrics JSON. `eval/score.py --efficiency path/to/eff.json` is the
  supported merge mechanism.
- `config.setting` must be `fine-tuned` for every entry in the survey tables.

## 6. The single evaluator and runner

There are **two scripts**, depending on whether you produce predictions
inside or outside the runner.

### 6.1 `evaluate.py` (recommended) — runner with built-in timing

`evaluate.py` at the repo root is what every teammate runs. It:

1. Loads the unified test JSONL (§4).
2. Wraps your model behind the :class:`Predictor` protocol.
3. Times each `predict()` call (mean / median / p95 / std / throughput).
4. Tracks peak GPU memory (`torch.cuda.max_memory_allocated`) if CUDA is
   available.
5. Validates the resulting predictions file against `eval/schema.py`.
6. Computes the metrics defined in §5 via `eval/score.py`.
7. Writes both `results/predictions/{method}_{dataset}.jsonl` and
   `results/metrics/{method}_{dataset}.json` — already merged with the
   `efficiency` block.

Each teammate only writes a small wrapper class, e.g. `predictors/dot.py`:

```python
class DOTPredictor:
    method   = "DOT"
    paradigm = "Generative-Seq2Seq"
    backbone = "VietAI/vit5-base"

    def __init__(self, ckpt_path):
        ...  # load model

    def predict(self, text, aspect=None) -> tuple[str, str, str]:
        # The method must predict BOTH the aspect/topic and the sentiment
        # (SPEC v1.2). The runner does not provide ``aspect`` — it is
        # passed only as a hint when available (currently always None).
        # Return (pred_aspect, pred_sentiment, raw_output).
        ...
```

Then run:

```bash
python evaluate.py \
    --predictor   predictors.dot:DOTPredictor \
    --predictor-kwargs '{"ckpt_path": "state_dict/dot_vsfc.pt"}' \
    --test-set    datasets/unified/uit_vsfc_test.jsonl \
    --output-dir  results/ \
    --warmup 5 \
    --params-million 247 \
    --training-hours 2.1
```

Skeleton wrappers for all 5 methods live in `predictors/` — they have
`TODO` markers where the model loading and inference go.

### 6.2 `eval/score.py` (post-hoc) — score an existing predictions file

If you have already produced a predictions JSONL by some other means, you
can score it directly:

```bash
python -m eval.score \
    --predictions results/predictions/dot_vsfc.jsonl \
    --output      results/metrics/dot_vsfc.json \
    --efficiency  results/efficiency/dot_vsfc.json
```

`eval/score.py` reads predictions, computes sentiment accuracy /
macro-F1 / per-class-F1, aspect accuracy (plus macro-F1 for UIT-VSFC),
joint (aspect, sentiment) accuracy, and parse-failure-rate; optionally
merges in an efficiency block; and writes the final metrics file.
Method authors **must not** compute these numbers themselves.

`eval/schema.py` provides `validate_predictions_file(path)` and is run by
both `score.py` and `evaluate.py` before scoring. Any schema violation is
a hard error.

## 7. Final comparison tables

Generated by `eval/aggregate_tables.py` (TBD) from the JSON metrics files.

### Table A — Accuracy

For each dataset, three columns: aspect accuracy (Asp-Acc), sentiment
macro-F1 (Sent-F1), and joint (aspect, sentiment) accuracy (Joint-Acc).

| # | Method         | Year | Paradigm            | SemEval-14 Asp-Acc | SemEval-14 Sent-F1 | SemEval-14 Joint-Acc | UIT-VSFC Asp-Acc | UIT-VSFC Sent-F1 | UIT-VSFC Joint-Acc |
|---|----------------|------|---------------------|--------------------|--------------------|----------------------|------------------|------------------|--------------------|
| 1 | LCF-BERT       | 2019 | Discriminative      | …                  | …                  | …                    | …                | …                | …                  |
| 2 | InstructABSA   | 2023 | Instruction Tuning  | …                  | …                  | …                    | …                | …                | …                  |
| 3 | SSIN           | 2024 | Graph (Syn + Sem)   | …                  | …                  | …                    | …                | …                | …                  |
| 4 | DOT            | 2025 | Generative Seq2Seq  | …                  | …                  | …                    | …                | …                | …                  |
| 5 | LLM-Reasoning  | 2025 | LLM + QLoRA + CoT   | …                  | …                  | …                    | …                | …                | …                  |

### Table B — Efficiency (1× T4 16GB, fp16, batch=1)

| # | Method         | Params (M) | GPU mem peak (GB) | Latency (ms / sample) | Train time (h) | Parse-fail % |
|---|----------------|------------|-------------------|-----------------------|----------------|--------------|
| 1 | LCF-BERT       | …          | …                 | …                     | …              | 0.0          |
| 2 | InstructABSA   | …          | …                 | …                     | …              | …            |
| 3 | SSIN           | …          | …                 | …                     | …              | 0.0          |
| 4 | DOT            | …          | …                 | …                     | …              | …            |
| 5 | LLM-Reasoning  | …          | …                 | …                     | …              | …            |

### Figures

For each dataset, an accuracy-vs-latency scatter plot
(`results/figures/tradeoff_{dataset}.png`) is rendered from the same metrics
files.

## 8. Repository layout

```
results/
├── predictions/   # 10 files: {method}_{dataset}.jsonl
├── metrics/       # 10 files: {method}_{dataset}.json
├── efficiency/    # 10 files: {method}_{dataset}.json (raw efficiency only)
├── tables/        # accuracy.md, efficiency.md
└── figures/       # tradeoff_{dataset}.png

evaluate.py        # standalone runner (recommended entry-point)

eval/
├── schema.py      # validation of prediction records
├── score.py       # the single evaluator (post-hoc scoring)
└── tests/         # smoke tests (stdlib unittest)

predictors/        # one wrapper per method (TODO skeletons today)
├── lcf_bert.py
├── instructabsa.py
├── ssin.py
├── dot.py
└── llm_reasoning.py

datasets/unified/  # team-shared test JSONL files (the same file across 5 runs)
```

Files under `results/` are not committed; only the schemas, evaluator and
final aggregated tables/figures are.
