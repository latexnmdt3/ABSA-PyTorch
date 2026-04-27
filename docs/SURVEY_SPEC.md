# ABSA Survey — Unified Output Specification

> Version: 1.0 (locked 2026-04-27)
>
> This document is the **single source of truth** for how every method in the
> survey reports predictions, metrics, and efficiency numbers. Every method
> implementation **must** emit files conforming to the schemas below so that a
> single evaluator can produce the comparison tables.

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
  `Conflict` label is removed. 3 sentiment classes.
- **UIT-VSFC (Vietnamese)** — task **ACSA single-label** (Aspect Category
  Sentiment Analysis) on student feedback. ~16,175 sentences. Each sentence
  has exactly one `(topic, sentiment)` pair.
  - topics: `lecturer` (0), `training_program` (1), `facility` (2), `others` (3)
  - sentiments: `negative` (0), `neutral` (1), `positive` (2)

## 2. Locked decisions

| # | Item                              | Decision                                                                                  |
|---|-----------------------------------|-------------------------------------------------------------------------------------------|
| 1 | Headline metric on UIT-VSFC       | Report **Joint (topic, sentiment) Accuracy** *and* **Sentiment Macro-F1** side by side. No combined ranking. |
| 2 | Generative parse failures         | `parse_ok=false` → predicted aspect/sentiment set to `__PARSE_ERROR__` and counted as **wrong**. No retries, no skips. |
| 3 | Hardware for efficiency reporting | **1× T4 16GB**, batch size **1**, **fp16**. (Closer to a real deployment than A100.)      |
| 4 | Setting for every method          | Every method **must fine-tune on the corresponding training set**. LLM-Reasoning is fine-tuned with QLoRA; **no zero-shot or few-shot rows** in the comparison tables. |

## 3. Common task contract

Both datasets are normalised to a single contract:

> A sample is `{text, [{aspect, sentiment}]}` where each sample contains
> exactly one `(aspect, sentiment)` pair.

| Dataset            | `aspect` field         | Aspect comes from   | Sentiment label space                                |
|--------------------|------------------------|---------------------|------------------------------------------------------|
| SemEval-2014 (EN)  | aspect term string     | **given in input**, copied verbatim into the output | `positive` / `neutral` / `negative`               |
| UIT-VSFC (VI)      | one of 4 topic strings | **predicted by model** | `positive` / `neutral` / `negative`               |

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
  "aspect":   {"accuracy": 0.913, "macro_f1": 0.892},
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

- The `aspect` block and `joint_aspect_sentiment_acc` field are present **only
  for ACSA datasets** (UIT-VSFC). They are omitted for ATSC datasets where the
  aspect is given.
- `parse_failure_rate` is `(# samples with parse_ok=false) / n_samples`.
- `efficiency.*` is filled in by the runner (not the evaluator) and merged
  into the metrics JSON. `eval/score.py --efficiency path/to/eff.json` is the
  supported merge mechanism.
- `config.setting` must be `fine-tuned` for every entry in the survey tables.

## 6. The single evaluator

To prevent each method computing its own metrics with a slightly different
formula, **only one evaluator is allowed**:

```
python eval/score.py \
    --predictions results/predictions/dot_vsfc.jsonl \
    --output      results/metrics/dot_vsfc.json \
    --efficiency  results/efficiency/dot_vsfc.json
```

`eval/score.py` reads predictions, computes accuracy / macro-F1 /
per-class-F1 / parse-failure-rate / (for ACSA) aspect metrics and joint
accuracy, optionally merges in an efficiency block, and writes the final
metrics file. Method authors **must not** compute these numbers themselves.

`eval/schema.py` provides `validate_predictions_file(path)` and is run by
`score.py` before scoring. Any schema violation is a hard error.

## 7. Final comparison tables

Generated by `eval/aggregate_tables.py` (TBD) from the JSON metrics files.

### Table A — Accuracy

| # | Method         | Year | Paradigm            | SemEval-14 Acc | SemEval-14 Macro-F1 | UIT-VSFC Joint-Acc | UIT-VSFC Sent-Macro-F1 |
|---|----------------|------|---------------------|----------------|---------------------|--------------------|------------------------|
| 1 | LCF-BERT       | 2019 | Discriminative      | …              | …                   | …                  | …                      |
| 2 | InstructABSA   | 2023 | Instruction Tuning  | …              | …                   | …                  | …                      |
| 3 | SSIN           | 2024 | Graph (Syn + Sem)   | …              | …                   | …                  | …                      |
| 4 | DOT            | 2025 | Generative Seq2Seq  | …              | …                   | …                  | …                      |
| 5 | LLM-Reasoning  | 2025 | LLM + QLoRA + CoT   | …              | …                   | …                  | …                      |

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

eval/
├── schema.py      # validation of prediction records
├── score.py       # the single evaluator
└── tests/         # smoke tests (stdlib unittest)
```

Files under `results/` are not committed; only the schemas, evaluator and
final aggregated tables/figures are.
