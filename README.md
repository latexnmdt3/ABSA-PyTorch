# ABSA-PyTorch (LCF-BERT only)

> Aspect Based Sentiment Analysis with LCF-BERT, PyTorch Implementation.
>
> 基于方面的情感分析，使用 PyTorch 实现 LCF-BERT 模型。

![LICENSE](https://img.shields.io/packagist/l/doctrine/orm.svg)

## Requirement

* pytorch >= 0.4.0
* numpy >= 1.13.3
* sklearn
* python 3.6 / 3.7
* transformers

To install requirements, run `pip install -r requirements.txt`.

## Usage

### Training

```sh
python train.py --model_name lcf_bert --dataset restaurant
```

* See [train.py](./train.py) for more training arguments.
* Refer to [train_k_fold_cross_val.py](./train_k_fold_cross_val.py) for k-fold cross validation support.

### Inference

* Refer to [infer_example.py](./infer_example.py) for an LCF-BERT inference example.

### Benchmark (time / memory / accuracy trade-off)

`benchmark.py` reports latency (median / p95 / mean / std), peak GPU memory,
throughput, accuracy and macro-F1 on the test set so you can reason about the
trade-off between speed and quality.

```sh
# Single config — uses an already-trained checkpoint to get meaningful metrics.
python benchmark.py \
    --dataset restaurant \
    --state_dict_path state_dict/lcf_bert_restaurant_val_acc_0.85 \
    --batch_size 16 --num_runs 200

# Grid sweep over focus mode / SRD / batch size, with CSV + scatter plot.
python benchmark.py --sweep \
    --dataset restaurant \
    --state_dict_path state_dict/lcf_bert_restaurant_val_acc_0.85 \
    --sweep_focus cdm cdw --sweep_srd 3 5 7 --sweep_batch_size 8 16 32 \
    --output_csv bench.csv --output_plot bench.png
```

Without `--state_dict_path` the model head is randomly initialised, so latency /
memory numbers are still valid but accuracy / F1 are not meaningful.

### Tips

* BERT-based models are sensitive to hyperparameters (especially learning rate) on small data sets.
* Fine-tuning on the specific task is necessary for releasing the true power of BERT.

### Framework
For flexible training/inference and aspect term extraction, try [PyABSA](https://github.com/yangheng95/PyABSA).

## Model

### LCF-BERT ([lcf_bert.py](./models/lcf_bert.py)) ([official](https://github.com/yangheng95/LCF-ABSA))

Zeng Biqing, Yang Heng, et al. "LCF: A Local Context Focus Mechanism for Aspect-Based Sentiment Classification." Applied Sciences. 2019, 9, 3389. [[pdf]](https://www.mdpi.com/2076-3417/9/16/3389/pdf)

## Note on running with RTX30*
If you are running on RTX30 series there may be some compatibility issues between installed/required versions of torch, cuda.
In that case try using `requirements_rtx30.txt` instead of `requirements.txt`.

## Licence

MIT
