# -*- coding: utf-8 -*-
# file: benchmark.py
# Benchmark LCF-BERT: latency + memory + accuracy/macro-F1 trade-off.

import argparse
import csv
import json
import logging
import time

import numpy as np
import torch
from sklearn import metrics
from torch.utils.data import DataLoader
from transformers import BertModel

from data_utils import Tokenizer4Bert, ABSADataset
from models import LCF_BERT


LCF_BERT_INPUT_COLS = [
    'concat_bert_indices',
    'concat_segments_indices',
    'text_bert_indices',
    'aspect_bert_indices',
]

DATASET_FILES = {
    'twitter': {
        'train': './datasets/acl-14-short-data/train.raw',
        'test': './datasets/acl-14-short-data/test.raw',
    },
    'restaurant': {
        'train': './datasets/semeval14/Restaurants_Train.xml.seg',
        'test': './datasets/semeval14/Restaurants_Test_Gold.xml.seg',
    },
    'laptop': {
        'train': './datasets/semeval14/Laptops_Train.xml.seg',
        'test': './datasets/semeval14/Laptops_Test_Gold.xml.seg',
    },
}


logger = logging.getLogger(__name__)


def _move_inputs(batch, device, keys):
    return [batch[k].to(device) for k in keys]


def benchmark(model, dataloader, device, *,
              input_cols=LCF_BERT_INPUT_COLS,
              label_key='polarity',
              warmup=20,
              num_runs=200,
              num_classes=3):
    """Benchmark a model for latency, peak memory and quality.

    Parameters
    ----------
    model : nn.Module
        Model whose ``forward`` accepts a single positional list of tensors,
        as is the case for ``LCF_BERT``.
    dataloader : torch.utils.data.DataLoader
        Iterable over batches. Each batch is a ``dict`` containing the keys
        listed in ``input_cols`` plus ``label_key``.
    device : torch.device
        CUDA or CPU device.
    input_cols : list[str]
        Order of dict keys to feed to the model.
    label_key : str
        Key holding integer class labels in each batch.
    warmup : int
        Number of warm-up batches before measurement (CUDA kernel cache).
    num_runs : int
        Number of measured batches.
    num_classes : int
        Number of polarity classes.

    Returns
    -------
    dict
        Aggregated latency / memory / quality metrics.
    """
    model.eval().to(device)
    is_cuda = device.type == 'cuda'

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

    # 1) WARM-UP — CUDA kernels are JIT-compiled / cached on first launch.
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= warmup:
                break
            inputs = _move_inputs(batch, device, input_cols)
            _ = model(inputs)
    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    # 2) MEASUREMENT — CUDA events are more precise than ``time.time``.
    latencies_ms = []
    preds_chunks, targets_chunks = [], []
    n_correct, n_total = 0, 0

    if is_cuda:
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_runs:
                break
            inputs = _move_inputs(batch, device, input_cols)
            targets = batch[label_key].to(device)

            if is_cuda:
                starter.record()
                outputs = model(inputs)
                ender.record()
                torch.cuda.synchronize()
                latencies_ms.append(starter.elapsed_time(ender))
            else:
                t0 = time.perf_counter()
                outputs = model(inputs)
                latencies_ms.append((time.perf_counter() - t0) * 1000.0)

            preds = torch.argmax(outputs, dim=-1)
            n_correct += (preds == targets).sum().item()
            n_total += targets.size(0)
            preds_chunks.append(preds.cpu())
            targets_chunks.append(targets.cpu())

    if n_total == 0:
        raise RuntimeError('Dataloader produced zero measured batches.')

    preds_all = torch.cat(preds_chunks).numpy()
    targets_all = torch.cat(targets_chunks).numpy()
    acc = n_correct / n_total
    macro_f1 = metrics.f1_score(
        targets_all, preds_all,
        labels=list(range(num_classes)),
        average='macro',
    )

    # 3) PEAK MEMORY + THROUGHPUT
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 ** 3 if is_cuda else 0.0
    total_time_s = sum(latencies_ms) / 1000.0
    throughput = n_total / total_time_s if total_time_s > 0 else float('nan')

    return {
        'median_ms': float(np.median(latencies_ms)),
        'p95_ms': float(np.percentile(latencies_ms, 95)),
        'mean_ms': float(np.mean(latencies_ms)),
        'std_ms': float(np.std(latencies_ms)),
        'peak_mem_gb': float(peak_mem_gb),
        'acc': float(acc),
        'macro_f1': float(macro_f1),
        'throughput_sps': float(throughput),
        'n_batches': len(latencies_ms),
        'n_samples': int(n_total),
    }


class _Opt:
    """Tiny attribute bag matching the ``opt`` shape used elsewhere."""


def _build_opt(args, *, local_context_focus=None, srd=None, batch_size=None):
    opt = _Opt()
    opt.pretrained_bert_name = args.pretrained_bert_name
    opt.bert_dim = 768
    opt.dropout = 0.1
    opt.polarities_dim = 3
    opt.max_seq_len = args.max_seq_len
    opt.dataset = args.dataset
    opt.dataset_file = DATASET_FILES[args.dataset]
    opt.local_context_focus = local_context_focus or args.local_context_focus
    opt.SRD = srd if srd is not None else args.SRD
    opt.batch_size = batch_size if batch_size is not None else args.batch_size
    opt.state_dict_path = args.state_dict_path
    opt.device = torch.device(args.device)
    return opt


def _build_model(opt):
    bert = BertModel.from_pretrained(opt.pretrained_bert_name)
    model = LCF_BERT(bert, opt).to(opt.device)
    if opt.state_dict_path:
        state = torch.load(opt.state_dict_path, map_location=opt.device)
        model.load_state_dict(state)
        logger.info('loaded state_dict from %s', opt.state_dict_path)
    else:
        logger.warning('no --state_dict_path provided; the head is random so '
                       'acc / macro_f1 will not be meaningful (latency / '
                       'memory are still valid)')
    return model


def _build_loader(opt, tokenizer):
    testset = ABSADataset(opt.dataset_file['test'], tokenizer)
    return DataLoader(dataset=testset, batch_size=opt.batch_size, shuffle=False)


def run_single(args):
    opt = _build_opt(args)
    tokenizer = Tokenizer4Bert(opt.max_seq_len, opt.pretrained_bert_name)
    loader = _build_loader(opt, tokenizer)
    model = _build_model(opt)
    result = benchmark(
        model, loader, opt.device,
        warmup=args.warmup, num_runs=args.num_runs,
        num_classes=opt.polarities_dim,
    )
    result['config'] = {
        'dataset': opt.dataset,
        'local_context_focus': opt.local_context_focus,
        'SRD': opt.SRD,
        'batch_size': opt.batch_size,
        'max_seq_len': opt.max_seq_len,
    }
    print(json.dumps(result, indent=2))
    return result


def run_sweep(args):
    tokenizer = Tokenizer4Bert(args.max_seq_len, args.pretrained_bert_name)

    rows = []
    for focus in args.sweep_focus:
        for srd in args.sweep_srd:
            for bs in args.sweep_batch_size:
                opt = _build_opt(args, local_context_focus=focus,
                                 srd=srd, batch_size=bs)
                loader = _build_loader(opt, tokenizer)
                model = _build_model(opt)
                logger.info('benchmarking focus=%s SRD=%d batch_size=%d',
                            focus, srd, bs)
                res = benchmark(
                    model, loader, opt.device,
                    warmup=args.warmup, num_runs=args.num_runs,
                    num_classes=opt.polarities_dim,
                )
                row = {
                    'dataset': args.dataset,
                    'local_context_focus': focus,
                    'SRD': srd,
                    'batch_size': bs,
                    **res,
                }
                rows.append(row)
                # Free GPU memory between configs.
                del model
                if opt.device.type == 'cuda':
                    torch.cuda.empty_cache()

    if args.output_csv:
        keys = list(rows[0].keys())
        with open(args.output_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        logger.info('wrote %s', args.output_csv)

    if args.output_plot:
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
        except ImportError:
            logger.warning('matplotlib not installed; skipping plot')
        else:
            fig, ax = plt.subplots(figsize=(7, 5))
            for r in rows:
                ax.scatter(r['median_ms'], r['macro_f1'])
                ax.annotate(
                    f"{r['local_context_focus']}/SRD{r['SRD']}/bs{r['batch_size']}",
                    (r['median_ms'], r['macro_f1']),
                    fontsize=8, xytext=(4, 4), textcoords='offset points',
                )
            ax.set_xlabel('Median latency per batch (ms)')
            ax.set_ylabel('Macro F1')
            ax.set_title(f'LCF-BERT trade-off — {args.dataset}')
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(args.output_plot, dpi=150)
            logger.info('wrote %s', args.output_plot)

    print(json.dumps(rows, indent=2))
    return rows


def main():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s %(message)s',
    )

    parser = argparse.ArgumentParser(
        description='Latency / memory / accuracy trade-off benchmark for LCF-BERT.',
    )
    parser.add_argument('--dataset', default='restaurant',
                        choices=list(DATASET_FILES.keys()))
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased')
    parser.add_argument('--state_dict_path', default=None,
                        help='trained LCF-BERT checkpoint to load')
    parser.add_argument('--max_seq_len', default=85, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--warmup', default=20, type=int)
    parser.add_argument('--num_runs', default=200, type=int)
    parser.add_argument('--local_context_focus', default='cdm',
                        choices=['cdm', 'cdw'])
    parser.add_argument('--SRD', default=3, type=int)

    parser.add_argument('--sweep', action='store_true',
                        help='grid-sweep over focus / SRD / batch_size')
    parser.add_argument('--sweep_focus', nargs='+', default=['cdm', 'cdw'])
    parser.add_argument('--sweep_srd', nargs='+', type=int, default=[3, 5, 7])
    parser.add_argument('--sweep_batch_size', nargs='+', type=int, default=[16])
    parser.add_argument('--output_csv', default=None,
                        help='path to write sweep results as CSV')
    parser.add_argument('--output_plot', default=None,
                        help='path to write a latency-vs-F1 scatter plot (PNG)')

    args = parser.parse_args()
    if args.sweep:
        run_sweep(args)
    else:
        run_single(args)


if __name__ == '__main__':
    main()
