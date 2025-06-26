#!/usr/bin/env python3
"""Run coreset experiments across multiple sizes and seeds.

This script loops over a set of coreset sizes and random seeds.
For each configuration it trains a classifier using both
:class:`UniformRandomCoreset` and :class:`SensitivityCoreset` from
``coreset.py``. Metrics are logged to CSV files and aggregated to
produce a plot comparing validation accuracy versus coreset size.
The full dataset baseline is included for reference.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Dict, Any

import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

import eda
from coreset import UniformRandomCoreset, SensitivityCoreset
import train_finetune as tf


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coreset experiment runner")
    p.add_argument(
        "--coreset-sizes",
        type=int,
        nargs="+",
        default=[100, 200, 500, 1000, 2000],
        help="Coreset sizes to evaluate",
    )
    p.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
        help="Random seeds to run",
    )
    p.add_argument("--model-size", default="124M", choices=["124M", "355M", "774M", "1558M"], help="GPT-2 model size")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p.add_argument("--bsz", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--seq-len", type=int, default=256, help="Maximum tokens per review")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--accum-steps", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--k-clusters-fraction", type=float, default=0.025, help="Cluster fraction for sensitivity coreset")
    p.add_argument("--pilot-fraction", type=float, default=0.1, help="Pilot fraction for sensitivity coreset")
    p.add_argument("--eval-every", type=int, default=0, help="Evaluation interval")
    p.add_argument("--max-num-batches", type=int, default=None, help="Optional dataloader limit")
    p.add_argument("--log-dir", type=str, default="logs", help="Directory for CSV logs")
    p.add_argument("--results-csv", type=str, default="coreset_results.csv", help="Aggregated results file")
    p.add_argument("--plot-file", type=str, default="coreset_plot.png", help="Output plot image")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_full_loader(dataset: torch.utils.data.Dataset, args: argparse.Namespace, collate_fn) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=args.bsz,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ---------------------------------------------------------------------------
# Main experiment routine
# ---------------------------------------------------------------------------

def run() -> None:
    args = get_args()
    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset once
    df_train, df_val, _ = eda.load_all_splits()
    df_train = eda.deduplicate(df_train)
    df_val = eda.deduplicate(df_val)

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    train_ids = eda.tokenise_with_cache(df_train, cache_dir / "train.pt", max_length=args.seq_len)
    val_ids = eda.tokenise_with_cache(df_val, cache_dir / "val.pt", max_length=args.seq_len)

    dset_train = tf.IMDBDataset(df_train, train_ids)
    dset_val = tf.IMDBDataset(df_val, val_ids)
    collate_fn = lambda b: tf.collate(b, max_len=args.seq_len)
    dl_val = DataLoader(
        dset_val,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )

    results: List[Dict[str, Any]] = []

    # --- Full data baseline -------------------------------------------------
    dl_full = make_full_loader(dset_train, args, collate_fn)
    tf_args = argparse.Namespace(**vars(args), coreset_fraction=1.0, log_file=str(Path(args.log_dir)/"full.csv"))
    metrics, log_path = tf.run_training("full", dl_full, tf_args, device, dl_val)
    results.append({"method": "full", "coreset_size": len(dset_train), "seed": 0, **metrics})

    # --- Coreset loops ------------------------------------------------------
    for size in args.coreset_sizes:
        frac = size / len(dset_train)
        for seed in args.seeds:
            # UniformRandomCoreset
            uni = UniformRandomCoreset(dset_train, fraction=frac, seed=seed)
            idx, w = uni.select_coreset()
            dl_uni = tf.make_subset_loader(dset_train, idx, args.bsz, args.workers, collate_fn, device, weights=w)
            tf_args = argparse.Namespace(**vars(args), coreset_fraction=frac, log_file=str(Path(args.log_dir)/f"uniform_{size}_{seed}.csv"))
            metrics, log_path = tf.run_training(f"uniform_{size}_{seed}", dl_uni, tf_args, device, dl_val)
            results.append({"method": "uniform", "coreset_size": size, "seed": seed, **metrics})

            # SensitivityCoreset
            sens_model = tf.init_model(args.model_size, device)
            sens = SensitivityCoreset(
                dataset=dset_train,
                coreset_fraction=frac,
                k_clusters_fraction=args.k_clusters_fraction,
                pilot_fraction=args.pilot_fraction,
                model=sens_model,
                seed=seed,
            )
            s_idx, s_w = sens.select_coreset(sens_model, collate_fn=collate_fn)
            dl_sens = tf.make_subset_loader(dset_train, s_idx, args.bsz, args.workers, collate_fn, device, weights=s_w)
            tf_args = argparse.Namespace(**vars(args), coreset_fraction=frac, log_file=str(Path(args.log_dir)/f"sensitivity_{size}_{seed}.csv"))
            metrics, log_path = tf.run_training(f"sens_{size}_{seed}", dl_sens, tf_args, device, dl_val)
            results.append({"method": "sensitivity", "coreset_size": size, "seed": seed, **metrics})

    # --- Aggregate and save results ----------------------------------------
    df_results = pd.DataFrame(results)
    df_results.to_csv(args.results_csv, index=False)

    agg = df_results.groupby(["method", "coreset_size"]) ["val_acc"].mean().reset_index()
    baseline = agg[agg["method"] == "full"]["val_acc"].iloc[0]

    plt.figure(figsize=(8, 5))
    for method in ["uniform", "sensitivity"]:
        sub = agg[agg["method"] == method]
        plt.plot(sub["coreset_size"], sub["val_acc"], marker="o", label=method)
    plt.axhline(baseline, color="k", linestyle="--", label="full data")
    plt.xlabel("Coreset size")
    plt.ylabel("Validation accuracy")
    plt.legend()
    plt.title("Accuracy vs. Coreset Size")
    plt.tight_layout()
    plt.savefig(args.plot_file)
    print(f"✓ Results saved to {args.results_csv} and {args.plot_file}")


if __name__ == "__main__":  # pragma: no cover
    run()
