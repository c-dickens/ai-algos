#!/usr/bin/env python3
"""Finetune GPT-2 on IMDB using coreset sampling strategies.

This script runs one epoch of training using two strategies:
1. UniformRandomCoreset
2. SensitivityCoreset

Metrics (loss and accuracy) are logged after each epoch.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import (
    DataLoader,
    Dataset,
    SubsetRandomSampler,
    WeightedRandomSampler,
    Subset,
)
from tqdm.auto import tqdm

import eda
import gpt_download as gpt_dl
from coreset import UniformRandomCoreset, SensitivityCoreset
from utils import evaluate_accuracy

_PAD = 50256


class IMDBDataset(Dataset):
    """Tiny wrapper around tokenised IMDB texts."""

    def __init__(self, df: pd.DataFrame, toks: List[List[int]]) -> None:
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.toks = toks

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.toks[idx], dtype=torch.long), self.labels[idx]


class WeightedSubset(Dataset):
    """Dataset subset that yields ``(x, y, weight)`` for weighted sampling.

    When used alongside :func:`make_subset_loader`, each item includes its
    associated sampling weight so the training loop can scale the loss.
    This dataset pairs with :class:`~torch.utils.data.WeightedRandomSampler`
    to draw weighted batches while still returning the weight tensor.
    """

    def __init__(self, dataset: Dataset, indices: List[int], weights: torch.Tensor):
        self.dataset = dataset
        self.indices = indices
        self.weights = weights

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        x, y = self.dataset[self.indices[idx]]
        return x, y, self.weights[idx]


def collate(batch, max_len: int = 256):
    """Pad a batch and optionally return per-sample weights."""

    has_weight = len(batch[0]) == 3
    if has_weight:
        seqs, labels, weights = zip(*batch)
    else:
        seqs, labels = zip(*batch)
        weights = None

    L = min(max(len(s) for s in seqs), max_len) + 1
    padded = torch.full((len(seqs), L), _PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        trunc = seq[: L - 1]
        padded[i, : len(trunc)] = trunc

    out = [padded[:, :-1], torch.stack(labels)]
    if has_weight:
        out.append(torch.tensor(weights, dtype=torch.float))
    return tuple(out)


def build_backbone(size: str, cache_dir: str = "gpt2") -> Tuple[nn.Module, int]:
    settings, params = gpt_dl.download_and_load_gpt2(size, cache_dir)
    from gpt_download import GPTModel, load_weights_into_gpt

    model = GPTModel(
        {
            "vocab_size": settings["n_vocab"],
            "context_length": settings["n_ctx"],
            "emb_dim": settings["n_embd"],
            "n_layers": settings["n_layer"],
            "n_heads": settings["n_head"],
            "drop_rate": 0.0,
            "qkv_bias": True,
        }
    )
    load_weights_into_gpt(model, params)
    return model, settings["n_embd"]


def init_model(size: str, device: torch.device) -> nn.Module:
    backbone, hidden = build_backbone(size)
    backbone.out_head = nn.Linear(hidden, 2)

    def get_embeddings(self, x):
        seq_len = x.size(1)
        tok = self.tok_emb(x)
        pos = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = self.drop_emb(tok + pos)
        x = self.trf_blocks(x)
        return self.final_norm(x)

    backbone.get_embeddings = get_embeddings.__get__(backbone, type(backbone))
    return backbone.to(device)


def train_epoch(loader: DataLoader, model: nn.Module, crit: nn.Module,
                optimiser: optim.Optimizer, scaler: torch.cuda.amp.GradScaler,
                device: torch.device, accum_steps: int) -> float:
    model.train()
    meter_loss = 0.0
    total = 0
    step = 0

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.autocast
    for x, y in tqdm(loader, leave=False):
        x, y = x.to(device), y.to(device)
        with autocast():
            logits = model(x)
            loss = crit(logits[:, -1, :], y) / accum_steps
        scaler.scale(loss).backward()
        if (step + 1) % accum_steps == 0:
            scaler.step(optimiser)
            scaler.update()
            optimiser.zero_grad()
        meter_loss += loss.item() * y.size(0)
        total += y.size(0)
        step += 1
    return meter_loss / total


def evaluate(loader: DataLoader, model: nn.Module, crit: nn.Module,
             device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 3:
                x, y, _ = batch
            else:
                x, y = batch
            x, y = x.to(device), y.to(device)
            logits = model(x)
            l = crit(logits[:, -1, :], y)
            total_loss += l.item() * y.size(0)
            total += y.size(0)
    return total_loss / total if total > 0 else float("nan")


def make_subset_loader(
    dataset: Dataset,
    indices: List[int],
    batch_size: int,
    workers: int,
    collate_fn,
    device: torch.device,
    weights: torch.Tensor | None = None,
) -> DataLoader:
    """Return a DataLoader over ``indices`` with optional weights.

    If ``weights`` is provided, the loader uses a
    :class:`~torch.utils.data.WeightedRandomSampler` with replacement to sample
    from ``indices`` according to these weights and yields batches of
    ``(x, y, weight)``.  Otherwise a normal :class:`SubsetRandomSampler` is used
    and the batches contain only ``(x, y)``.
    """

    if weights is not None:
        sub_dset = WeightedSubset(dataset, indices, weights)
        sampler = WeightedRandomSampler(weights=weights, num_samples=len(weights), replacement=True)
    else:
        sub_dset = Subset(dataset, indices)
        sampler = SubsetRandomSampler(range(len(indices)))

    return DataLoader(
        sub_dset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )


def run_training(tag: str, loader: DataLoader, args: argparse.Namespace, device: torch.device, dl_val: DataLoader) -> None:
    model = init_model(args.model_size, device)
    crit = nn.CrossEntropyLoss()
    crit_none = nn.CrossEntropyLoss(reduction="none")
    opt = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    train_eval_loader = DataLoader(
        loader.dataset,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=loader.collate_fn,
        pin_memory=device.type == "cuda",
    )

    eval_every = args.eval_every if args.eval_every > 0 else len(loader)
    global_step = 0

    with open(args.log_file, "w") as fh:
        fh.write("step,epoch,train_loss,train_acc,val_loss,val_acc\n")

        autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.autocast

        for ep in range(1, args.epochs + 1):
            running_loss = 0.0
            running_correct = 0
            running_total = 0

            for batch in tqdm(loader, leave=False):
                if len(batch) == 3:
                    x, y, w = batch
                    w = w.to(device)
                else:
                    x, y = batch
                    w = None
                x, y = x.to(device), y.to(device)
                with autocast():
                    logits = model(x)
                    losses = crit_none(logits[:, -1, :], y)
                    if w is not None:
                        losses = losses * w
                    loss = losses.mean() / args.accum_steps
                scaler.scale(loss).backward()
                if (global_step + 1) % args.accum_steps == 0:
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad()

                preds = logits[:, -1, :].argmax(dim=-1)
                running_correct += (preds == y).sum().item()
                running_loss += loss.item() * y.size(0)
                running_total += y.size(0)
                global_step += 1

                if args.eval_every > 0 and global_step % eval_every == 0:
                    train_acc = evaluate_accuracy(model, train_eval_loader, device)
                    val_loss = evaluate(dl_val, model, crit, device)
                    val_acc = evaluate_accuracy(model, dl_val, device)
                    train_loss = running_loss / running_total if running_total > 0 else float("nan")
                    fh.write(f"{global_step},{ep},{train_loss},{train_acc},{val_loss},{val_acc}\n")
                    fh.flush()
                    print(
                        f"[{tag} step {global_step} epoch {ep}] "
                        f"train {train_loss:.4f} acc {train_acc:.3%} | "
                        f"val {val_loss:.4f} acc {val_acc:.3%}"
                    )
                    running_loss = 0.0
                    running_correct = 0
                    running_total = 0

            if args.eval_every == 0 or running_total > 0:
                train_acc = evaluate_accuracy(model, train_eval_loader, device)
                val_loss = evaluate(dl_val, model, crit, device)
                val_acc = evaluate_accuracy(model, dl_val, device)
                train_loss = running_loss / running_total if running_total > 0 else float("nan")
                fh.write(f"{global_step},{ep},{train_loss},{train_acc},{val_loss},{val_acc}\n")
                fh.flush()
                print(
                    f"[{tag} step {global_step} epoch {ep}] "
                    f"train {train_loss:.4f} acc {train_acc:.3%} | "
                    f"val {val_loss:.4f} acc {val_acc:.3%}"
                )


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune GPT-2 using coresets")
    p.add_argument("--model-size", default="124M",
                   choices=["124M", "355M", "774M", "1558M"],
                   help="GPT-2 checkpoint to start from")
    p.add_argument("--epochs", type=int, default=1, help="Training epochs")
    p.add_argument("--bsz", type=int, default=8, help="Batch size")
    p.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Maximum tokens per review")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation steps")
    p.add_argument("--coreset-fraction", type=float, default=0.1,
                   help="Fraction of data for the coreset")
    p.add_argument("--k-clusters-fraction", type=float, default=0.025,
                   help="Clusters fraction for sensitivity coreset")
    p.add_argument("--pilot-fraction", type=float, default=0.1,
                   help="Pilot fraction for sensitivity coreset")
    p.add_argument("--max-num-batches", type=int, default=None,
                   help="Optional limit for number of batches")
    p.add_argument("--eval-every", type=int, default=0,
                   help="Evaluation interval in steps (0 -> once per epoch)")
    p.add_argument(
        "--log-file",
        type=str,
        default=f"training_metrics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        help="CSV file for metrics",
    )
    p.add_argument("--coreset-type", choices=["uniform", "sensitivity", "all"], default="all",
                   help="Which coreset strategy to use")
    return p.parse_args()


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda")
    print(f"✓ Using device: {device}")

    df_train, df_val, _ = eda.load_all_splits()
    df_train = eda.deduplicate(df_train)
    df_val = eda.deduplicate(df_val)

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    train_ids = eda.tokenise_with_cache(df_train, cache_dir / "train.pt",
                                        max_length=args.seq_len)
    val_ids = eda.tokenise_with_cache(df_val, cache_dir / "val.pt",
                                      max_length=args.seq_len)

    dset_train = IMDBDataset(df_train, train_ids)
    dset_val = IMDBDataset(df_val, val_ids)

    collate_fn = lambda b: collate(b, max_len=args.seq_len)

    dl_val = DataLoader(
        dset_val,
        batch_size=args.bsz,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn,
        pin_memory=device.type == "cuda",
    )

    if args.coreset_type == "uniform":
        # Uniform coreset
        uniform = UniformRandomCoreset(dset_train, fraction=args.coreset_fraction)
        uniform_idx, uniform_w = uniform.select_coreset()
        dl_uniform = make_subset_loader(
            dset_train,
            uniform_idx,
            args.bsz,
            args.workers,
            collate_fn,
            device,
            weights=uniform_w,
        )
        print("\n== Training with UniformRandomCoreset ==")
        run_training("uniform", dl_uniform, args, device, dl_val)

    elif args.coreset_type == "sensitivity":
        # Sensitivity coreset
        sens_model = init_model(args.model_size, device)
        sens = SensitivityCoreset(
            dataset=dset_train,
            coreset_fraction=args.coreset_fraction,
            k_clusters_fraction=args.k_clusters_fraction,
            pilot_fraction=args.pilot_fraction,
            model=sens_model,
            seed=42,
        )
        sens_idx, sens_w = sens.select_coreset(sens_model, collate_fn=collate_fn)
        dl_sens = make_subset_loader(
            dset_train,
            sens_idx,
            args.bsz,
            args.workers,
            collate_fn,
            device,
            weights=sens_w,
        )
        print("\n== Training with SensitivityCoreset ==")
        run_training("sensitivity", dl_sens, args, device, dl_val)

    else:  # args.coreset_type == "all" or fallback
        # Full dataset (no coreset)
        dl_full = DataLoader(
            dset_train,
            batch_size=args.bsz,
            shuffle=True,
            num_workers=args.workers,
            collate_fn=collate_fn,
            pin_memory=device.type == "cuda",
        )
        print("\n== Training with Full Dataset (no coreset) ==")
        run_training("full", dl_full, args, device, dl_val)


if __name__ == "__main__":  # pragma: no cover
    main(get_args())
