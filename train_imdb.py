#!/usr/bin/env python3
"""
train_imdb.py – minimal finetuning of GPT‑2 on IMDB sentiment

This script depends only on:
  • PyTorch ≥ 2.0
  • tqdm
  • pandas
  • tiktoken
  • The companion modules:
        - eda.py      (for data wrangling & tokenising)
        - gpt_download.py (for grabbing the public GPT‑2 weights)
"""

from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
import tiktoken

# --- local helpers --------------------------------------------------------- #
import eda                                   # your slimmed‑down EDA module :contentReference[oaicite:0]{index=0}
import gpt_download as gpt_dl                # weight‑downloader from earlier      :contentReference[oaicite:1]{index=1}

# --------------------------------------------------------------------------- #
# 0. Arguments
# --------------------------------------------------------------------------- #
def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Finetune GPT‑2 for IMDB sentiment")
    p.add_argument("--model-size",
                   default="124M",
                   choices=["124M", "355M", "774M", "1558M"],
                   help="Which GPT‑2 checkpoint to start from")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--bsz",    type=int, default=8, help="Batch size")
    p.add_argument("--lr",     type=float, default=5e-5)
    p.add_argument("--seq-len", type=int, default=256,
                   help="Max tokens per review (longer will be truncated)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--accum-steps", type=int, default=1,
                   help="Gradient accumulation for big models on small GPUs")
    p.add_argument("--max-num-batches", type=int, default=None,
                   help="Maximum number of batches to process per epoch (for testing)")
    return p.parse_args()


# --------------------------------------------------------------------------- #
# 1. Dataset & collate
# --------------------------------------------------------------------------- #
_PAD = 50256   # GPT‑2's  end‑of‑text  token
_IGNORE = -100 # label id to mask out padding positions

class IMDBDataset(Dataset):
    """Tiny wrapper around pre‑tokenised IMDB texts"""
    def __init__(self, df: pd.DataFrame, toks: List[List[int]]) -> None:
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.toks   = toks

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.toks[idx], dtype=torch.long), self.labels[idx]


def collate(batch, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest item in *this* batch – no global re‑padding needed"""
    seqs, labels = zip(*batch)
    L = min(max(len(s) for s in seqs), max_len) + 1  # +1 for appended EOS

    padded = torch.full((len(seqs), L), _PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        trunc = seq[: L - 1]            # reserve one slot for EOS
        padded[i, : len(trunc)] = trunc
    return padded[:, :-1], torch.stack(labels)       # predict after EOS


# --------------------------------------------------------------------------- #
# 2. Model
# --------------------------------------------------------------------------- #
def build_backbone(size: str, cache_dir: str = "gpt2") -> Tuple[nn.Module, int]:
    """
    Downloads (if needed) and returns a bare GPT‑2 Transformer
    + its hidden size so we can bolt a classification head on.
    """
    settings, params = gpt_dl.download_and_load_gpt2(size, cache_dir)
    # tiny dependency‑free GPT‑2 implementation from the Manning repo
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


# --------------------------------------------------------------------------- #
# 3. Training helpers
# --------------------------------------------------------------------------- #
def loop(loader, model, crit, optimiser, scaler, device, is_train: bool, max_num_batches: int = None):
    """
    A single pass over *loader*. If `is_train` we update parameters.
    """
    model.train(is_train)
    meter_loss = 0.0
    batch_count = 0

    autocast = torch.cuda.amp.autocast if device.type == "cuda" else torch.autocast
    step_fn  = optimiser.step if is_train else lambda: None

    for x, y in tqdm(loader, leave=False):
        x = x.to(device)
        y = y.to(device)

        with autocast():
            logits = model(x)  # (B, T, 2)
            last_token_logits = logits[:, -1, :]  # (B, 2)
            loss = crit(last_token_logits, y) / args.accum_steps

        if is_train:
            scaler.scale(loss).backward()
            if loop.step % args.accum_steps == 0:   # type: ignore[attr-defined]
                scaler.step(optimiser)
                scaler.update()
                optimiser.zero_grad()
        meter_loss += loss.item() * y.size(0)
        loop.step += 1     # type: ignore[attr-defined]
        batch_count += 1
        
        # Break early if we've reached the maximum number of batches
        if max_num_batches is not None and batch_count >= max_num_batches:
            break
            
    return meter_loss / len(loader.dataset)
loop.step = 0            # type: ignore[attr-defined]


def accuracy(loader, model, device, max_num_batches: int = None) -> float:
    model.eval()
    correct = total = 0
    batch_count = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            last_token_logits = logits[:, -1, :]
            pred = last_token_logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.numel()
            batch_count += 1
            
            # Break early if we've reached the maximum number of batches
            if max_num_batches is not None and batch_count >= max_num_batches:
                break
    return correct / total


# --------------------------------------------------------------------------- #
# 4. Main
# --------------------------------------------------------------------------- #
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # ---------- data ------------------------------------------------------- #
    df_train, df_val, _ = eda.load_all_splits()
    df_train = eda.deduplicate(df_train)
    df_val   = eda.deduplicate(df_val)

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)

    train_ids = eda.tokenise_with_cache(df_train, cache_dir / "train.pt",
                                        max_length=args.seq_len)
    val_ids   = eda.tokenise_with_cache(df_val,   cache_dir / "val.pt",
                                        max_length=args.seq_len)

    dset_train = IMDBDataset(df_train, train_ids)
    dset_val   = IMDBDataset(df_val,   val_ids)

    collate_fn = lambda b: collate(b, max_len=args.seq_len)

    dl_train = DataLoader(dset_train, batch_size=args.bsz,
                          shuffle=True,  num_workers=args.workers,
                          collate_fn=collate_fn, pin_memory=device.type == "cuda")
    dl_val   = DataLoader(dset_val,   batch_size=args.bsz,
                          shuffle=False, num_workers=args.workers,
                          collate_fn=collate_fn, pin_memory=device.type == "cuda")

    # ---------- model ------------------------------------------------------ #
    print("↓ Loading GPT‑2 backbone …")
    backbone, hidden = build_backbone(args.model_size)
    # Replace the output head for classification
    backbone.out_head = torch.nn.Linear(hidden, 2)
    model = backbone.to(device)

    crit   = nn.CrossEntropyLoss()
    opt    = optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    # ---------- train ------------------------------------------------------ #
    for ep in range(1, args.epochs + 1):
        train_loss = loop(dl_train, model, crit, opt, scaler, device, True, args.max_num_batches)
        val_loss   = loop(dl_val,   model, crit, opt, scaler, device, False, args.max_num_batches)
        val_acc    = accuracy(dl_val, model, device, args.max_num_batches)
        print(f"[epoch {ep}]  train {train_loss:.4f} | "
              f"val {val_loss:.4f} | acc {val_acc:.3%}")

    # ---------- save ------------------------------------------------------- #
    out = Path(f"trained_models/gpt2_imdb_{args.model_size}.pt")
    torch.save(model.state_dict(), out)
    print(f"✓ Model saved to {out.resolve()}")


if __name__ == "__main__":
    args = get_args()
    main(args)
