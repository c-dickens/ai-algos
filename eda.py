#!/usr/bin/env python3
# eda.py ─ a minimal, fast utility module for IMDB sentiment data
#
# Copyright (c) 2025
# Licensed under the Apache 2.0 licence (same as original project).

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterable, List, Tuple

import pandas as pd
import torch
import tiktoken
from tqdm.auto import tqdm

# --------------------------------------------------------------------------- #
# 0. Globals & constants
# --------------------------------------------------------------------------- #
_DATA_DIR = Path("data")
_CACHE_DIR = Path("cache")
_CACHE_DIR.mkdir(exist_ok=True)
_ENCODER = tiktoken.get_encoding("gpt2")  # created once, re‑used everywhere


# --------------------------------------------------------------------------- #
# 1. Data loading utilities
# --------------------------------------------------------------------------- #
def load_split(split: str) -> pd.DataFrame:
    """
    Load one of the IMDB CSV splits (train / val / test).

    Parameters
    ----------
    split : {"train", "val", "test"}

    Returns
    -------
    pd.DataFrame
    """
    split = split.lower()
    if split not in {"train", "val", "test"}:
        raise ValueError("split must be 'train', 'val' or 'test'")

    path = _DATA_DIR / f"imdb_{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Expected file {path} – did you download the dataset?")

    return pd.read_csv(path)


def load_all_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Convenience wrapper that returns (train, val, test) dataframes."""
    return load_split("train"), load_split("val"), load_split("test")


# --------------------------------------------------------------------------- #
# 2. Quick‑and‑clean EDA helpers
# --------------------------------------------------------------------------- #
def describe_dataframe(df: pd.DataFrame) -> dict:
    """
    Return a *minimal* set of dataset statistics as a dictionary.

    The caller can `print(json.dumps(stats, indent=2))` or log as needed.
    """
    stats = {
        "rows": len(df),
        "cols": df.shape[1],
        "memory_mb": round(df.memory_usage(deep=True).sum() / 2**20, 2),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "n_missing": int(df.isna().any(axis=1).sum()),
        "n_duplicates": int(df.duplicated().sum()),
    }

    # If a 'label' column is present we add a very small class breakdown
    if "label" in df.columns:
        counts = df["label"].value_counts(dropna=False).to_dict()
        total = sum(counts.values())
        stats["class_distribution"] = {k: f"{v} ({v/total:.1%})" for k, v in counts.items()}

    return stats


def deduplicate(df: pd.DataFrame, *, keep: str = "first") -> pd.DataFrame:
    """
    Drop duplicate rows and return a copy (the original frame is untouched).

    Parameters
    ----------
    keep : {"first", "last", False}
        Matches the `DataFrame.drop_duplicates` API.
    """
    return df.drop_duplicates(keep=keep, ignore_index=True)


# --------------------------------------------------------------------------- #
# 3. Tokenisation (with optional disk cache)
# --------------------------------------------------------------------------- #
def _encode(texts: Iterable[str], *, max_length: int) -> List[List[int]]:
    """
    Vectorise raw texts → list of GPT‑2 token‑id lists.

    We keep this private so the outer helper can handle progress bars and caching.
    """
    encoded: List[List[int]] = []
    for t in tqdm(texts, desc="Tokenising", unit="docs"):
        ids = _ENCODER.encode(t)[:max_length]
        encoded.append(ids)
    return encoded


def tokenise_with_cache(
    df: pd.DataFrame,
    cache_path: Path,
    *,
    max_length: int = 512,
    overwrite: bool = False,
) -> List[List[int]]:
    """
    Tokenise the `"text"` column and cache the result to disk.

    Any Torch serialisable object would work; we stick with `torch.save`.
    """
    if cache_path.exists() and not overwrite:
        return torch.load(cache_path)

    encoded = _encode(df["text"].astype(str), max_length=max_length)
    torch.save(encoded, cache_path)
    return encoded


# --------------------------------------------------------------------------- #
# 4. Small CLI for quick exploration
# --------------------------------------------------------------------------- #
def _main() -> None:
    parser = argparse.ArgumentParser(
        description="Tiny EDA + tokeniser helper for the IMDB dataset."
    )
    parser.add_argument(
        "--split",
        default="train",
        choices=["train", "val", "test"],
        help="Which split to run the quick report on (default: train).",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length when tokenising (default: 256).",
    )
    args = parser.parse_args()

    df = load_split(args.split)
    stats = describe_dataframe(df)
    print(json.dumps(stats, indent=2))

    cache_file = _CACHE_DIR / f"{args.split}_toklen{args.max_length}.pt"
    tokenised = tokenise_with_cache(df, cache_file, max_length=args.max_length)
    print(f"\nTokenised {len(tokenised):,} documents → "
          f"{cache_file} (mean length: {sum(map(len, tokenised))/len(tokenised):.1f} tokens)")


if __name__ == "__main__":  # pragma: no cover
    _main()
