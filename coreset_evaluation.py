#!/usr/bin/env python3
"""
coreset_evaluation.py – Basic scaffolding for coreset methods on IMDB sentiment

This script:
1. Reads training data from data/imdb_train.csv
2. Instantiates a pretrained model
3. Provides scaffolding to iterate through the dataloader
"""

from __future__ import annotations
import argparse
import os
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from tqdm.auto import tqdm
from typing import List

# Local imports
import eda
import gpt_download as gpt_dl
from coreset import UniformRandomCoreset, SensitivityCoreset
from utils import evaluate_accuracy

# Constants
_PAD = 50256   # GPT-2's end-of-text token
_IGNORE = -100 # label id to mask out padding positions

class IMDBDataset(Dataset):
    """Tiny wrapper around pre-tokenised IMDB texts"""
    def __init__(self, df: pd.DataFrame, toks: List[List[int]]) -> None:
        self.labels = torch.tensor(df["label"].values, dtype=torch.long)
        self.toks   = toks

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return torch.tensor(self.toks[idx], dtype=torch.long), self.labels[idx]


def collate(batch, max_len: int = 256) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad to the longest item in *this* batch – no global re-padding needed"""
    seqs, labels = zip(*batch)
    L = min(max(len(s) for s in seqs), max_len) + 1  # +1 for appended EOS

    padded = torch.full((len(seqs), L), _PAD, dtype=torch.long)
    for i, seq in enumerate(seqs):
        trunc = seq[: L - 1]            # reserve one slot for EOS
        padded[i, : len(trunc)] = trunc
    return padded[:, :-1], torch.stack(labels)


def build_backbone(size: str, cache_dir: str = "gpt2") -> Tuple[nn.Module, int]:
    """
    Downloads (if needed) and returns a bare GPT-2 Transformer
    + its hidden size so we can bolt a classification head on.
    """
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


def load_pretrained_model(model_size: str = "124M") -> nn.Module:
    """
    Load a pretrained model from the trained_models directory.
    If not found, create a new model with pretrained backbone.
    """
    model_path = Path(f"trained_models/gpt2_imdb_{model_size}.pt")
    
    if model_path.exists():
        print(f"Loading pretrained model from {model_path}")
        # Load the backbone first
        backbone, hidden = build_backbone(model_size)
        # Replace the output head for classification
        backbone.out_head = torch.nn.Linear(hidden, 2)
        # Load the trained weights
        backbone.load_state_dict(torch.load(model_path, map_location='cpu'))
    else:
        print(f"Pretrained model not found at {model_path}, creating new model with pretrained backbone")
        backbone, hidden = build_backbone(model_size)
        # Replace the output head for classification
        backbone.out_head = torch.nn.Linear(hidden, 2)
    
    # Add a method to get embeddings (penultimate layer output)
    def get_embeddings(self, x):
        """Get embeddings from the penultimate layer (before out_head)"""
        batch_size, seq_len = x.shape
        tok_embeds = self.tok_emb(x)
        pos_embeds = self.pos_emb(torch.arange(seq_len, device=x.device))
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)  # This is the penultimate layer output
        return x
    
    # Attach the method to the model
    backbone.get_embeddings = get_embeddings.__get__(backbone, type(backbone))
    
    return backbone


def get_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Coreset methods for IMDB sentiment")
    p.add_argument("--model-size",
                   default="124M",
                   choices=["124M", "355M", "774M", "1558M"],
                   help="Which GPT-2 checkpoint to use")
    p.add_argument("--bsz", type=int, default=32, help="Batch size")
    p.add_argument("--seq-len", type=int, default=256,
                   help="Max tokens per review (longer will be truncated)")
    p.add_argument("--workers", type=int, default=max(1, os.cpu_count() // 2))
    p.add_argument("--max-num-batches", type=int, default=None,
                   help="Maximum number of batches to process (for testing)")
    return p.parse_args()


def evaluate_model(model, dataloader, device, weights=None, max_num_batches=None):
    """
    Evaluate the model on a dataloader, optionally with per-sample or per-batch weights.
    Args:
        model: The model to evaluate.
        dataloader: DataLoader to iterate over.
        device: torch.device.
        weights: None (no weighting), float (uniform weight), or tensor/list (per-sample weights).
        max_num_batches: Optional[int], stop after this many batches.
    Returns:
        total_weighted_loss: float
        average_loss: float
        total_samples: int
    """
    criterion_none = nn.CrossEntropyLoss(reduction='none')
    model.eval()
    total_weighted_loss = 0.0
    total_samples = 0
    batch_count = 0
    batch_start = 0
    with torch.no_grad():
        for x, y in tqdm(dataloader, desc="Evaluating"):
            x = x.to(device)
            y = y.to(device)
            batch_size = y.size(0)
            logits = model(x)
            last_token_logits = logits[:, -1, :]
            loss_per_sample = criterion_none(last_token_logits, y)
            # Determine weighting
            if weights is None:
                weighted_loss = loss_per_sample.sum()
            elif isinstance(weights, float):
                weighted_loss = (loss_per_sample * weights).sum()
            elif isinstance(weights, (torch.Tensor, list)):
                # Assume weights is a tensor/list of per-sample weights
                batch_weights = torch.as_tensor(weights[batch_start:batch_start+batch_size], dtype=torch.float, device=device)
                weighted_loss = (loss_per_sample * batch_weights).sum()
                batch_start += batch_size
            else:
                raise ValueError(f"Unsupported weights type: {type(weights)}. Expected None, float, torch.Tensor, or list.")
            total_weighted_loss += weighted_loss.item()
            total_samples += batch_size
            batch_count += 1
            if max_num_batches is not None and batch_count >= max_num_batches:
                break
    average_loss = total_weighted_loss / total_samples if total_samples > 0 else float('nan')
    return total_weighted_loss, average_loss, total_samples


def main():
    args = get_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")

    # ---------- Load training data ------------------------------------------------------- #
    print("Loading training data...")
    df_train, _, _ = eda.load_all_splits()
    df_train = eda.deduplicate(df_train)
    
    print(f"Original training data shape: {df_train.shape}")
    print(f"Class distribution: {df_train['label'].value_counts().to_dict()}")

    # ---------- Use only 25% of training data for debugging/testing ------------------------------------------------------- #
    print("\n" + "="*60)
    print("USING ONLY 25% OF TRAINING DATA FOR EVALUATION")
    print("="*60)
    
    # Take only 25% of the data
    subset_size = len(df_train) // 4
    df_train_subset = df_train.head(subset_size).reset_index(drop=True)
    
    print(f"Subset training data shape: {df_train_subset.shape}")
    print(f"Subset class distribution: {df_train_subset['label'].value_counts().to_dict()}")
    

    # ---------- Tokenize data ------------------------------------------------------- #
    cache_dir = Path("exercises/cache")
    cache_dir.mkdir(exist_ok=True)

    train_ids = eda.tokenise_with_cache(df_train_subset, cache_dir / "train_subset.pt", max_length=args.seq_len)
    print(f"train_ids length: {len(train_ids)}")
    print(f"train_subset.pt exists: {(cache_dir / 'train_subset.pt').exists()}")
    # ---------- Create dataset and dataloader ------------------------------------------------------- #
    dset_train = IMDBDataset(df_train_subset, train_ids)
    collate_fn = lambda b: collate(b, max_len=args.seq_len)
    
    dl_train = DataLoader(dset_train, batch_size=args.bsz,
                          shuffle=True, num_workers=args.workers,
                          collate_fn=collate_fn, pin_memory=device.type == "cuda")

    # ---------- Load pretrained model ------------------------------------------------------- #
    print("Loading pretrained model...")
    model = load_pretrained_model(args.model_size)
    model = model.to(device)

    #---------- Basic evaluation on subset ------------------------------------------------------- #
    print("\n" + "="*60)
    print("EVALUATING ON 25% SUBSET")
    print("="*60)
    total_loss, avg_loss, total_samples = evaluate_model(model, dl_train, device, weights=None, max_num_batches=args.max_num_batches)
    print(f"Total loss on subset: {total_loss:.4f}")
    print(f"Average loss on subset: {avg_loss:.4f}")
    print(f"Processed {total_samples} samples")

    # ---------- Test SensitivityCoreset ------------------------------------------------------- #
    print("\n" + "="*60)
    print("TESTING SENSITIVITY CORESET")
    print("="*60)
    
    # Test coreset creation
    print("Creating SensitivityCoreset...")
    coreset_fraction   :float = 0.1
    k_clusters_fraction:float = 0.025
    pilot_fraction     :float = 0.1
    
    sensitivity_coreset = SensitivityCoreset(
        dataset=dset_train, 
        coreset_fraction=coreset_fraction,
        k_clusters_fraction=k_clusters_fraction,
        pilot_fraction=pilot_fraction,
        model=model, 
        seed=42
    )
    print(f"SensitivityCoreset created successfully")
    print(f"Target coreset size (m): {sensitivity_coreset.total_coreset_size}")
    print(f"Number of clusters (k): {sensitivity_coreset.k_clusters}")
    print(f"Pilot size: {sensitivity_coreset.pilot_size}")

    sampled_idx: List[int]
    sample_weights: List[float]
    sampled_idx, sample_weights = sensitivity_coreset.build(model, collate_fn=collate_fn)
    print(f"Build method completed successfully")
    print(f"Embeddings shape: {len(sampled_idx)}")
    print(f"Sample weights:\n{sample_weights}")

    # ---------- Test sensitivity random coreset evaluation ------------------------------------------------------- #
    print("\n" + "="*60)
    print("TESTING SENSITIVITY CORESET EVALUATION")
    print("="*60)
    sensitivity_sampler = SubsetRandomSampler(sampled_idx)
    dl_sensitivity_coreset = DataLoader(dset_train, batch_size=args.bsz, shuffle=False,
                           sampler=sensitivity_sampler, num_workers=args.workers,
                           collate_fn=collate_fn, pin_memory=device.type == "cuda")
    sample_weights_tensor = torch.tensor(sample_weights, dtype=torch.float)
    coreset_total_weighted_loss, coreset_avg_loss, coreset_total_samples = evaluate_model(
        model, dl_sensitivity_coreset, device, weights=sample_weights_tensor,
        max_num_batches=args.max_num_batches)
    print(f"Coreset total weighted loss: {coreset_total_weighted_loss:.4f}")
    print(f"Coreset average loss: {coreset_avg_loss:.4f}")
    print(f"Full subset loss: {total_loss:.4f}")
    print(f"Absolute error (coreset - full)/full: {abs(coreset_total_weighted_loss - total_loss)/total_loss:.4f}")
    print(f"Full dataset size N: {total_samples}")
    print(f"Coreset size: {len(sampled_idx)}")
    
    # ---------- Test UniformRandomCoreset ------------------------------------------------------- #
    print("\n" + "="*60)
    print("TESTING UNIFORM RANDOM CORESET")
    print("="*60)
    
    # Test coreset creation
    print("Creating UniformRandomCoreset...")
    coreset = UniformRandomCoreset(dset_train, fraction=0.1, seed=42)
    
    sel_idx: List[int]
    weight: float
    sel_idx, weight = coreset.select_coreset()
    print(f"Selected indices: {len(sel_idx)} samples")
    print(f"Sampling weights: {weight:.2f}")
    
    # Verify the weight calculation
    expected_weight = len(dset_train) / len(sel_idx)
    print(f"Expected weight: {expected_weight:.2f}")
    print(f"Weight matches expected: {abs(weight - expected_weight) < 1e-6}")
    
    # Geenrate the dataloaders for the coreset datasets
    print("\nTesting coreset data loader creation...")
    sampler = SubsetRandomSampler(sel_idx)
    dl_uniform = DataLoader(dset_train, batch_size=args.bsz, shuffle=False,
                           sampler=sampler, num_workers=args.workers,
                           collate_fn=collate_fn, pin_memory=device.type=="cuda")
    
    # Evaluate on coreset with weighted loss
    coreset_total_weighted_loss, coreset_avg_loss, coreset_total_samples = evaluate_model(
        model, dl_uniform, device, weights=weight, max_num_batches=args.max_num_batches)
    print(f"Coreset total weighted loss: {coreset_total_weighted_loss:.4f}")
    print(f"Coreset average loss: {coreset_avg_loss:.4f}")
    print(f"Full subset loss: {total_loss:.4f}")
    print(f"Absolute error (coreset - full)/full: {abs(coreset_total_weighted_loss - total_loss)/total_loss:.4f}")
    print(f"Full dataset size N: {total_samples}")
    print(f"Coreset size: {len(sel_idx)}")


if __name__ == "__main__":
    main() 