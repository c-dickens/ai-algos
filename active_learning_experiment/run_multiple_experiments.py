#!/usr/bin/env python3
"""
Multiple experiment runner for active learning with different seeds
Runs experiments with sensitivity and uniform sampling methods across multiple seeds
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import pandas as pd
import os
import sys
import argparse
from typing import Dict, List, Tuple
import time
from datetime import datetime

# Import functions from main.py
sys.path.append('.')
from main import MLP, CNN, get_device, load_dataset, train_model, compute_losses, coreset_sampling
from models import MnistNet

def run_single_experiment(seed: int, coreset_method: str, k: int = 1024) -> Dict:
    """
    Run a single experiment with given seed and coreset method
    
    Args:
        seed: Random seed for reproducibility
        coreset_method: Either "sensitivity" or "uniform"
        k: Total budget size
    
    Returns:
        Dictionary containing experiment results
    """
    print(f"\n{'='*60}")
    print(f"Running experiment with seed {seed}, method: {coreset_method}")
    print(f"{'='*60}")
    
    # Set seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
    device = get_device()
    print(f"Using device: {device}")
    
    # Load MNIST dataset
    train_dataset, test_dataset = load_dataset("mnist")
    
    # Use a small budget for testing
    k_prime = int(k >> 2)  # 256 initial points
    
    print(f"Total budget: {k}, Initial samples: {k_prime}")
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Step 1: Randomly sample k_prime points
    indices = torch.randperm(len(train_subset))[:k_prime]
    initial_subset = Subset(train_subset, indices)
    
    # Step 2: Train initial model
    BATCH_SIZE = 32
    EVALUATION_BATCH_SIZE = 128
    model = MnistNet()
    train_loader = DataLoader(initial_subset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f"Training MnistNet model on {len(initial_subset)} total points...")
    train_accs, val_accs, initial_steps = train_model(model, train_loader, val_loader, device, epochs=10)
    initial_val_acc = val_accs[-1]
    print(f"Initial validation accuracy: {initial_val_acc:.2f}%")
    print(f"Initial training steps: {initial_steps}")
    
    # Step 3: Test coreset sampling
    if coreset_method == "sensitivity":
        kwargs = {"coreset_method": "sensitivity", "k_clusters": 16, "pilot_size": k_prime, "model": model}
    else:  # uniform
        kwargs = {"coreset_method": "uniform"}
    
    print(f"Testing coreset sampling {coreset_method} with kwargs\n{kwargs if kwargs else 'None'}...")
    selected_indices = coreset_sampling(train_subset, k, k_prime, used_indices=indices.numpy(), batch_size=train_loader.batch_size, **kwargs)

    # Step 4: Retrain on selected data
    print("Retraining on selected data...")
    all_selected_indices = np.concatenate([indices.numpy(), selected_indices])
    final_subset = Subset(train_subset, all_selected_indices)
    
    print(f"Training MnistNet model on {len(final_subset)} total points...")
    new_model = MnistNet()  # Fresh model
    new_train_loader = DataLoader(final_subset, batch_size=train_loader.batch_size, shuffle=True)
    _, val_accs, final_steps = train_model(new_model, new_train_loader, val_loader, device, epochs=10)
    final_val_acc = val_accs[-1]
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    print(f"Final training steps: {final_steps}")
    print(f"Validation improvement: {final_val_acc - initial_val_acc:.2f} % points")
    
    # Step 5: Evaluate on test dataset
    print("Evaluating on test dataset...")
    test_loader = DataLoader(test_dataset, batch_size=EVALUATION_BATCH_SIZE, shuffle=False)
    new_model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = new_model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
    
    test_acc = 100. * correct / total
    print(f"Test accuracy: {test_acc:.2f}%")
    
    # Return results
    results = {
        'seed': seed,
        'method': coreset_method,
        'initial_val_acc': initial_val_acc,
        'final_val_acc': final_val_acc,
        'test_acc': test_acc,
        'val_improvement': final_val_acc - initial_val_acc,
        'initial_steps': initial_steps,
        'final_steps': final_steps,
        'total_budget': k,
        'initial_samples': k_prime,
        'final_samples': len(final_subset)
    }
    
    return results

def run_multiple_experiments(num_experiments: int = 3, methods: List[str] = None) -> pd.DataFrame:
    """
    Run multiple experiments with different seeds and methods
    
    Args:
        num_experiments: Number of experiments to run (will generate this many seeds)
        methods: List of coreset methods to test
    
    Returns:
        DataFrame with all results
    """
    # Generate seeds based on number of experiments
    seeds = [i * 1000000 + 123456 for i in range(num_experiments)]
    
    if methods is None:
        methods = ["sensitivity", "uniform"]
    
    all_results = []
    
    for method in methods:
        print(f"\n{'='*80}")
        print(f"Testing method: {method.upper()}")
        print(f"{'='*80}")
        
        for i, seed in enumerate(seeds):
            print(f"\nRun {i+1}/{len(seeds)} for {method}")
            try:
                result = run_single_experiment(seed, method)
                all_results.append(result)
                print(f"✓ Completed run {i+1} for {method}")
            except Exception as e:
                print(f"✗ Error in run {i+1} for {method}: {e}")
                continue
    
    # Convert to DataFrame
    df = pd.DataFrame(all_results)
    return df

def print_results_table(df: pd.DataFrame, metadata: dict):
    """Print a pretty formatted results table with metadata"""
    print(f"\n{'='*100}")
    print("EXPERIMENT RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Group by method and calculate statistics
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        print(f"\n{method.upper()} SAMPLING:")
        print(f"{'='*60}")
        print(f"{'Seed':<10} {'Init Val':<10} {'Final Val':<10} {'Test Acc':<10} {'Improvement':<12}")
        print(f"{'='*60}")
        
        for _, row in method_df.iterrows():
            print(f"{row['seed']:<10} {row['initial_val_acc']:<10.2f} {row['final_val_acc']:<10.2f} "
                  f"{row['test_acc']:<10.2f} {row['val_improvement']:<12.2f}")
        
        # Calculate medians
        median_init = method_df['initial_val_acc'].median()
        median_final = method_df['final_val_acc'].median()
        median_test = method_df['test_acc'].median()
        median_improvement = method_df['val_improvement'].median()
        
        std_init = method_df['initial_val_acc'].std()
        std_final = method_df['final_val_acc'].std()
        std_test = method_df['test_acc'].std()
        std_improvement = method_df['val_improvement'].std()
        
        print(f"{'='*60}")
        print(f"{'MEDIAN':<10} {median_init:<10.2f} {median_final:<10.2f} {median_test:<10.2f} {median_improvement:<12.2f}")
        print(f"{'STD':<10} {std_init:<10.2f} {std_final:<10.2f} {std_test:<10.2f} {std_improvement:<12.2f}")
    
    # Overall comparison
    print(f"\n{'='*100}")
    print("OVERALL COMPARISON")
    print(f"{'='*100}")
    # Print metadata here
    print("Experiment Metadata:")
    for k, v in metadata.items():
        print(f"  {k}: {v}")
    print(f"{'='*100}")
    print(f"{'Method':<15} {'Median Test Acc':<15} {'Std Test Acc':<15} {'Median Improvement':<15}")
    print(f"{'='*100}")
    
    for method in df['method'].unique():
        method_df = df[df['method'] == method]
        median_test = method_df['test_acc'].median()
        std_test = method_df['test_acc'].std()
        median_improvement = method_df['val_improvement'].median()
        print(f"{method:<15} {median_test:<15.2f} {std_test:<15.2f} {median_improvement:<15.2f}")

def save_results_to_csv(df: pd.DataFrame, filename: str, metadata: dict):
    """Save results to CSV file with metadata as commented lines at the top"""
    with open(filename, 'w') as f:
        for k, v in metadata.items():
            f.write(f"# {k}: {v}\n")
        df.to_csv(f, index=False)
    print(f"\nResults saved to {filename}")

def main():
    """Main function to run all experiments"""
    parser = argparse.ArgumentParser(description="Run multiple active learning experiments")
    parser.add_argument("--num_experiments", type=int, default=3, 
                       help="Number of experiments to run (default: 3)")
    parser.add_argument("--methods", nargs="+", default=["sensitivity", "uniform"],
                       help="Coreset methods to test (default: sensitivity uniform)")
    parser.add_argument("--k", type=int, default=1024, help="Coreset size (default: 1024)")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs (default: 10)")
    args = parser.parse_args()
    
    print("Starting multiple active learning experiments...")
    print(f"Number of experiments: {args.num_experiments}")
    print(f"Methods to test: {args.methods}")
    print("Testing sensitivity vs uniform sampling with multiple seeds")
    
    # Run experiments
    df = run_multiple_experiments(num_experiments=args.num_experiments, methods=args.methods)
    
    # Metadata for output
    k_prime = int(args.k >> 2)
    dt_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"mnist_active_learning_{dt_str}.csv"
    metadata = {
        'dataset': 'mnist',
        'methods': ', '.join(args.methods),
        'num_epochs': args.epochs,
        'coreset_size': args.k,
        'initial_sample_size': k_prime,
        'num_experiments': args.num_experiments,
        'timestamp': dt_str
    }
    
    # Print results table with metadata
    print_results_table(df, metadata)
    
    # Save to CSV with metadata
    save_results_to_csv(df, filename, metadata)
    
    print("\nAll experiments completed!")

if __name__ == "__main__":
    main() 