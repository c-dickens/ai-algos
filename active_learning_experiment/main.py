#!/usr/bin/env python3
"""
Active Learning Classification Experiment
Reproducing results from Section 5.2.2 of the clustering-based sensitivity sampling paper.

This script implements:
1. Coreset sampling using Algorithm 1 from the paper
2. Margin sampling for comparison
3. Entropy sampling for comparison
4. Training on MNIST, Fashion-MNIST, and CIFAR-10 datasets
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import os
import time
from typing import List, Tuple, Dict, Any
import argparse
from tqdm import tqdm
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from coreset import UniformRandomCoreset, SensitivityCoreset

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MLP(nn.Module):
    """128-unit MLP for MNIST and Fashion-MNIST"""
    def __init__(self, input_dim: int, num_classes: int):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)
        self.dropout = nn.Dropout(0.5)
        
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return the 128-D hidden representation (before dropout & logits).

        Args
        ----
        x : (B, 1, 28, 28) or (B, 784) tensor on *any* device

        Returns
        -------
        emb : (B, 128) tensor on the **same** device as `x`
        """
        x = x.view(x.size(0), -1)
        emb = F.relu(self.fc1(x))
        return emb


class CNN(nn.Module):
    """3-conv-layer CNN for CIFAR-10"""
    def __init__(self, num_classes: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def load_dataset(dataset_name: str, data_dir: str = "./data") -> Tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]:
    """Load and prepare dataset"""
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name == "mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == "fashion_mnist":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        train_dataset = datasets.FashionMNIST(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.FashionMNIST(data_dir, train=False, download=True, transform=transform)
        
    elif dataset_name == "cifar10":
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_dataset = datasets.CIFAR10(data_dir, train=True, download=True, transform=transform)
        test_dataset = datasets.CIFAR10(data_dir, train=False, download=True, transform=transform)
        
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return train_dataset, test_dataset

def train_model(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, 
                device: torch.device, epochs: int = 10) -> Tuple[List[float], List[float], int]:
    """Train model and return training/validation accuracies and total training steps"""
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    train_accs = []
    val_accs = []
    total_steps = 0
    
    for epoch in range(epochs):
        # Training
        model.train()
        correct = 0
        total = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            total_steps += target.size(0)  # Count individual samples, not batches
        
        train_acc = 100. * correct / total
        train_accs.append(train_acc)
        
        # Validation
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)
        
        val_acc = 100. * correct / total
        val_accs.append(val_acc)
        
        print(f'Epoch {epoch+1}/{epochs}: Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%')
    
    return train_accs, val_accs, total_steps

def compute_losses(model: nn.Module, dataset: torch.utils.data.Dataset, 
                  device: torch.device, batch_size: int = 32) -> np.ndarray:
    """Compute per-example losses for all data points"""
    model.eval()
    losses = []
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.cross_entropy(output, target, reduction='none')
            losses.extend(loss.cpu().numpy())
    
    return np.array(losses)

def coreset_sampling(dataset: torch.utils.data.Dataset, k: int, k_prime: int, 
                    used_indices: np.ndarray = None, seed: int = 293870, batch_size: int = 32,
                    coreset_method: str = "uniform", k_clusters:int=None, pilot_size:int=None, model:nn.Module=None) -> np.ndarray:
    """Coreset sampling using UniformRandomCoreset from coreset.py"""
    if used_indices is None:
        used_indices = np.array([])
    
    # Create available indices (all indices minus used ones)
    all_indices = np.arange(len(dataset))
    available_indices = np.setdiff1d(all_indices, used_indices)
    
    # Create subset of available data
    available_dataset = Subset(dataset, available_indices)
    
    # Sample from available data
    if coreset_method == "uniform":
        print(f"Using uniform coreset with coreset_size={k - k_prime}")
        coreset = UniformRandomCoreset(available_dataset, k - k_prime, seed=seed)
    elif coreset_method == "sensitivity":
        kwargs = {
            "coreset_size": k - k_prime,
            "k_clusters": k_clusters, 
            "pilot_size": pilot_size, 
            "model": model,
            "seed" : seed}
        print(f"Using sensitivity coreset with k_clusters={k_clusters} and pilot_size={pilot_size}")
        coreset = SensitivityCoreset(available_dataset, **kwargs)
    else:
        raise ValueError(f"Unknown coreset method: {coreset_method}")
    selected_available_indices, _ = coreset.select_coreset(batch_size=batch_size)
    
    # Map back to original dataset indices
    selected_indices = available_indices[selected_available_indices]
    print(f"Final 5 selected indices: {selected_indices[-5:]}")
    return selected_indices

def run_active_learning_experiment(dataset_name: str, sampling_method: str, 
                                 device: torch.device, k: int = None) -> float:
    """Run a single active learning experiment"""
    print(f"\nRunning {sampling_method} sampling on {dataset_name}")
    
    # Load dataset
    train_dataset, test_dataset = load_dataset(dataset_name)
    
    # Set budget if not provided
    if k is None:
        k = len(train_dataset)
    
    k_prime = int(0.2 * k)
    print(f"Total budget: {k}, Initial samples: {k_prime}")
    
    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_subset, val_subset = random_split(train_dataset, [train_size, val_size])
    
    # Step 1: Randomly sample k_prime points
    indices = torch.randperm(len(train_subset))[:k_prime]
    initial_subset = Subset(train_subset, indices)
    
    # Step 2: Train initial model
    if dataset_name in ["mnist", "fashion_mnist"]:
        model = MLP(input_dim=28*28, num_classes=10)
    else:  # cifar10
        model = CNN(num_classes=10)
    
    train_loader = DataLoader(initial_subset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=32, shuffle=False)
    
    print("Training initial model...")
    train_accs, val_accs, total_steps = train_model(model, train_loader, val_loader, device, epochs=10)
    initial_val_acc = val_accs[-1]
    print(f"Initial validation accuracy: {initial_val_acc:.2f}%")
    
    # Step 3: Select remaining points based on sampling method
    print(f"Selecting remaining {k-k_prime} points using {sampling_method}...")
    
    if sampling_method == "coreset":
        # Compute losses for all training data
        losses = compute_losses(model, train_subset, device)
        selected_indices = coreset_sampling(train_subset, k, k_prime, used_indices=indices.numpy(), batch_size=train_loader.batch_size)
        
    elif sampling_method == "margin":
        selected_indices = margin_sampling(model, train_subset, device, k, k_prime)
        
    elif sampling_method == "entropy":
        selected_indices = entropy_sampling(model, train_subset, device, k, k_prime)
        
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    # Combine initial and selected indices
    all_selected_indices = np.concatenate([indices.numpy(), selected_indices])
    final_subset = Subset(train_subset, all_selected_indices)
    
    # Step 4: Retrain model from scratch on selected data
    print("Retraining model on selected data...")
    model = type(model)(**{k: v for k, v in model.__dict__.items() if not k.startswith('_')})
    if dataset_name in ["mnist", "fashion_mnist"]:
        model = MLP(input_dim=28*28, num_classes=10)
    else:  # cifar10
        model = CNN(num_classes=10)
    
    train_loader = DataLoader(final_subset, batch_size=32, shuffle=True)
    train_accs, val_accs, total_steps = train_model(model, train_loader, val_loader, device, epochs=10)
    final_val_acc = val_accs[-1]
    
    print(f"Final validation accuracy: {final_val_acc:.2f}%")
    return final_val_acc

def main():
    parser = argparse.ArgumentParser(description="Active Learning Classification Experiment")
    parser.add_argument("--datasets", nargs="+", default=["mnist", "fashion_mnist", "cifar10"],
                       help="Datasets to run experiments on")
    parser.add_argument("--methods", nargs="+", default=["coreset", "margin", "entropy"],
                       help="Sampling methods to compare")
    parser.add_argument("--budget", type=int, default=None,
                       help="Total budget (default: full dataset size)")
    parser.add_argument("--runs", type=int, default=3,
                       help="Number of runs per method for averaging")
    args = parser.parse_args()
    
    device = get_device()
    print(f"Using device: {device}")
    
    results = {}
    
    for dataset_name in args.datasets:
        print(f"\n{'='*50}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*50}")
        
        results[dataset_name] = {}
        
        for method in args.methods:
            print(f"\nMethod: {method}")
            accuracies = []
            
            for run in range(args.runs):
                print(f"Run {run+1}/{args.runs}")
                try:
                    acc = run_active_learning_experiment(dataset_name, method, device, args.budget)
                    accuracies.append(acc)
                except Exception as e:
                    print(f"Error in run {run+1}: {e}")
                    continue
            
            if accuracies:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)
                results[dataset_name][method] = {
                    'mean': mean_acc,
                    'std': std_acc,
                    'runs': accuracies
                }
                print(f"{method}: {mean_acc:.2f}% ± {std_acc:.2f}%")
    
    # Plot results
    plot_results(results, args.datasets, args.methods)
    
    # Save results
    save_results(results, args.datasets, args.methods)

def plot_results(results: Dict, datasets: List[str], methods: List[str]):
    """Plot comparison of methods across datasets"""
    fig, axes = plt.subplots(1, len(datasets), figsize=(5*len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]
    
    colors = ['blue', 'red', 'green']
    
    for i, dataset in enumerate(datasets):
        ax = axes[i]
        
        means = []
        stds = []
        labels = []
        
        for method in methods:
            if method in results[dataset]:
                means.append(results[dataset][method]['mean'])
                stds.append(results[dataset][method]['std'])
                labels.append(method)
        
        if means:
            x = np.arange(len(means))
            bars = ax.bar(x, means, yerr=stds, capsize=5, color=colors[:len(means)])
            ax.set_xlabel('Sampling Method')
            ax.set_ylabel('Validation Accuracy (%)')
            ax.set_title(f'{dataset.upper()}')
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=45)
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean in zip(bars, means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{mean:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('active_learning_results.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_results(results: Dict, datasets: List[str], methods: List[str]):
    """Save results to file"""
    with open('active_learning_results.txt', 'w') as f:
        f.write("Active Learning Classification Results\n")
        f.write("="*50 + "\n\n")
        
        for dataset in datasets:
            f.write(f"Dataset: {dataset.upper()}\n")
            f.write("-"*30 + "\n")
            
            for method in methods:
                if method in results[dataset]:
                    result = results[dataset][method]
                    f.write(f"{method}: {result['mean']:.2f}% ± {result['std']:.2f}%\n")
                    f.write(f"  Runs: {result['runs']}\n")
            
            f.write("\n")

if __name__ == "__main__":
    main() 