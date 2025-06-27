# Active Learning Classification Experiment

This directory contains an implementation to reproduce the results from Section 5.2.2 of the paper "Data-Efficient Learning via Clustering-Based Sensitivity Sampling: Foundation Models and Beyond" (https://arxiv.org/pdf/2402.17327).

## Overview

The experiment compares three active learning sampling methods:
1. **Coreset sampling** - Using Algorithm 1 from the paper (clustering-based sensitivity sampling)
2. **Margin sampling** - Selecting points with smallest margin between top-2 predictions
3. **Entropy sampling** - Selecting points with highest entropy

## Experiment Design

For each dataset (MNIST, Fashion-MNIST, CIFAR-10):

1. Fix a total labeled budget k (default: full training split)
2. Randomly sample k_prime = int(0.2 * k) points
3. Train a model M on them for 10 epochs
4. Use the per-example cross-entropy losses predicted by M as scores inside Algorithm 1 to select the remaining k - k_prime points
5. Retrain the same architecture from scratch on the entire selected set of k points for 10 epochs
6. Evaluate on the standard validation split and record accuracy

## Model Architectures

- **MNIST/Fashion-MNIST**: 128-unit MLP (one hidden layer)
- **CIFAR-10**: 3-conv-layer CNN with max pooling

## Requirements

```bash
pip install -r requirements.txt
```

## Usage

### Quick Test

First, run a simple test to verify everything works:

```bash
cd active_learning_experiment
python test_simple.py
```

This will run a small experiment on MNIST with 1000 total points to verify the implementation.

### Full Experiment

Run the complete experiment comparing all three methods:

```bash
python main.py
```

### Custom Options

```bash
# Run only on specific datasets
python main.py --datasets mnist fashion_mnist

# Run only specific methods
python main.py --methods coreset margin

# Set custom budget (default: full dataset)
python main.py --budget 5000

# Run multiple times for averaging
python main.py --runs 5
```

## Output

The experiment will generate:
1. **Console output** showing progress and results
2. **`active_learning_results.png`** - Bar plot comparing methods across datasets
3. **`active_learning_results.txt`** - Detailed numerical results

## Expected Results

Based on the paper, you should expect:
- Coreset sampling to perform competitively with or better than margin/entropy sampling
- All methods to show improvement over random sampling
- Results to vary by dataset, with coreset showing particular strength on complex datasets

## Implementation Details

### Coreset Sampling (Algorithm 1)

1. **Clustering**: Use k-means clustering on loss values
2. **Sensitivity Sampling**: Sample from each cluster proportionally to loss values
3. **Diversity**: Ensures coverage across different loss regions

### Key Features

- **Device Agnostic**: Automatically uses MPS (Apple Silicon), CUDA, or CPU
- **Reproducible**: Fixed random seeds for consistent results
- **Modular**: Easy to extend with new sampling methods or datasets
- **Self-contained**: Downloads datasets automatically

## Troubleshooting

### Common Issues

1. **Out of Memory**: Reduce batch size or budget
2. **Slow Training**: Use GPU if available
3. **Import Errors**: Ensure all requirements are installed

### Performance Tips

- Use GPU acceleration when available
- For faster testing, reduce epochs or budget
- The experiment can take several hours on CPU for full datasets

## File Structure

```
active_learning_experiment/
├── main.py              # Main experiment script
├── test_simple.py       # Simple test script
├── requirements.txt     # Dependencies
├── README.md           # This file
└── data/               # Downloaded datasets (created automatically)
```

## References

- Original paper: https://arxiv.org/pdf/2402.17327
- Section 5.2.2: Classification Tasks
- Algorithm 1: Clustering-based sensitivity sampling 