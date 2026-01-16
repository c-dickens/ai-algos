"""
Integration test script for TensorSketchFusion.

Simulates a training loop to verify backpropagation works correctly
with the TensorSketchFusion layer.
"""

import torch
import torch.nn as nn

from tensor_sketch_fusion import TensorSketchFusion


def test_backpropagation():
    """
    Verify that gradients flow correctly through TensorSketchFusion.

    Creates synthetic "Image" and "Text" data, passes them through the fusion
    layer and a classifier, computes loss, and verifies gradients are non-zero.
    """
    print("=" * 60)
    print("TensorSketchFusion Backpropagation Test")
    print("=" * 60)

    # Configuration
    batch_size = 32
    image_dim = 1024  # Simulating ResNet-like features
    text_dim = 512  # Simulating BERT-like features
    sketch_dim = 2000
    num_classes = 10

    # Determine device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create the fusion layer
    fusion = TensorSketchFusion(
        input_dim1=image_dim,
        input_dim2=text_dim,
        sketch_dim=sketch_dim,
        device=device,
        seed=42,
    )

    # Create a simple classifier on top
    classifier = nn.Linear(sketch_dim, num_classes).to(device)

    # Generate synthetic data with requires_grad=True to track gradients
    x_image = torch.randn(batch_size, image_dim, device=device, requires_grad=True)
    x_text = torch.randn(batch_size, text_dim, device=device, requires_grad=True)

    # Generate random labels
    labels = torch.randint(0, num_classes, (batch_size,), device=device)

    print(f"\nInput shapes:")
    print(f"  Image features: {x_image.shape}")
    print(f"  Text features:  {x_text.shape}")

    # Forward pass
    fused = fusion(x_image, x_text)
    print(f"  Fused output:   {fused.shape}")

    logits = classifier(fused)
    print(f"  Logits:         {logits.shape}")

    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, labels)
    print(f"\nLoss: {loss.item():.4f}")

    # Backward pass
    loss.backward()

    # Check gradients
    print("\nGradient check:")
    image_grad_ok = x_image.grad is not None and x_image.grad.abs().sum() > 0
    text_grad_ok = x_text.grad is not None and x_text.grad.abs().sum() > 0

    print(f"  x_image.grad exists and non-zero: {image_grad_ok}")
    print(f"  x_text.grad exists and non-zero:  {text_grad_ok}")

    if image_grad_ok:
        print(f"  x_image.grad mean: {x_image.grad.abs().mean().item():.6f}")
        print(f"  x_image.grad max:  {x_image.grad.abs().max().item():.6f}")

    if text_grad_ok:
        print(f"  x_text.grad mean:  {x_text.grad.abs().mean().item():.6f}")
        print(f"  x_text.grad max:   {x_text.grad.abs().max().item():.6f}")

    # Final verdict
    print("\n" + "=" * 60)
    if image_grad_ok and text_grad_ok:
        print("SUCCESS: Gradients flow correctly through TensorSketchFusion!")
    else:
        print("FAILURE: Gradient flow is broken!")
    print("=" * 60)

    return image_grad_ok and text_grad_ok


def test_training_loop():
    """
    Simulate a mini training loop to verify the layer works in practice.
    """
    print("\n" + "=" * 60)
    print("TensorSketchFusion Training Loop Simulation")
    print("=" * 60)

    # Configuration
    batch_size = 32
    image_dim = 1024
    text_dim = 512
    sketch_dim = 2000
    num_classes = 10
    num_epochs = 3
    num_batches = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    # Create model components
    fusion = TensorSketchFusion(
        input_dim1=image_dim,
        input_dim2=text_dim,
        sketch_dim=sketch_dim,
        device=device,
        seed=42,
    )
    classifier = nn.Linear(sketch_dim, num_classes).to(device)

    # Optimizer
    params = list(classifier.parameters())
    optimizer = torch.optim.Adam(params, lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining for {num_epochs} epochs, {num_batches} batches each...")

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for batch_idx in range(num_batches):
            # Generate synthetic batch
            x_image = torch.randn(batch_size, image_dim, device=device)
            x_text = torch.randn(batch_size, text_dim, device=device)
            labels = torch.randint(0, num_classes, (batch_size,), device=device)

            # Forward pass
            optimizer.zero_grad()
            fused = fusion(x_image, x_text)
            logits = classifier(fused)
            loss = criterion(logits, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / num_batches
        print(f"  Epoch {epoch + 1}/{num_epochs}: Avg Loss = {avg_loss:.4f}")

    print("\nSUCCESS: Training loop completed without errors!")
    print("=" * 60)
    return True


if __name__ == "__main__":
    backprop_ok = test_backpropagation()
    training_ok = test_training_loop()

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Backpropagation test: {'PASSED' if backprop_ok else 'FAILED'}")
    print(f"  Training loop test:   {'PASSED' if training_ok else 'FAILED'}")
    print("=" * 60)
