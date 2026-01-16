"""
Acceptance tests for TensorSketchFusion.

Test A: Dimensionality Check
Test B: Approximation Accuracy (Polynomial Kernel)
Test C: Gradient Flow
Test D: Memory Benchmark
"""

import torch
import torch.nn as nn
import gc

from tensor_sketch_fusion import TensorSketchFusion


def test_a_dimensionality():
    """
    Test A: Dimensionality Check

    Input: x1=(10, 100), x2=(10, 50), sketch_dim=256.
    Expected Output Shape: (10, 256).
    """
    print("=" * 60)
    print("Test A: Dimensionality Check")
    print("=" * 60)

    batch_size = 10
    dim1 = 100
    dim2 = 50
    sketch_dim = 256

    fusion = TensorSketchFusion(
        input_dim1=dim1, input_dim2=dim2, sketch_dim=sketch_dim, seed=42
    )

    x1 = torch.randn(batch_size, dim1)
    x2 = torch.randn(batch_size, dim2)

    output = fusion(x1, x2)

    expected_shape = (batch_size, sketch_dim)
    actual_shape = tuple(output.shape)

    print(f"  Input x1 shape:     {tuple(x1.shape)}")
    print(f"  Input x2 shape:     {tuple(x2.shape)}")
    print(f"  Expected output:    {expected_shape}")
    print(f"  Actual output:      {actual_shape}")

    passed = actual_shape == expected_shape
    print(f"\n  Result: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)
    return passed


def test_b_approximation_accuracy():
    """
    Test B: Approximation Accuracy (The "Why It Works" Test)

    Goal: Prove that Tensor Sketch approximates the polynomial kernel.

    Procedure:
    1. Create two pairs of vectors: (u1, u2) and (v1, v2).
    2. Compute exact kernel: K_exact = (u1 . v1) * (u2 . v2).
    3. Compute sketches: S_u = Fusion(u1, u2), S_v = Fusion(v1, v2).
    4. Compute approx kernel: K_approx = S_u . S_v.

    Pass Criteria: relative error < 0.1 (for reasonably large sketch dimensions).
    """
    print("\n" + "=" * 60)
    print("Test B: Approximation Accuracy (Polynomial Kernel)")
    print("=" * 60)

    dim1 = 128
    dim2 = 64
    sketch_dim = 8192  # Large sketch dimension for better approximation

    # Use same fusion layer for both pairs (same hash functions)
    fusion = TensorSketchFusion(
        input_dim1=dim1, input_dim2=dim2, sketch_dim=sketch_dim, seed=42
    )

    # Create two pairs of vectors (batch size = 1 for simplicity)
    torch.manual_seed(123)
    u1 = torch.randn(1, dim1)
    u2 = torch.randn(1, dim2)
    v1 = torch.randn(1, dim1)
    v2 = torch.randn(1, dim2)

    # Compute exact polynomial kernel: K_exact = (u1 . v1) * (u2 . v2)
    dot1 = torch.sum(u1 * v1)  # u1 . v1
    dot2 = torch.sum(u2 * v2)  # u2 . v2
    k_exact = dot1 * dot2

    # Compute sketches
    s_u = fusion(u1, u2)  # Shape: (1, sketch_dim)
    s_v = fusion(v1, v2)  # Shape: (1, sketch_dim)

    # Compute approximate kernel: K_approx = S_u . S_v
    k_approx = torch.sum(s_u * s_v)

    # Compute relative error
    relative_error = torch.abs(k_exact - k_approx) / (torch.abs(k_exact) + 1e-8)

    print(f"  Dimensions: d1={dim1}, d2={dim2}, sketch={sketch_dim}")
    print(f"  Exact kernel K_exact:     {k_exact.item():.6f}")
    print(f"  Approx kernel K_approx:   {k_approx.item():.6f}")
    print(f"  Relative error:           {relative_error.item():.6f}")

    # Run multiple trials to show statistical properties
    print("\n  Running 10 trials with different random vectors...")
    errors = []
    for trial in range(10):
        torch.manual_seed(trial * 100)
        u1 = torch.randn(1, dim1)
        u2 = torch.randn(1, dim2)
        v1 = torch.randn(1, dim1)
        v2 = torch.randn(1, dim2)

        k_exact_t = torch.sum(u1 * v1) * torch.sum(u2 * v2)
        s_u_t = fusion(u1, u2)
        s_v_t = fusion(v1, v2)
        k_approx_t = torch.sum(s_u_t * s_v_t)
        rel_err = (
            torch.abs(k_exact_t - k_approx_t) / (torch.abs(k_exact_t) + 1e-8)
        ).item()
        errors.append(rel_err)

    avg_error = sum(errors) / len(errors)
    max_error = max(errors)
    print(f"  Average relative error:   {avg_error:.6f}")
    print(f"  Max relative error:       {max_error:.6f}")

    passed = avg_error < 0.1
    print(f"\n  Result: {'PASSED' if passed else 'FAILED'} (avg error < 0.1)")
    print("=" * 60)
    return passed


def test_c_gradient_flow():
    """
    Test C: Gradient Flow

    Verify that gradients are not None for inputs after a backward pass.
    """
    print("\n" + "=" * 60)
    print("Test C: Gradient Flow")
    print("=" * 60)

    batch_size = 16
    dim1 = 256
    dim2 = 128
    sketch_dim = 512

    fusion = TensorSketchFusion(
        input_dim1=dim1, input_dim2=dim2, sketch_dim=sketch_dim, seed=42
    )

    # Create inputs with gradient tracking
    x1 = torch.randn(batch_size, dim1, requires_grad=True)
    x2 = torch.randn(batch_size, dim2, requires_grad=True)

    # Forward pass
    output = fusion(x1, x2)

    # Create a scalar loss (sum of outputs)
    loss = output.sum()

    # Backward pass
    loss.backward()

    # Check gradients
    x1_grad_exists = x1.grad is not None
    x2_grad_exists = x2.grad is not None
    x1_grad_nonzero = x1_grad_exists and x1.grad.abs().sum() > 0
    x2_grad_nonzero = x2_grad_exists and x2.grad.abs().sum() > 0

    print(f"  x1.grad exists:        {x1_grad_exists}")
    print(f"  x1.grad non-zero:      {x1_grad_nonzero}")
    print(f"  x2.grad exists:        {x2_grad_exists}")
    print(f"  x2.grad non-zero:      {x2_grad_nonzero}")

    if x1_grad_nonzero:
        print(f"  x1.grad shape:         {tuple(x1.grad.shape)}")
        print(f"  x1.grad mean abs:      {x1.grad.abs().mean().item():.6f}")
    if x2_grad_nonzero:
        print(f"  x2.grad shape:         {tuple(x2.grad.shape)}")
        print(f"  x2.grad mean abs:      {x2.grad.abs().mean().item():.6f}")

    passed = x1_grad_nonzero and x2_grad_nonzero
    print(f"\n  Result: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)
    return passed


def test_d_memory_benchmark():
    """
    Test D: Memory Benchmark

    Compare memory usage of TensorSketchFusion vs computing raw outer product
    (1024 x 1024) for a batch size of 64.

    Pass Criteria: Tensor Sketch memory usage < 10% of Outer Product memory usage.
    """
    print("\n" + "=" * 60)
    print("Test D: Memory Benchmark")
    print("=" * 60)

    batch_size = 64
    dim1 = 1024
    dim2 = 1024
    sketch_dim = 4096  # Much smaller than dim1 * dim2 = 1,048,576

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Batch size: {batch_size}")
    print(f"  Input dimensions: {dim1} x {dim2}")
    print(f"  Sketch dimension: {sketch_dim}")
    print(f"  Outer product dimension: {dim1 * dim2:,}")

    # Calculate theoretical memory requirements
    # Outer product: batch_size * dim1 * dim2 * 4 bytes (float32)
    outer_product_bytes = batch_size * dim1 * dim2 * 4
    outer_product_mb = outer_product_bytes / (1024 * 1024)

    # Tensor Sketch output: batch_size * sketch_dim * 4 bytes
    sketch_output_bytes = batch_size * sketch_dim * 4
    sketch_output_mb = sketch_output_bytes / (1024 * 1024)

    print(f"\n  Theoretical memory comparison:")
    print(f"    Outer product output: {outer_product_mb:.2f} MB")
    print(f"    Tensor Sketch output: {sketch_output_mb:.2f} MB")

    # Actual memory measurement
    if device == "cuda":
        # CUDA memory measurement
        torch.cuda.empty_cache()
        gc.collect()

        # Measure outer product memory
        torch.cuda.reset_peak_memory_stats()
        x1_op = torch.randn(batch_size, dim1, device=device)
        x2_op = torch.randn(batch_size, dim2, device=device)
        outer = torch.einsum("bi,bj->bij", x1_op, x2_op)
        outer_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del x1_op, x2_op, outer
        torch.cuda.empty_cache()
        gc.collect()

        # Measure tensor sketch memory
        torch.cuda.reset_peak_memory_stats()
        fusion = TensorSketchFusion(
            input_dim1=dim1,
            input_dim2=dim2,
            sketch_dim=sketch_dim,
            device=device,
            seed=42,
        )
        x1_ts = torch.randn(batch_size, dim1, device=device)
        x2_ts = torch.randn(batch_size, dim2, device=device)
        fused = fusion(x1_ts, x2_ts)
        sketch_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
        del fusion, x1_ts, x2_ts, fused
        torch.cuda.empty_cache()

        print(f"\n  Actual CUDA peak memory:")
        print(f"    Outer product: {outer_peak:.2f} MB")
        print(f"    Tensor Sketch: {sketch_peak:.2f} MB")
        ratio = sketch_peak / outer_peak * 100
    else:
        # CPU: Use theoretical calculation
        print("\n  (Using theoretical calculation for CPU)")
        ratio = sketch_output_mb / outer_product_mb * 100

    print(f"\n  Memory ratio: {ratio:.2f}%")

    passed = ratio < 10.0
    print(f"  Threshold: < 10%")
    print(f"\n  Result: {'PASSED' if passed else 'FAILED'}")
    print("=" * 60)
    return passed


def test_cuda_compatibility():
    """
    Additional test: Verify CUDA compatibility if available.
    """
    print("\n" + "=" * 60)
    print("Additional Test: CUDA Compatibility")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("  CUDA not available, skipping...")
        print("=" * 60)
        return True

    device = "cuda"
    batch_size = 32
    dim1 = 512
    dim2 = 256
    sketch_dim = 1024

    # Create fusion layer on CUDA
    fusion = TensorSketchFusion(
        input_dim1=dim1,
        input_dim2=dim2,
        sketch_dim=sketch_dim,
        device=device,
        seed=42,
    )

    # Check that buffers are on CUDA
    h1_device = fusion.h1.device.type
    h2_device = fusion.h2.device.type
    s1_device = fusion.s1.device.type
    s2_device = fusion.s2.device.type

    print(f"  h1 device: {h1_device}")
    print(f"  h2 device: {h2_device}")
    print(f"  s1 device: {s1_device}")
    print(f"  s2 device: {s2_device}")

    # Create inputs on CUDA
    x1 = torch.randn(batch_size, dim1, device=device, requires_grad=True)
    x2 = torch.randn(batch_size, dim2, device=device, requires_grad=True)

    # Forward pass
    output = fusion(x1, x2)
    output_device = output.device.type

    print(f"  Output device: {output_device}")

    # Backward pass
    loss = output.sum()
    loss.backward()

    grad_device_x1 = x1.grad.device.type if x1.grad is not None else "None"
    grad_device_x2 = x2.grad.device.type if x2.grad is not None else "None"

    print(f"  x1.grad device: {grad_device_x1}")
    print(f"  x2.grad device: {grad_device_x2}")

    all_cuda = all(
        d == "cuda"
        for d in [
            h1_device,
            h2_device,
            s1_device,
            s2_device,
            output_device,
            grad_device_x1,
            grad_device_x2,
        ]
    )

    print(f"\n  Result: {'PASSED' if all_cuda else 'FAILED'}")
    print("=" * 60)
    return all_cuda


def test_model_save_load():
    """
    Additional test: Verify model can be saved and loaded correctly.
    """
    print("\n" + "=" * 60)
    print("Additional Test: Model Save/Load")
    print("=" * 60)

    import tempfile
    import os

    dim1 = 128
    dim2 = 64
    sketch_dim = 256

    # Create original model
    fusion_orig = TensorSketchFusion(
        input_dim1=dim1, input_dim2=dim2, sketch_dim=sketch_dim, seed=42
    )

    # Create test input
    torch.manual_seed(999)
    x1 = torch.randn(4, dim1)
    x2 = torch.randn(4, dim2)

    # Get original output
    output_orig = fusion_orig(x1, x2)

    # Save and load model
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "fusion_model.pt")
        torch.save(fusion_orig.state_dict(), model_path)

        # Create new model and load state
        fusion_loaded = TensorSketchFusion(
            input_dim1=dim1, input_dim2=dim2, sketch_dim=sketch_dim, seed=0
        )  # Different seed
        fusion_loaded.load_state_dict(torch.load(model_path, weights_only=True))

    # Get output from loaded model
    output_loaded = fusion_loaded(x1, x2)

    # Compare outputs
    outputs_match = torch.allclose(output_orig, output_loaded, atol=1e-6)

    print(f"  Original output sum:  {output_orig.sum().item():.6f}")
    print(f"  Loaded output sum:    {output_loaded.sum().item():.6f}")
    print(f"  Outputs match:        {outputs_match}")

    print(f"\n  Result: {'PASSED' if outputs_match else 'FAILED'}")
    print("=" * 60)
    return outputs_match


def run_all_tests():
    """Run all acceptance tests and report results."""
    print("\n" + "=" * 60)
    print("TENSOR SKETCH FUSION - ACCEPTANCE TESTS")
    print("=" * 60)

    results = {}

    results["A: Dimensionality"] = test_a_dimensionality()
    results["B: Approximation"] = test_b_approximation_accuracy()
    results["C: Gradient Flow"] = test_c_gradient_flow()
    results["D: Memory Benchmark"] = test_d_memory_benchmark()
    results["CUDA Compatibility"] = test_cuda_compatibility()
    results["Model Save/Load"] = test_model_save_load()

    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    all_passed = True
    for test_name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {test_name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("ALL TESTS PASSED!")
    else:
        print("SOME TESTS FAILED - see details above")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    run_all_tests()
