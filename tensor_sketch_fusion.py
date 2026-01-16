"""
TensorSketchFusion: Efficient fusion of high-dimensional feature vectors using Tensor Sketch.

Implements Compact Bilinear Pooling via Tensor Sketch algorithm (Count Sketch + FFT)
to approximate the outer product of two vectors in O(n log n) time and O(d_out) space.
"""

import torch
import torch.nn as nn
from typing import Optional


class TensorSketchFusion(nn.Module):
    """
    A PyTorch module that fuses two high-dimensional feature vectors into a compact
    representation using the Tensor Sketch algorithm.

    The module approximates the polynomial kernel (outer product) of two input vectors
    using Count Sketch and FFT convolution, enabling efficient high-order interaction
    modeling without exploding memory usage.

    Attributes:
        input_dim1: Dimension of first input vector.
        input_dim2: Dimension of second input vector.
        sketch_dim: Output dimension of the sketch.
    """

    def __init__(
        self,
        input_dim1: int,
        input_dim2: int,
        sketch_dim: int,
        device: str = "cpu",
        seed: Optional[int] = None,
    ):
        """
        Initialize the TensorSketchFusion layer.

        Args:
            input_dim1: Dimension of first input vector.
            input_dim2: Dimension of second input vector.
            sketch_dim: Output dimension of the sketch.
            device: 'cpu' or 'cuda'.
            seed: Optional random seed for reproducible hash generation.
        """
        super().__init__()

        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.sketch_dim = sketch_dim

        # Set seed for reproducibility if provided
        generator = torch.Generator(device=device)
        if seed is not None:
            generator.manual_seed(seed)

        # Pre-compute hash indices h1, h2 in range [0, sketch_dim)
        h1 = torch.randint(
            0, sketch_dim, (input_dim1,), generator=generator, device=device
        )
        h2 = torch.randint(
            0, sketch_dim, (input_dim2,), generator=generator, device=device
        )

        # Pre-compute sign vectors s1, s2 with values {-1, 1}
        # Generate random {0, 1} and convert to {-1, 1}
        s1 = (
            torch.randint(0, 2, (input_dim1,), generator=generator, device=device)
            * 2
            - 1
        ).float()
        s2 = (
            torch.randint(0, 2, (input_dim2,), generator=generator, device=device)
            * 2
            - 1
        ).float()

        # Register as buffers so they move with the model to GPU automatically
        # and are not considered trainable parameters
        self.register_buffer("h1", h1)
        self.register_buffer("h2", h2)
        self.register_buffer("s1", s1)
        self.register_buffer("s2", s2)

    def _sketch(
        self, x: torch.Tensor, h: torch.Tensor, s: torch.Tensor
    ) -> torch.Tensor:
        """
        Perform Count Sketch on input tensor.

        Args:
            x: Input tensor of shape (Batch, InDim).
            h: Hash indices of shape (InDim,) mapping each input dimension
               to a sketch dimension in [0, sketch_dim).
            s: Sign vector of shape (InDim,) with values {-1, 1}.

        Returns:
            Sketched tensor of shape (Batch, sketch_dim).
        """
        batch_size = x.shape[0]
        in_dim = x.shape[1]

        # Apply signs to input: element-wise multiplication
        # x: (Batch, InDim), s: (InDim,) -> signed_x: (Batch, InDim)
        signed_x = x * s.unsqueeze(0)

        # Initialize output tensor
        out = torch.zeros(
            batch_size, self.sketch_dim, dtype=x.dtype, device=x.device
        )

        # Expand hash indices for batch dimension
        # h: (InDim,) -> h_expanded: (Batch, InDim)
        h_expanded = h.unsqueeze(0).expand(batch_size, -1)

        # Use scatter_add_ to accumulate values at hash positions
        # This maps each input dimension to its corresponding sketch dimension
        out.scatter_add_(dim=1, index=h_expanded, src=signed_x)

        return out

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse two input vectors using Tensor Sketch.

        Args:
            x1: Batch of vectors of shape (Batch_Size, input_dim1).
            x2: Batch of vectors of shape (Batch_Size, input_dim2).

        Returns:
            Fused vector of shape (Batch_Size, sketch_dim).
        """
        # Validate input dimensions
        if x1.shape[1] != self.input_dim1:
            raise ValueError(
                f"Expected x1 to have {self.input_dim1} features, got {x1.shape[1]}"
            )
        if x2.shape[1] != self.input_dim2:
            raise ValueError(
                f"Expected x2 to have {self.input_dim2} features, got {x2.shape[1]}"
            )
        if x1.shape[0] != x2.shape[0]:
            raise ValueError(
                f"Batch sizes must match: x1 has {x1.shape[0]}, x2 has {x2.shape[0]}"
            )

        # Step 1: Compute Count Sketches
        sketch1 = self._sketch(x1, self.h1, self.s1)
        sketch2 = self._sketch(x2, self.h2, self.s2)

        # Step 2: Compute FFT convolution
        # Using rfft (Real-to-Complex FFT) for efficiency since inputs are real
        fft1 = torch.fft.rfft(sketch1, n=self.sketch_dim)
        fft2 = torch.fft.rfft(sketch2, n=self.sketch_dim)

        # Element-wise product in frequency domain
        fft_product = fft1 * fft2

        # Step 3: Inverse FFT to get fused vector
        # irfft returns real tensor, n=sketch_dim ensures correct output length
        fused = torch.fft.irfft(fft_product, n=self.sketch_dim)

        return fused

    def extra_repr(self) -> str:
        """Return a string representation of the module's configuration."""
        return (
            f"input_dim1={self.input_dim1}, input_dim2={self.input_dim2}, "
            f"sketch_dim={self.sketch_dim}"
        )
