"""
Script 02: Core homomorphic filtering pipeline for grayscale images.
Implements the full pipeline: ln -> DFT -> H(u, v) filter -> IDFT -> exp
Applies to a single input image and displays before/after results.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


if __name__ == "__main__":
    image_path = "images/synthetic_uneven.png"
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    gamma_l = 0.5
    gamma_h = 1.5
    d0 = 30
    filter_type = "butterworth"
    print(f"  Filter: {filter_type}, gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")

    print("  Step 1: Taking logarithm...")
    # TODO: add a small constant to avoid log(0), then take the logarithm
    # f_log = np.log(f + 1.0)
    print("  Step 1: Logarithm taken")

    print("  Step 2: Computing DFT and centering...")
    # TODO: use np.fft.fft2 and np.fft.fftshift
    # F = ...
    print("  Step 2: DFT computed")

    print("  Step 3: Creating and applying filter...")
    # TODO: create the filter using utils.filters, then multiply with F
    # H = ...
    # G = H * F
    print("  Step 3: Filter applied")

    print("  Step 4: Computing inverse DFT...")
    # TODO: use np.fft.ifftshift and np.fft.ifft2, then take the real part
    # g_log = ...
    print("  Step 4: Inverse DFT computed")

    print("  Step 5: Taking exponential...")
    # TODO: take exp to undo the logarithm
    g = f.copy()
    # g = np.exp(g_log) - 1.0
    print("  Step 5: Exponential taken")

    print("Normalizing result to display range...")
    # TODO: scale output to 0-255 for visualization
    print(f"  Output range: [{g.min():.2f}, {g.max():.2f}]")

    print("Saving figures to results/ ...")
    os.makedirs("results", exist_ok=True)
    # TODO: create a figure with original image, filter visualization, and result
    # TODO: create histograms before and after filtering

    print("Done!")
