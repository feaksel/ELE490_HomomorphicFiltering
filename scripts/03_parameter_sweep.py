"""
Script 03: Sweep homomorphic filter parameters and compare results.
Runs the grayscale homomorphic pipeline for multiple gamma_L, gamma_H,
and D0 values, then saves comparison grids for visual inspection.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


if __name__ == "__main__":
    image_path = "images/rice.png"
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    gamma_l_values = [0.3, 0.5, 0.7, 0.9]
    gamma_h_values = [1.2, 1.5, 1.8, 2.0, 2.5]
    d0_values = [10, 20, 30, 50, 80]
    filter_type = "butterworth"
    print("Parameter sweep setup:")
    print(f"  gamma_L values: {gamma_l_values}")
    print(f"  gamma_H values: {gamma_h_values}")
    print(f"  D0 values: {d0_values}")
    print(f"  Filter type: {filter_type}")

    print("Preparing logarithm and Fourier transform...")
    # TODO: compute log image and shifted DFT once before the sweep
    # f_log = ...
    # F = ...

    print("Beginning parameter sweep...")
    os.makedirs("results", exist_ok=True)

    # TODO: loop over parameter combinations
    # TODO: create filter for each setting
    # TODO: apply inverse transform and save individual outputs if helpful
    # TODO: build comparison figures with titles showing parameter values

    print("Done! Parameter sweep results saved.")
