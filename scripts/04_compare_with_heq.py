"""
Script 04: Compare homomorphic filtering with histogram equalization.
Loads one grayscale image, produces a homomorphic filtering result,
produces a histogram equalization result, and saves a comparison figure.
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

    gamma_l = 0.5
    gamma_h = 1.8
    d0 = 30
    filter_type = "gaussian"
    print("Comparison setup:")
    print(f"  Homomorphic filter type: {filter_type}")
    print(f"  gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")

    print("Running homomorphic filtering pipeline...")
    # TODO: reuse the same textbook steps from script 02
    # TODO: compute homomorphic output and normalize to 0-255

    print("Running histogram equalization...")
    # TODO: implement histogram equalization in a simple step-by-step way
    # TODO: avoid extra dependencies if possible

    print("Saving comparison figures...")
    os.makedirs("results", exist_ok=True)
    # TODO: save side-by-side image comparison
    # TODO: save histogram comparison for original, homomorphic, and equalized outputs

    print("Done! Comparison results saved.")
