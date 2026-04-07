"""
Script 06: Compute quantitative metrics for the synthetic test case.
Uses the original cameraman image as reference and compares it against
the synthetic corrupted image and the homomorphic filtering result.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


if __name__ == "__main__":
    reference_path = "images/cameraman.tif"
    corrupted_path = "images/synthetic_uneven.png"
    restored_path = "results/homomorphic_restored.png"

    print("Loading images for evaluation...")
    reference = np.array(Image.open(reference_path).convert("L"), dtype=np.float64)
    corrupted = np.array(Image.open(corrupted_path).convert("L"), dtype=np.float64)
    restored = np.array(Image.open(restored_path).convert("L"), dtype=np.float64)
    print(f"  Reference shape: {reference.shape}")
    print(f"  Corrupted shape: {corrupted.shape}")
    print(f"  Restored shape: {restored.shape}")

    print("Checking shape consistency...")
    # TODO: make sure all images have the same shape before computing metrics

    print("Computing PSNR...")
    # TODO: compute mean squared error and PSNR manually
    # TODO: report PSNR for corrupted vs reference and restored vs reference

    print("Computing SSIM...")
    # TODO: compute SSIM, either manually or using scikit-image if needed later

    print("Saving metric summary figure...")
    os.makedirs("results", exist_ok=True)
    # TODO: create a figure showing reference, corrupted, restored, and metric values

    print("Done! Metrics saved.")
