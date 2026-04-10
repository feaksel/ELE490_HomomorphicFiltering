"""
Script 09: Blind homomorphic filtering on the flashlight image.
This script converts the flashlight scene to grayscale, applies a
balanced blind filter and a slightly stronger variant, and saves
comparison figures to the results folder.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


def normalize_to_uint8(image_array):
    min_value = np.min(image_array)
    max_value = np.max(image_array)

    if max_value - min_value < 1e-10:
        return np.zeros_like(image_array, dtype=np.uint8)

    scaled = 255.0 * (image_array - min_value) / (max_value - min_value)
    return scaled.astype(np.uint8)


def apply_homomorphic(image_array, gamma_l, gamma_h, d0, filter_type, order=2):
    image_normalized = image_array / 255.0
    log_image = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = image_array.shape

    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    return normalize_to_uint8(filtered), H


if __name__ == "__main__":
    image_path = "images/flashlight.jpeg"
    print(f"Loading flashlight image: {image_path}")
    rgb = Image.open(image_path).convert("RGB")
    gray = rgb.convert("L")
    gray_array = np.array(gray, dtype=np.float64)
    print("  RGB image converted to grayscale for blind grayscale filtering")
    print(f"  Shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")

    balanced_filter_type = "gaussian"
    balanced_gamma_l = 0.4
    balanced_gamma_h = 1.0
    balanced_d0 = 180

    aggressive_filter_type = "butterworth"
    aggressive_gamma_l = 0.3
    aggressive_gamma_h = 1.2
    aggressive_d0 = 180
    aggressive_order = 2

    print("Running balanced blind filtering...")
    balanced_result, balanced_filter = apply_homomorphic(
        gray_array,
        balanced_gamma_l,
        balanced_gamma_h,
        balanced_d0,
        balanced_filter_type,
        2,
    )
    print(f"  Balanced output range: [{balanced_result.min()}, {balanced_result.max()}]")

    print("Running stronger blind filtering...")
    aggressive_result, aggressive_filter = apply_homomorphic(
        gray_array,
        aggressive_gamma_l,
        aggressive_gamma_h,
        aggressive_d0,
        aggressive_filter_type,
        aggressive_order,
    )
    print(f"  Aggressive output range: [{aggressive_result.min()}, {aggressive_result.max()}]")

    print("Saving flashlight demo figures...")
    os.makedirs("results", exist_ok=True)
    Image.fromarray(balanced_result).save("results/flashlight_homomorphic_restored.png")
    Image.fromarray(aggressive_result).save("results/flashlight_homomorphic_restored_aggressive.png")

    crop_row_start = 480
    crop_row_end = 1140
    crop_col_start = 640
    crop_col_end = 1560

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Grayscale")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    balanced_plot = axes[0, 1].imshow(balanced_filter, cmap="viridis")
    axes[0, 1].set_title("Balanced Filter")
    axes[0, 1].set_xlabel("Frequency Column")
    axes[0, 1].set_ylabel("Frequency Row")
    figure.colorbar(balanced_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(balanced_result, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Balanced Restoration")
    axes[0, 2].set_xlabel("Column")
    axes[0, 2].set_ylabel("Row")

    axes[1, 0].imshow(
        gray_array[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 0].set_title("Crop: Original")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")

    aggressive_plot = axes[1, 1].imshow(aggressive_filter, cmap="viridis")
    axes[1, 1].set_title("Aggressive Filter")
    axes[1, 1].set_xlabel("Frequency Column")
    axes[1, 1].set_ylabel("Frequency Row")
    figure.colorbar(aggressive_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(
        aggressive_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 2].set_title("Crop: Aggressive Restoration")
    axes[1, 2].set_xlabel("Column")
    axes[1, 2].set_ylabel("Row")

    figure.suptitle("Flashlight Image Blind Homomorphic Filtering Demo")
    figure.tight_layout()
    figure.savefig("results/flashlight_homomorphic_comparison.png", dpi=200)
    plt.close(figure)

    print("Done! Flashlight demo saved.")
