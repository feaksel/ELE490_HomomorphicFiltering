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
from matplotlib.patches import Rectangle

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
    f_log = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(f_log))
    rows, cols = image_array.shape

    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    g_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    g = np.expm1(g_log)
    g = np.clip(g, 0, None)

    return normalize_to_uint8(g)


def histogram_equalization(image_array):
    image_uint8 = image_array.astype(np.uint8)
    histogram, _ = np.histogram(image_uint8.flatten(), bins=256, range=(0, 256))
    cdf = histogram.cumsum()

    nonzero_cdf = cdf[cdf > 0]
    cdf_min = nonzero_cdf[0]
    total_pixels = image_uint8.size

    if total_pixels == cdf_min:
        return image_uint8.copy()

    lut = np.round((cdf - cdf_min) * 255.0 / (total_pixels - cdf_min))
    lut = np.clip(lut, 0, 255).astype(np.uint8)

    equalized = lut[image_uint8]
    return equalized


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
    aggressive_gamma_l = 0.1
    aggressive_gamma_h = 1.5
    aggressive_d0 = 80
    aggressive_filter_type = "butterworth"
    aggressive_order = 4
    print("Comparison setup:")
    print(f"  Homomorphic filter type: {filter_type}")
    print(f"  gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")
    print(
        f"  Aggressive blind demo: {aggressive_filter_type}, gamma_L={aggressive_gamma_l}, "
        f"gamma_H={aggressive_gamma_h}, D0={aggressive_d0}, order={aggressive_order}"
    )

    print("Running homomorphic filtering pipeline...")
    homomorphic_result = apply_homomorphic(f, gamma_l, gamma_h, d0, filter_type)
    print(f"  Homomorphic output range: [{homomorphic_result.min()}, {homomorphic_result.max()}]")

    print("Running aggressive blind homomorphic pipeline...")
    aggressive_result = apply_homomorphic(
        f,
        aggressive_gamma_l,
        aggressive_gamma_h,
        aggressive_d0,
        aggressive_filter_type,
        aggressive_order,
    )
    print(f"  Aggressive output range: [{aggressive_result.min()}, {aggressive_result.max()}]")

    print("Running histogram equalization...")
    equalized_result = histogram_equalization(f)
    print(f"  Equalized output range: [{equalized_result.min()}, {equalized_result.max()}]")

    print("Saving comparison figures...")
    os.makedirs("results", exist_ok=True)

    comparison_figure, comparison_axes = plt.subplots(1, 3, figsize=(15, 5))

    comparison_axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
    comparison_axes[0].set_title("Original Rice Image")
    comparison_axes[0].set_xlabel("Column")
    comparison_axes[0].set_ylabel("Row")

    comparison_axes[1].imshow(homomorphic_result, cmap="gray", vmin=0, vmax=255)
    comparison_axes[1].set_title("Homomorphic Filtering")
    comparison_axes[1].set_xlabel("Column")
    comparison_axes[1].set_ylabel("Row")

    comparison_axes[2].imshow(equalized_result, cmap="gray", vmin=0, vmax=255)
    comparison_axes[2].set_title("Histogram Equalization")
    comparison_axes[2].set_xlabel("Column")
    comparison_axes[2].set_ylabel("Row")

    comparison_figure.suptitle("Homomorphic Filtering vs Histogram Equalization")
    comparison_figure.tight_layout()
    comparison_figure.savefig("results/compare_homomorphic_vs_heq.png", dpi=200)
    plt.close(comparison_figure)

    histogram_figure, histogram_axes = plt.subplots(1, 3, figsize=(15, 4))

    histogram_axes[0].hist(f.ravel(), bins=256, range=(0, 255), color="gray")
    histogram_axes[0].set_title("Original Histogram")
    histogram_axes[0].set_xlabel("Intensity")
    histogram_axes[0].set_ylabel("Pixel Count")

    histogram_axes[1].hist(homomorphic_result.ravel(), bins=256, range=(0, 255), color="black")
    histogram_axes[1].set_title("Homomorphic Histogram")
    histogram_axes[1].set_xlabel("Intensity")
    histogram_axes[1].set_ylabel("Pixel Count")

    histogram_axes[2].hist(equalized_result.ravel(), bins=256, range=(0, 255), color="dimgray")
    histogram_axes[2].set_title("Equalized Histogram")
    histogram_axes[2].set_xlabel("Intensity")
    histogram_axes[2].set_ylabel("Pixel Count")

    histogram_figure.suptitle("Histogram Comparison")
    histogram_figure.tight_layout()
    histogram_figure.savefig("results/compare_hists_homomorphic_vs_heq.png", dpi=200)
    plt.close(histogram_figure)

    print("Saving grain-focused aggressive comparison...")
    crop_row_start = 88
    crop_row_end = 176
    crop_col_start = 88
    crop_col_end = 176

    aggressive_figure, aggressive_axes = plt.subplots(2, 4, figsize=(18, 9))

    aggressive_axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 0].add_patch(
        Rectangle(
            (crop_col_start, crop_row_start),
            crop_col_end - crop_col_start,
            crop_row_end - crop_row_start,
            fill=False,
            edgecolor="red",
            linewidth=2,
        )
    )
    aggressive_axes[0, 0].set_title("Original Rice")
    aggressive_axes[0, 0].set_xlabel("Column")
    aggressive_axes[0, 0].set_ylabel("Row")

    aggressive_axes[0, 1].imshow(homomorphic_result, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 1].set_title("Balanced Blind")
    aggressive_axes[0, 1].set_xlabel("Column")
    aggressive_axes[0, 1].set_ylabel("Row")

    aggressive_axes[0, 2].imshow(aggressive_result, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 2].set_title("Aggressive Blind")
    aggressive_axes[0, 2].set_xlabel("Column")
    aggressive_axes[0, 2].set_ylabel("Row")

    aggressive_axes[0, 3].imshow(equalized_result, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 3].set_title("Histogram Equalization")
    aggressive_axes[0, 3].set_xlabel("Column")
    aggressive_axes[0, 3].set_ylabel("Row")

    aggressive_axes[1, 0].imshow(
        f[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    aggressive_axes[1, 0].set_title("Crop: Original")
    aggressive_axes[1, 0].set_xlabel("Column")
    aggressive_axes[1, 0].set_ylabel("Row")

    aggressive_axes[1, 1].imshow(
        homomorphic_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    aggressive_axes[1, 1].set_title("Crop: Balanced Blind")
    aggressive_axes[1, 1].set_xlabel("Column")
    aggressive_axes[1, 1].set_ylabel("Row")

    aggressive_axes[1, 2].imshow(
        aggressive_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    aggressive_axes[1, 2].set_title("Crop: Aggressive Blind")
    aggressive_axes[1, 2].set_xlabel("Column")
    aggressive_axes[1, 2].set_ylabel("Row")

    aggressive_axes[1, 3].imshow(
        equalized_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    aggressive_axes[1, 3].set_title("Crop: Histogram Eq")
    aggressive_axes[1, 3].set_xlabel("Column")
    aggressive_axes[1, 3].set_ylabel("Row")

    aggressive_figure.suptitle("Rice Grain Demo: Balanced vs Aggressive Blind Filtering")
    aggressive_figure.tight_layout()
    aggressive_figure.savefig("results/rice_aggressive_comparison.png", dpi=200)
    plt.close(aggressive_figure)

    Image.fromarray(aggressive_result).save("results/rice_aggressive_blind.png")
    print("  Saved aggressive rice demo to results/rice_aggressive_comparison.png")

    print("Done! Comparison results saved.")
