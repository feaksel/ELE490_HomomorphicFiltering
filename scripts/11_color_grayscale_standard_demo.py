"""
Script 11: Batch grayscale homomorphic filtering for the current photo set.
This script converts the active color photos in images/ to grayscale, applies
the current standard blind homomorphic setting, adds a brightness lift for
display, and saves per-image plus overview comparisons.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_gaussian_homomorphic


MAX_DIMENSION = 1600
SHOWCASE_IMAGES = [
    ("cardboard.jpg", "cardboard"),
    ("carboard_uniform.jpg", "carboard_uniform"),
    ("carpet.jpg", "carpet"),
    ("markers.jpg", "markers"),
    ("markers_uniform.jpg", "markers_uniform"),
    ("pillar.jpg", "pillar"),
    ("seat.jpg", "seat"),
]


def normalize_percentile_to_uint8(image_array, low_percentile=1.0, high_percentile=99.5, display_gamma=0.75):
    low_value = np.percentile(image_array, low_percentile)
    high_value = np.percentile(image_array, high_percentile)

    if high_value - low_value < 1e-10:
        return np.zeros_like(image_array, dtype=np.uint8)

    scaled = (image_array - low_value) / (high_value - low_value)
    scaled = np.clip(scaled, 0, 1)
    scaled = scaled ** display_gamma
    return (255.0 * scaled).astype(np.uint8)


def brighten_uint8(image_uint8, gamma=0.72):
    image_float = image_uint8.astype(np.float64) / 255.0
    brightened = np.clip(image_float, 0, 1) ** gamma
    return (255.0 * brightened).astype(np.uint8)


def load_resized_grayscale(image_path, max_dimension=MAX_DIMENSION):
    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size

    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized_size = (int(round(width * scale)), int(round(height * scale)))
        rgb = rgb.resize(resized_size, Image.Resampling.LANCZOS)

    gray = rgb.convert("L")
    return np.array(gray, dtype=np.float64)


def apply_standard_homomorphic(image_array):
    gamma_l = 0.06
    gamma_h = 1.00
    d0 = 320

    image_normalized = image_array / 255.0
    log_image = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = image_array.shape

    H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)
    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    restored = normalize_percentile_to_uint8(filtered)
    brightened = brighten_uint8(restored, gamma=0.72)
    return restored, brightened


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    overview_rows = len(SHOWCASE_IMAGES)
    overview_figure, overview_axes = plt.subplots(overview_rows, 2, figsize=(10, 3.6 * overview_rows))
    if overview_rows == 1:
        overview_axes = np.array([overview_axes])

    print("Running batch grayscale standard homomorphic demo...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")

    for index, (image_name, base_name) in enumerate(SHOWCASE_IMAGES):
        image_path = os.path.join("images", image_name)
        print(f"Loading image: {image_path}")

        gray_array = load_resized_grayscale(image_path)
        restored, brightened = apply_standard_homomorphic(gray_array)

        print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
        print(f"  HF output range: [{restored.min()}, {restored.max()}]")
        print(f"  HF+bright range: [{brightened.min()}, {brightened.max()}]")

        grayscale_output_path = os.path.join("results", f"{base_name}_grayscale.png")
        restored_output_path = os.path.join("results", f"{base_name}_homomorphic_standard.png")
        brightened_output_path = os.path.join("results", f"{base_name}_homomorphic_standard_bright.png")
        comparison_output_path = os.path.join("results", f"{base_name}_grayscale_vs_standard.png")

        Image.fromarray(gray_array.astype(np.uint8)).save(grayscale_output_path)
        Image.fromarray(restored).save(restored_output_path)
        Image.fromarray(brightened).save(brightened_output_path)

        comparison_figure, comparison_axes = plt.subplots(1, 2, figsize=(10, 5))

        comparison_axes[0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
        comparison_axes[0].set_title(f"{base_name}: Grayscale")
        comparison_axes[0].set_xlabel("Column")
        comparison_axes[0].set_ylabel("Row")

        comparison_axes[1].imshow(brightened, cmap="gray", vmin=0, vmax=255)
        comparison_axes[1].set_title(f"{base_name}: Standard HF + Brightness")
        comparison_axes[1].set_xlabel("Column")
        comparison_axes[1].set_ylabel("Row")

        comparison_figure.suptitle(f"{base_name}: Grayscale vs Standard HF")
        comparison_figure.tight_layout()
        comparison_figure.savefig(comparison_output_path, dpi=200)
        plt.close(comparison_figure)

        overview_axes[index, 0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
        overview_axes[index, 0].set_title(f"{base_name}: Grayscale")
        overview_axes[index, 0].set_xlabel("Column")
        overview_axes[index, 0].set_ylabel("Row")

        overview_axes[index, 1].imshow(brightened, cmap="gray", vmin=0, vmax=255)
        overview_axes[index, 1].set_title(f"{base_name}: Standard HF + Brightness")
        overview_axes[index, 1].set_xlabel("Column")
        overview_axes[index, 1].set_ylabel("Row")

    overview_figure.suptitle("Current Photo Set: Grayscale vs Brightened Standard Blind Homomorphic Filtering")
    overview_figure.tight_layout()
    overview_figure.savefig("results/color_grayscale_standard_overview.png", dpi=200)
    plt.close(overview_figure)

    print("Done! Batch grayscale standard demo saved.")
