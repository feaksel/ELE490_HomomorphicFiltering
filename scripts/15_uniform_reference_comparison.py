"""
Script 15: Compare homomorphic filtering against uniform-lighting references.
This script uses the current non-uniform images together with the matching
uniform-lighting examples to show whether the homomorphic result moves closer
to the more evenly illuminated scene.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_gaussian_homomorphic


MAX_DIMENSION = 1600
REFERENCE_PAIRS = [
    ("cardboard", "cardboard.jpg", "carboard_uniform.jpg"),
    ("markers", "markers.jpg", "markers_uniform.jpg"),
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

    return np.array(rgb.convert("L"), dtype=np.float64)


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
    return brightened


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running uniform-reference comparison...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")
    overview_figure, overview_axes = plt.subplots(len(REFERENCE_PAIRS), 3, figsize=(15, 4 * len(REFERENCE_PAIRS)))
    if len(REFERENCE_PAIRS) == 1:
        overview_axes = np.array([overview_axes])

    for row_index, (label, source_name, reference_name) in enumerate(REFERENCE_PAIRS):
        source_gray = load_resized_grayscale(os.path.join("images", source_name))
        reference_gray = load_resized_grayscale(os.path.join("images", reference_name))
        source_hf = apply_standard_homomorphic(source_gray)

        print(f"Processed pair: {label}")
        print(f"  Source grayscale shape: {source_gray.shape}")
        print(f"  Reference grayscale shape: {reference_gray.shape}")

        per_image_figure, per_image_axes = plt.subplots(1, 3, figsize=(15, 5))

        per_image_axes[0].imshow(source_gray, cmap="gray", vmin=0, vmax=255)
        per_image_axes[0].set_title(f"{label}: Original grayscale")
        per_image_axes[0].set_xlabel("Column")
        per_image_axes[0].set_ylabel("Row")

        per_image_axes[1].imshow(source_hf, cmap="gray", vmin=0, vmax=255)
        per_image_axes[1].set_title(f"{label}: Standard HF + Brightness")
        per_image_axes[1].set_xlabel("Column")
        per_image_axes[1].set_ylabel("Row")

        per_image_axes[2].imshow(reference_gray, cmap="gray", vmin=0, vmax=255)
        per_image_axes[2].set_title(f"{label}: Uniform-lighting reference")
        per_image_axes[2].set_xlabel("Column")
        per_image_axes[2].set_ylabel("Row")

        per_image_figure.suptitle(f"{label}: Homomorphic Filtering vs Uniform-Lighting Reference")
        per_image_figure.tight_layout()
        per_image_figure.savefig(os.path.join("results", f"{label}_uniform_reference_comparison.png"), dpi=200)
        plt.close(per_image_figure)

        overview_axes[row_index, 0].imshow(source_gray, cmap="gray", vmin=0, vmax=255)
        overview_axes[row_index, 0].set_title(f"{label}: Original grayscale")
        overview_axes[row_index, 0].set_xlabel("Column")
        overview_axes[row_index, 0].set_ylabel("Row")

        overview_axes[row_index, 1].imshow(source_hf, cmap="gray", vmin=0, vmax=255)
        overview_axes[row_index, 1].set_title(f"{label}: Standard HF + Brightness")
        overview_axes[row_index, 1].set_xlabel("Column")
        overview_axes[row_index, 1].set_ylabel("Row")

        overview_axes[row_index, 2].imshow(reference_gray, cmap="gray", vmin=0, vmax=255)
        overview_axes[row_index, 2].set_title(f"{label}: Uniform-lighting reference")
        overview_axes[row_index, 2].set_xlabel("Column")
        overview_axes[row_index, 2].set_ylabel("Row")

    overview_figure.suptitle("Uniform-Lighting Reference Comparison for the Current Photo Set")
    overview_figure.tight_layout()
    overview_figure.savefig("results/uniform_reference_comparison_overview.png", dpi=200)
    plt.close(overview_figure)

    print("Done! Uniform-reference comparison saved.")
