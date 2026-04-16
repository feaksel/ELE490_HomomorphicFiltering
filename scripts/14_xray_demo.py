"""
Script 14: Chest X-ray homomorphic filtering demo.
This script processes the user-added X-ray images with the current
standard blind homomorphic setting, a presentation brightness lift, and
manual histogram equalization, then saves per-image and combined
comparison figures.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_gaussian_homomorphic


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
    return lut[image_uint8]


if __name__ == "__main__":
    image_names = ["xray1.png", "xray2.png"]
    os.makedirs("results", exist_ok=True)

    print("Running chest X-ray homomorphic filtering demo...")
    combined_figure, combined_axes = plt.subplots(len(image_names), 4, figsize=(18, 5 * len(image_names)))

    if len(image_names) == 1:
        combined_axes = np.array([combined_axes])

    for row_index, image_name in enumerate(image_names):
        image_path = os.path.join("images", image_name)
        base_name = os.path.splitext(image_name)[0]

        print(f"Loading image: {image_path}")
        gray_array = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
        print(f"  Shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")

        hf_result, hf_bright = apply_standard_homomorphic(gray_array)
        heq_result = histogram_equalization(gray_array)
        print(f"  HF result range: [{hf_result.min()}, {hf_result.max()}]")
        print(f"  HF+bright range: [{hf_bright.min()}, {hf_bright.max()}]")
        print(f"  Histogram-equalized range: [{heq_result.min()}, {heq_result.max()}]")

        Image.fromarray(gray_array.astype(np.uint8)).save(os.path.join("results", f"{base_name}_grayscale.png"))
        Image.fromarray(hf_result).save(os.path.join("results", f"{base_name}_homomorphic_standard.png"))
        Image.fromarray(hf_bright).save(os.path.join("results", f"{base_name}_homomorphic_standard_bright.png"))
        Image.fromarray(heq_result).save(os.path.join("results", f"{base_name}_hist_eq.png"))

        per_image_figure, per_image_axes = plt.subplots(1, 4, figsize=(18, 5))

        per_image_axes[0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
        per_image_axes[0].set_title(f"{base_name}: Original")
        per_image_axes[0].set_xlabel("Column")
        per_image_axes[0].set_ylabel("Row")

        per_image_axes[1].imshow(hf_result, cmap="gray", vmin=0, vmax=255)
        per_image_axes[1].set_title(f"{base_name}: Standard HF")
        per_image_axes[1].set_xlabel("Column")
        per_image_axes[1].set_ylabel("Row")

        per_image_axes[2].imshow(hf_bright, cmap="gray", vmin=0, vmax=255)
        per_image_axes[2].set_title(f"{base_name}: HF + Bright")
        per_image_axes[2].set_xlabel("Column")
        per_image_axes[2].set_ylabel("Row")

        per_image_axes[3].imshow(heq_result, cmap="gray", vmin=0, vmax=255)
        per_image_axes[3].set_title(f"{base_name}: Histogram Eq")
        per_image_axes[3].set_xlabel("Column")
        per_image_axes[3].set_ylabel("Row")

        per_image_figure.suptitle(f"{base_name}: Chest X-ray Comparison")
        per_image_figure.tight_layout()
        per_image_figure.savefig(os.path.join("results", f"{base_name}_hf_vs_heq.png"), dpi=200)
        plt.close(per_image_figure)

        combined_axes[row_index, 0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
        combined_axes[row_index, 0].set_title(f"{base_name}: Original")
        combined_axes[row_index, 0].set_xlabel("Column")
        combined_axes[row_index, 0].set_ylabel("Row")

        combined_axes[row_index, 1].imshow(hf_result, cmap="gray", vmin=0, vmax=255)
        combined_axes[row_index, 1].set_title(f"{base_name}: Standard HF")
        combined_axes[row_index, 1].set_xlabel("Column")
        combined_axes[row_index, 1].set_ylabel("Row")

        combined_axes[row_index, 2].imshow(hf_bright, cmap="gray", vmin=0, vmax=255)
        combined_axes[row_index, 2].set_title(f"{base_name}: HF + Bright")
        combined_axes[row_index, 2].set_xlabel("Column")
        combined_axes[row_index, 2].set_ylabel("Row")

        combined_axes[row_index, 3].imshow(heq_result, cmap="gray", vmin=0, vmax=255)
        combined_axes[row_index, 3].set_title(f"{base_name}: Histogram Eq")
        combined_axes[row_index, 3].set_xlabel("Column")
        combined_axes[row_index, 3].set_ylabel("Row")

    combined_figure.suptitle("Chest X-rays: Original vs Standard HF vs HF + Bright vs Histogram Equalization")
    combined_figure.tight_layout()
    combined_figure.savefig("results/xray_hf_vs_heq.png", dpi=200)
    plt.close(combined_figure)

    print("Done! Chest X-ray comparison saved.")
