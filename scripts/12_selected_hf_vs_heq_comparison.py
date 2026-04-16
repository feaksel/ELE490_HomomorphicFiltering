"""
Script 12: Brightened HF comparison on selected real images.
This script converts selected color images to grayscale, applies the
current standard blind homomorphic setting with the presentation
brightness lift, and saves one combined comparison figure.
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
    ("markers.jpg", "markers"),
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
    return brighten_uint8(restored)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running selected real-image grayscale vs brightened homomorphic comparison...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")
    figure, axes = plt.subplots(len(SHOWCASE_IMAGES), 2, figsize=(10, 4 * len(SHOWCASE_IMAGES)))
    if len(SHOWCASE_IMAGES) == 1:
        axes = np.array([axes])

    for row_index, (image_name, base_name) in enumerate(SHOWCASE_IMAGES):
        image_path = os.path.join("images", image_name)

        print(f"Loading image: {image_path}")
        gray_array = load_resized_grayscale(image_path)
        hf_result = apply_standard_homomorphic(gray_array)

        print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
        print(f"  HF+bright range: [{hf_result.min()}, {hf_result.max()}]")

        axes[row_index, 0].imshow(gray_array, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 0].set_title(f"{base_name}: Grayscale")
        axes[row_index, 0].set_xlabel("Column")
        axes[row_index, 0].set_ylabel("Row")

        axes[row_index, 1].imshow(hf_result, cmap="gray", vmin=0, vmax=255)
        axes[row_index, 1].set_title(f"{base_name}: Standard HF + Brightness")
        axes[row_index, 1].set_xlabel("Column")
        axes[row_index, 1].set_ylabel("Row")

    figure.suptitle("Selected Real Images: Grayscale vs Standard Homomorphic Filtering + Brightness")
    figure.tight_layout()
    figure.savefig("results/selected_real_images_hf_bright_overview.png", dpi=200)
    plt.close(figure)

    print("Done! Selected brightened HF comparison saved.")
