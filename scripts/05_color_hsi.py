"""
Script 05: Apply homomorphic filtering to color images in HSI space.
Converts an RGB image to HSI, processes only the intensity channel with
homomorphic filtering, and then reconstructs the color image.
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

    scaled = (image_array - min_value) / (max_value - min_value)
    scaled = np.clip(scaled, 0, 1)
    return (255.0 * scaled).astype(np.uint8)


def rgb_to_hsi(rgb_image):
    eps = 1e-10
    rgb_norm = rgb_image / 255.0
    r = rgb_norm[:, :, 0]
    g = rgb_norm[:, :, 1]
    b = rgb_norm[:, :, 2]

    intensity = (r + g + b) / 3.0
    minimum = np.minimum(np.minimum(r, g), b)
    saturation = 1.0 - (3.0 * minimum / (r + g + b + eps))
    saturation = np.clip(saturation, 0, 1)

    numerator = 0.5 * ((r - g) + (r - b))
    denominator = np.sqrt((r - g) ** 2 + (r - b) * (g - b)) + eps
    theta = np.arccos(np.clip(numerator / denominator, -1, 1))
    hue = np.where(b <= g, theta, 2.0 * np.pi - theta)

    return hue, saturation, intensity


def hsi_to_rgb(hue, saturation, intensity):
    eps = 1e-10
    r = np.zeros_like(intensity)
    g = np.zeros_like(intensity)
    b = np.zeros_like(intensity)

    mask_rg = (hue >= 0) & (hue < 2.0 * np.pi / 3.0)
    mask_gb = (hue >= 2.0 * np.pi / 3.0) & (hue < 4.0 * np.pi / 3.0)
    mask_br = (hue >= 4.0 * np.pi / 3.0) & (hue <= 2.0 * np.pi)

    h1 = hue[mask_rg]
    b[mask_rg] = intensity[mask_rg] * (1.0 - saturation[mask_rg])
    r[mask_rg] = intensity[mask_rg] * (
        1.0 + (saturation[mask_rg] * np.cos(h1)) / (np.cos(np.pi / 3.0 - h1) + eps)
    )
    g[mask_rg] = 3.0 * intensity[mask_rg] - (r[mask_rg] + b[mask_rg])

    h2 = hue[mask_gb] - 2.0 * np.pi / 3.0
    r[mask_gb] = intensity[mask_gb] * (1.0 - saturation[mask_gb])
    g[mask_gb] = intensity[mask_gb] * (
        1.0 + (saturation[mask_gb] * np.cos(h2)) / (np.cos(np.pi / 3.0 - h2) + eps)
    )
    b[mask_gb] = 3.0 * intensity[mask_gb] - (r[mask_gb] + g[mask_gb])

    h3 = hue[mask_br] - 4.0 * np.pi / 3.0
    g[mask_br] = intensity[mask_br] * (1.0 - saturation[mask_br])
    b[mask_br] = intensity[mask_br] * (
        1.0 + (saturation[mask_br] * np.cos(h3)) / (np.cos(np.pi / 3.0 - h3) + eps)
    )
    r[mask_br] = 3.0 * intensity[mask_br] - (g[mask_br] + b[mask_br])

    rgb = np.stack([r, g, b], axis=2)
    rgb = np.clip(rgb, 0, 1)
    return (255.0 * rgb).astype(np.uint8)


def apply_homomorphic_to_intensity(intensity_channel, gamma_l, gamma_h, d0, filter_type, order):
    intensity_log = np.log1p(intensity_channel)
    F = np.fft.fftshift(np.fft.fft2(intensity_log))
    rows, cols = intensity_channel.shape

    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    min_value = np.min(filtered)
    max_value = np.max(filtered)
    if max_value - min_value < 1e-10:
        return np.zeros_like(filtered), H

    filtered = (filtered - min_value) / (max_value - min_value)
    return filtered, H


if __name__ == "__main__":
    image_path = "images/photo_1.jpg"
    if not os.path.exists(image_path):
        print(f"Color image not found: {image_path}")
        print("Place a real uneven-lighting photo in images/photo_1.jpg and rerun the script.")
        raise SystemExit(0)

    print(f"Loading color image: {image_path}")
    img = Image.open(image_path).convert("RGB")
    rgb = np.array(img, dtype=np.float64)
    print(f"  Shape: {rgb.shape}, range: [{rgb.min()}, {rgb.max()}]")

    gamma_l = 0.2
    gamma_h = 1.1
    d0 = 180
    filter_type = "butterworth"
    order = 4
    print("HSI homomorphic filtering setup:")
    print(f"  Filter type: {filter_type}")
    print(f"  gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")
    print(f"  Butterworth order: {order}")

    print("Converting RGB image to HSI...")
    hue, saturation, intensity = rgb_to_hsi(rgb)
    print(f"  Hue range: [{hue.min():.4f}, {hue.max():.4f}]")
    print(f"  Saturation range: [{saturation.min():.4f}, {saturation.max():.4f}]")
    print(f"  Intensity range: [{intensity.min():.4f}, {intensity.max():.4f}]")

    print("Applying homomorphic filtering to intensity channel...")
    filtered_intensity, filter_visual = apply_homomorphic_to_intensity(
        intensity,
        gamma_l,
        gamma_h,
        d0,
        filter_type,
        order,
    )
    print(f"  Filtered intensity range: [{filtered_intensity.min():.4f}, {filtered_intensity.max():.4f}]")

    print("Reconstructing RGB image from filtered HSI channels...")
    rgb_restored = hsi_to_rgb(hue, saturation, filtered_intensity)
    print(f"  Restored RGB range: [{rgb_restored.min()}, {rgb_restored.max()}]")

    print("Saving color comparison figures...")
    os.makedirs("results", exist_ok=True)
    Image.fromarray(rgb_restored).save("results/color_hsi_restored.png")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(rgb.astype(np.uint8))
    axes[0, 0].set_title("Original RGB Image")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    intensity_plot = axes[0, 1].imshow(intensity, cmap="gray")
    axes[0, 1].set_title("Original Intensity Channel")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")
    figure.colorbar(intensity_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

    filter_plot = axes[1, 0].imshow(filter_visual, cmap="viridis")
    axes[1, 0].set_title("Homomorphic Filter")
    axes[1, 0].set_xlabel("Frequency Column")
    axes[1, 0].set_ylabel("Frequency Row")
    figure.colorbar(filter_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(rgb_restored)
    axes[1, 1].set_title("Restored RGB Image")
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Row")

    figure.suptitle("HSI Color Homomorphic Filtering")
    figure.tight_layout()
    figure.savefig("results/color_hsi_overview.png", dpi=200)
    plt.close(figure)

    print("Done! Color processing results saved.")
