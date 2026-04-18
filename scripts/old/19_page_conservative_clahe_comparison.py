"""
Script 19: CLAHE comparison built around the conservative page-specific
homomorphic setting.
This script compares grayscale, conservative HF + brightness, CLAHE-only,
conservative HF + brightness -> CLAHE, and CLAHE -> conservative HF +
brightness for the page image.
"""
import os
import sys

import numpy as np
from PIL import Image
from skimage import exposure

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_row
from utils.filters import make_gaussian_homomorphic


MAX_DIMENSION = 1600
IMAGE_PATH = os.path.join("images", "page.jpeg")
CONSERVATIVE_GAMMA_L = 0.25
CONSERVATIVE_GAMMA_H = 1.00
CONSERVATIVE_D0 = 160
CONSERVATIVE_BRIGHT_GAMMA = 0.84
CLAHE_CLIP_LIMIT = 0.015
CLAHE_KERNEL_SIZE = (128, 128)


def normalize_percentile_to_uint8(image_array, low_percentile=1.0, high_percentile=99.5, display_gamma=0.75):
    low_value = np.percentile(image_array, low_percentile)
    high_value = np.percentile(image_array, high_percentile)

    if high_value - low_value < 1e-10:
        return np.zeros_like(image_array, dtype=np.uint8)

    scaled = (image_array - low_value) / (high_value - low_value)
    scaled = np.clip(scaled, 0, 1)
    scaled = scaled ** display_gamma
    return (255.0 * scaled).astype(np.uint8)


def brighten_uint8(image_uint8, gamma):
    image_float = image_uint8.astype(np.float64) / 255.0
    brightened = np.clip(image_float, 0, 1) ** gamma
    return (255.0 * brightened).astype(np.uint8)


def apply_clahe(image_array, clip_limit=CLAHE_CLIP_LIMIT, kernel_size=CLAHE_KERNEL_SIZE):
    image_normalized = image_array.astype(np.float64) / 255.0
    clahe_result = exposure.equalize_adapthist(
        image_normalized,
        clip_limit=clip_limit,
        kernel_size=kernel_size,
    )
    return (255.0 * np.clip(clahe_result, 0, 1)).astype(np.uint8)


def load_resized_grayscale(image_path, max_dimension=MAX_DIMENSION):
    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size

    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized_size = (int(round(width * scale)), int(round(height * scale)))
        rgb = rgb.resize(resized_size, Image.Resampling.LANCZOS)

    return np.array(rgb.convert("L"), dtype=np.float64)


def apply_conservative_homomorphic(image_array):
    image_normalized = image_array / 255.0
    log_image = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = image_array.shape

    H = make_gaussian_homomorphic(
        rows,
        cols,
        CONSERVATIVE_D0,
        CONSERVATIVE_GAMMA_L,
        CONSERVATIVE_GAMMA_H,
    )
    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    restored = normalize_percentile_to_uint8(filtered)
    return brighten_uint8(restored, CONSERVATIVE_BRIGHT_GAMMA)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running conservative page CLAHE comparison...")
    gray_array = load_resized_grayscale(IMAGE_PATH)
    clahe_result = apply_clahe(gray_array)
    conservative_hf_result = apply_conservative_homomorphic(gray_array)
    clahe_then_conservative_hf = apply_conservative_homomorphic(clahe_result.astype(np.float64))
    conservative_hf_then_clahe = apply_clahe(conservative_hf_result.astype(np.float64))

    print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
    print(f"  CLAHE range: [{clahe_result.min()}, {clahe_result.max()}]")
    print(f"  Conservative HF+bright range: [{conservative_hf_result.min()}, {conservative_hf_result.max()}]")
    print(f"  CLAHE->Conservative HF range: [{clahe_then_conservative_hf.min()}, {clahe_then_conservative_hf.max()}]")
    print(f"  Conservative HF->CLAHE range: [{conservative_hf_then_clahe.min()}, {conservative_hf_then_clahe.max()}]")

    Image.fromarray(clahe_result).save(os.path.join("results", "page_clahe.png"))
    Image.fromarray(conservative_hf_result).save(os.path.join("results", "page_conservative_homomorphic_bright.png"))
    Image.fromarray(clahe_then_conservative_hf).save(os.path.join("results", "page_clahe_then_conservative_hf_bright.png"))
    Image.fromarray(conservative_hf_then_clahe).save(os.path.join("results", "page_conservative_hf_bright_then_clahe.png"))

    save_comparison_row(
        [
            gray_array,
            conservative_hf_result,
            clahe_result,
            conservative_hf_then_clahe,
            clahe_then_conservative_hf,
        ],
        [
            "Page: Grayscale",
            "Page: Conservative HF + Bright",
            "Page: CLAHE",
            "Page: Conservative HF -> CLAHE",
            "Page: CLAHE -> Conservative HF",
        ],
        os.path.join("results", "page_conservative_clahe_comparison.png"),
    )

    print("Done! Conservative page CLAHE comparison saved.")
