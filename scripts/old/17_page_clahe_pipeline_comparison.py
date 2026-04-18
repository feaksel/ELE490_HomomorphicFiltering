"""
Script 17: Page image comparison with CLAHE before and after homomorphic
filtering.
This script focuses on page.jpeg as a document-style real-life example and
compares grayscale, CLAHE-only, standard HF + brightness, CLAHE -> HF +
brightness, and HF + brightness -> CLAHE.
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


def apply_clahe(image_array, clip_limit=0.015, kernel_size=(128, 128)):
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


def apply_standard_homomorphic(image_array, brighten=False):
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
    if brighten:
        return brighten_uint8(restored)
    return restored


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running page CLAHE / HF pipeline comparison...")
    gray_array = load_resized_grayscale(IMAGE_PATH)
    clahe_result = apply_clahe(gray_array)
    hf_bright_result = apply_standard_homomorphic(gray_array, brighten=True)
    clahe_then_hf_bright_result = apply_standard_homomorphic(clahe_result.astype(np.float64), brighten=True)
    hf_bright_then_clahe_result = apply_clahe(hf_bright_result.astype(np.float64))

    print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
    print(f"  CLAHE range: [{clahe_result.min()}, {clahe_result.max()}]")
    print(f"  HF+bright range: [{hf_bright_result.min()}, {hf_bright_result.max()}]")
    print(f"  CLAHE->HF+bright range: [{clahe_then_hf_bright_result.min()}, {clahe_then_hf_bright_result.max()}]")
    print(f"  HF+bright->CLAHE range: [{hf_bright_then_clahe_result.min()}, {hf_bright_then_clahe_result.max()}]")

    Image.fromarray(clahe_result).save(os.path.join("results", "page_clahe.png"))
    Image.fromarray(clahe_then_hf_bright_result).save(os.path.join("results", "page_clahe_then_hf_bright.png"))
    Image.fromarray(hf_bright_then_clahe_result).save(os.path.join("results", "page_hf_bright_then_clahe.png"))

    save_comparison_row(
        [
            gray_array,
            clahe_result,
            hf_bright_result,
            clahe_then_hf_bright_result,
            hf_bright_then_clahe_result,
        ],
        [
            "Page: Grayscale",
            "Page: CLAHE",
            "Page: Standard HF + Bright",
            "Page: CLAHE -> HF + Bright",
            "Page: HF + Bright -> CLAHE",
        ],
        os.path.join("results", "clahe_page_pipeline_comparison.png"),
    )

    print("Done! Page CLAHE / HF comparison saved.")
