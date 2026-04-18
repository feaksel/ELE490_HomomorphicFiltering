"""
Shared real-scene showcase pipeline helpers.
"""
import numpy as np
from PIL import Image
from skimage.filters import gaussian

from utils.filters import make_gaussian_homomorphic


MAX_DIMENSION = 1600


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


def load_resized_grayscale(image_path, max_dimension=MAX_DIMENSION):
    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size

    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized_size = (int(round(width * scale)), int(round(height * scale)))
        rgb = rgb.resize(resized_size, Image.Resampling.LANCZOS)

    return np.array(rgb.convert("L"), dtype=np.float64)


def apply_homomorphic(image_array, gamma_l, gamma_h, d0, brighten_gamma):
    image_normalized = image_array / 255.0
    log_image = np.log1p(image_normalized)
    transformed = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = image_array.shape

    homomorphic_filter = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)
    filtered_frequency = homomorphic_filter * transformed
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_frequency)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    restored = normalize_percentile_to_uint8(filtered)
    brightened = brighten_uint8(restored, gamma=brighten_gamma)
    return restored, brightened


def tone_adjust_shadows_highlights(
    image_uint8,
    blur_sigma=32,
    shadow_strength=0.30,
    highlight_strength=0.18,
    shadow_pivot=0.55,
    highlight_pivot=0.45,
):
    image_float = image_uint8.astype(np.float64) / 255.0
    base = gaussian(image_float, sigma=blur_sigma, preserve_range=True)

    shadow_mask = np.clip((shadow_pivot - base) / max(shadow_pivot, 1e-6), 0, 1)
    highlight_mask = np.clip((base - highlight_pivot) / max(1.0 - highlight_pivot, 1e-6), 0, 1)

    shadow_mask = shadow_mask ** 1.4
    highlight_mask = highlight_mask ** 1.4

    adjusted = image_float.copy()
    adjusted = adjusted + shadow_strength * shadow_mask * (1.0 - adjusted)
    adjusted = adjusted - highlight_strength * highlight_mask * adjusted
    adjusted = np.clip(adjusted, 0, 1)

    return normalize_percentile_to_uint8(adjusted, low_percentile=0.5, high_percentile=99.7, display_gamma=0.95)


def apply_regular_showcase_pipeline(image_array, base_name):
    if base_name == "page":
        restored, brightened = apply_homomorphic(
            image_array,
            gamma_l=0.25,
            gamma_h=1.00,
            d0=160,
            brighten_gamma=0.84,
        )
        final_result = tone_adjust_shadows_highlights(
            brightened,
            blur_sigma=28,
            shadow_strength=0.24,
            highlight_strength=0.20,
            shadow_pivot=0.58,
            highlight_pivot=0.42,
        )
        pipeline_title = "Conservative HF + Tone Equalization"
    else:
        restored, brightened = apply_homomorphic(
            image_array,
            gamma_l=0.06,
            gamma_h=1.00,
            d0=320,
            brighten_gamma=0.72,
        )
        final_result = tone_adjust_shadows_highlights(brightened)
        pipeline_title = "Standard HF + Tone Equalization"

    return {
        "restored": restored,
        "brightened": brightened,
        "final": final_result,
        "title": pipeline_title,
    }
