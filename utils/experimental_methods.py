"""
Helpers for the next-phase local-equalization and hard-case evaluation branch.
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from skimage import exposure
from skimage.filters import gaussian

from utils.filters import make_butterworth_homomorphic
from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline


def resolve_existing_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(candidate_paths[0])


def load_hard_case_manifest(manifest_path=os.path.join("configs", "hard_case_manifest.json")):
    with open(manifest_path, "r", encoding="utf-8") as manifest_file:
        return json.load(manifest_file)


def load_resized_rgb(image_path, max_dimension=MAX_DIMENSION):
    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size

    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized_size = (int(round(width * scale)), int(round(height * scale)))
        rgb = rgb.resize(resized_size, Image.Resampling.LANCZOS)

    return np.array(rgb, dtype=np.uint8)


def rgb_to_grayscale_uint8(rgb_array):
    return np.array(Image.fromarray(rgb_array, mode="RGB").convert("L"), dtype=np.uint8)


def load_resized_grayscale(image_path, max_dimension=MAX_DIMENSION):
    return rgb_to_grayscale_uint8(load_resized_rgb(image_path, max_dimension=max_dimension))


def normalize_to_uint8(image_array):
    array = np.asarray(image_array, dtype=np.float64)
    min_value = float(array.min())
    max_value = float(array.max())

    if max_value - min_value < 1e-10:
        return np.zeros_like(array, dtype=np.uint8)

    scaled = 255.0 * (array - min_value) / (max_value - min_value)
    return np.clip(scaled, 0, 255).astype(np.uint8)


def equalize_local_histogram(image_array, tile_size):
    image_float = image_array.astype(np.float64) / 255.0
    equalized = exposure.equalize_adapthist(
        image_float,
        kernel_size=(tile_size, tile_size),
        clip_limit=1.0,
        nbins=256,
    )
    return np.clip(255.0 * equalized, 0, 255).astype(np.uint8)


def equalize_clahe(image_array, tile_size, clip_limit):
    image_float = image_array.astype(np.float64) / 255.0
    equalized = exposure.equalize_adapthist(
        image_float,
        kernel_size=(tile_size, tile_size),
        clip_limit=clip_limit,
        nbins=256,
    )
    return np.clip(255.0 * equalized, 0, 255).astype(np.uint8)


def apply_sauvola_binarization(image_uint8, window_size=25, k=0.2):
    """
    Sauvola (2000) local adaptive thresholding. Standard classical baseline for
    document-binarization preprocessing.

    Returns a uint8 RGB image where ink pixels are black (0) and paper is white
    (255), broadcast to 3 channels so downstream consumers (TrOCR) can take it
    as RGB without a separate code path.
    """
    from skimage.filters import threshold_sauvola

    # skimage's threshold_sauvola defaults its `r` (dynamic-range divisor)
    # from the dtype range, which becomes 1.0 for float64 input. Pass r=128
    # explicitly so the threshold is computed on the standard 0-255 scale.
    image_float = image_uint8.astype(np.float64)
    threshold_map = threshold_sauvola(image_float, window_size=window_size, k=k, r=128.0)
    binary_mask = image_float >= threshold_map
    binary_uint8 = (binary_mask.astype(np.uint8)) * 255
    return np.stack([binary_uint8] * 3, axis=-1)


def apply_high_boost(image_uint8, sigma=1.2, amount=0.65):
    image_float = image_uint8.astype(np.float64) / 255.0
    blurred = gaussian(image_float, sigma=sigma, preserve_range=True)
    boosted = image_float + amount * (image_float - blurred)
    boosted = np.clip(boosted, 0, 1)
    return np.clip(255.0 * boosted, 0, 255).astype(np.uint8)


def apply_synthetic_homomorphic_baseline(image_array):
    image_normalized = image_array.astype(np.float64) / 255.0
    log_image = np.log1p(image_normalized)
    transformed = np.fft.fftshift(np.fft.fft2(log_image))

    rows, cols = image_array.shape
    homomorphic_filter = make_butterworth_homomorphic(
        rows,
        cols,
        d0=180,
        gamma_l=0.2,
        gamma_h=1.1,
        order=4,
    )
    filtered_frequency = homomorphic_filter * transformed
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(filtered_frequency)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)
    return normalize_to_uint8(filtered)


def apply_real_hard_case_baseline(image_array, case_id):
    return apply_regular_showcase_pipeline(image_array.astype(np.float64), case_id)["final"]


def iter_local_method_specs():
    specs = []
    for tile_size in [16, 32, 64]:
        specs.append(
            {
                "id": f"ahe_t{tile_size}",
                "family": "ahe",
                "display_name": f"AHE {tile_size}x{tile_size}",
                "tile_size": tile_size,
            }
        )

    for tile_size in [16, 32, 64]:
        for clip_limit in [0.01, 0.03]:
            clip_tag = str(clip_limit).replace(".", "")
            specs.append(
                {
                    "id": f"clahe_t{tile_size}_c{clip_tag}",
                    "family": "clahe",
                    "display_name": f"CLAHE {tile_size}x{tile_size} clip={clip_limit:.2f}",
                    "tile_size": tile_size,
                    "clip_limit": clip_limit,
                }
            )

    return specs


def get_local_method_spec(method_id):
    for method_spec in iter_local_method_specs():
        if method_spec["id"] == method_id:
            return method_spec
    raise KeyError(method_id)


def apply_local_method(image_array, method_spec):
    if method_spec["family"] == "ahe":
        return equalize_local_histogram(image_array, tile_size=method_spec["tile_size"])
    return equalize_clahe(
        image_array,
        tile_size=method_spec["tile_size"],
        clip_limit=method_spec["clip_limit"],
    )


def resolve_crop_box(image_shape, crop_rel=None):
    rows, cols = image_shape[:2]
    if crop_rel is None:
        width = int(round(cols * 0.35))
        height = int(round(rows * 0.35))
        start_x = max(0, (cols - width) // 2)
        start_y = max(0, (rows - height) // 2)
        return start_x, start_y, width, height

    start_x = int(round(crop_rel[0] * cols))
    start_y = int(round(crop_rel[1] * rows))
    width = int(round(crop_rel[2] * cols))
    height = int(round(crop_rel[3] * rows))

    start_x = max(0, min(start_x, cols - 1))
    start_y = max(0, min(start_y, rows - 1))
    width = max(1, min(width, cols - start_x))
    height = max(1, min(height, rows - start_y))
    return start_x, start_y, width, height


def crop_image(image_array, crop_box):
    start_x, start_y, width, height = crop_box
    return image_array[start_y:start_y + height, start_x:start_x + width]
