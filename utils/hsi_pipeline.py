"""
HSI color homomorphic-filtering pipeline.

Wraps the regular grayscale showcase pipeline so it operates only on the
intensity channel of an RGB image. Hue and saturation are preserved, then the
processed intensity is recombined to give a color result.

The intensity-channel processing reuses `utils.showcase_pipeline.apply_regular_showcase_pipeline`,
which means the HSI color pipeline inherits the project's accepted real-scene
homomorphic settings (D-002) and the conservative page-specific override
(D-006).
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline


def load_resized_rgb(image_path, max_dimension=MAX_DIMENSION):
    rgb = Image.open(image_path).convert("RGB")
    width, height = rgb.size

    longest_side = max(width, height)
    if longest_side > max_dimension:
        scale = max_dimension / float(longest_side)
        resized_size = (int(round(width * scale)), int(round(height * scale)))
        rgb = rgb.resize(resized_size, Image.Resampling.LANCZOS)

    return np.array(rgb, dtype=np.float64)


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


def apply_hsi_showcase_pipeline(rgb_image_float64, base_name):
    """
    Run the project's accepted homomorphic + tone pipeline on the intensity
    channel of an RGB image. Returns a dict with the recombined `final` color
    image alongside the intermediate intensity arrays for inspection.
    """
    hue, saturation, intensity_01 = rgb_to_hsi(rgb_image_float64)
    intensity_uint_scale = intensity_01 * 255.0

    intensity_pipeline = apply_regular_showcase_pipeline(intensity_uint_scale, base_name)
    final_intensity_01 = intensity_pipeline["final"].astype(np.float64) / 255.0
    final_rgb = hsi_to_rgb(hue, saturation, final_intensity_01)

    return {
        "hue": hue,
        "saturation": saturation,
        "intensity_original": intensity_01,
        "intensity_final": final_intensity_01,
        "final_rgb": final_rgb,
        "title": intensity_pipeline["title"],
        "config": intensity_pipeline["config"],
    }
