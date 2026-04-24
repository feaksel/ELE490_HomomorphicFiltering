"""
Metric helpers for synthetic-reference and real hard-case evaluation.
"""
from __future__ import annotations

import numpy as np
from skimage.metrics import structural_similarity

from utils.analysis_visuals import compute_frequency_statistics, compute_intensity_statistics


def compute_mse_psnr(reference_image, test_image):
    reference = reference_image.astype(np.float64)
    test = test_image.astype(np.float64)
    mse = float(np.mean((reference - test) ** 2))

    if mse < 1e-12:
        return mse, float("inf")

    psnr = float(10.0 * np.log10((255.0 ** 2) / mse))
    return mse, psnr


def compute_reference_metrics(reference_image, test_image):
    mse, psnr = compute_mse_psnr(reference_image, test_image)
    ssim = float(
        structural_similarity(
            reference_image.astype(np.float64),
            test_image.astype(np.float64),
            data_range=255,
        )
    )
    return {
        "mse": mse,
        "psnr": psnr,
        "ssim": ssim,
    }


def compute_mean_gradient(image_array):
    image_float = image_array.astype(np.float64)
    dx = np.diff(image_float, axis=1)
    dy = np.diff(image_float, axis=0)
    gradient = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
    return float(np.mean(gradient))


def compute_clipped_pixel_ratio(image_array, low_threshold=5, high_threshold=250):
    image_uint8 = np.clip(image_array, 0, 255).astype(np.uint8)
    clipped = (image_uint8 <= low_threshold) | (image_uint8 >= high_threshold)
    return float(np.mean(clipped))


def compute_proxy_metrics(image_array):
    intensity_stats = compute_intensity_statistics(image_array)
    frequency_stats = compute_frequency_statistics(image_array)

    return {
        "mean": intensity_stats["mean"],
        "std": intensity_stats["std"],
        "entropy": intensity_stats["entropy"],
        "spread_1_99": intensity_stats["spread_1_99"],
        "high_frequency_share": frequency_stats["high_frequency_share"],
        "mean_gradient": compute_mean_gradient(image_array),
        "clipped_pixel_ratio": compute_clipped_pixel_ratio(image_array),
    }
