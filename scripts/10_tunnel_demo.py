"""
Script 10: Blind homomorphic filtering on the tunnel image.
This script uses fixed blind parameter sets so it runs quickly and
produces both a balanced restoration and a stronger presentation demo.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic, make_gaussian_homomorphic


def normalize_percentile_to_uint8(image_array, low_percentile=1.0, high_percentile=99.0, display_gamma=1.0):
    low_value = np.percentile(image_array, low_percentile)
    high_value = np.percentile(image_array, high_percentile)

    if high_value - low_value < 1e-10:
        return np.zeros_like(image_array, dtype=np.uint8)

    normalized = (image_array - low_value) / (high_value - low_value)
    normalized = np.clip(normalized, 0, 1)
    normalized = normalized ** display_gamma
    return (255.0 * normalized).astype(np.uint8)


def raw_to_uint8(image_array):
    clipped = np.clip(image_array, 0, 1)
    return (255.0 * clipped).astype(np.uint8)


def post_enhance(image_uint8, gamma=0.75, low_percentile=1.0, high_percentile=99.5):
    image_float = image_uint8.astype(np.float64) / 255.0
    low_value = np.percentile(image_float, low_percentile)
    high_value = np.percentile(image_float, high_percentile)

    if high_value - low_value < 1e-10:
        return image_uint8.copy()

    stretched = (image_float - low_value) / (high_value - low_value)
    stretched = np.clip(stretched, 0, 1)
    stretched = stretched ** gamma
    return (255.0 * stretched).astype(np.uint8)


def plot_tunnel_sweep(
    original_image,
    crop_bounds,
    sweep_images,
    sweep_titles,
    output_path,
    figure_title,
):
    crop_row_start, crop_row_end, crop_col_start, crop_col_end = crop_bounds
    columns = len(sweep_images) + 1
    figure, axes = plt.subplots(2, columns, figsize=(4.5 * columns, 9))

    axes[0, 0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Tunnel")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    axes[1, 0].imshow(
        original_image[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 0].set_title("Crop: Original")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")

    for index, (sweep_image, sweep_title) in enumerate(zip(sweep_images, sweep_titles), start=1):
        axes[0, index].imshow(sweep_image, cmap="gray", vmin=0, vmax=255)
        axes[0, index].set_title(sweep_title)
        axes[0, index].set_xlabel("Column")
        axes[0, index].set_ylabel("Row")

        axes[1, index].imshow(
            sweep_image[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        axes[1, index].set_title(f"Crop: {sweep_title}")
        axes[1, index].set_xlabel("Column")
        axes[1, index].set_ylabel("Row")

    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


def apply_homomorphic(
    image_array,
    gamma_l,
    gamma_h,
    d0,
    filter_type,
    order=2,
    low_percentile=1.0,
    high_percentile=99.0,
    display_gamma=1.0,
):
    image_normalized = image_array / 255.0
    log_image = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = image_array.shape

    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)
    raw_output = raw_to_uint8(filtered)
    stretched_output = normalize_percentile_to_uint8(filtered, low_percentile, high_percentile, display_gamma)

    return raw_output, stretched_output, H


if __name__ == "__main__":
    image_path = "images/tun.jpg"
    print(f"Loading tunnel image: {image_path}")
    image = np.array(Image.open(image_path).convert("L"), dtype=np.float64)
    print(f"  Shape: {image.shape}, range: [{image.min()}, {image.max()}]")

    print("Applying balanced blind filter...")
    balanced_raw, balanced_result, balanced_filter = apply_homomorphic(
        image,
        gamma_l=0.35,
        gamma_h=1.0,
        d0=150,
        filter_type="gaussian",
        order=2,
        low_percentile=1.0,
        high_percentile=99.0,
        display_gamma=1.0,
    )
    print(f"  Balanced raw range: [{balanced_raw.min()}, {balanced_raw.max()}]")
    print(f"  Balanced stretched range: [{balanced_result.min()}, {balanced_result.max()}]")

    print("Applying stronger blind demo filter...")
    aggressive_raw, aggressive_result, aggressive_filter = apply_homomorphic(
        image,
        gamma_l=0.15,
        gamma_h=1.3,
        d0=120,
        filter_type="butterworth",
        order=2,
        low_percentile=1.0,
        high_percentile=99.5,
        display_gamma=0.75,
    )
    print(f"  Aggressive raw range: [{aggressive_raw.min()}, {aggressive_raw.max()}]")
    print(f"  Aggressive stretched range: [{aggressive_result.min()}, {aggressive_result.max()}]")

    print("Saving tunnel demo outputs...")
    os.makedirs("results", exist_ok=True)
    Image.fromarray(balanced_raw).save("results/tun_homomorphic_restored_raw.png")
    Image.fromarray(balanced_result).save("results/tun_homomorphic_restored.png")
    Image.fromarray(aggressive_raw).save("results/tun_homomorphic_restored_aggressive_raw.png")
    Image.fromarray(aggressive_result).save("results/tun_homomorphic_restored_aggressive.png")

    print("Applying post-enhancement to the aggressive blind result...")
    post_enhanced = post_enhance(aggressive_result, gamma=0.72, low_percentile=0.5, high_percentile=99.7)
    Image.fromarray(post_enhanced).save("results/tun_homomorphic_restored_post_enhanced.png")

    crop_row_start = 150
    crop_row_end = 620
    crop_col_start = 20
    crop_col_end = 530

    figure, axes = plt.subplots(2, 3, figsize=(16, 10))

    axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Original Tunnel Image")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    balanced_plot = axes[0, 1].imshow(balanced_filter, cmap="viridis")
    axes[0, 1].set_title("Balanced Filter")
    axes[0, 1].set_xlabel("Frequency Column")
    axes[0, 1].set_ylabel("Frequency Row")
    figure.colorbar(balanced_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

    axes[0, 2].imshow(balanced_result, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Balanced Restoration")
    axes[0, 2].set_xlabel("Column")
    axes[0, 2].set_ylabel("Row")

    axes[1, 0].imshow(
        image[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 0].set_title("Crop: Original")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")

    aggressive_plot = axes[1, 1].imshow(aggressive_filter, cmap="viridis")
    axes[1, 1].set_title("Aggressive Filter")
    axes[1, 1].set_xlabel("Frequency Column")
    axes[1, 1].set_ylabel("Frequency Row")
    figure.colorbar(aggressive_plot, ax=axes[1, 1], fraction=0.046, pad=0.04)

    axes[1, 2].imshow(
        aggressive_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    axes[1, 2].set_title("Crop: Aggressive Restoration")
    axes[1, 2].set_xlabel("Column")
    axes[1, 2].set_ylabel("Row")

    figure.suptitle("Tunnel Image Blind Homomorphic Filtering Demo")
    figure.tight_layout()
    figure.savefig("results/tun_homomorphic_comparison.png", dpi=200)
    plt.close(figure)

    print("Saving raw-vs-stretched comparison...")
    raw_compare_figure, raw_compare_axes = plt.subplots(2, 3, figsize=(16, 10))

    raw_compare_axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    raw_compare_axes[0, 0].set_title("Original Tunnel")
    raw_compare_axes[0, 0].set_xlabel("Column")
    raw_compare_axes[0, 0].set_ylabel("Row")

    raw_compare_axes[0, 1].imshow(balanced_raw, cmap="gray", vmin=0, vmax=255)
    raw_compare_axes[0, 1].set_title("Balanced Raw Output")
    raw_compare_axes[0, 1].set_xlabel("Column")
    raw_compare_axes[0, 1].set_ylabel("Row")

    raw_compare_axes[0, 2].imshow(balanced_result, cmap="gray", vmin=0, vmax=255)
    raw_compare_axes[0, 2].set_title("Balanced Display-Stretched")
    raw_compare_axes[0, 2].set_xlabel("Column")
    raw_compare_axes[0, 2].set_ylabel("Row")

    raw_compare_axes[1, 0].imshow(
        image[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    raw_compare_axes[1, 0].set_title("Crop: Original")
    raw_compare_axes[1, 0].set_xlabel("Column")
    raw_compare_axes[1, 0].set_ylabel("Row")

    raw_compare_axes[1, 1].imshow(
        aggressive_raw[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    raw_compare_axes[1, 1].set_title("Crop: Aggressive Raw Output")
    raw_compare_axes[1, 1].set_xlabel("Column")
    raw_compare_axes[1, 1].set_ylabel("Row")

    raw_compare_axes[1, 2].imshow(
        aggressive_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    raw_compare_axes[1, 2].set_title("Crop: Aggressive Display-Stretched")
    raw_compare_axes[1, 2].set_xlabel("Column")
    raw_compare_axes[1, 2].set_ylabel("Row")

    raw_compare_figure.suptitle("Tunnel Demo: Raw Processed Output vs Display Stretching")
    raw_compare_figure.tight_layout()
    raw_compare_figure.savefig("results/tun_raw_vs_stretched.png", dpi=200)
    plt.close(raw_compare_figure)

    print("Saving post-enhancement comparison...")
    post_figure, post_axes = plt.subplots(2, 3, figsize=(16, 10))

    post_axes[0, 0].imshow(image, cmap="gray", vmin=0, vmax=255)
    post_axes[0, 0].set_title("Original Tunnel")
    post_axes[0, 0].set_xlabel("Column")
    post_axes[0, 0].set_ylabel("Row")

    post_axes[0, 1].imshow(aggressive_raw, cmap="gray", vmin=0, vmax=255)
    post_axes[0, 1].set_title("Blind Raw Output")
    post_axes[0, 1].set_xlabel("Column")
    post_axes[0, 1].set_ylabel("Row")

    post_axes[0, 2].imshow(aggressive_result, cmap="gray", vmin=0, vmax=255)
    post_axes[0, 2].set_title("Blind Display-Stretched")
    post_axes[0, 2].set_xlabel("Column")
    post_axes[0, 2].set_ylabel("Row")

    post_axes[1, 0].imshow(
        image[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    post_axes[1, 0].set_title("Crop: Original")
    post_axes[1, 0].set_xlabel("Column")
    post_axes[1, 0].set_ylabel("Row")

    post_axes[1, 1].imshow(
        aggressive_result[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    post_axes[1, 1].set_title("Crop: Blind Stretched")
    post_axes[1, 1].set_xlabel("Column")
    post_axes[1, 1].set_ylabel("Row")

    post_axes[1, 2].imshow(
        post_enhanced[crop_row_start:crop_row_end, crop_col_start:crop_col_end],
        cmap="gray",
        vmin=0,
        vmax=255,
    )
    post_axes[1, 2].set_title("Crop: Blind + Post Enhancement")
    post_axes[1, 2].set_xlabel("Column")
    post_axes[1, 2].set_ylabel("Row")

    post_figure.suptitle("Tunnel Demo: Blind Result vs Post-Enhanced Blind Result")
    post_figure.tight_layout()
    post_figure.savefig("results/tun_post_enhancement_comparison.png", dpi=200)
    plt.close(post_figure)

    print("Saving full-frame post-enhancement comparison...")
    full_post_figure, full_post_axes = plt.subplots(1, 4, figsize=(20, 5.5))

    full_post_axes[0].imshow(image, cmap="gray", vmin=0, vmax=255)
    full_post_axes[0].set_title("Original Tunnel")
    full_post_axes[0].set_xlabel("Column")
    full_post_axes[0].set_ylabel("Row")

    full_post_axes[1].imshow(aggressive_raw, cmap="gray", vmin=0, vmax=255)
    full_post_axes[1].set_title("Blind Raw Output")
    full_post_axes[1].set_xlabel("Column")
    full_post_axes[1].set_ylabel("Row")

    full_post_axes[2].imshow(aggressive_result, cmap="gray", vmin=0, vmax=255)
    full_post_axes[2].set_title("Blind Display-Stretched")
    full_post_axes[2].set_xlabel("Column")
    full_post_axes[2].set_ylabel("Row")

    full_post_axes[3].imshow(post_enhanced, cmap="gray", vmin=0, vmax=255)
    full_post_axes[3].set_title("Blind + Post Enhancement")
    full_post_axes[3].set_xlabel("Column")
    full_post_axes[3].set_ylabel("Row")

    full_post_figure.suptitle("Tunnel Demo: Full-Frame Comparison of Blind and Post-Enhanced Results")
    full_post_figure.tight_layout()
    full_post_figure.savefig("results/tun_post_enhancement_fullframe_comparison.png", dpi=200)
    plt.close(full_post_figure)

    gamma_l_values = [0.05, 0.10, 0.15, 0.25, 0.35]
    gamma_h_values = [1.00, 1.10, 1.30, 1.50, 1.80]

    print("Running tunnel gamma_L sweep...")
    gamma_l_images = []
    gamma_l_titles = []
    for gamma_l_value in gamma_l_values:
        _, gamma_l_result, _ = apply_homomorphic(
            image,
            gamma_l=gamma_l_value,
            gamma_h=1.30,
            d0=120,
            filter_type="butterworth",
            order=2,
            low_percentile=1.0,
            high_percentile=99.5,
            display_gamma=0.75,
        )
        gamma_l_images.append(gamma_l_result)
        gamma_l_titles.append(f"gamma_L={gamma_l_value:.2f}\ngamma_H=1.30")

    plot_tunnel_sweep(
        image,
        (crop_row_start, crop_row_end, crop_col_start, crop_col_end),
        gamma_l_images,
        gamma_l_titles,
        "results/tun_gamma_l_sweep.png",
        "Tunnel Demo: gamma_L Sweep with Fixed gamma_H=1.30",
    )

    print("Running tunnel gamma_H sweep...")
    gamma_h_images = []
    gamma_h_titles = []
    for gamma_h_value in gamma_h_values:
        _, gamma_h_result, _ = apply_homomorphic(
            image,
            gamma_l=0.15,
            gamma_h=gamma_h_value,
            d0=120,
            filter_type="butterworth",
            order=2,
            low_percentile=1.0,
            high_percentile=99.5,
            display_gamma=0.75,
        )
        gamma_h_images.append(gamma_h_result)
        gamma_h_titles.append(f"gamma_L=0.15\ngamma_H={gamma_h_value:.2f}")

    plot_tunnel_sweep(
        image,
        (crop_row_start, crop_row_end, crop_col_start, crop_col_end),
        gamma_h_images,
        gamma_h_titles,
        "results/tun_gamma_h_sweep.png",
        "Tunnel Demo: gamma_H Sweep with Fixed gamma_L=0.15",
    )

    print("Done! Tunnel demo saved.")
