"""
Script 02: Core homomorphic filtering pipeline for grayscale images.
Implements the full pipeline: ln -> DFT -> H(u, v) filter -> IDFT -> exp
Applies to a single input image and displays before/after results.
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

    scaled = 255.0 * (image_array - min_value) / (max_value - min_value)
    return scaled.astype(np.uint8)


def make_difference_map(reference_image, test_image):
    difference = np.abs(reference_image.astype(np.float64) - test_image.astype(np.float64))
    return normalize_to_uint8(difference)


def apply_homomorphic_filter(image_array, gamma_l, gamma_h, d0, filter_type, order):
    image_normalized = image_array / 255.0
    f_log = np.log1p(image_normalized)
    F = np.fft.fftshift(np.fft.fft2(f_log))
    spectrum = np.log1p(np.abs(F))

    rows, cols = image_array.shape
    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    g_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    g = np.expm1(g_log)
    g = np.clip(g, 0, None)
    g_uint8 = normalize_to_uint8(g)

    return f_log, F, spectrum, H, g_log, g, g_uint8


if __name__ == "__main__":
    image_path = "images/synthetic_uneven.png"
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    gamma_l = 0.2
    gamma_h = 1.1
    d0 = 180
    filter_type = "butterworth"
    order = 4
    print(f"  Filter: {filter_type}, gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}")

    print("  Step 1: Taking logarithm...")
    # Eq. 4.5-4 style preprocessing: log turns multiplication into addition.
    f_log, F, spectrum, H, g_log, g, g_uint8 = apply_homomorphic_filter(
        f, gamma_l, gamma_h, d0, filter_type, order
    )
    print(f"    Log image range: [{f_log.min():.4f}, {f_log.max():.4f}]")
    print("  Step 1: Logarithm taken")

    print("  Step 2: Computing DFT and centering...")
    print(f"    Spectrum range: [{spectrum.min():.4f}, {spectrum.max():.4f}]")
    print("  Step 2: DFT computed")

    print("  Step 3: Creating and applying filter...")
    print(f"    Filter range: [{H.min():.4f}, {H.max():.4f}]")
    print("  Step 3: Filter applied")

    print("  Step 4: Computing inverse DFT...")
    print(f"    Filtered log image range: [{g_log.min():.4f}, {g_log.max():.4f}]")
    print("  Step 4: Inverse DFT computed")

    print("  Step 5: Taking exponential...")
    print(f"    Exponential output range: [{g.min():.4f}, {g.max():.4f}]")
    print("  Step 5: Exponential taken")

    print("Normalizing result to display range...")
    print(f"  Output range after normalization: [{g_uint8.min()}, {g_uint8.max()}]")

    print("Saving figures to results/ ...")
    os.makedirs("results", exist_ok=True)
    Image.fromarray(g_uint8).save("results/homomorphic_restored.png")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(f, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Input Image")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    filter_plot = axes[0, 1].imshow(H, cmap="viridis")
    axes[0, 1].set_title("Homomorphic Filter H(u, v)")
    axes[0, 1].set_xlabel("Frequency Column")
    axes[0, 1].set_ylabel("Frequency Row")
    figure.colorbar(filter_plot, ax=axes[0, 1], fraction=0.046, pad=0.04)

    spectrum_plot = axes[1, 0].imshow(spectrum, cmap="magma")
    axes[1, 0].set_title("Log Magnitude Spectrum")
    axes[1, 0].set_xlabel("Frequency Column")
    axes[1, 0].set_ylabel("Frequency Row")
    figure.colorbar(spectrum_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(g_uint8, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Restored Image")
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Row")

    figure.suptitle("Homomorphic Filtering Pipeline")
    figure.tight_layout()
    figure.savefig("results/homomorphic_pipeline_overview.png", dpi=200)
    plt.close(figure)

    histogram_figure, histogram_axes = plt.subplots(1, 2, figsize=(12, 4))

    histogram_axes[0].hist(f.ravel(), bins=256, range=(0, 255), color="gray")
    histogram_axes[0].set_title("Histogram Before Filtering")
    histogram_axes[0].set_xlabel("Intensity")
    histogram_axes[0].set_ylabel("Pixel Count")

    histogram_axes[1].hist(g_uint8.ravel(), bins=256, range=(0, 255), color="black")
    histogram_axes[1].set_title("Histogram After Filtering")
    histogram_axes[1].set_xlabel("Intensity")
    histogram_axes[1].set_ylabel("Pixel Count")

    histogram_figure.suptitle("Homomorphic Filtering Histograms")
    histogram_figure.tight_layout()
    histogram_figure.savefig("results/homomorphic_histograms.png", dpi=200)
    plt.close(histogram_figure)

    print("Generating aggressive synthetic demo...")
    aggressive_gamma_l = 0.1
    aggressive_gamma_h = 1.3
    aggressive_d0 = 180
    aggressive_order = 4
    print(
        f"  Aggressive demo filter: butterworth, gamma_L={aggressive_gamma_l}, "
        f"gamma_H={aggressive_gamma_h}, D0={aggressive_d0}, order={aggressive_order}"
    )

    _, _, _, _, _, _, aggressive_uint8 = apply_homomorphic_filter(
        f,
        aggressive_gamma_l,
        aggressive_gamma_h,
        aggressive_d0,
        "butterworth",
        aggressive_order,
    )
    Image.fromarray(aggressive_uint8).save("results/homomorphic_restored_aggressive.png")

    baseline_image = np.array(Image.open("images/cameraman.png").convert("L"), dtype=np.float64)
    corrupted_difference = make_difference_map(baseline_image, f)
    balanced_difference = make_difference_map(baseline_image, g_uint8)
    aggressive_difference = make_difference_map(baseline_image, aggressive_uint8)

    aggressive_figure, aggressive_axes = plt.subplots(2, 3, figsize=(15, 10))

    aggressive_axes[0, 0].imshow(baseline_image, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 0].set_title("Baseline")
    aggressive_axes[0, 0].set_xlabel("Column")
    aggressive_axes[0, 0].set_ylabel("Row")

    aggressive_axes[0, 1].imshow(f, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 1].set_title("Corrupted Synthetic")
    aggressive_axes[0, 1].set_xlabel("Column")
    aggressive_axes[0, 1].set_ylabel("Row")

    aggressive_axes[0, 2].imshow(g_uint8, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[0, 2].set_title("Balanced Blind Restoration")
    aggressive_axes[0, 2].set_xlabel("Column")
    aggressive_axes[0, 2].set_ylabel("Row")

    aggressive_axes[1, 0].imshow(aggressive_uint8, cmap="gray", vmin=0, vmax=255)
    aggressive_axes[1, 0].set_title("Aggressive Blind Restoration")
    aggressive_axes[1, 0].set_xlabel("Column")
    aggressive_axes[1, 0].set_ylabel("Row")

    diff_plot_1 = aggressive_axes[1, 1].imshow(corrupted_difference, cmap="inferno")
    aggressive_axes[1, 1].set_title("Abs Difference: Baseline vs Corrupted")
    aggressive_axes[1, 1].set_xlabel("Column")
    aggressive_axes[1, 1].set_ylabel("Row")
    aggressive_figure.colorbar(diff_plot_1, ax=aggressive_axes[1, 1], fraction=0.046, pad=0.04)

    diff_plot_2 = aggressive_axes[1, 2].imshow(aggressive_difference, cmap="inferno")
    aggressive_axes[1, 2].set_title("Abs Difference: Baseline vs Aggressive")
    aggressive_axes[1, 2].set_xlabel("Column")
    aggressive_axes[1, 2].set_ylabel("Row")
    aggressive_figure.colorbar(diff_plot_2, ax=aggressive_axes[1, 2], fraction=0.046, pad=0.04)

    aggressive_figure.suptitle("Blind Synthetic Demo: Balanced vs Aggressive Restoration")
    aggressive_figure.tight_layout()
    aggressive_figure.savefig("results/synthetic_aggressive_demo.png", dpi=200)
    plt.close(aggressive_figure)

    difference_figure, difference_axes = plt.subplots(1, 3, figsize=(15, 5))

    diff_plot_3 = difference_axes[0].imshow(corrupted_difference, cmap="inferno")
    difference_axes[0].set_title("Baseline vs Corrupted")
    difference_axes[0].set_xlabel("Column")
    difference_axes[0].set_ylabel("Row")
    difference_figure.colorbar(diff_plot_3, ax=difference_axes[0], fraction=0.046, pad=0.04)

    diff_plot_4 = difference_axes[1].imshow(balanced_difference, cmap="inferno")
    difference_axes[1].set_title("Baseline vs Balanced")
    difference_axes[1].set_xlabel("Column")
    difference_axes[1].set_ylabel("Row")
    difference_figure.colorbar(diff_plot_4, ax=difference_axes[1], fraction=0.046, pad=0.04)

    diff_plot_5 = difference_axes[2].imshow(aggressive_difference, cmap="inferno")
    difference_axes[2].set_title("Baseline vs Aggressive")
    difference_axes[2].set_xlabel("Column")
    difference_axes[2].set_ylabel("Row")
    difference_figure.colorbar(diff_plot_5, ax=difference_axes[2], fraction=0.046, pad=0.04)

    difference_figure.suptitle("Synthetic Difference Maps")
    difference_figure.tight_layout()
    difference_figure.savefig("results/synthetic_difference_maps.png", dpi=200)
    plt.close(difference_figure)

    print("Generating multi-case restoration overview...")
    case_paths = [
        ("Vertical", "images/synthetic_vertical.png"),
        ("Rotated", "images/synthetic_rotated.png"),
        ("Sine", "images/synthetic_sine.png"),
    ]

    multicase_figure, multicase_axes = plt.subplots(3, 3, figsize=(14, 14))
    for row_index, (case_name, case_path) in enumerate(case_paths):
        case_image = np.array(Image.open(case_path).convert("L"), dtype=np.float64)
        _, _, _, _, _, _, case_restored = apply_homomorphic_filter(
            case_image, gamma_l, gamma_h, d0, filter_type, order
        )

        case_output_path = f"results/homomorphic_restored_{case_name.lower()}.png"
        Image.fromarray(case_restored).save(case_output_path)
        print(f"  Saved {case_name.lower()} restoration to {case_output_path}")

        multicase_axes[row_index, 0].imshow(
            np.array(Image.open("images/cameraman.png").convert("L"), dtype=np.float64),
            cmap="gray",
            vmin=0,
            vmax=255,
        )
        multicase_axes[row_index, 0].set_title(f"{case_name}: Baseline")
        multicase_axes[row_index, 0].set_xlabel("Column")
        multicase_axes[row_index, 0].set_ylabel("Row")

        multicase_axes[row_index, 1].imshow(case_image, cmap="gray", vmin=0, vmax=255)
        multicase_axes[row_index, 1].set_title(f"{case_name}: Corrupted")
        multicase_axes[row_index, 1].set_xlabel("Column")
        multicase_axes[row_index, 1].set_ylabel("Row")

        multicase_axes[row_index, 2].imshow(case_restored, cmap="gray", vmin=0, vmax=255)
        multicase_axes[row_index, 2].set_title(f"{case_name}: Restored")
        multicase_axes[row_index, 2].set_xlabel("Column")
        multicase_axes[row_index, 2].set_ylabel("Row")

    multicase_figure.suptitle("Homomorphic Filtering Results for Multiple Illumination Cases")
    multicase_figure.tight_layout()
    multicase_figure.savefig("results/homomorphic_multicase_overview.png", dpi=200)
    plt.close(multicase_figure)
    print("  Saved multi-case overview to results/homomorphic_multicase_overview.png")

    print("Done!")
