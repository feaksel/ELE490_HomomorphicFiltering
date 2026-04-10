"""
Script 07: Compare padding and windowing choices for blind homomorphic filtering.
This script evaluates whether simple FFT-padding and Hann-window options help
the blind pipeline on a real grayscale benchmark and one synthetic case.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.filters import make_butterworth_homomorphic


def normalize_to_uint8(image_array):
    min_value = np.min(image_array)
    max_value = np.max(image_array)

    if max_value - min_value < 1e-10:
        return np.zeros_like(image_array, dtype=np.uint8)

    scaled = 255.0 * (image_array - min_value) / (max_value - min_value)
    return scaled.astype(np.uint8)


def apply_variant(image_array, gamma_l, gamma_h, d0, order, pad_mode="none", use_window=False):
    image_normalized = image_array / 255.0

    if pad_mode == "reflect":
        pad_rows = image_array.shape[0] // 2
        pad_cols = image_array.shape[1] // 2
        padded = np.pad(image_normalized, ((pad_rows, pad_rows), (pad_cols, pad_cols)), mode="reflect")
    else:
        pad_rows = 0
        pad_cols = 0
        padded = image_normalized.copy()

    log_image = np.log1p(padded)

    if use_window:
        window_row = np.hanning(log_image.shape[0])
        window_col = np.hanning(log_image.shape[1])
        window_2d = np.outer(window_row, window_col)
        log_image = log_image * window_2d

    F = np.fft.fftshift(np.fft.fft2(log_image))
    rows, cols = log_image.shape
    H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    G = H * F
    filtered_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    filtered = np.expm1(filtered_log)
    filtered = np.clip(filtered, 0, None)

    if pad_mode == "reflect":
        filtered = filtered[pad_rows:pad_rows + image_array.shape[0], pad_cols:pad_cols + image_array.shape[1]]

    return normalize_to_uint8(filtered)


def mean_gradient(image_array):
    image_float = image_array.astype(np.float64)
    dx = np.diff(image_float, axis=1)
    dy = np.diff(image_float, axis=0)
    gradient = np.sqrt(dx[:-1, :] ** 2 + dy[:, :-1] ** 2)
    return np.mean(gradient)


if __name__ == "__main__":
    print("Loading images for padding/windowing experiment...")
    rice = np.array(Image.open("images/rice.png").convert("L"), dtype=np.float64)
    synthetic = np.array(Image.open("images/synthetic_rotated.png").convert("L"), dtype=np.float64)
    baseline = np.array(Image.open("images/cameraman.png").convert("L"), dtype=np.float64)
    print(f"  Rice shape: {rice.shape}")
    print(f"  Synthetic shape: {synthetic.shape}")

    gamma_l = 0.2
    gamma_h = 1.1
    d0 = 180
    order = 4
    print(f"  Blind parameters: gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}, order={order}")

    print("Running no-padding variant...")
    rice_no_pad = apply_variant(rice, gamma_l, gamma_h, d0, order, pad_mode="none", use_window=False)
    synthetic_no_pad = apply_variant(synthetic, gamma_l, gamma_h, d0, order, pad_mode="none", use_window=False)

    print("Running reflect-padding variant...")
    rice_reflect = apply_variant(rice, gamma_l, gamma_h, d0, order, pad_mode="reflect", use_window=False)
    synthetic_reflect = apply_variant(synthetic, gamma_l, gamma_h, d0, order, pad_mode="reflect", use_window=False)

    print("Running reflect-padding + Hann-window variant...")
    rice_window = apply_variant(rice, gamma_l, gamma_h, d0, order, pad_mode="reflect", use_window=True)
    synthetic_window = apply_variant(synthetic, gamma_l, gamma_h, d0, order, pad_mode="reflect", use_window=True)

    print("Saving comparison figure...")
    os.makedirs("results", exist_ok=True)

    crop_row_start = 88
    crop_row_end = 176
    crop_col_start = 88
    crop_col_end = 176

    figure, axes = plt.subplots(3, 4, figsize=(18, 13))

    axes[0, 0].imshow(rice, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Rice: Original")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    axes[0, 1].imshow(rice_no_pad, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Rice: No Padding")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")

    axes[0, 2].imshow(rice_reflect, cmap="gray", vmin=0, vmax=255)
    axes[0, 2].set_title("Rice: Reflect Padding")
    axes[0, 2].set_xlabel("Column")
    axes[0, 2].set_ylabel("Row")

    axes[0, 3].imshow(rice_window, cmap="gray", vmin=0, vmax=255)
    axes[0, 3].set_title("Rice: Reflect + Hann")
    axes[0, 3].set_xlabel("Column")
    axes[0, 3].set_ylabel("Row")

    axes[1, 0].imshow(rice[crop_row_start:crop_row_end, crop_col_start:crop_col_end], cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Rice Crop: Original")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")

    axes[1, 1].imshow(rice_no_pad[crop_row_start:crop_row_end, crop_col_start:crop_col_end], cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("Rice Crop: No Padding")
    axes[1, 1].set_xlabel("Column")
    axes[1, 1].set_ylabel("Row")

    axes[1, 2].imshow(rice_reflect[crop_row_start:crop_row_end, crop_col_start:crop_col_end], cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title("Rice Crop: Reflect Padding")
    axes[1, 2].set_xlabel("Column")
    axes[1, 2].set_ylabel("Row")

    axes[1, 3].imshow(rice_window[crop_row_start:crop_row_end, crop_col_start:crop_col_end], cmap="gray", vmin=0, vmax=255)
    axes[1, 3].set_title("Rice Crop: Reflect + Hann")
    axes[1, 3].set_xlabel("Column")
    axes[1, 3].set_ylabel("Row")

    axes[2, 0].imshow(synthetic, cmap="gray", vmin=0, vmax=255)
    axes[2, 0].set_title("Synthetic Rotated: Corrupted")
    axes[2, 0].set_xlabel("Column")
    axes[2, 0].set_ylabel("Row")

    axes[2, 1].imshow(synthetic_no_pad, cmap="gray", vmin=0, vmax=255)
    axes[2, 1].set_title("Synthetic: No Padding")
    axes[2, 1].set_xlabel("Column")
    axes[2, 1].set_ylabel("Row")

    axes[2, 2].imshow(synthetic_reflect, cmap="gray", vmin=0, vmax=255)
    axes[2, 2].set_title("Synthetic: Reflect Padding")
    axes[2, 2].set_xlabel("Column")
    axes[2, 2].set_ylabel("Row")

    axes[2, 3].imshow(synthetic_window, cmap="gray", vmin=0, vmax=255)
    axes[2, 3].set_title("Synthetic: Reflect + Hann")
    axes[2, 3].set_xlabel("Column")
    axes[2, 3].set_ylabel("Row")

    figure.suptitle("Padding and Windowing Experiment for Blind Homomorphic Filtering")
    figure.tight_layout()
    figure.savefig("results/padding_windowing_experiment.png", dpi=200)
    plt.close(figure)

    print("Writing experiment summary table...")
    lines = [
        "# Padding and Windowing Experiment",
        "",
        "| Variant | Rice Std | Rice Mean Gradient | Synthetic PSNR (dB) | Synthetic SSIM |",
        "|---------|----------|--------------------|---------------------|----------------|",
    ]

    variants = [
        ("No padding", rice_no_pad, synthetic_no_pad),
        ("Reflect padding", rice_reflect, synthetic_reflect),
        ("Reflect + Hann", rice_window, synthetic_window),
    ]

    for label, rice_variant, synthetic_variant in variants:
        mse = np.mean((baseline - synthetic_variant.astype(np.float64)) ** 2)
        psnr = 10.0 * np.log10((255.0 ** 2) / mse)
        mu_x = baseline.mean()
        mu_y = synthetic_variant.mean()
        sigma_x = baseline.var()
        sigma_y = synthetic_variant.var()
        sigma_xy = ((baseline - mu_x) * (synthetic_variant - mu_y)).mean()
        c1 = (0.01 * 255) ** 2
        c2 = (0.03 * 255) ** 2
        ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
        )

        lines.append(
            f"| {label} | {np.std(rice_variant):.4f} | {mean_gradient(rice_variant):.4f} | {psnr:.4f} | {ssim:.4f} |"
        )

    with open("results/padding_windowing_metrics.md", "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")

    print("Done! Padding/windowing experiment saved.")
