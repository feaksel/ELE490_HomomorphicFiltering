"""
Script 03: Sweep homomorphic filter parameters and compare results.
Runs the grayscale homomorphic pipeline for multiple gamma_L, gamma_H,
and D0 values, then saves comparison grids for visual inspection.
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


def apply_homomorphic_from_spectrum(F, rows, cols, gamma_l, gamma_h, d0, order, filter_type):
    if filter_type == "butterworth":
        H = make_butterworth_homomorphic(rows, cols, d0, gamma_l, gamma_h, order=order)
    else:
        H = make_gaussian_homomorphic(rows, cols, d0, gamma_l, gamma_h)

    G = H * F
    g_log = np.real(np.fft.ifft2(np.fft.ifftshift(G)))
    g = np.expm1(g_log)
    g = np.clip(g, 0, None)

    return normalize_to_uint8(g)


def save_grid(images, titles, figure_title, output_path, nrows, ncols):
    figure, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for index, axis in enumerate(axes):
        if index < len(images):
            axis.imshow(images[index], cmap="gray", vmin=0, vmax=255)
            axis.set_title(titles[index])
            axis.set_xlabel("Column")
            axis.set_ylabel("Row")
        else:
            axis.axis("off")

    figure.suptitle(figure_title)
    figure.tight_layout()
    figure.savefig(output_path, dpi=200)
    plt.close(figure)


if __name__ == "__main__":
    image_path = "images/rice.png"
    print(f"Loading image: {image_path}")
    img = Image.open(image_path).convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    gamma_l_values = [0.3, 0.5, 0.7, 0.9]
    gamma_h_values = [1.2, 1.5, 1.8, 2.0, 2.5]
    d0_values = [10, 20, 30, 50, 80]
    filter_type = "butterworth"
    print("Parameter sweep setup:")
    print(f"  gamma_L values: {gamma_l_values}")
    print(f"  gamma_H values: {gamma_h_values}")
    print(f"  D0 values: {d0_values}")
    print(f"  Filter type: {filter_type}")

    print("Preparing logarithm and Fourier transform...")
    f_normalized = f / 255.0
    f_log = np.log1p(f_normalized)
    F = np.fft.fftshift(np.fft.fft2(f_log))
    rows, cols = f.shape
    print(f"  Log image range: [{f_log.min():.4f}, {f_log.max():.4f}]")

    print("Beginning parameter sweep...")
    os.makedirs("results", exist_ok=True)

    gamma_l_images = []
    gamma_l_titles = []
    for gamma_l in gamma_l_values:
        print(f"  Sweeping gamma_L = {gamma_l}")
        output = apply_homomorphic_from_spectrum(F, rows, cols, gamma_l, 1.8, 30, 2, filter_type)
        gamma_l_images.append(output)
        gamma_l_titles.append(f"gamma_L={gamma_l}, gamma_H=1.8, D0=30")

    save_grid(
        gamma_l_images,
        gamma_l_titles,
        "Parameter Sweep: gamma_L",
        "results/sweep_gamma_l.png",
        2,
        2,
    )

    gamma_h_images = []
    gamma_h_titles = []
    for gamma_h in gamma_h_values:
        print(f"  Sweeping gamma_H = {gamma_h}")
        output = apply_homomorphic_from_spectrum(F, rows, cols, 0.5, gamma_h, 30, 2, filter_type)
        gamma_h_images.append(output)
        gamma_h_titles.append(f"gamma_L=0.5, gamma_H={gamma_h}, D0=30")

    save_grid(
        gamma_h_images,
        gamma_h_titles,
        "Parameter Sweep: gamma_H",
        "results/sweep_gamma_h.png",
        2,
        3,
    )

    d0_images = []
    d0_titles = []
    for d0 in d0_values:
        print(f"  Sweeping D0 = {d0}")
        output = apply_homomorphic_from_spectrum(F, rows, cols, 0.5, 1.8, d0, 2, filter_type)
        d0_images.append(output)
        d0_titles.append(f"gamma_L=0.5, gamma_H=1.8, D0={d0}")

    save_grid(
        d0_images,
        d0_titles,
        "Parameter Sweep: D0",
        "results/sweep_d0.png",
        2,
        3,
    )

    order_images = []
    order_titles = []
    for order in [1, 2, 4]:
        print(f"  Sweeping Butterworth order = {order}")
        output = apply_homomorphic_from_spectrum(F, rows, cols, 0.5, 1.8, 30, order, filter_type)
        order_images.append(output)
        order_titles.append(f"gamma_L=0.5, gamma_H=1.8, D0=30, n={order}")

    save_grid(
        order_images,
        order_titles,
        "Parameter Sweep: Butterworth Order",
        "results/sweep_butterworth_order.png",
        1,
        3,
    )

    print("Done! Parameter sweep results saved.")
