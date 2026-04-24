"""
Utilities for building before/after analysis visuals and summary metrics.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


EPSILON = 1e-12


def _to_uint8(image_array):
    array = np.asarray(image_array, dtype=np.float64)
    return np.clip(np.rint(array), 0, 255).astype(np.uint8)


def _normalized_radius_grid(rows, cols):
    y_coords, x_coords = np.indices((rows, cols))
    center_y = (rows - 1) / 2.0
    center_x = (cols - 1) / 2.0
    radius = np.sqrt((y_coords - center_y) ** 2 + (x_coords - center_x) ** 2)
    max_radius = max(radius.max(), EPSILON)
    return radius / max_radius


def compute_intensity_statistics(image_array):
    image_uint8 = _to_uint8(image_array)
    histogram = np.bincount(image_uint8.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / max(histogram.sum(), 1.0)
    nonzero = probabilities > 0

    percentile_01, percentile_99 = np.percentile(image_uint8, [1, 99])

    return {
        "min": float(image_uint8.min()),
        "max": float(image_uint8.max()),
        "mean": float(image_uint8.mean()),
        "std": float(image_uint8.std()),
        "entropy": float(-np.sum(probabilities[nonzero] * np.log2(probabilities[nonzero]))),
        "spread_1_99": float(percentile_99 - percentile_01),
    }


def compute_frequency_statistics(image_array, radial_bins=128, high_frequency_threshold=0.35):
    image_uint8 = _to_uint8(image_array)
    centered = image_uint8.astype(np.float64) / 255.0
    centered = centered - centered.mean()

    spectrum = np.fft.fftshift(np.fft.fft2(centered))
    power = np.abs(spectrum) ** 2
    magnitude_view = np.log1p(np.abs(spectrum))

    radius_norm = _normalized_radius_grid(*image_uint8.shape)
    bin_indices = np.minimum((radius_norm * radial_bins).astype(np.int32), radial_bins - 1)
    flat_bins = bin_indices.ravel()
    flat_power = power.ravel()

    radial_power_sum = np.bincount(flat_bins, weights=flat_power, minlength=radial_bins)
    radial_counts = np.bincount(flat_bins, minlength=radial_bins)
    radial_profile = radial_power_sum / np.maximum(radial_counts, 1)
    radial_positions = (np.arange(radial_bins, dtype=np.float64) + 0.5) / radial_bins

    total_power = float(power.sum())
    high_frequency_mask = radius_norm >= high_frequency_threshold
    high_frequency_share = float(power[high_frequency_mask].sum() / max(total_power, EPSILON))

    return {
        "magnitude_view": magnitude_view,
        "radial_positions": radial_positions,
        "radial_profile": radial_profile,
        "high_frequency_share": high_frequency_share,
    }


def summarize_before_after_metrics(before_image, after_image, label):
    before_intensity = compute_intensity_statistics(before_image)
    after_intensity = compute_intensity_statistics(after_image)
    before_frequency = compute_frequency_statistics(before_image)
    after_frequency = compute_frequency_statistics(after_image)

    before_uint8 = _to_uint8(before_image).astype(np.float64)
    after_uint8 = _to_uint8(after_image).astype(np.float64)
    absolute_difference = np.abs(after_uint8 - before_uint8)

    return {
        "label": label,
        "before": before_intensity,
        "after": after_intensity,
        "before_frequency": before_frequency,
        "after_frequency": after_frequency,
        "mean_abs_difference": float(absolute_difference.mean()),
        "max_abs_difference": float(absolute_difference.max()),
        "delta_mean": after_intensity["mean"] - before_intensity["mean"],
        "delta_std": after_intensity["std"] - before_intensity["std"],
        "delta_entropy": after_intensity["entropy"] - before_intensity["entropy"],
        "delta_spread_1_99": after_intensity["spread_1_99"] - before_intensity["spread_1_99"],
        "delta_high_frequency_share": (
            after_frequency["high_frequency_share"] - before_frequency["high_frequency_share"]
        ),
    }


def save_before_after_analysis_figure(
    before_image,
    after_image,
    output_path,
    case_title,
    after_label="After Processing",
):
    before_uint8 = _to_uint8(before_image)
    after_uint8 = _to_uint8(after_image)
    metrics = summarize_before_after_metrics(before_uint8, after_uint8, label=case_title)

    difference_map = np.abs(after_uint8.astype(np.float64) - before_uint8.astype(np.float64))
    difference_mean = metrics["mean_abs_difference"]
    difference_max = metrics["max_abs_difference"]

    before_hist, _ = np.histogram(before_uint8.ravel(), bins=256, range=(0, 256), density=True)
    after_hist, _ = np.histogram(after_uint8.ravel(), bins=256, range=(0, 256), density=True)
    intensity_axis = np.arange(256)

    spectrum_min = min(
        metrics["before_frequency"]["magnitude_view"].min(),
        metrics["after_frequency"]["magnitude_view"].min(),
    )
    spectrum_max = max(
        metrics["before_frequency"]["magnitude_view"].max(),
        metrics["after_frequency"]["magnitude_view"].max(),
    )

    figure, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(before_uint8, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Before")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(after_uint8, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title(after_label)
    axes[0, 1].axis("off")

    diff_plot = axes[0, 2].imshow(difference_map, cmap="inferno")
    axes[0, 2].set_title("|After - Before|")
    axes[0, 2].axis("off")
    figure.colorbar(diff_plot, ax=axes[0, 2], fraction=0.046, pad=0.04)

    axes[0, 3].axis("off")
    stats_lines = [
        f"Case: {case_title}",
        "",
        f"Mean: {metrics['before']['mean']:.1f} -> {metrics['after']['mean']:.1f}",
        f"Std: {metrics['before']['std']:.1f} -> {metrics['after']['std']:.1f}",
        f"Entropy: {metrics['before']['entropy']:.3f} -> {metrics['after']['entropy']:.3f} bits",
        f"P1-P99 spread: {metrics['before']['spread_1_99']:.1f} -> {metrics['after']['spread_1_99']:.1f}",
        (
            "HF energy share: "
            f"{100.0 * metrics['before_frequency']['high_frequency_share']:.1f}% -> "
            f"{100.0 * metrics['after_frequency']['high_frequency_share']:.1f}%"
        ),
        f"Mean abs diff: {difference_mean:.1f}",
        f"Max abs diff: {difference_max:.1f}",
    ]
    axes[0, 3].text(
        0.02,
        0.98,
        "\n".join(stats_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    axes[1, 0].plot(intensity_axis, before_hist, label="Before", color="dimgray", linewidth=2.0)
    axes[1, 0].plot(intensity_axis, after_hist, label=after_label, color="tab:blue", linewidth=2.0)
    axes[1, 0].axvline(metrics["before"]["mean"], color="dimgray", linestyle="--", linewidth=1.2)
    axes[1, 0].axvline(metrics["after"]["mean"], color="tab:blue", linestyle="--", linewidth=1.2)
    axes[1, 0].set_title("Intensity Histogram")
    axes[1, 0].set_xlabel("Gray Level")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_xlim(0, 255)
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.25, linewidth=0.5)

    axes[1, 1].semilogy(
        metrics["before_frequency"]["radial_positions"],
        metrics["before_frequency"]["radial_profile"] + EPSILON,
        color="dimgray",
        linewidth=2.0,
        label="Before",
    )
    axes[1, 1].semilogy(
        metrics["after_frequency"]["radial_positions"],
        metrics["after_frequency"]["radial_profile"] + EPSILON,
        color="tab:blue",
        linewidth=2.0,
        label=after_label,
    )
    axes[1, 1].set_title("Radial Frequency Energy")
    axes[1, 1].set_xlabel("Normalized Radius")
    axes[1, 1].set_ylabel("Mean Spectral Power")
    axes[1, 1].set_xlim(0, 1)
    axes[1, 1].legend()
    axes[1, 1].grid(alpha=0.25, linewidth=0.5)

    spectrum_before_plot = axes[1, 2].imshow(
        metrics["before_frequency"]["magnitude_view"],
        cmap="magma",
        vmin=spectrum_min,
        vmax=spectrum_max,
    )
    axes[1, 2].set_title("FFT Magnitude Before")
    axes[1, 2].axis("off")
    figure.colorbar(spectrum_before_plot, ax=axes[1, 2], fraction=0.046, pad=0.04)

    spectrum_after_plot = axes[1, 3].imshow(
        metrics["after_frequency"]["magnitude_view"],
        cmap="magma",
        vmin=spectrum_min,
        vmax=spectrum_max,
    )
    axes[1, 3].set_title("FFT Magnitude After")
    axes[1, 3].axis("off")
    figure.colorbar(spectrum_after_plot, ax=axes[1, 3], fraction=0.046, pad=0.04)

    figure.suptitle(f"{case_title}: Before / After Analysis")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_metric_delta_dashboard(records, output_path):
    labels = [record["label"] for record in records]
    mean_deltas = [record["delta_mean"] for record in records]
    std_deltas = [record["delta_std"] for record in records]
    entropy_deltas = [record["delta_entropy"] for record in records]
    hf_share_deltas = [100.0 * record["delta_high_frequency_share"] for record in records]
    x_positions = np.arange(len(labels))

    figure, axes = plt.subplots(2, 2, figsize=(14, 10))

    dashboard_items = [
        (axes[0, 0], mean_deltas, "Mean Intensity Delta", "Gray Levels", "tab:blue"),
        (axes[0, 1], std_deltas, "Std Delta", "Gray Levels", "tab:orange"),
        (axes[1, 0], entropy_deltas, "Entropy Delta", "Bits", "tab:green"),
        (axes[1, 1], hf_share_deltas, "High-Frequency Share Delta", "Percentage Points", "tab:red"),
    ]

    for axis, values, title, ylabel, color in dashboard_items:
        axis.bar(x_positions, values, color=color, alpha=0.85)
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x_positions, labels, rotation=20)
        axis.grid(axis="y", alpha=0.25, linewidth=0.5)

    figure.suptitle("Showcase Before/After Metric Deltas")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)


def save_pipeline_math_figure(
    original_image,
    diagnostics,
    final_image,
    output_path,
    case_title,
    pipeline_title,
    gamma_l,
    gamma_h,
    d0,
    brighten_gamma,
):
    original_uint8 = _to_uint8(original_image)
    restored_uint8 = _to_uint8(diagnostics["restored"])
    brightened_uint8 = _to_uint8(diagnostics["brightened"])
    final_uint8 = _to_uint8(final_image)

    input_spectrum = np.log1p(np.abs(diagnostics["frequency"]))
    filtered_spectrum = np.log1p(np.abs(diagnostics["filtered_frequency"]))
    spectrum_min = min(input_spectrum.min(), filtered_spectrum.min())
    spectrum_max = max(input_spectrum.max(), filtered_spectrum.max())

    log_image_display = 255.0 * diagnostics["log_image"] / max(diagnostics["log_image"].max(), EPSILON)

    figure, axes = plt.subplots(2, 4, figsize=(20, 10))

    axes[0, 0].imshow(original_uint8, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("1. Input Grayscale f(x, y)")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(log_image_display, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("2. Log Domain ln(1 + f)")
    axes[0, 1].axis("off")

    input_spectrum_plot = axes[0, 2].imshow(input_spectrum, cmap="magma", vmin=spectrum_min, vmax=spectrum_max)
    axes[0, 2].set_title("3. FFT Magnitude |F(u, v)|")
    axes[0, 2].axis("off")
    figure.colorbar(input_spectrum_plot, ax=axes[0, 2], fraction=0.046, pad=0.04)

    filter_plot = axes[0, 3].imshow(diagnostics["filter"], cmap="viridis")
    axes[0, 3].set_title("4. Homomorphic Filter H(u, v)")
    axes[0, 3].axis("off")
    figure.colorbar(filter_plot, ax=axes[0, 3], fraction=0.046, pad=0.04)

    filtered_spectrum_plot = axes[1, 0].imshow(
        filtered_spectrum,
        cmap="magma",
        vmin=spectrum_min,
        vmax=spectrum_max,
    )
    axes[1, 0].set_title("5. Filtered Spectrum |H.F|")
    axes[1, 0].axis("off")
    figure.colorbar(filtered_spectrum_plot, ax=axes[1, 0], fraction=0.046, pad=0.04)

    axes[1, 1].imshow(restored_uint8, cmap="gray", vmin=0, vmax=255)
    axes[1, 1].set_title("6. After IDFT + exp")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(final_uint8, cmap="gray", vmin=0, vmax=255)
    axes[1, 2].set_title(f"7. Final {pipeline_title}")
    axes[1, 2].axis("off")

    axes[1, 3].axis("off")
    text_lines = [
        f"Case: {case_title}",
        "",
        "Pipeline math:",
        "z = ln(1 + f)",
        "Z = FFT{z}",
        "S = H(u, v) * Z",
        "s = IFFT{S}",
        "g = exp(s) - 1",
        "",
        f"gamma_L = {gamma_l:.2f}",
        f"gamma_H = {gamma_h:.2f}",
        f"D0 = {d0:.0f}",
        f"brighten gamma = {brighten_gamma:.2f}",
        "",
        f"Filter range: [{diagnostics['filter'].min():.2f}, {diagnostics['filter'].max():.2f}]",
        f"Input mean/std: {original_uint8.mean():.1f} / {original_uint8.std():.1f}",
        f"Final mean/std: {final_uint8.mean():.1f} / {final_uint8.std():.1f}",
        f"Brightened mean/std: {brightened_uint8.mean():.1f} / {brightened_uint8.std():.1f}",
    ]
    axes[1, 3].text(
        0.02,
        0.98,
        "\n".join(text_lines),
        va="top",
        ha="left",
        fontsize=11,
        family="monospace",
    )

    figure.suptitle(f"{case_title}: Homomorphic Pipeline Math View")
    figure.tight_layout()
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)
