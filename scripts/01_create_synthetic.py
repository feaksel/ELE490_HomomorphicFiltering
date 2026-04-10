"""
Script 01: Create synthetic unevenly illuminated image.
Takes the cameraman image and multiplies it by a synthetic illumination
pattern to simulate non-uniform lighting. The script can generate
paper-style patterns such as vertical, rotated, and sine-wave
illumination so the homomorphic filtering experiment is easier to see.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def make_illumination_pattern(rows, cols, pattern_type, illumination_min, illumination_max, power, cycles=1.0):
    y = np.arange(rows)
    x = np.arange(cols)
    X, Y = np.meshgrid(x, y)

    if pattern_type == "horizontal_ramp":
        base_pattern = X / (cols - 1)
    elif pattern_type == "vertical_ramp":
        base_pattern = Y / (rows - 1)
    elif pattern_type == "rotated_ramp":
        # A 45-degree ramp is a simple low-frequency illumination field that
        # stays close to the examples shown in the reference paper.
        base_pattern = (X + Y) / ((cols - 1) + (rows - 1))
    elif pattern_type == "gaussian":
        center_row = 0.32 * rows
        center_col = 0.18 * cols
        sigma_row = 0.45 * rows
        sigma_col = 0.32 * cols
        base_pattern = np.exp(
            -(
                ((Y - center_row) ** 2) / (2.0 * sigma_row ** 2)
                + ((X - center_col) ** 2) / (2.0 * sigma_col ** 2)
            )
        )
        base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min())
    elif pattern_type == "sine_wave":
        # Blind homomorphic filtering assumes illumination changes slowly, so
        # the sinusoidal test is intentionally kept low frequency here.
        sine_x = 0.5 * (1.0 + np.sin(2.0 * np.pi * cycles * X / cols + 0.2))
        sine_y = 0.5 * (1.0 + np.sin(2.0 * np.pi * cycles * Y / rows + 0.2))
        base_pattern = sine_x * sine_y
        base_pattern = (base_pattern - base_pattern.min()) / (base_pattern.max() - base_pattern.min())
    else:
        raise ValueError(f"Unsupported pattern type: {pattern_type}")

    strengthened_pattern = base_pattern ** power
    illumination = illumination_min + (illumination_max - illumination_min) * strengthened_pattern
    return illumination


def save_image(image_array, output_path):
    Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8)).save(output_path)


if __name__ == "__main__":
    print("Loading cameraman image...")
    img = Image.open("images/cameraman.tif").convert("L")
    f = np.array(img, dtype=np.float64)
    print(f"  Image shape: {f.shape}, range: [{f.min()}, {f.max()}]")

    rows, cols = f.shape
    case_settings = [
        {
            "name": "vertical",
            "pattern_type": "vertical_ramp",
            "illumination_min": 0.50,
            "illumination_max": 1.30,
            "power": 1.0,
            "cycles": 1.0,
            "image_path": "images/synthetic_vertical.png",
        },
        {
            "name": "rotated",
            "pattern_type": "rotated_ramp",
            "illumination_min": 0.45,
            "illumination_max": 1.35,
            "power": 1.0,
            "cycles": 1.0,
            "image_path": "images/synthetic_rotated.png",
        },
        {
            "name": "sine",
            "pattern_type": "sine_wave",
            "illumination_min": 0.50,
            "illumination_max": 1.25,
            "power": 1.0,
            "cycles": 1.0,
            "image_path": "images/synthetic_sine.png",
        },
    ]

    print("Creating synthetic illumination cases...")
    case_results = []
    for case in case_settings:
        print(f"  Case: {case['name']}")
        print(
            f"    Pattern={case['pattern_type']}, min={case['illumination_min']:.2f}, "
            f"max={case['illumination_max']:.2f}, power={case['power']:.2f}, cycles={case['cycles']:.2f}"
        )

        illumination = make_illumination_pattern(
            rows,
            cols,
            case["pattern_type"],
            case["illumination_min"],
            case["illumination_max"],
            case["power"],
            case["cycles"],
        )
        f_corrupted = np.clip(f * illumination, 0, 255)
        saturation_fraction = np.mean(f_corrupted >= 255.0)

        print(f"    Illumination range: [{illumination.min():.2f}, {illumination.max():.2f}]")
        print(f"    Corrupted range: [{f_corrupted.min():.2f}, {f_corrupted.max():.2f}]")
        print(f"    Saturated pixel fraction: {100.0 * saturation_fraction:.2f}%")

        save_image(f_corrupted, case["image_path"])
        case_results.append(
            {
                "name": case["name"],
                "pattern_type": case["pattern_type"],
                "illumination": illumination,
                "corrupted": f_corrupted,
                "saturation_fraction": saturation_fraction,
                "image_path": case["image_path"],
            }
        )

    default_case = case_results[1]

    print("Saving results...")
    os.makedirs("results", exist_ok=True)
    save_image(f, "images/cameraman.png")
    save_image(default_case["corrupted"], "images/synthetic_uneven.png")
    print("  Saved baseline PNG copy to images/cameraman.png")
    print("  Saved default synthetic image to images/synthetic_uneven.png")

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(f, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original Cameraman")
    axes[0].set_xlabel("Column")
    axes[0].set_ylabel("Row")

    illumination_plot = axes[1].imshow(default_case["illumination"], cmap="viridis")
    axes[1].set_title("Default Illumination Field")
    axes[1].set_xlabel("Column")
    axes[1].set_ylabel("Row")
    figure.colorbar(illumination_plot, ax=axes[1], fraction=0.046, pad=0.04)

    axes[2].imshow(default_case["corrupted"], cmap="gray", vmin=0, vmax=255)
    axes[2].set_title("Default Synthetic Uneven Image")
    axes[2].set_xlabel("Column")
    axes[2].set_ylabel("Row")

    figure.suptitle("Synthetic Uneven Illumination Generation")
    figure.tight_layout()
    figure.savefig("results/synthetic_generation_overview.png", dpi=200)
    plt.close(figure)
    print("  Saved overview figure to results/synthetic_generation_overview.png")

    print("Saving paper-style illumination pattern gallery...")
    gallery_figure, gallery_axes = plt.subplots(2, 3, figsize=(15, 9))

    for index, case in enumerate(case_results):
        pattern_axis = gallery_axes[0, index]
        corrupted_axis = gallery_axes[1, index]

        pattern_plot = pattern_axis.imshow(case["illumination"], cmap="gray")
        pattern_axis.set_title(f"{case['name'].title()} Illumination")
        pattern_axis.set_xlabel("Column")
        pattern_axis.set_ylabel("Row")
        gallery_figure.colorbar(pattern_plot, ax=pattern_axis, fraction=0.046, pad=0.04)

        corrupted_axis.imshow(case["corrupted"], cmap="gray", vmin=0, vmax=255)
        corrupted_axis.set_title(f"{case['name'].title()} Corrupted Image")
        corrupted_axis.set_xlabel("Column")
        corrupted_axis.set_ylabel("Row")

    gallery_figure.suptitle("Paper-Style Illumination Patterns and Corrupted Images")
    gallery_figure.tight_layout()
    gallery_figure.savefig("results/paper_style_illumination_patterns.png", dpi=200)
    plt.close(gallery_figure)
    print("  Saved pattern gallery to results/paper_style_illumination_patterns.png")

    print("Done! Synthetic image saved.")
