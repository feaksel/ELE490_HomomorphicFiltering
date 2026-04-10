"""
Script 08: Build a polished grayscale demo pack from the generated blind results.
This script assembles the strongest current grayscale outputs into one figure
that is easier to use in a report or presentation.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def load_image(path):
    return np.array(Image.open(path))


if __name__ == "__main__":
    print("Loading key grayscale result figures...")
    rice_demo = load_image("results/rice_aggressive_comparison.png")
    synthetic_demo = load_image("results/synthetic_aggressive_demo.png")
    padding_demo = load_image("results/padding_windowing_experiment.png")
    metrics_demo = load_image("results/blind_multicase_metrics.png")

    print("Creating final grayscale pack...")
    os.makedirs("results", exist_ok=True)

    figure, axes = plt.subplots(2, 2, figsize=(18, 14))

    axes[0, 0].imshow(rice_demo)
    axes[0, 0].set_title("Main Visual Demo: Rice Image")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(synthetic_demo)
    axes[0, 1].set_title("Synthetic Demo: Balanced vs Aggressive")
    axes[0, 1].axis("off")

    axes[1, 0].imshow(padding_demo)
    axes[1, 0].set_title("Padding and Windowing Experiment")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(metrics_demo)
    axes[1, 1].set_title("Blind Multi-Case Metrics")
    axes[1, 1].axis("off")

    figure.suptitle("Final Grayscale Blind Homomorphic Filtering Pack")
    figure.tight_layout()
    figure.savefig("results/final_grayscale_demo_pack.png", dpi=200)
    plt.close(figure)

    print("Writing summary note...")
    summary_lines = [
        "# Final Grayscale Summary",
        "",
        "## Recommended Narrative",
        "",
        "- Use `rice_aggressive_comparison.png` as the main visual demonstration.",
        "- Use `synthetic_aggressive_demo.png` and `synthetic_difference_maps.png` to show controlled blind synthetic behavior.",
        "- Use `blind_results_table.md` and `blind_multicase_metrics.png` for the quantitative synthetic summary.",
        "- Use `padding_windowing_experiment.png` as an additional engineering experiment showing that FFT boundary handling was also examined.",
        "",
        "## Recommended Submission Figures",
        "",
        "- `results/final_grayscale_demo_pack.png`",
        "- `results/rice_aggressive_comparison.png`",
        "- `results/homomorphic_multicase_overview.png`",
        "- `results/blind_results_table.md`",
    ]

    with open("results/final_grayscale_summary.md", "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(summary_lines) + "\n")

    print("Done! Final grayscale pack saved.")
