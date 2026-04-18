"""
Script 11: Batch grayscale homomorphic filtering for the current photo set.
This script converts the active color photos in images/ to grayscale, applies
the regular showcase pipeline, and saves per-image plus overview comparisons.
"""
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid, save_comparison_row
from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline, load_resized_grayscale
SHOWCASE_IMAGES = [
    ("cardboard.jpg", "cardboard"),
    ("carboard_uniform.jpg", "carboard_uniform"),
    ("carpet.jpg", "carpet"),
    ("markers.jpg", "markers"),
    ("markers_uniform.jpg", "markers_uniform"),
    ("page.jpeg", "page"),
    ("pillar.jpg", "pillar"),
    ("seat.jpg", "seat"),
]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    overview_rows = []

    print("Running batch grayscale regular showcase demo...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")

    for index, (image_name, base_name) in enumerate(SHOWCASE_IMAGES):
        image_path = os.path.join("images", image_name)
        print(f"Loading image: {image_path}")

        gray_array = load_resized_grayscale(image_path)
        pipeline_result = apply_regular_showcase_pipeline(gray_array, base_name)
        restored = pipeline_result["restored"]
        brightened = pipeline_result["brightened"]
        final_result = pipeline_result["final"]
        pipeline_title = pipeline_result["title"]

        print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
        print(f"  HF output range: [{restored.min()}, {restored.max()}]")
        print(f"  HF+bright range: [{brightened.min()}, {brightened.max()}]")
        print(f"  Final pipeline range: [{final_result.min()}, {final_result.max()}]")

        grayscale_output_path = os.path.join("results", f"{base_name}_grayscale.png")
        restored_output_path = os.path.join("results", f"{base_name}_homomorphic_standard.png")
        brightened_output_path = os.path.join("results", f"{base_name}_homomorphic_standard_bright.png")
        final_output_path = os.path.join("results", f"{base_name}_homomorphic_tone_adjusted.png")
        comparison_output_path = os.path.join("results", f"{base_name}_grayscale_vs_standard.png")

        Image.fromarray(gray_array.astype(np.uint8)).save(grayscale_output_path)
        Image.fromarray(restored).save(restored_output_path)
        Image.fromarray(brightened).save(brightened_output_path)
        Image.fromarray(final_result).save(final_output_path)

        save_comparison_row(
            [gray_array, final_result],
            [f"{base_name}: Grayscale", f"{base_name}: {pipeline_title}"],
            comparison_output_path,
        )

        overview_rows.append(
            [
                (gray_array, f"{base_name}: Grayscale"),
                (final_result, f"{base_name}: {pipeline_title}"),
            ]
        )

    save_comparison_grid(overview_rows, "results/color_grayscale_standard_overview.png")

    print("Done! Batch grayscale regular showcase demo saved.")
