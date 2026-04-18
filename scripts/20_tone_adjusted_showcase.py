"""
Script 20: Tone-adjusted showcase on top of homomorphic filtering.
This script applies the regular showcase pipeline and compares grayscale,
HF+brightness, and the final tone-equalized output across the active showcase.
"""
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid, save_comparison_row
from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline, load_resized_grayscale
SHOWCASE_IMAGES = [
    ("cardboard.jpg", "cardboard"),
    ("markers.jpg", "markers"),
    ("page.jpeg", "page"),
    ("pillar.jpg", "pillar"),
    ("seat.jpg", "seat"),
]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    overview_rows = []

    print("Running tone-adjusted showcase...")

    for row_index, (image_name, base_name) in enumerate(SHOWCASE_IMAGES):
        image_path = os.path.join("images", image_name)
        print(f"Loading image: {image_path}")
        gray_array = load_resized_grayscale(image_path)
        pipeline_result = apply_regular_showcase_pipeline(gray_array, base_name)
        restored = pipeline_result["restored"]
        hf_bright = pipeline_result["brightened"]
        tone_result = pipeline_result["final"]

        print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
        print(f"  HF+bright range: [{hf_bright.min()}, {hf_bright.max()}]")
        print(f"  Tone-adjusted range: [{tone_result.min()}, {tone_result.max()}]")

        Image.fromarray(restored).save(os.path.join("results", f"{base_name}_homomorphic_tone_base.png"))
        Image.fromarray(hf_bright).save(os.path.join("results", f"{base_name}_homomorphic_tone_bright.png"))
        Image.fromarray(tone_result).save(os.path.join("results", f"{base_name}_homomorphic_tone_adjusted.png"))

        save_comparison_row(
            [gray_array, hf_bright, tone_result],
            [
                f"{base_name}: Grayscale",
                f"{base_name}: HF + Bright",
                f"{base_name}: HF + Tone Adjust",
            ],
            os.path.join("results", f"{base_name}_grayscale_vs_hf_vs_tone.png"),
        )

        overview_rows.append(
            [
                (gray_array, f"{base_name}: Grayscale"),
                (hf_bright, f"{base_name}: HF + Bright"),
                (tone_result, f"{base_name}: HF + Tone"),
            ]
        )

    save_comparison_grid(overview_rows, os.path.join("results", "tone_adjusted_showcase_overview.png"))

    print("Done! Tone-adjusted showcase saved.")
