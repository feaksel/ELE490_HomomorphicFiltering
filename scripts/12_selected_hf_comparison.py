"""
Script 12: Regular-pipeline comparison on selected real images.
This script converts selected color images to grayscale, applies the regular
showcase pipeline, and saves one combined comparison figure.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid
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

    print("Running selected real-image regular-pipeline comparison...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")
    comparison_rows = []

    for row_index, (image_name, base_name) in enumerate(SHOWCASE_IMAGES):
        image_path = os.path.join("images", image_name)

        print(f"Loading image: {image_path}")
        gray_array = load_resized_grayscale(image_path)
        pipeline_result = apply_regular_showcase_pipeline(gray_array, base_name)
        final_result = pipeline_result["final"]
        final_title = f"{base_name}: {pipeline_result['title']}"

        print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
        print(f"  Final pipeline range: [{final_result.min()}, {final_result.max()}]")

        comparison_rows.append(
            [
                (gray_array, f"{base_name}: Grayscale"),
                (final_result, final_title),
            ]
        )

    save_comparison_grid(comparison_rows, "results/selected_real_images_hf_bright_overview.png")

    print("Done! Selected regular-pipeline comparison saved.")
