"""
Script 15: Compare homomorphic filtering against uniform-lighting references.
This script uses the current non-uniform images together with the matching
uniform-lighting examples to show whether the regular showcase pipeline moves
closer to the more evenly illuminated scene.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid, save_comparison_row
from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline, load_resized_grayscale
REFERENCE_PAIRS = [
    ("cardboard", "cardboard.jpg", "carboard_uniform.jpg"),
    ("markers", "markers.jpg", "markers_uniform.jpg"),
]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running uniform-reference comparison...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")
    overview_rows = []

    for row_index, (label, source_name, reference_name) in enumerate(REFERENCE_PAIRS):
        source_gray = load_resized_grayscale(os.path.join("images", source_name))
        reference_gray = load_resized_grayscale(os.path.join("images", reference_name))
        source_pipeline = apply_regular_showcase_pipeline(source_gray, label)
        source_final = source_pipeline["final"]

        print(f"Processed pair: {label}")
        print(f"  Source grayscale shape: {source_gray.shape}")
        print(f"  Reference grayscale shape: {reference_gray.shape}")

        save_comparison_row(
            [source_gray, source_final, reference_gray],
            [
                f"{label}: Original grayscale",
                f"{label}: {source_pipeline['title']}",
                f"{label}: Uniform-lighting reference",
            ],
            os.path.join("results", f"{label}_uniform_reference_comparison.png"),
        )

        overview_rows.append(
            [
                (source_gray, f"{label}: Original grayscale"),
                (source_final, f"{label}: {source_pipeline['title']}"),
                (reference_gray, f"{label}: Uniform-lighting reference"),
            ]
        )

    save_comparison_grid(overview_rows, "results/uniform_reference_comparison_overview.png")

    print("Done! Uniform-reference comparison saved.")
