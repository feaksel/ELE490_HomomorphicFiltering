"""
Script 18: Conservative regular-pipeline comparison for the page image.
This script tests milder page-specific homomorphic settings and then applies
the regular tone-equalization stage so document readability can be judged more
fairly.
"""
import os
import sys

from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_row
from utils.showcase_pipeline import (
    MAX_DIMENSION,
    apply_homomorphic,
    load_resized_grayscale,
    tone_adjust_shadows_highlights,
)


IMAGE_PATH = os.path.join("images", "page.jpeg")
PAGE_CASES = [
    ("standard", "Standard HF + Tone Equalization", 0.06, 1.00, 320, 0.72),
    ("mild", "Mild HF + Tone Equalization", 0.15, 1.00, 220, 0.78),
    ("conservative", "Conservative HF + Tone Equalization", 0.25, 1.00, 160, 0.84),
]


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running conservative page regular-pipeline comparison...")
    gray_array = load_resized_grayscale(IMAGE_PATH, max_dimension=MAX_DIMENSION)
    print(f"  Grayscale shape: {gray_array.shape}, range: [{gray_array.min()}, {gray_array.max()}]")
    comparison_images = [gray_array]
    comparison_titles = ["Page: Grayscale"]

    for case_name, title, gamma_l, gamma_h, d0, brighten_gamma in PAGE_CASES:
        restored, brightened = apply_homomorphic(gray_array, gamma_l, gamma_h, d0, brighten_gamma)
        result = tone_adjust_shadows_highlights(
            brightened,
            blur_sigma=28,
            shadow_strength=0.24,
            highlight_strength=0.20,
            shadow_pivot=0.58,
            highlight_pivot=0.42,
        )
        print(
            f"  {case_name}: gamma_L={gamma_l}, gamma_H={gamma_h}, D0={d0}, "
            f"bright_gamma={brighten_gamma}, range=[{result.min()}, {result.max()}]"
        )

        Image.fromarray(restored).save(os.path.join("results", f"page_{case_name}_homomorphic_base.png"))
        Image.fromarray(brightened).save(os.path.join("results", f"page_{case_name}_homomorphic_bright.png"))
        Image.fromarray(result).save(os.path.join("results", f"page_{case_name}_homomorphic_tone_adjusted.png"))
        comparison_images.append(result)
        comparison_titles.append(f"Page: {title}")

    save_comparison_row(comparison_images, comparison_titles, os.path.join("results", "page_conservative_hf_comparison.png"))

    print("Done! Conservative page regular-pipeline comparison saved.")
