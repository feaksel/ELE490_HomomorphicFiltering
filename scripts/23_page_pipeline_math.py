"""
Script 23: Build a professor-friendly homomorphic pipeline math figure for the page image.
This script visualizes the main transform stages for the active page pipeline:
input, log domain, spectrum, filter response, filtered spectrum, restored output,
and final tone-equalized result.
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.analysis_visuals import save_pipeline_math_figure
from utils.showcase_pipeline import (
    MAX_DIMENSION,
    analyze_homomorphic,
    get_showcase_pipeline_config,
    load_resized_grayscale,
    tone_adjust_shadows_highlights,
)


if __name__ == "__main__":
    results_dir = os.path.join("results", "analysis")
    os.makedirs(results_dir, exist_ok=True)

    image_path = os.path.join("images", "page.jpeg")
    print("Building homomorphic pipeline math figure for the page image...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels.")

    gray_array = load_resized_grayscale(image_path, max_dimension=MAX_DIMENSION)
    config = get_showcase_pipeline_config("page")
    diagnostics = analyze_homomorphic(
        gray_array,
        gamma_l=config["gamma_l"],
        gamma_h=config["gamma_h"],
        d0=config["d0"],
        brighten_gamma=config["brighten_gamma"],
    )
    final_result = tone_adjust_shadows_highlights(diagnostics["brightened"], **config["tone_kwargs"])

    output_path = os.path.join(results_dir, "page_pipeline_math_figure.png")
    save_pipeline_math_figure(
        gray_array,
        diagnostics,
        final_result,
        output_path,
        case_title="Page",
        pipeline_title=config["title"],
        gamma_l=config["gamma_l"],
        gamma_h=config["gamma_h"],
        d0=config["d0"],
        brighten_gamma=config["brighten_gamma"],
    )

    print(f"Saved math figure: {output_path}")
    print(
        f"  filter range=[{diagnostics['filter'].min():.3f}, {diagnostics['filter'].max():.3f}], "
        f"final range=[{final_result.min()}, {final_result.max()}]"
    )
    print("Done! Page pipeline math figure is ready.")
