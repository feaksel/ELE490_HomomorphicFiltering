"""
Script 21: Detail comparison for the page image.
This script shows the page grayscale image and the conservative page-specific
regular pipeline result with two zoomed text/detail regions.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid


PAGE_GRAYSCALE_PATH = os.path.join("results", "real_batch", "page_grayscale.png")
PAGE_PIPELINE_PATH = os.path.join("results", "real_batch", "page_homomorphic_tone_adjusted.png")
PAGE_DETAIL_CASES = [
    {
        "name": "Upper text region",
        "crop": (110, 210, 520, 260),
        "detail_scale": 3.2,
    },
    {
        "name": "Lower footnote region",
        "crop": (70, 1280, 650, 190),
        "detail_scale": 3.2,
    },
]


def load_uint8_image(image_path):
    return np.array(Image.open(image_path).convert("L"), dtype=np.uint8)


def resolve_existing_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(candidate_paths[0])


def add_crop_outline(image_array, crop_box, outline_color=(220, 20, 20), outline_width=8):
    image = Image.fromarray(image_array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = crop_box

    for offset in range(outline_width):
        draw.rectangle(
            [x + offset, y + offset, x + w - offset - 1, y + h - offset - 1],
            outline=outline_color,
        )

    return np.array(image, dtype=np.uint8)


def upscale_detail_crop(image_array, scale_factor):
    crop_image = Image.fromarray(image_array, mode="L")
    upscaled_size = (
        int(round(crop_image.width * scale_factor)),
        int(round(crop_image.height * scale_factor)),
    )
    return np.array(crop_image.resize(upscaled_size, Image.Resampling.LANCZOS), dtype=np.uint8)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)

    print("Running page detail comparison...")
    gray = load_uint8_image(resolve_existing_path(PAGE_GRAYSCALE_PATH, os.path.join("results", os.path.basename(PAGE_GRAYSCALE_PATH))))
    pipeline_result = load_uint8_image(
        resolve_existing_path(PAGE_PIPELINE_PATH, os.path.join("results", os.path.basename(PAGE_PIPELINE_PATH)))
    )
    comparison_rows = []

    for case in PAGE_DETAIL_CASES:
        crop_x, crop_y, crop_w, crop_h = case["crop"]
        detail_scale = case["detail_scale"]

        gray_crop = gray[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        pipeline_crop = pipeline_result[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]

        print(f"Processed page detail case: {case['name']}")
        print(f"  Crop rectangle: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        print(f"  Detail-panel scale: {detail_scale}x")

        comparison_rows.append(
            [
                (add_crop_outline(gray, case["crop"]), "Page: Grayscale"),
                (add_crop_outline(pipeline_result, case["crop"]), "Page: Conservative HF + Tone Equalization"),
                (upscale_detail_crop(gray_crop, detail_scale), f"{case['name']}: Grayscale detail"),
                (upscale_detail_crop(pipeline_crop, detail_scale), f"{case['name']}: Pipeline detail"),
            ]
        )

    save_comparison_grid(comparison_rows, os.path.join("results", "page_detail_comparison.png"))

    print("Done! Page detail comparison saved.")
