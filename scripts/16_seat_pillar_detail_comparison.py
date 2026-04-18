"""
Script 16: Detail comparison for seat and pillar.
This script highlights texture/detail changes using the regular showcase
pipeline outputs, showing both full-frame context and zoomed crops.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid

DETAIL_CASES = [
    {
        "name": "seat",
        "grayscale_path": os.path.join("results", "real_batch", "seat_grayscale.png"),
        "pipeline_path": os.path.join("results", "real_batch", "seat_homomorphic_tone_adjusted.png"),
        "crop": (260, 430, 520, 520),
        "detail_scale": 3.0,
    },
    {
        "name": "pillar",
        "grayscale_path": os.path.join("results", "real_batch", "pillar_grayscale.png"),
        "pipeline_path": os.path.join("results", "real_batch", "pillar_homomorphic_tone_adjusted.png"),
        "crop": (560, 200, 520, 520),
        "detail_scale": 3.0,
    },
]


def load_uint8_image(image_path):
    return np.array(Image.open(image_path).convert("L"), dtype=np.uint8)


def resolve_existing_path(*candidate_paths):
    for candidate_path in candidate_paths:
        if os.path.exists(candidate_path):
            return candidate_path
    raise FileNotFoundError(candidate_paths[0])


def add_crop_outline(image_array, crop_box, outline_width=8):
    image = Image.fromarray(image_array, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    x, y, w, h = crop_box

    for offset in range(outline_width):
        draw.rectangle(
            [x + offset, y + offset, x + w - offset - 1, y + h - offset - 1],
            outline=(220, 20, 20),
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

    print("Running seat and pillar detail comparison...")
    comparison_rows = []

    for row_index, case in enumerate(DETAIL_CASES):
        gray = load_uint8_image(resolve_existing_path(case["grayscale_path"], os.path.join("results", os.path.basename(case["grayscale_path"]))))
        pipeline_result = load_uint8_image(
            resolve_existing_path(case["pipeline_path"], os.path.join("results", os.path.basename(case["pipeline_path"])))
        )
        crop_x, crop_y, crop_w, crop_h = case["crop"]
        detail_scale = case.get("detail_scale", 2.0)

        gray_crop = gray[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        pipeline_crop = pipeline_result[crop_y : crop_y + crop_h, crop_x : crop_x + crop_w]
        gray_crop_upscaled = upscale_detail_crop(gray_crop, detail_scale)
        pipeline_crop_upscaled = upscale_detail_crop(pipeline_crop, detail_scale)

        print(f"Processed detail case: {case['name']}")
        print(f"  Crop rectangle: x={crop_x}, y={crop_y}, w={crop_w}, h={crop_h}")
        print(f"  Detail-panel scale: {detail_scale}x")
        comparison_rows.append(
            [
                (add_crop_outline(gray, case["crop"]), f"{case['name']}: Grayscale"),
                (add_crop_outline(pipeline_result, case["crop"]), f"{case['name']}: HF + Tone Equalization"),
                (gray_crop_upscaled, f"{case['name']}: Grayscale detail"),
                (pipeline_crop_upscaled, f"{case['name']}: HF + Tone detail"),
            ]
        )

    save_comparison_grid(comparison_rows, os.path.join("results", "seat_pillar_detail_comparison.png"))

    print("Done! Seat and pillar detail comparison saved.")
