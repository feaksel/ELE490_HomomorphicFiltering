"""
Script 30: HSI color homomorphic pipeline on the hard-case set.

Runs the accepted regular showcase pipeline (homomorphic filter + brightness
lift + tone equalization) on the intensity channel of each hard-case image
and reconstructs an RGB result. Writes per-case color outputs plus proxy
metrics computed on the luminance of the result.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation_metrics import compute_proxy_metrics
from utils.experimental_methods import (
    load_hard_case_manifest,
    resolve_existing_path,
    rgb_to_grayscale_uint8,
)
from utils.hsi_pipeline import apply_hsi_showcase_pipeline, load_resized_rgb


RESULTS_ROOT = os.path.join("results", "experimental", "hsi")


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def write_csv(output_path, fieldnames, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    rgb_dir = os.path.join(RESULTS_ROOT, "hard_cases_rgb")
    gray_dir = os.path.join(RESULTS_ROOT, "hard_cases_gray")
    for directory in [RESULTS_ROOT, rgb_dir, gray_dir]:
        ensure_directory(directory)

    manifest = load_hard_case_manifest()
    hard_case_rows = []
    config_records = []

    for case in manifest:
        image_path = resolve_existing_path(case["path"])
        rgb_image = load_resized_rgb(image_path)

        start_time = time.perf_counter()
        result = apply_hsi_showcase_pipeline(rgb_image, case["id"])
        runtime_ms = 1000.0 * (time.perf_counter() - start_time)

        final_rgb = result["final_rgb"]
        final_gray = rgb_to_grayscale_uint8(final_rgb)
        Image.fromarray(final_rgb).save(os.path.join(rgb_dir, f"{case['id']}.png"))
        Image.fromarray(final_gray).save(os.path.join(gray_dir, f"{case['id']}.png"))

        proxy_metrics = compute_proxy_metrics(final_gray)
        hard_case_rows.append(
            {
                "case_id": case["id"],
                "method_id": "hsi_homomorphic_color",
                "display_name": "HSI Homomorphic Color",
                "runtime_ms": runtime_ms,
                **proxy_metrics,
            }
        )
        config_records.append(
            {
                "case_id": case["id"],
                "title": result["title"],
                "gamma_l": result["config"]["gamma_l"],
                "gamma_h": result["config"]["gamma_h"],
                "d0": result["config"]["d0"],
                "brighten_gamma": result["config"]["brighten_gamma"],
            }
        )
        print(f"  HSI {case['id']}: {runtime_ms:.1f} ms ({result['title']})")

    write_csv(
        os.path.join(RESULTS_ROOT, "hsi_hard_case_metrics.csv"),
        [
            "case_id",
            "method_id",
            "display_name",
            "runtime_ms",
            "mean",
            "std",
            "entropy",
            "spread_1_99",
            "high_frequency_share",
            "mean_gradient",
            "clipped_pixel_ratio",
        ],
        hard_case_rows,
    )

    with open(os.path.join(RESULTS_ROOT, "hsi_pipeline_config.json"), "w", encoding="utf-8") as output_file:
        json.dump(config_records, output_file, indent=2)

    print(f"Saved HSI hard-case outputs to {RESULTS_ROOT}")
    print("Done! HSI color pipeline branch is ready.")
