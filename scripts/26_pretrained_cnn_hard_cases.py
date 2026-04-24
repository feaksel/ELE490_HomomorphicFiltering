"""
Script 26: Optional pretrained CNN comparison on the hard-case set.
This script expects local TorchScript weights for Zero-DCE++ or RetinexNet,
runs RGB inference, and saves both RGB and grayscale outputs plus metrics.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.cnn_baseline import (
    describe_expected_model_locations,
    find_available_model_spec,
    load_torchscript_model,
    run_torchscript_model,
)
from utils.evaluation_metrics import compute_proxy_metrics, compute_reference_metrics
from utils.experimental_methods import (
    load_hard_case_manifest,
    load_resized_rgb,
    resolve_existing_path,
    rgb_to_grayscale_uint8,
)


RESULTS_ROOT = os.path.join("results", "experimental", "cnn")
SYNTHETIC_CASES = [
    ("uneven", ["images/synthetic_uneven.png", "images/old/synthetic_uneven.png"]),
    ("vertical", ["images/synthetic_vertical.png", "images/old/synthetic_vertical.png"]),
    ("rotated", ["images/synthetic_rotated.png", "images/old/synthetic_rotated.png"]),
    ("sine", ["images/synthetic_sine.png", "images/old/synthetic_sine.png"]),
]


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def timed_call(function, *args, **kwargs):
    start_time = time.perf_counter()
    result = function(*args, **kwargs)
    elapsed_ms = 1000.0 * (time.perf_counter() - start_time)
    return result, elapsed_ms


def write_csv(output_path, fieldnames, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    ensure_directory(RESULTS_ROOT)
    ensure_directory(os.path.join(RESULTS_ROOT, "hard_cases"))
    ensure_directory(os.path.join(RESULTS_ROOT, "synthetic"))

    print("Running optional pretrained CNN comparison...")
    model_spec = find_available_model_spec()
    status_path = os.path.join(RESULTS_ROOT, "cnn_status.md")

    if model_spec is None:
        with open(status_path, "w", encoding="utf-8") as output_file:
            output_file.write(
                "# CNN Status\n\n"
                "No pretrained TorchScript model was found locally, so the CNN comparison branch was skipped.\n\n"
                + describe_expected_model_locations()
                + "\n"
            )
        print("No local CNN weights found. Wrote status note and skipped inference.")
        sys.exit(0)

    model = load_torchscript_model(model_spec["path"], device="cpu")
    print(f"Loaded {model_spec['display_name']} from {model_spec['path']}")

    model_root = os.path.join(RESULTS_ROOT, model_spec["id"])
    hard_rgb_dir = os.path.join(model_root, "hard_cases_rgb")
    hard_gray_dir = os.path.join(model_root, "hard_cases_gray")
    synthetic_rgb_dir = os.path.join(model_root, "synthetic_rgb")
    synthetic_gray_dir = os.path.join(model_root, "synthetic_gray")
    for directory in [model_root, hard_rgb_dir, hard_gray_dir, synthetic_rgb_dir, synthetic_gray_dir]:
        ensure_directory(directory)

    manifest = load_hard_case_manifest()
    hard_case_rows = []
    for case in manifest:
        image_path = resolve_existing_path(case["path"])
        rgb_image = load_resized_rgb(image_path)
        output_rgb, runtime_ms = timed_call(run_torchscript_model, model, rgb_image, device="cpu")
        output_gray = rgb_to_grayscale_uint8(output_rgb)

        Image.fromarray(output_rgb).save(os.path.join(hard_rgb_dir, f"{case['id']}.png"))
        Image.fromarray(output_gray).save(os.path.join(hard_gray_dir, f"{case['id']}.png"))

        proxy_metrics = compute_proxy_metrics(output_gray)
        hard_case_rows.append(
            {
                "case_id": case["id"],
                "model_id": model_spec["id"],
                "display_name": model_spec["display_name"],
                "runtime_ms": runtime_ms,
                **proxy_metrics,
            }
        )
        print(f"  Saved CNN hard-case output for {case['id']}")

    synthetic_reference_path = resolve_existing_path(
        "images/cameraman.tif",
        "images/cameraman.png",
        "images/old/cameraman.tif",
        "images/old/cameraman.png",
    )
    synthetic_reference = np.array(Image.open(synthetic_reference_path).convert("L"), dtype=np.uint8)
    synthetic_rows = []
    for case_id, candidate_paths in SYNTHETIC_CASES:
        corrupted_path = resolve_existing_path(*candidate_paths)
        corrupted_gray = np.array(Image.open(corrupted_path).convert("L"), dtype=np.uint8)
        corrupted_rgb = np.repeat(corrupted_gray[:, :, None], 3, axis=2)

        output_rgb, runtime_ms = timed_call(run_torchscript_model, model, corrupted_rgb, device="cpu")
        output_gray = rgb_to_grayscale_uint8(output_rgb)

        Image.fromarray(output_rgb).save(os.path.join(synthetic_rgb_dir, f"{case_id}.png"))
        Image.fromarray(output_gray).save(os.path.join(synthetic_gray_dir, f"{case_id}.png"))

        reference_metrics = compute_reference_metrics(synthetic_reference, output_gray)
        synthetic_rows.append(
            {
                "case_id": case_id,
                "model_id": model_spec["id"],
                "display_name": model_spec["display_name"],
                "runtime_ms": runtime_ms,
                **reference_metrics,
            }
        )
        print(f"  Saved CNN synthetic output for {case_id}")

    write_csv(
        os.path.join(RESULTS_ROOT, "cnn_hard_case_metrics.csv"),
        [
            "case_id",
            "model_id",
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
    write_csv(
        os.path.join(RESULTS_ROOT, "cnn_synthetic_metrics.csv"),
        ["case_id", "model_id", "display_name", "runtime_ms", "mse", "psnr", "ssim"],
        synthetic_rows,
    )

    with open(os.path.join(RESULTS_ROOT, "cnn_model.json"), "w", encoding="utf-8") as output_file:
        json.dump(model_spec, output_file, indent=2)

    with open(status_path, "w", encoding="utf-8") as output_file:
        output_file.write(
            "# CNN Status\n\n"
            f"- Model: `{model_spec['display_name']}`\n"
            f"- Path: `{model_spec['path']}`\n"
            "- Inference device: `cpu`\n"
            f"- Hard cases processed: `{len(hard_case_rows)}`\n"
            f"- Synthetic cases processed: `{len(synthetic_rows)}`\n"
        )

    print(f"Saved CNN hard-case metrics to {os.path.join(RESULTS_ROOT, 'cnn_hard_case_metrics.csv')}")
    print(f"Saved CNN synthetic metrics to {os.path.join(RESULTS_ROOT, 'cnn_synthetic_metrics.csv')}")
    print("Done! CNN comparison branch is ready.")
