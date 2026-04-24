"""
Script 25: Local histogram equalization branch on hard cases.
This script evaluates AHE and CLAHE on synthetic reference cases and on the
curated hard-case manifest, then promotes the best local method to a lightweight
high-boost branch for further comparison.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.evaluation_metrics import compute_proxy_metrics, compute_reference_metrics
from utils.experimental_methods import (
    apply_high_boost,
    apply_local_method,
    apply_real_hard_case_baseline,
    apply_synthetic_homomorphic_baseline,
    get_local_method_spec,
    iter_local_method_specs,
    load_hard_case_manifest,
    load_resized_grayscale,
    resolve_existing_path,
)


RESULTS_ROOT = os.path.join("results", "experimental", "local_eq")
BOOST_SIGMA = 1.2
BOOST_AMOUNT = 0.65
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


def apply_boosted_local_method(image_array, method_spec):
    local_output = apply_local_method(image_array, method_spec)
    return apply_high_boost(local_output, sigma=BOOST_SIGMA, amount=BOOST_AMOUNT)


def write_csv(output_path, fieldnames, rows):
    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(output_path, title, headers, rows):
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


def summarize_rows(rows, keys):
    summary = {}
    for key in keys:
        summary[key] = float(np.mean([row[key] for row in rows]))
    return summary


if __name__ == "__main__":
    ensure_directory(RESULTS_ROOT)
    ensure_directory(os.path.join(RESULTS_ROOT, "hard_cases"))
    ensure_directory(os.path.join(RESULTS_ROOT, "synthetic"))

    print("Running local equalization branch on synthetic and hard cases...")

    manifest = load_hard_case_manifest()
    hard_cases = []
    for case in manifest:
        image_path = resolve_existing_path(case["path"])
        gray_image = load_resized_grayscale(image_path)
        hard_cases.append(
            {
                "id": case["id"],
                "label": case["label"],
                "category": case["category"],
                "image": gray_image,
            }
        )

    synthetic_reference_path = resolve_existing_path(
        "images/cameraman.tif",
        "images/cameraman.png",
        "images/old/cameraman.tif",
        "images/old/cameraman.png",
    )
    synthetic_reference = np.array(Image.open(synthetic_reference_path).convert("L"), dtype=np.uint8)
    synthetic_cases = []
    for case_id, candidate_paths in SYNTHETIC_CASES:
        corrupted_path = resolve_existing_path(*candidate_paths)
        corrupted_image = np.array(Image.open(corrupted_path).convert("L"), dtype=np.uint8)
        synthetic_cases.append({"id": case_id, "image": corrupted_image})

    print(f"  Hard cases loaded: {len(hard_cases)}")
    print(f"  Synthetic cases loaded: {len(synthetic_cases)}")

    hard_case_rows = []
    synthetic_rows = []
    method_summary_rows = []

    baseline_hard_dir = os.path.join(RESULTS_ROOT, "hard_cases", "hf_tone_baseline")
    baseline_synthetic_dir = os.path.join(RESULTS_ROOT, "synthetic", "homomorphic_synthetic_baseline")
    ensure_directory(baseline_hard_dir)
    ensure_directory(baseline_synthetic_dir)

    print("Computing hard-case homomorphic baseline outputs...")
    baseline_hard_metrics = []
    for case in hard_cases:
        baseline_output, runtime_ms = timed_call(apply_real_hard_case_baseline, case["image"], case["id"])
        Image.fromarray(baseline_output).save(os.path.join(baseline_hard_dir, f"{case['id']}.png"))
        proxy_metrics = compute_proxy_metrics(baseline_output)
        baseline_row = {
            "case_id": case["id"],
            "method_id": "hf_tone_baseline",
            "display_name": "HF + Tone Baseline",
            "family": "baseline",
            "runtime_ms": runtime_ms,
            **proxy_metrics,
        }
        hard_case_rows.append(baseline_row)
        baseline_hard_metrics.append(baseline_row)

    print("Computing synthetic homomorphic baseline metrics...")
    baseline_synthetic_metrics = []
    for case in synthetic_cases:
        baseline_output, runtime_ms = timed_call(apply_synthetic_homomorphic_baseline, case["image"])
        Image.fromarray(baseline_output).save(os.path.join(baseline_synthetic_dir, f"{case['id']}.png"))
        reference_metrics = compute_reference_metrics(synthetic_reference, baseline_output)
        baseline_row = {
            "case_id": case["id"],
            "method_id": "homomorphic_synthetic_baseline",
            "display_name": "Homomorphic Synthetic Baseline",
            "family": "baseline",
            "runtime_ms": runtime_ms,
            **reference_metrics,
        }
        synthetic_rows.append(baseline_row)
        baseline_synthetic_metrics.append(baseline_row)

    print("Sweeping local equalization methods...")
    for method_spec in iter_local_method_specs():
        print(f"  Evaluating {method_spec['display_name']}...")

        hard_dir = os.path.join(RESULTS_ROOT, "hard_cases", method_spec["id"])
        ensure_directory(hard_dir)

        synthetic_metric_rows = []
        hard_metric_rows = []

        for synthetic_case in synthetic_cases:
            local_output, runtime_ms = timed_call(apply_local_method, synthetic_case["image"], method_spec)
            reference_metrics = compute_reference_metrics(synthetic_reference, local_output)
            row = {
                "case_id": synthetic_case["id"],
                "method_id": method_spec["id"],
                "display_name": method_spec["display_name"],
                "family": method_spec["family"],
                "tile_size": method_spec["tile_size"],
                "clip_limit": method_spec.get("clip_limit", ""),
                "runtime_ms": runtime_ms,
                **reference_metrics,
            }
            synthetic_rows.append(row)
            synthetic_metric_rows.append(row)

        for hard_case in hard_cases:
            local_output, runtime_ms = timed_call(apply_local_method, hard_case["image"], method_spec)
            Image.fromarray(local_output).save(os.path.join(hard_dir, f"{hard_case['id']}.png"))
            proxy_metrics = compute_proxy_metrics(local_output)
            row = {
                "case_id": hard_case["id"],
                "method_id": method_spec["id"],
                "display_name": method_spec["display_name"],
                "family": method_spec["family"],
                "tile_size": method_spec["tile_size"],
                "clip_limit": method_spec.get("clip_limit", ""),
                "runtime_ms": runtime_ms,
                **proxy_metrics,
            }
            hard_case_rows.append(row)
            hard_metric_rows.append(row)

        synthetic_summary = summarize_rows(synthetic_metric_rows, ["mse", "psnr", "ssim", "runtime_ms"])
        hard_summary = summarize_rows(
            hard_metric_rows,
            [
                "entropy",
                "spread_1_99",
                "high_frequency_share",
                "mean_gradient",
                "clipped_pixel_ratio",
                "runtime_ms",
            ],
        )
        method_summary_rows.append(
            {
                "method_id": method_spec["id"],
                "display_name": method_spec["display_name"],
                "family": method_spec["family"],
                "tile_size": method_spec["tile_size"],
                "clip_limit": method_spec.get("clip_limit", ""),
                "avg_synthetic_mse": synthetic_summary["mse"],
                "avg_synthetic_psnr": synthetic_summary["psnr"],
                "avg_synthetic_ssim": synthetic_summary["ssim"],
                "avg_synthetic_runtime_ms": synthetic_summary["runtime_ms"],
                "avg_entropy": hard_summary["entropy"],
                "avg_spread_1_99": hard_summary["spread_1_99"],
                "avg_high_frequency_share": hard_summary["high_frequency_share"],
                "avg_mean_gradient": hard_summary["mean_gradient"],
                "avg_clipped_pixel_ratio": hard_summary["clipped_pixel_ratio"],
                "avg_hard_runtime_ms": hard_summary["runtime_ms"],
            }
        )

    best_method_summary = max(
        method_summary_rows,
        key=lambda row: (
            row["avg_synthetic_ssim"],
            row["avg_synthetic_psnr"],
            -row["avg_hard_runtime_ms"],
        ),
    )
    best_method_spec = get_local_method_spec(best_method_summary["method_id"])
    boosted_method_id = f"{best_method_spec['id']}_boosted"
    boosted_display_name = f"{best_method_spec['display_name']} + High Boost"

    print(f"Best local method: {best_method_spec['display_name']}")
    boosted_hard_dir = os.path.join(RESULTS_ROOT, "hard_cases", boosted_method_id)
    boosted_synthetic_dir = os.path.join(RESULTS_ROOT, "synthetic", boosted_method_id)
    ensure_directory(boosted_hard_dir)
    ensure_directory(boosted_synthetic_dir)

    boosted_synthetic_rows = []
    boosted_hard_rows = []
    for synthetic_case in synthetic_cases:
        boosted_output, runtime_ms = timed_call(apply_boosted_local_method, synthetic_case["image"], best_method_spec)
        Image.fromarray(boosted_output).save(os.path.join(boosted_synthetic_dir, f"{synthetic_case['id']}.png"))
        reference_metrics = compute_reference_metrics(synthetic_reference, boosted_output)
        row = {
            "case_id": synthetic_case["id"],
            "method_id": boosted_method_id,
            "display_name": boosted_display_name,
            "family": "boosted_local",
            "tile_size": best_method_spec["tile_size"],
            "clip_limit": best_method_spec.get("clip_limit", ""),
            "runtime_ms": runtime_ms,
            **reference_metrics,
        }
        synthetic_rows.append(row)
        boosted_synthetic_rows.append(row)

    for hard_case in hard_cases:
        boosted_output, runtime_ms = timed_call(apply_boosted_local_method, hard_case["image"], best_method_spec)
        Image.fromarray(boosted_output).save(os.path.join(boosted_hard_dir, f"{hard_case['id']}.png"))
        proxy_metrics = compute_proxy_metrics(boosted_output)
        row = {
            "case_id": hard_case["id"],
            "method_id": boosted_method_id,
            "display_name": boosted_display_name,
            "family": "boosted_local",
            "tile_size": best_method_spec["tile_size"],
            "clip_limit": best_method_spec.get("clip_limit", ""),
            "runtime_ms": runtime_ms,
            **proxy_metrics,
        }
        hard_case_rows.append(row)
        boosted_hard_rows.append(row)

    method_summary_rows.append(
        {
            "method_id": boosted_method_id,
            "display_name": boosted_display_name,
            "family": "boosted_local",
            "tile_size": best_method_spec["tile_size"],
            "clip_limit": best_method_spec.get("clip_limit", ""),
            "avg_synthetic_mse": summarize_rows(boosted_synthetic_rows, ["mse"])["mse"],
            "avg_synthetic_psnr": summarize_rows(boosted_synthetic_rows, ["psnr"])["psnr"],
            "avg_synthetic_ssim": summarize_rows(boosted_synthetic_rows, ["ssim"])["ssim"],
            "avg_synthetic_runtime_ms": summarize_rows(boosted_synthetic_rows, ["runtime_ms"])["runtime_ms"],
            "avg_entropy": summarize_rows(boosted_hard_rows, ["entropy"])["entropy"],
            "avg_spread_1_99": summarize_rows(boosted_hard_rows, ["spread_1_99"])["spread_1_99"],
            "avg_high_frequency_share": summarize_rows(boosted_hard_rows, ["high_frequency_share"])["high_frequency_share"],
            "avg_mean_gradient": summarize_rows(boosted_hard_rows, ["mean_gradient"])["mean_gradient"],
            "avg_clipped_pixel_ratio": summarize_rows(boosted_hard_rows, ["clipped_pixel_ratio"])["clipped_pixel_ratio"],
            "avg_hard_runtime_ms": summarize_rows(boosted_hard_rows, ["runtime_ms"])["runtime_ms"],
        }
    )

    synthetic_fieldnames = [
        "case_id",
        "method_id",
        "display_name",
        "family",
        "tile_size",
        "clip_limit",
        "runtime_ms",
        "mse",
        "psnr",
        "ssim",
    ]
    hard_fieldnames = [
        "case_id",
        "method_id",
        "display_name",
        "family",
        "tile_size",
        "clip_limit",
        "runtime_ms",
        "mean",
        "std",
        "entropy",
        "spread_1_99",
        "high_frequency_share",
        "mean_gradient",
        "clipped_pixel_ratio",
    ]
    summary_fieldnames = [
        "method_id",
        "display_name",
        "family",
        "tile_size",
        "clip_limit",
        "avg_synthetic_mse",
        "avg_synthetic_psnr",
        "avg_synthetic_ssim",
        "avg_synthetic_runtime_ms",
        "avg_entropy",
        "avg_spread_1_99",
        "avg_high_frequency_share",
        "avg_mean_gradient",
        "avg_clipped_pixel_ratio",
        "avg_hard_runtime_ms",
    ]

    write_csv(os.path.join(RESULTS_ROOT, "synthetic_local_metrics.csv"), synthetic_fieldnames, synthetic_rows)
    write_csv(os.path.join(RESULTS_ROOT, "hard_case_local_metrics.csv"), hard_fieldnames, hard_case_rows)
    write_csv(os.path.join(RESULTS_ROOT, "local_method_summary.csv"), summary_fieldnames, method_summary_rows)

    sorted_summaries = sorted(
        method_summary_rows,
        key=lambda row: (
            row["avg_synthetic_ssim"],
            row["avg_synthetic_psnr"],
            -row["avg_hard_runtime_ms"],
        ),
        reverse=True,
    )
    markdown_rows = []
    for summary_row in sorted_summaries:
        markdown_rows.append(
            [
                summary_row["display_name"],
                f"{summary_row['avg_synthetic_psnr']:.3f}",
                f"{summary_row['avg_synthetic_ssim']:.4f}",
                f"{summary_row['avg_mean_gradient']:.3f}",
                f"{summary_row['avg_hard_runtime_ms']:.2f}",
            ]
        )

    write_markdown_table(
        os.path.join(RESULTS_ROOT, "local_method_summary.md"),
        "Local Equalization Method Summary",
        ["Method", "Avg Synthetic PSNR", "Avg Synthetic SSIM", "Avg Hard Mean Gradient", "Avg Hard Runtime (ms)"],
        markdown_rows,
    )

    best_method_record = {
        "selected_method_id": best_method_spec["id"],
        "selected_display_name": best_method_spec["display_name"],
        "selection_rule": "highest average synthetic SSIM, then highest average synthetic PSNR, then lowest average hard-case runtime",
        "boosted_method_id": boosted_method_id,
        "boosted_display_name": boosted_display_name,
        "boost_sigma": BOOST_SIGMA,
        "boost_amount": BOOST_AMOUNT,
    }
    with open(os.path.join(RESULTS_ROOT, "best_local_method.json"), "w", encoding="utf-8") as output_file:
        json.dump(best_method_record, output_file, indent=2)

    with open(os.path.join(RESULTS_ROOT, "best_local_method.md"), "w", encoding="utf-8") as output_file:
        output_file.write(
            "# Best Local Method\n\n"
            f"- Selected method: `{best_method_spec['display_name']}`\n"
            f"- Boosted branch: `{boosted_display_name}`\n"
            "- Selection rule: highest average synthetic SSIM, then PSNR, then lowest hard-case runtime\n"
            f"- High-boost sigma: `{BOOST_SIGMA}`\n"
            f"- High-boost amount: `{BOOST_AMOUNT}`\n"
        )

    print(f"Saved synthetic metrics to {os.path.join(RESULTS_ROOT, 'synthetic_local_metrics.csv')}")
    print(f"Saved hard-case metrics to {os.path.join(RESULTS_ROOT, 'hard_case_local_metrics.csv')}")
    print(f"Saved best method record to {os.path.join(RESULTS_ROOT, 'best_local_method.json')}")
    print("Done! Local equalization branch is ready.")
