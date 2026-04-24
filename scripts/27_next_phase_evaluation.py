"""
Script 27: Unified evaluation for the next-phase experimental branch.
This script aggregates the local-equalization branch, optional CNN outputs,
and the homomorphic baseline into synthetic tables, hard-case proxy tables,
summary figures, and a short recommendation note.
"""
import csv
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid
from utils.experimental_methods import (
    apply_high_boost,
    apply_local_method,
    apply_real_hard_case_baseline,
    crop_image,
    get_local_method_spec,
    load_hard_case_manifest,
    load_resized_grayscale,
    load_resized_rgb,
    resolve_crop_box,
    resolve_existing_path,
    rgb_to_grayscale_uint8,
)

from utils.cnn_baseline import load_torchscript_model, run_torchscript_model


RESULTS_ROOT = os.path.join("results", "experimental", "evaluation")
LOCAL_ROOT = os.path.join("results", "experimental", "local_eq")
CNN_ROOT = os.path.join("results", "experimental", "cnn")
REPRESENTATIVE_CASE_IDS = ["tun", "page", "seat", "markers"]


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as input_file:
        reader = csv.DictReader(input_file)
        rows = []
        for row in reader:
            parsed = {}
            for key, value in row.items():
                if value is None or value == "":
                    parsed[key] = value
                    continue
                try:
                    parsed[key] = float(value)
                except ValueError:
                    parsed[key] = value
            rows.append(parsed)
        return rows


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


def aggregate_by_method(rows, method_key, numeric_keys):
    grouped = {}
    for row in rows:
        grouped.setdefault(row[method_key], []).append(row)

    aggregated = {}
    for method_id, group_rows in grouped.items():
        aggregated[method_id] = {
            key: float(np.mean([group_row[key] for group_row in group_rows])) for key in numeric_keys
        }
        aggregated[method_id]["display_name"] = group_rows[0].get("display_name", method_id)
    return aggregated


def rank_methods(method_rows, metric_key, descending=True):
    ordered = sorted(
        method_rows.items(),
        key=lambda item: item[1][metric_key],
        reverse=descending,
    )
    return {method_id: index + 1 for index, (method_id, _) in enumerate(ordered)}


def load_uint8(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


def benchmark_runtime(function, repeats=3):
    runtimes = []
    for _ in range(repeats):
        start_time = time.perf_counter()
        function()
        runtimes.append(1000.0 * (time.perf_counter() - start_time))
    return float(np.mean(runtimes))


if __name__ == "__main__":
    ensure_directory(RESULTS_ROOT)

    best_method_path = os.path.join(LOCAL_ROOT, "best_local_method.json")
    synthetic_metrics_path = os.path.join(LOCAL_ROOT, "synthetic_local_metrics.csv")
    hard_metrics_path = os.path.join(LOCAL_ROOT, "hard_case_local_metrics.csv")
    if not os.path.exists(best_method_path) or not os.path.exists(synthetic_metrics_path) or not os.path.exists(hard_metrics_path):
        raise FileNotFoundError(
            "Local-equalization outputs are missing. Run scripts/25_local_equalization_hard_cases.py first."
        )

    with open(best_method_path, "r", encoding="utf-8") as input_file:
        best_method_record = json.load(input_file)

    manifest = load_hard_case_manifest()
    synthetic_rows = read_csv_rows(synthetic_metrics_path)
    hard_rows = read_csv_rows(hard_metrics_path)

    cnn_available = False
    cnn_model_id = None
    cnn_model_name = None
    cnn_synthetic_rows = []
    cnn_hard_rows = []
    cnn_model_path = os.path.join(CNN_ROOT, "cnn_model.json")
    cnn_synthetic_path = os.path.join(CNN_ROOT, "cnn_synthetic_metrics.csv")
    cnn_hard_path = os.path.join(CNN_ROOT, "cnn_hard_case_metrics.csv")
    if os.path.exists(cnn_model_path) and os.path.exists(cnn_synthetic_path) and os.path.exists(cnn_hard_path):
        with open(cnn_model_path, "r", encoding="utf-8") as input_file:
            cnn_model = json.load(input_file)
        cnn_model_id = cnn_model["id"]
        cnn_model_name = cnn_model["display_name"]
        cnn_synthetic_rows = read_csv_rows(cnn_synthetic_path)
        cnn_hard_rows = read_csv_rows(cnn_hard_path)
        cnn_available = True

    selected_synthetic_ids = [
        "homomorphic_synthetic_baseline",
        best_method_record["selected_method_id"],
        best_method_record["boosted_method_id"],
    ]
    selected_hard_ids = [
        "hf_tone_baseline",
        best_method_record["selected_method_id"],
        best_method_record["boosted_method_id"],
    ]
    if cnn_available:
        selected_synthetic_ids.append(cnn_model_id)
        selected_hard_ids.append(cnn_model_id)
        synthetic_rows.extend(cnn_synthetic_rows)
        hard_rows.extend(cnn_hard_rows)

    if cnn_available:
        for row in cnn_synthetic_rows:
            row["method_id"] = row["model_id"]
            row["display_name"] = row["display_name"]
        for row in cnn_hard_rows:
            row["method_id"] = row["model_id"]
            row["display_name"] = row["display_name"]

    synthetic_rows = [dict(row, method_id=row.get("method_id", row.get("model_id"))) for row in synthetic_rows]
    hard_rows = [dict(row, method_id=row.get("method_id", row.get("model_id"))) for row in hard_rows]

    synthetic_aggregates = aggregate_by_method(
        [row for row in synthetic_rows if row["method_id"] in selected_synthetic_ids],
        "method_id",
        ["mse", "psnr", "ssim", "runtime_ms"],
    )
    hard_aggregates = aggregate_by_method(
        [row for row in hard_rows if row["method_id"] in selected_hard_ids],
        "method_id",
        [
            "entropy",
            "spread_1_99",
            "high_frequency_share",
            "mean_gradient",
            "clipped_pixel_ratio",
            "runtime_ms",
        ],
    )

    runtime_cases = []
    for case in manifest:
        case_path = resolve_existing_path(case["path"])
        runtime_cases.append(
            {
                "id": case["id"],
                "gray": load_resized_grayscale(case_path),
                "rgb": load_resized_rgb(case_path),
            }
        )

    best_method_spec = get_local_method_spec(best_method_record["selected_method_id"])

    runtime_functions = {
        "hf_tone_baseline": lambda case: apply_real_hard_case_baseline(case["gray"], case["id"]),
        best_method_record["selected_method_id"]: lambda case: apply_local_method(case["gray"], best_method_spec),
        best_method_record["boosted_method_id"]: lambda case: apply_high_boost(
            apply_local_method(case["gray"], best_method_spec),
            sigma=best_method_record["boost_sigma"],
            amount=best_method_record["boost_amount"],
        ),
    }

    if cnn_available:
        cnn_model_instance = load_torchscript_model(cnn_model["path"], device="cpu")
        runtime_functions[cnn_model_id] = lambda case: rgb_to_grayscale_uint8(
            run_torchscript_model(cnn_model_instance, case["rgb"], device="cpu")
        )

    for method_id, runtime_function in runtime_functions.items():
        per_case_runtimes = [benchmark_runtime(lambda case=case: runtime_function(case), repeats=3) for case in runtime_cases]
        if method_id in hard_aggregates:
            hard_aggregates[method_id]["runtime_ms"] = float(np.mean(per_case_runtimes))

    synthetic_table_rows = []
    synthetic_markdown_rows = []
    for method_id in selected_synthetic_ids:
        if method_id not in synthetic_aggregates:
            continue
        aggregate = synthetic_aggregates[method_id]
        synthetic_table_rows.append(
            {
                "method_id": method_id,
                "display_name": aggregate["display_name"],
                "avg_mse": aggregate["mse"],
                "avg_psnr": aggregate["psnr"],
                "avg_ssim": aggregate["ssim"],
                "avg_runtime_ms": aggregate["runtime_ms"],
            }
        )
        synthetic_markdown_rows.append(
            [
                aggregate["display_name"],
                f"{aggregate['mse']:.3f}",
                f"{aggregate['psnr']:.3f}",
                f"{aggregate['ssim']:.4f}",
                f"{aggregate['runtime_ms']:.2f}",
            ]
        )

    hard_table_rows = []
    hard_markdown_rows = []
    for method_id in selected_hard_ids:
        if method_id not in hard_aggregates:
            continue
        aggregate = hard_aggregates[method_id]
        hard_table_rows.append(
            {
                "method_id": method_id,
                "display_name": aggregate["display_name"],
                "avg_entropy": aggregate["entropy"],
                "avg_spread_1_99": aggregate["spread_1_99"],
                "avg_high_frequency_share": aggregate["high_frequency_share"],
                "avg_mean_gradient": aggregate["mean_gradient"],
                "avg_clipped_pixel_ratio": aggregate["clipped_pixel_ratio"],
                "avg_runtime_ms": aggregate["runtime_ms"],
            }
        )
        hard_markdown_rows.append(
            [
                aggregate["display_name"],
                f"{aggregate['entropy']:.3f}",
                f"{aggregate['spread_1_99']:.2f}",
                f"{100.0 * aggregate['high_frequency_share']:.2f}",
                f"{aggregate['mean_gradient']:.3f}",
                f"{100.0 * aggregate['clipped_pixel_ratio']:.2f}",
                f"{aggregate['runtime_ms']:.2f}",
            ]
        )

    write_csv(
        os.path.join(RESULTS_ROOT, "synthetic_method_table.csv"),
        ["method_id", "display_name", "avg_mse", "avg_psnr", "avg_ssim", "avg_runtime_ms"],
        synthetic_table_rows,
    )
    write_csv(
        os.path.join(RESULTS_ROOT, "hard_case_method_table.csv"),
        [
            "method_id",
            "display_name",
            "avg_entropy",
            "avg_spread_1_99",
            "avg_high_frequency_share",
            "avg_mean_gradient",
            "avg_clipped_pixel_ratio",
            "avg_runtime_ms",
        ],
        hard_table_rows,
    )

    write_markdown_table(
        os.path.join(RESULTS_ROOT, "synthetic_method_table.md"),
        "Synthetic Method Table",
        ["Method", "Avg MSE", "Avg PSNR", "Avg SSIM", "Avg Runtime (ms)"],
        synthetic_markdown_rows,
    )
    write_markdown_table(
        os.path.join(RESULTS_ROOT, "hard_case_method_table.md"),
        "Hard-Case Proxy Metric Table",
        ["Method", "Avg Entropy", "Avg P99-P1", "Avg HF Share (%)", "Avg Mean Gradient", "Avg Clipped (%)", "Avg Runtime (ms)"],
        hard_markdown_rows,
    )

    method_ids_for_tradeoff = [row["method_id"] for row in synthetic_table_rows if row["method_id"] in hard_aggregates]
    tradeoff_rows = {
        method_id: {
            "display_name": synthetic_aggregates[method_id]["display_name"],
            "avg_ssim": synthetic_aggregates[method_id]["ssim"],
            "avg_runtime_ms": hard_aggregates[method_id]["runtime_ms"],
            "avg_entropy": hard_aggregates[method_id]["entropy"],
            "avg_spread_1_99": hard_aggregates[method_id]["spread_1_99"],
            "avg_high_frequency_share": hard_aggregates[method_id]["high_frequency_share"],
            "avg_mean_gradient": hard_aggregates[method_id]["mean_gradient"],
            "avg_clipped_pixel_ratio": hard_aggregates[method_id]["clipped_pixel_ratio"],
        }
        for method_id in method_ids_for_tradeoff
    }

    ssim_ranks = rank_methods(tradeoff_rows, "avg_ssim", descending=True)
    runtime_ranks = rank_methods(tradeoff_rows, "avg_runtime_ms", descending=False)
    entropy_ranks = rank_methods(tradeoff_rows, "avg_entropy", descending=True)
    spread_ranks = rank_methods(tradeoff_rows, "avg_spread_1_99", descending=True)
    hf_ranks = rank_methods(tradeoff_rows, "avg_high_frequency_share", descending=True)
    gradient_ranks = rank_methods(tradeoff_rows, "avg_mean_gradient", descending=True)
    clipped_ranks = rank_methods(tradeoff_rows, "avg_clipped_pixel_ratio", descending=False)

    for method_id in tradeoff_rows:
        hard_quality_rank = np.mean(
            [
                entropy_ranks[method_id],
                spread_ranks[method_id],
                hf_ranks[method_id],
                gradient_ranks[method_id],
                clipped_ranks[method_id],
            ]
        )
        tradeoff_rows[method_id]["synthetic_rank"] = ssim_ranks[method_id]
        tradeoff_rows[method_id]["hard_quality_rank"] = hard_quality_rank
        tradeoff_rows[method_id]["runtime_rank"] = runtime_ranks[method_id]
        tradeoff_rows[method_id]["tradeoff_rank"] = ssim_ranks[method_id] + hard_quality_rank + runtime_ranks[method_id]

    tradeoff_winner_id = min(
        tradeoff_rows,
        key=lambda method_id: (
            tradeoff_rows[method_id]["tradeoff_rank"],
            tradeoff_rows[method_id]["synthetic_rank"],
            tradeoff_rows[method_id]["runtime_rank"],
        ),
    )

    manifest_by_id = {case["id"]: case for case in manifest}
    representative_ids = [case_id for case_id in REPRESENTATIVE_CASE_IDS if case_id in manifest_by_id]
    if not representative_ids:
        representative_ids = [case["id"] for case in manifest[:4]]

    method_visuals = [
        ("Original", None),
        ("HF + Tone", os.path.join(LOCAL_ROOT, "hard_cases", "hf_tone_baseline")),
        (best_method_record["selected_display_name"], os.path.join(LOCAL_ROOT, "hard_cases", best_method_record["selected_method_id"])),
        (best_method_record["boosted_display_name"], os.path.join(LOCAL_ROOT, "hard_cases", best_method_record["boosted_method_id"])),
    ]
    if cnn_available:
        method_visuals.append((cnn_model_name, os.path.join(CNN_ROOT, cnn_model_id, "hard_cases_gray")))

    full_rows = []
    crop_rows = []
    for case_id in representative_ids:
        case_record = manifest_by_id[case_id]
        original_image = load_resized_grayscale(resolve_existing_path(case_record["path"]))
        crop_box = resolve_crop_box(original_image.shape, case_record.get("crop_rel"))

        full_row = [(original_image, f"{case_id}: Original")]
        crop_row = [(crop_image(original_image, crop_box), f"{case_id}: Crop")]
        for title, directory in method_visuals[1:]:
            method_image = load_uint8(os.path.join(directory, f"{case_id}.png"))
            full_row.append((method_image, f"{case_id}: {title}"))
            crop_row.append((crop_image(method_image, crop_box), f"{case_id}: {title}"))

        full_rows.append(full_row)
        crop_rows.append(crop_row)

    save_comparison_grid(full_rows, os.path.join(RESULTS_ROOT, "hard_case_visual_overview.png"))
    save_comparison_grid(crop_rows, os.path.join(RESULTS_ROOT, "hard_case_crop_overview.png"))

    fastest_method_id = min(hard_aggregates, key=lambda method_id: hard_aggregates[method_id]["runtime_ms"])
    best_synthetic_method_id = max(synthetic_aggregates, key=lambda method_id: synthetic_aggregates[method_id]["ssim"])

    summary_lines = [
        "# Next-Phase Evaluation Summary",
        "",
        f"- Synthetic quality leader: `{synthetic_aggregates[best_synthetic_method_id]['display_name']}`",
        f"- Hard-case runtime leader: `{hard_aggregates[fastest_method_id]['display_name']}`",
        (
            "- Balanced quality/runtime winner: "
            f"`{tradeoff_rows[tradeoff_winner_id]['display_name']}` "
            "(ranked by synthetic SSIM, aggregate hard-case proxy quality, and CPU runtime)"
        ),
        f"- Selected best local method from script 25: `{best_method_record['selected_display_name']}`",
        f"- Selected boosted local branch: `{best_method_record['boosted_display_name']}`",
    ]
    if cnn_available:
        summary_lines.append(f"- CNN comparator included: `{cnn_model_name}`")
    else:
        summary_lines.append("- CNN comparator not included: no local TorchScript weights were available.")

    with open(os.path.join(RESULTS_ROOT, "next_phase_summary.md"), "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(summary_lines) + "\n")

    print(f"Saved synthetic table to {os.path.join(RESULTS_ROOT, 'synthetic_method_table.csv')}")
    print(f"Saved hard-case table to {os.path.join(RESULTS_ROOT, 'hard_case_method_table.csv')}")
    print(f"Saved visual overview to {os.path.join(RESULTS_ROOT, 'hard_case_visual_overview.png')}")
    print(f"Saved crop overview to {os.path.join(RESULTS_ROOT, 'hard_case_crop_overview.png')}")
    print("Done! Next-phase evaluation package is ready.")
