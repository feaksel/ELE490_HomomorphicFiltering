"""
Script 22: Generate frequency-domain and histogram analysis for the active showcase.
This script compares each original grayscale input against the current final
regular-pipeline output and saves report-ready plots plus numeric summaries.
"""
import csv
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.analysis_visuals import (
    save_before_after_analysis_figure,
    save_metric_delta_dashboard,
    summarize_before_after_metrics,
)
from utils.showcase_pipeline import MAX_DIMENSION, apply_regular_showcase_pipeline, load_resized_grayscale


SHOWCASE_IMAGES = [
    ("cardboard.jpg", "cardboard"),
    ("markers.jpg", "markers"),
    ("page.jpeg", "page"),
    ("pillar.jpg", "pillar"),
    ("seat.jpg", "seat"),
]


def write_summary_csv(records, output_path):
    fieldnames = [
        "label",
        "before_mean",
        "after_mean",
        "delta_mean",
        "before_std",
        "after_std",
        "delta_std",
        "before_entropy",
        "after_entropy",
        "delta_entropy",
        "before_spread_1_99",
        "after_spread_1_99",
        "delta_spread_1_99",
        "before_high_frequency_share",
        "after_high_frequency_share",
        "delta_high_frequency_share",
        "mean_abs_difference",
        "max_abs_difference",
    ]

    with open(output_path, "w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "label": record["label"],
                    "before_mean": f"{record['before']['mean']:.4f}",
                    "after_mean": f"{record['after']['mean']:.4f}",
                    "delta_mean": f"{record['delta_mean']:.4f}",
                    "before_std": f"{record['before']['std']:.4f}",
                    "after_std": f"{record['after']['std']:.4f}",
                    "delta_std": f"{record['delta_std']:.4f}",
                    "before_entropy": f"{record['before']['entropy']:.6f}",
                    "after_entropy": f"{record['after']['entropy']:.6f}",
                    "delta_entropy": f"{record['delta_entropy']:.6f}",
                    "before_spread_1_99": f"{record['before']['spread_1_99']:.4f}",
                    "after_spread_1_99": f"{record['after']['spread_1_99']:.4f}",
                    "delta_spread_1_99": f"{record['delta_spread_1_99']:.4f}",
                    "before_high_frequency_share": f"{record['before_frequency']['high_frequency_share']:.6f}",
                    "after_high_frequency_share": f"{record['after_frequency']['high_frequency_share']:.6f}",
                    "delta_high_frequency_share": f"{record['delta_high_frequency_share']:.6f}",
                    "mean_abs_difference": f"{record['mean_abs_difference']:.4f}",
                    "max_abs_difference": f"{record['max_abs_difference']:.4f}",
                }
            )


def write_summary_markdown(records, output_path):
    lines = [
        "# Showcase Analysis Summary",
        "",
        "| Image | Mean | Std | Entropy | HF Share | Mean Abs Diff |",
        "| --- | --- | --- | --- | --- | --- |",
    ]

    for record in records:
        lines.append(
            "| "
            f"{record['label']} | "
            f"{record['before']['mean']:.1f} -> {record['after']['mean']:.1f} | "
            f"{record['before']['std']:.1f} -> {record['after']['std']:.1f} | "
            f"{record['before']['entropy']:.3f} -> {record['after']['entropy']:.3f} | "
            f"{100.0 * record['before_frequency']['high_frequency_share']:.1f}% -> "
            f"{100.0 * record['after_frequency']['high_frequency_share']:.1f}% | "
            f"{record['mean_abs_difference']:.1f} |"
        )

    with open(output_path, "w", encoding="utf-8") as output_file:
        output_file.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    results_dir = os.path.join("results", "analysis")
    os.makedirs(results_dir, exist_ok=True)

    print("Running before/after frequency analysis for the active showcase...")
    print(f"  Resizing long image dimension to at most {MAX_DIMENSION} pixels for the showcase run.")

    records = []
    for image_name, base_name in SHOWCASE_IMAGES:
        image_path = os.path.join("images", image_name)
        display_name = base_name.replace("_", " ").title()

        print(f"Analyzing image: {image_path}")
        gray_array = load_resized_grayscale(image_path)
        pipeline_result = apply_regular_showcase_pipeline(gray_array, base_name)
        final_result = pipeline_result["final"]

        analysis_path = os.path.join(results_dir, f"{base_name}_before_after_analysis.png")
        save_before_after_analysis_figure(
            gray_array,
            final_result,
            analysis_path,
            case_title=display_name,
            after_label=pipeline_result["title"],
        )

        record = summarize_before_after_metrics(gray_array, final_result, label=base_name)
        records.append(record)

        print(f"  Saved analysis figure: {analysis_path}")
        print(
            f"  mean={record['before']['mean']:.1f}->{record['after']['mean']:.1f}, "
            f"std={record['before']['std']:.1f}->{record['after']['std']:.1f}, "
            f"entropy={record['before']['entropy']:.3f}->{record['after']['entropy']:.3f}"
        )

    dashboard_path = os.path.join(results_dir, "showcase_metric_deltas.png")
    csv_path = os.path.join(results_dir, "showcase_analysis_metrics.csv")
    markdown_path = os.path.join(results_dir, "showcase_analysis_summary.md")

    save_metric_delta_dashboard(records, dashboard_path)
    write_summary_csv(records, csv_path)
    write_summary_markdown(records, markdown_path)

    print(f"Saved metric dashboard: {dashboard_path}")
    print(f"Saved summary CSV: {csv_path}")
    print(f"Saved summary markdown: {markdown_path}")
    print("Done! Frequency-domain analysis outputs are ready.")
