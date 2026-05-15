"""
Script 33: Build the report-facing OCR comparison figure.

Reads the expanded OCR outputs from script 32 and produces:

- results/final/ocr_handwriting_comparison.png

The figure shows representative writing.jpeg and Bentham line crops for each
method, plus dataset-split CER/WER bar charts.
"""
from __future__ import annotations

import csv
import os
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


OCR_ROOT = Path("results/experimental/ocr")
LINE_CROPS_ROOT = OCR_ROOT / "line_crops"
PREDICTIONS_PATH = OCR_ROOT / "per_line_predictions.csv"
METHOD_TABLE_PATH = OCR_ROOT / "ocr_method_table.csv"
FINAL_PATH = Path("results/final/ocr_handwriting_comparison.png")
METHOD_ORDER = ["original", "hf_tone", "hsi_color", "sauvola", "zerodcepp", "retinexnet"]
DATASET_ORDER = ["all", "writing_jpeg", "bentham"]
PREFERRED_WRITING_LINES = ["writing_l2", "writing_l6"]


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def parse_float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def short_text(value: str, width: int = 42) -> str:
    wrapped = textwrap.shorten(value.replace("\n", " "), width=width, placeholder="...")
    return wrapped.replace("$", r"\$")


def choose_representative_lines(predictions: list[dict[str, str]]) -> list[str]:
    lines_by_dataset: dict[str, list[str]] = {}
    original_rows_by_line: dict[str, dict[str, str]] = {}
    for row in predictions:
        if row["method_id"] != "original":
            continue
        lines_by_dataset.setdefault(row["dataset"], []).append(row["line_id"])
        original_rows_by_line[row["line_id"]] = row

    selected = [line_id for line_id in PREFERRED_WRITING_LINES if line_id in set(lines_by_dataset.get("writing_jpeg", []))]
    bentham_candidates = []
    for line_id in lines_by_dataset.get("bentham", []):
        gt = original_rows_by_line[line_id]["ground_truth"].strip()
        if gt and gt[0].isalnum() and 18 <= len(gt) <= 80:
            bentham_candidates.append(line_id)
    selected.extend(bentham_candidates[:2])
    if len(selected) < 4:
        for row in predictions:
            line_id = row["line_id"]
            if row["method_id"] == "original" and line_id not in selected:
                selected.append(line_id)
            if len(selected) >= 4:
                break
    return selected[:4]


def dataset_label(dataset: str) -> str:
    return {
        "all": "All",
        "writing_jpeg": "writing.jpeg",
        "bentham": "Bentham",
    }.get(dataset, dataset)


if __name__ == "__main__":
    if not PREDICTIONS_PATH.exists():
        raise FileNotFoundError(f"{PREDICTIONS_PATH} missing - run scripts/32 first.")
    if not METHOD_TABLE_PATH.exists():
        raise FileNotFoundError(f"{METHOD_TABLE_PATH} missing - run scripts/32 first.")

    predictions = read_csv_rows(PREDICTIONS_PATH)
    method_rows = read_csv_rows(METHOD_TABLE_PATH)

    all_rows = [row for row in method_rows if row["dataset"] == "all"]
    method_display_by_id = {row["method_id"]: row["method_display_name"] for row in all_rows}
    if not method_display_by_id:
        method_display_by_id = {row["method_id"]: row["method_display_name"] for row in method_rows}

    method_ids_used = [method_id for method_id in METHOD_ORDER if method_id in method_display_by_id]
    predictions_by_line_method = {(row["line_id"], row["method_id"]): row for row in predictions}
    representative_line_ids = choose_representative_lines(predictions)
    if not representative_line_ids:
        raise RuntimeError("No representative lines found in predictions CSV.")

    n_example_rows = len(representative_line_ids)
    n_cols = len(method_ids_used)
    figure = plt.figure(figsize=(3.0 * n_cols + 1.0, 1.85 * n_example_rows + 4.5))
    gs = figure.add_gridspec(
        n_example_rows + 2,
        n_cols,
        height_ratios=[1.65] * n_example_rows + [1.25, 1.25],
        hspace=0.62,
        wspace=0.18,
    )

    for row_index, line_id in enumerate(representative_line_ids):
        original_row = predictions_by_line_method.get((line_id, "original"))
        dataset = original_row["dataset"] if original_row else ""
        gt = original_row["ground_truth"] if original_row else ""
        for col_index, method_id in enumerate(method_ids_used):
            ax = figure.add_subplot(gs[row_index, col_index])
            crop_path = LINE_CROPS_ROOT / method_id / f"{line_id}.png"
            if crop_path.exists():
                ax.imshow(np.array(Image.open(crop_path).convert("L")), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(method_display_by_id[method_id], fontsize=10, pad=8)
            prediction_row = predictions_by_line_method.get((line_id, method_id))
            if prediction_row is None:
                continue
            cer_value = 100.0 * parse_float(prediction_row["cer"])
            pred = short_text(prediction_row["prediction"])
            label = f"{dataset_label(dataset)} / {line_id}\nGT: {short_text(gt)}\nPred: {pred}\nCER {cer_value:.1f}%"
            ax.set_xlabel(label, fontsize=7.4, labelpad=3)

    table_by_dataset_method = {(row["dataset"], row["method_id"]): row for row in method_rows}
    dataset_ids = [dataset for dataset in DATASET_ORDER if any(row["dataset"] == dataset for row in method_rows)]
    colors = {"all": "#3f6fa6", "writing_jpeg": "#8a9b35", "bentham": "#c05a46"}
    x_positions = np.arange(len(method_ids_used))
    width = 0.24 if len(dataset_ids) >= 3 else 0.32
    offsets = np.linspace(-width * (len(dataset_ids) - 1) / 2, width * (len(dataset_ids) - 1) / 2, len(dataset_ids))

    for metric_index, (metric_key, title) in enumerate([("corpus_cer", "CER (%)"), ("corpus_wer", "WER (%)")]):
        ax = figure.add_subplot(gs[n_example_rows + metric_index, :])
        max_value = 0.0
        for dataset, offset in zip(dataset_ids, offsets):
            values = []
            for method_id in method_ids_used:
                row = table_by_dataset_method.get((dataset, method_id))
                value = 100.0 * parse_float(row[metric_key]) if row else float("nan")
                values.append(value)
                if not np.isnan(value):
                    max_value = max(max_value, value)
            ax.bar(x_positions + offset, values, width, label=dataset_label(dataset), color=colors.get(dataset))
        ax.set_title(f"Corpus {title} by dataset (lower is better)", fontsize=11)
        ax.set_ylabel(title)
        ax.set_xticks(x_positions)
        ax.set_xticklabels([method_display_by_id[method_id] for method_id in method_ids_used], rotation=0, fontsize=9)
        ax.set_ylim(0, max_value * 1.18 if max_value > 0 else 1)
        ax.grid(axis="y", alpha=0.24)
        ax.legend(loc="upper right", ncols=len(dataset_ids), fontsize=8)

    model_id = method_rows[0].get("model_id", "TrOCR") if method_rows else "TrOCR"
    figure.suptitle(
        f"Handwriting OCR comparison on bad-scan writing plus Bentham historical lines ({model_id})",
        fontsize=13,
        y=0.995,
    )
    figure.subplots_adjust(top=0.94, bottom=0.055, left=0.035, right=0.985)
    FINAL_PATH.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(FINAL_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved OCR comparison figure to {FINAL_PATH}")
