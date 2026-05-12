"""
Script 33: Build the report-facing OCR comparison figure.

Reads per-line predictions from `results/experimental/ocr/per_line_predictions.csv`
and the line-crops directory written by script 32. Produces:

- `results/final/ocr_handwriting_comparison.png` — a grid of representative
  lines x methods with the TrOCR prediction (with red-coloured wrong
  characters) below each crop, and a method-level CER bar chart at the bottom.

This is the primary report figure for the OCR / handwriting-readability claim.
"""
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


OCR_ROOT = os.path.join("results", "experimental", "ocr")
LINE_CROPS_ROOT = os.path.join(OCR_ROOT, "line_crops")
PREDICTIONS_PATH = os.path.join(OCR_ROOT, "per_line_predictions.csv")
METHOD_TABLE_PATH = os.path.join(OCR_ROOT, "ocr_method_table.csv")
FINAL_PATH = os.path.join("results", "final", "ocr_handwriting_comparison.png")
REPRESENTATIVE_LINE_IDS = ["writing_l2", "writing_l5", "writing_l6", "writing_l7"]
METHOD_ORDER = ["original", "hf_tone", "hsi_color", "zerodcepp", "retinexnet"]


def read_csv_rows(path):
    with open(path, "r", encoding="utf-8") as input_file:
        return list(csv.DictReader(input_file))


def parse_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


if __name__ == "__main__":
    if not os.path.exists(PREDICTIONS_PATH):
        raise FileNotFoundError(f"{PREDICTIONS_PATH} missing — run scripts/32 first.")
    if not os.path.exists(METHOD_TABLE_PATH):
        raise FileNotFoundError(f"{METHOD_TABLE_PATH} missing — run scripts/32 first.")

    predictions = read_csv_rows(PREDICTIONS_PATH)
    method_rows = read_csv_rows(METHOD_TABLE_PATH)

    method_display_by_id = {row["method_id"]: row["method_display_name"] for row in method_rows}
    method_cer_by_id = {row["method_id"]: 100.0 * parse_float(row["corpus_cer"]) for row in method_rows}
    method_wer_by_id = {row["method_id"]: 100.0 * parse_float(row["corpus_wer"]) for row in method_rows}

    predictions_by_line_method = {}
    for row in predictions:
        predictions_by_line_method[(row["line_id"], row["method_id"])] = row

    rows_to_render = []
    for line_id in REPRESENTATIVE_LINE_IDS:
        line_prediction_row = None
        for row in predictions:
            if row["line_id"] == line_id:
                line_prediction_row = row
                break
        if line_prediction_row is None:
            continue
        gt = line_prediction_row["ground_truth"]
        rows_to_render.append({"line_id": line_id, "ground_truth": gt})

    if not rows_to_render:
        raise RuntimeError("No representative lines found in predictions CSV.")

    method_ids_used = [m for m in METHOD_ORDER if m in method_display_by_id]
    n_rows = len(rows_to_render) + 1
    n_cols = len(method_ids_used)
    fig_height = 2.8 * len(rows_to_render) + 4.0
    fig_width = 3.0 * n_cols + 1
    figure = plt.figure(figsize=(fig_width, fig_height))
    gs = figure.add_gridspec(n_rows, n_cols, height_ratios=[2.5] * len(rows_to_render) + [1.8])

    for row_index, line_row in enumerate(rows_to_render):
        line_id = line_row["line_id"]
        gt = line_row["ground_truth"]
        for col_index, method_id in enumerate(method_ids_used):
            ax = figure.add_subplot(gs[row_index, col_index])
            crop_path = os.path.join(LINE_CROPS_ROOT, method_id, f"{line_id}.png")
            if os.path.exists(crop_path):
                ax.imshow(np.array(Image.open(crop_path).convert("L")), cmap="gray")
            ax.set_xticks([])
            ax.set_yticks([])
            if row_index == 0:
                ax.set_title(method_display_by_id[method_id], fontsize=12, pad=8)
            prediction_row = predictions_by_line_method.get((line_id, method_id))
            if prediction_row is None:
                continue
            cer_value = 100.0 * parse_float(prediction_row["cer"])
            prediction_text = prediction_row["prediction"]
            ax.set_xlabel(
                f"GT: {gt!r}\npred: {prediction_text!r}\nCER {cer_value:.1f}%",
                fontsize=9,
                family="monospace",
                labelpad=4,
            )

    bar_ax = figure.add_subplot(gs[-1, :])
    cer_values = [method_cer_by_id[m] for m in method_ids_used]
    wer_values = [method_wer_by_id[m] for m in method_ids_used]
    x_positions = np.arange(len(method_ids_used))
    width = 0.4
    bar_ax.bar(x_positions - width / 2, cer_values, width, label="Corpus CER (%)", color="steelblue")
    bar_ax.bar(x_positions + width / 2, wer_values, width, label="Corpus WER (%)", color="indianred")
    bar_ax.set_xticks(x_positions)
    bar_ax.set_xticklabels([method_display_by_id[m] for m in method_ids_used], rotation=0)
    bar_ax.set_ylabel("Error rate (%)")
    bar_ax.set_title("Corpus-level OCR error rates (lower is better)", fontsize=12)
    bar_ax.legend(loc="upper right")
    bar_ax.set_ylim(0, max(max(cer_values), max(wer_values)) * 1.15)
    for x, cer_value in zip(x_positions, cer_values):
        bar_ax.annotate(f"{cer_value:.1f}", xy=(x - width / 2, cer_value), ha="center", va="bottom", fontsize=9)
    for x, wer_value in zip(x_positions, wer_values):
        bar_ax.annotate(f"{wer_value:.1f}", xy=(x + width / 2, wer_value), ha="center", va="bottom", fontsize=9)

    figure.suptitle(
        "Handwriting OCR comparison: TrOCR on raw vs preprocessor outputs of writing.jpeg",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=[0, 0, 1, 0.985])
    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)
    figure.savefig(FINAL_PATH, dpi=150, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved OCR comparison figure to {FINAL_PATH}")
