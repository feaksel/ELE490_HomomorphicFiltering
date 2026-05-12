"""
Script 32: Handwriting OCR readability pipeline.

For every line in `configs/handwriting_lines_manifest.json`:
- crop the line out of the original AND out of each preprocessor's output
- run TrOCR (microsoft/trocr-base-handwritten) on each crop
- compute CER and WER against the manually-transcribed ground truth

Writes per-line predictions + per-method aggregate metrics to
`results/experimental/ocr/`.

Methods compared:
- original       : raw writing.jpeg
- hf_tone        : results/experimental/local_eq/hard_cases/hf_tone_baseline/writing.png  (HF + Tone grayscale)
- hsi_color      : results/experimental/hsi/hard_cases_rgb/writing.png                    (HSI HF + Tone color)
- zerodcepp      : results/experimental/cnn/zerodcepp/hard_cases_rgb/writing.png          (Zero-DCE++)
- retinexnet     : results/experimental/cnn/retinexnet/hard_cases_rgb/writing.png         (RetinexNet)
"""
import csv
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiwer import cer, wer


MANIFEST_PATH = os.path.join("configs", "handwriting_lines_manifest.json")
RESULTS_ROOT = os.path.join("results", "experimental", "ocr")
TROCR_MODEL_ID = os.environ.get("TROCR_MODEL_ID", "microsoft/trocr-large-handwritten")

METHODS = [
    {
        "id": "original",
        "display_name": "Original",
        "image_path": "images/writing.jpeg",
    },
    {
        "id": "hf_tone",
        "display_name": "HF + Tone (Gray)",
        "image_path": os.path.join("results", "experimental", "local_eq", "hard_cases", "hf_tone_baseline", "writing.png"),
    },
    {
        "id": "hsi_color",
        "display_name": "HSI HF + Tone",
        "image_path": os.path.join("results", "experimental", "hsi", "hard_cases_rgb", "writing.png"),
    },
    {
        "id": "zerodcepp",
        "display_name": "Zero-DCE++",
        "image_path": os.path.join("results", "experimental", "cnn", "zerodcepp", "hard_cases_rgb", "writing.png"),
    },
    {
        "id": "retinexnet",
        "display_name": "RetinexNet",
        "image_path": os.path.join("results", "experimental", "cnn", "retinexnet", "hard_cases_rgb", "writing.png"),
    },
]


def ensure_directory(path):
    os.makedirs(path, exist_ok=True)


def load_method_image(path):
    return Image.open(path).convert("RGB")


def crop_line(image, bbox_abs):
    x1, y1, x2, y2 = bbox_abs
    return image.crop((x1, y1, x2, y2))


def trocr_predict(processor, model, image, device):
    pixel_values = processor(images=image, return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=64)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


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


if __name__ == "__main__":
    ensure_directory(RESULTS_ROOT)
    for method in METHODS:
        ensure_directory(os.path.join(RESULTS_ROOT, "line_crops", method["id"]))

    with open(MANIFEST_PATH, "r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)

    print(f"Loading TrOCR model {TROCR_MODEL_ID}...")
    processor = TrOCRProcessor.from_pretrained(TROCR_MODEL_ID)
    model = VisionEncoderDecoderModel.from_pretrained(TROCR_MODEL_ID)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  device = {device}")

    per_line_rows = []
    aggregate_inputs = {method["id"]: {"hyps": [], "refs": []} for method in METHODS}
    method_runtimes = {method["id"]: 0.0 for method in METHODS}
    method_runtime_counts = {method["id"]: 0 for method in METHODS}

    for method in METHODS:
        method_id = method["id"]
        print(f"\n=== Method: {method['display_name']} ===")
        image_path = method["image_path"]
        if not os.path.exists(image_path):
            print(f"  Skipping — missing input: {image_path}")
            continue
        method_image = load_method_image(image_path)
        for line in manifest:
            crop = crop_line(method_image, line["bbox_abs"])
            crop_path = os.path.join(RESULTS_ROOT, "line_crops", method_id, f"{line['id']}.png")
            crop.save(crop_path)

            start_time = time.perf_counter()
            prediction = trocr_predict(processor, model, crop, device)
            runtime_ms = 1000.0 * (time.perf_counter() - start_time)
            method_runtimes[method_id] += runtime_ms
            method_runtime_counts[method_id] += 1

            ref = line["transcript"]
            line_cer = float(cer(ref, prediction))
            line_wer = float(wer(ref, prediction))

            per_line_rows.append(
                {
                    "line_id": line["id"],
                    "method_id": method_id,
                    "method_display_name": method["display_name"],
                    "ground_truth": ref,
                    "prediction": prediction,
                    "cer": line_cer,
                    "wer": line_wer,
                    "runtime_ms": runtime_ms,
                }
            )
            aggregate_inputs[method_id]["hyps"].append(prediction)
            aggregate_inputs[method_id]["refs"].append(ref)
            print(f"  {line['id']:>10s} | GT: {ref!r:<40s} | pred: {prediction!r:<40s} | CER={line_cer:.3f}")

    aggregate_rows = []
    aggregate_markdown_rows = []
    for method in METHODS:
        method_id = method["id"]
        hyps = aggregate_inputs[method_id]["hyps"]
        refs = aggregate_inputs[method_id]["refs"]
        if not hyps:
            continue
        method_cer = float(cer(refs, hyps))
        method_wer = float(wer(refs, hyps))
        method_lines = method_runtime_counts[method_id]
        method_total_runtime = method_runtimes[method_id]
        method_avg_runtime = method_total_runtime / max(method_lines, 1)
        aggregate_rows.append(
            {
                "method_id": method_id,
                "method_display_name": method["display_name"],
                "num_lines": method_lines,
                "corpus_cer": method_cer,
                "corpus_wer": method_wer,
                "avg_runtime_ms": method_avg_runtime,
            }
        )
        aggregate_markdown_rows.append(
            [
                method["display_name"],
                str(method_lines),
                f"{100.0 * method_cer:.2f}",
                f"{100.0 * method_wer:.2f}",
                f"{method_avg_runtime:.1f}",
            ]
        )

    write_csv(
        os.path.join(RESULTS_ROOT, "per_line_predictions.csv"),
        ["line_id", "method_id", "method_display_name", "ground_truth", "prediction", "cer", "wer", "runtime_ms"],
        per_line_rows,
    )
    write_csv(
        os.path.join(RESULTS_ROOT, "ocr_method_table.csv"),
        ["method_id", "method_display_name", "num_lines", "corpus_cer", "corpus_wer", "avg_runtime_ms"],
        aggregate_rows,
    )
    write_markdown_table(
        os.path.join(RESULTS_ROOT, "ocr_method_table.md"),
        f"Handwriting OCR Readability ({TROCR_MODEL_ID} on writing.jpeg)",
        ["Method", "Lines", "Corpus CER (%)", "Corpus WER (%)", "Avg runtime (ms / line)"],
        aggregate_markdown_rows,
    )

    print("\nSaved:")
    print(f"  {os.path.join(RESULTS_ROOT, 'per_line_predictions.csv')}")
    print(f"  {os.path.join(RESULTS_ROOT, 'ocr_method_table.csv')}")
    print(f"  {os.path.join(RESULTS_ROOT, 'ocr_method_table.md')}")
    print("Done! OCR handwriting evaluation complete.")
