"""
Script 32: Handwriting OCR readability pipeline.

Runs TrOCR on every line in configs/handwriting_lines_manifest.json after each
preprocessing method. The manifest can mix full line images and page/image
crops; a null bbox_abs means the source image is already a line crop.

Outputs:
- results/experimental/ocr/per_line_predictions.csv
- results/experimental/ocr/ocr_method_table.csv
- results/experimental/ocr/ocr_method_table.md
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jiwer import cer, wer

from utils.cnn_baseline import MODEL_SPECS, load_torchscript_model, run_torchscript_model
from utils.experimental_methods import apply_sauvola_binarization
from utils.hsi_pipeline import apply_hsi_showcase_pipeline
from utils.showcase_pipeline import apply_regular_showcase_pipeline


MANIFEST_PATH = Path("configs/handwriting_lines_manifest.json")
RESULTS_ROOT = Path("results/experimental/ocr")
DEFAULT_TROCR_MODEL_ID = os.environ.get("TROCR_MODEL_ID", "microsoft/trocr-large-handwritten")

METHODS = [
    {"id": "original", "display_name": "Original", "family": "original"},
    {"id": "hf_tone", "display_name": "HF + Tone (Gray)", "family": "hf_tone"},
    {"id": "hsi_color", "display_name": "HSI HF + Tone", "family": "hsi_color"},
    {"id": "sauvola", "display_name": "Sauvola", "family": "sauvola"},
    {"id": "zerodcepp", "display_name": "Zero-DCE++", "family": "cnn", "model_spec_id": "zerodcepp"},
    {"id": "retinexnet", "display_name": "RetinexNet", "family": "cnn", "model_spec_id": "retinexnet"},
]

DATASET_ORDER = ["all", "writing_jpeg", "bentham"]


def ensure_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_manifest(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as input_file:
        manifest = json.load(input_file)
    for line in manifest:
        missing = {"id", "dataset", "source_image", "bbox_abs", "transcript", "split", "source"} - set(line)
        if missing:
            raise ValueError(f"Manifest line {line.get('id', '<unknown>')} missing keys: {sorted(missing)}")
    return manifest


def select_lines(manifest: list[dict[str, object]], max_lines_per_dataset: int | None) -> list[dict[str, object]]:
    if max_lines_per_dataset is None:
        return manifest
    counts: dict[str, int] = defaultdict(int)
    selected = []
    for line in manifest:
        dataset = str(line["dataset"])
        if counts[dataset] >= max_lines_per_dataset:
            continue
        selected.append(line)
        counts[dataset] += 1
    return selected


def load_rgb(path: str | Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def image_to_rgb_uint8(image: Image.Image) -> np.ndarray:
    return np.array(image.convert("RGB"), dtype=np.uint8)


def gray_to_rgb(gray_uint8: np.ndarray) -> np.ndarray:
    return np.stack([gray_uint8] * 3, axis=-1)


def crop_line(image: Image.Image, bbox_abs: list[int] | None) -> Image.Image:
    if bbox_abs is None:
        return image
    x1, y1, x2, y2 = [int(value) for value in bbox_abs]
    return image.crop((x1, y1, x2, y2))


def get_case_name(source_image: str, dataset: str) -> str:
    if dataset == "writing_jpeg":
        return "writing"
    return Path(source_image).stem


def resolve_cnn_models(device: str) -> dict[str, object]:
    models = {}
    for model_spec in MODEL_SPECS:
        model_path = None
        configured_path = os.environ.get(model_spec["env_var"])
        candidates = []
        if configured_path:
            candidates.append(configured_path)
        candidates.extend(model_spec["candidate_paths"])
        for candidate in candidates:
            if os.path.exists(candidate):
                model_path = candidate
                break
        if model_path is None:
            continue
        print(f"Loading {model_spec['display_name']} from {model_path}...")
        models[model_spec["id"]] = load_torchscript_model(model_path, device=device)
    return models


def preprocess_source(
    source_image_path: str,
    dataset: str,
    method: dict[str, str],
    cnn_models: dict[str, object],
    device: str,
) -> tuple[Image.Image, float]:
    family = method["family"]
    start_time = time.perf_counter()
    source_image = load_rgb(source_image_path)
    source_rgb = image_to_rgb_uint8(source_image)

    if family == "original":
        output_rgb = source_rgb
    elif family == "hf_tone":
        gray = np.array(source_image.convert("L"), dtype=np.float64)
        final_gray = apply_regular_showcase_pipeline(gray, get_case_name(source_image_path, dataset))["final"]
        output_rgb = gray_to_rgb(final_gray)
    elif family == "hsi_color":
        result = apply_hsi_showcase_pipeline(source_rgb.astype(np.float64), get_case_name(source_image_path, dataset))
        output_rgb = result["final_rgb"]
    elif family == "sauvola":
        gray = np.array(source_image.convert("L"), dtype=np.uint8)
        output_rgb = apply_sauvola_binarization(gray, window_size=51, k=0.2)
    elif family == "cnn":
        model_id = method["model_spec_id"]
        if model_id not in cnn_models:
            raise RuntimeError(f"Missing CNN model for method {method['id']}")
        output_rgb = run_torchscript_model(cnn_models[model_id], source_rgb, device=device)
    else:
        raise KeyError(family)

    runtime_ms = 1000.0 * (time.perf_counter() - start_time)
    return Image.fromarray(np.asarray(output_rgb, dtype=np.uint8), mode="RGB"), runtime_ms


def trocr_predict(processor, model, image: Image.Image, device: str) -> str:
    pixel_values = processor(images=image.convert("RGB"), return_tensors="pt").pixel_values.to(device)
    with torch.no_grad():
        generated_ids = model.generate(pixel_values, max_new_tokens=96)
    return processor.batch_decode(generated_ids, skip_special_tokens=True)[0]


def write_csv(output_path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown_table(output_path: Path, title: str, rows: list[dict[str, object]]) -> None:
    headers = [
        "Dataset",
        "Method",
        "Lines",
        "Corpus CER (%)",
        "Corpus WER (%)",
        "Avg preprocess (ms)",
        "Avg OCR (ms)",
    ]
    lines = [f"# {title}", "", "| " + " | ".join(headers) + " |", "|" + "|".join([" --- "] * len(headers)) + "|"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["dataset"]),
                    str(row["method_display_name"]),
                    str(row["num_lines"]),
                    f"{100.0 * float(row['corpus_cer']):.2f}",
                    f"{100.0 * float(row['corpus_wer']):.2f}",
                    f"{float(row['avg_preprocess_runtime_ms']):.1f}",
                    f"{float(row['avg_ocr_runtime_ms']):.1f}",
                ]
            )
            + " |"
        )
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def compute_aggregates(per_line_rows: list[dict[str, object]], methods: list[dict[str, str]], model_id: str) -> list[dict[str, object]]:
    rows = []
    datasets = ["all"] + sorted({str(row["dataset"]) for row in per_line_rows})
    datasets = [dataset for dataset in DATASET_ORDER if dataset in datasets] + [
        dataset for dataset in datasets if dataset not in DATASET_ORDER
    ]

    for dataset in datasets:
        dataset_rows = per_line_rows if dataset == "all" else [row for row in per_line_rows if row["dataset"] == dataset]
        if not dataset_rows:
            continue
        for method in methods:
            method_rows = [row for row in dataset_rows if row["method_id"] == method["id"]]
            if not method_rows:
                continue
            refs = [str(row["ground_truth"]) for row in method_rows]
            hyps = [str(row["prediction"]) for row in method_rows]
            rows.append(
                {
                    "dataset": dataset,
                    "method_id": method["id"],
                    "method_display_name": method["display_name"],
                    "model_id": model_id,
                    "num_lines": len(method_rows),
                    "corpus_cer": float(cer(refs, hyps)),
                    "corpus_wer": float(wer(refs, hyps)),
                    "total_preprocess_runtime_ms": sum(float(row["preprocess_runtime_ms"]) for row in method_rows),
                    "avg_preprocess_runtime_ms": sum(float(row["preprocess_runtime_ms"]) for row in method_rows) / len(method_rows),
                    "avg_ocr_runtime_ms": sum(float(row["ocr_runtime_ms"]) for row in method_rows) / len(method_rows),
                }
            )
    return rows


def parse_method_filter(methods_arg: str | None) -> list[dict[str, str]]:
    if not methods_arg:
        return METHODS
    requested = [item.strip() for item in methods_arg.split(",") if item.strip()]
    methods_by_id = {method["id"]: method for method in METHODS}
    missing = [method_id for method_id in requested if method_id not in methods_by_id]
    if missing:
        raise ValueError(f"Unknown method ids: {missing}")
    return [methods_by_id[method_id] for method_id in requested]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TrOCR handwriting OCR benchmark.")
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--output-root", default=str(RESULTS_ROOT))
    parser.add_argument("--trocr-model-id", default=DEFAULT_TROCR_MODEL_ID)
    parser.add_argument("--methods", default=None, help="Comma-separated method ids. Defaults to all methods.")
    parser.add_argument("--max-lines-per-dataset", type=int, default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    ensure_directory(output_root)

    manifest = select_lines(read_manifest(Path(args.manifest)), args.max_lines_per_dataset)
    methods = parse_method_filter(args.methods)
    for method in methods:
        ensure_directory(output_root / "line_crops" / method["id"])

    print(f"Lines: {len(manifest)}")
    print(f"Methods: {', '.join(method['id'] for method in methods)}")
    print(f"Loading TrOCR model {args.trocr_model_id}...")
    processor = TrOCRProcessor.from_pretrained(args.trocr_model_id)
    model = VisionEncoderDecoderModel.from_pretrained(args.trocr_model_id)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    print(f"  device = {device}")

    cnn_models = resolve_cnn_models(device) if any(method["family"] == "cnn" for method in methods) else {}
    source_cache: dict[tuple[str, str], tuple[Image.Image, float]] = {}
    per_line_rows = []

    for method in methods:
        print(f"\n=== Method: {method['display_name']} ===")
        for line in manifest:
            line_id = str(line["id"])
            dataset = str(line["dataset"])
            source_image = str(line["source_image"])
            bbox_abs = line["bbox_abs"]
            cache_key = (method["id"], source_image)
            if cache_key not in source_cache:
                if not os.path.exists(source_image):
                    raise FileNotFoundError(f"Missing source image for {line_id}: {source_image}")
                source_cache[cache_key] = preprocess_source(source_image, dataset, method, cnn_models, device)
            method_image, preprocess_runtime_ms = source_cache[cache_key]
            crop = crop_line(method_image, bbox_abs)
            crop_path = output_root / "line_crops" / method["id"] / f"{line_id}.png"
            crop.save(crop_path)

            start_time = time.perf_counter()
            prediction = trocr_predict(processor, model, crop, device)
            ocr_runtime_ms = 1000.0 * (time.perf_counter() - start_time)

            ref = str(line["transcript"])
            line_cer = float(cer(ref, prediction))
            line_wer = float(wer(ref, prediction))
            row = {
                "dataset": dataset,
                "line_id": line_id,
                "source_image": source_image,
                "method_id": method["id"],
                "method_display_name": method["display_name"],
                "model_id": args.trocr_model_id,
                "ground_truth": ref,
                "prediction": prediction,
                "cer": line_cer,
                "wer": line_wer,
                "preprocess_runtime_ms": preprocess_runtime_ms,
                "ocr_runtime_ms": ocr_runtime_ms,
            }
            per_line_rows.append(row)
            print(f"  {line_id:>24s} | CER={line_cer:.3f} | pred={prediction!r}")

    aggregate_rows = compute_aggregates(per_line_rows, methods, args.trocr_model_id)

    write_csv(
        output_root / "per_line_predictions.csv",
        [
            "dataset",
            "line_id",
            "source_image",
            "method_id",
            "method_display_name",
            "model_id",
            "ground_truth",
            "prediction",
            "cer",
            "wer",
            "preprocess_runtime_ms",
            "ocr_runtime_ms",
        ],
        per_line_rows,
    )
    write_csv(
        output_root / "ocr_method_table.csv",
        [
            "dataset",
            "method_id",
            "method_display_name",
            "model_id",
            "num_lines",
            "corpus_cer",
            "corpus_wer",
            "total_preprocess_runtime_ms",
            "avg_preprocess_runtime_ms",
            "avg_ocr_runtime_ms",
        ],
        aggregate_rows,
    )
    write_markdown_table(
        output_root / "ocr_method_table.md",
        f"Handwriting OCR Readability ({args.trocr_model_id})",
        aggregate_rows,
    )

    print("\nSaved:")
    print(f"  {output_root / 'per_line_predictions.csv'}")
    print(f"  {output_root / 'ocr_method_table.csv'}")
    print(f"  {output_root / 'ocr_method_table.md'}")
    print("Done! OCR handwriting evaluation complete.")


if __name__ == "__main__":
    main()
