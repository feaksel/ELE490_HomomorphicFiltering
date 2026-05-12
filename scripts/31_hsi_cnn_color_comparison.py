"""
Script 31: Report-facing color comparison between the HSI homomorphic
pipeline and the pretrained CNN baselines (Zero-DCE++ and RetinexNet).

Produces:
- results/final/hsi_cnn_color_comparison.png — three representative cases
  (`page`, `seat`, `markers`) across Original, HSI HF + Tone, Zero-DCE++,
  and RetinexNet, all in color.
- results/experimental/evaluation/hsi_cnn_all_color_overview.png — the full
  hard-case set with the same four columns, kept under the experimental
  evaluation folder as appendix material.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid
from utils.experimental_methods import load_hard_case_manifest, resolve_existing_path
from utils.hsi_pipeline import load_resized_rgb


HSI_ROOT = os.path.join("results", "experimental", "hsi")
CNN_ROOT = os.path.join("results", "experimental", "cnn")
FINAL_PATH = os.path.join("results", "final", "hsi_cnn_color_comparison.png")
ALL_PATH = os.path.join("results", "experimental", "evaluation", "hsi_cnn_all_color_overview.png")
REPRESENTATIVE_CASE_IDS = ["page", "seat", "markers"]


def load_rgb(path):
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def resize_to_match(image_uint8, target_shape):
    if image_uint8.shape[:2] == target_shape[:2]:
        return image_uint8
    target_height, target_width = target_shape[:2]
    resized = Image.fromarray(image_uint8).resize(
        (target_width, target_height), Image.Resampling.LANCZOS
    )
    return np.array(resized, dtype=np.uint8)


def build_row(case_record, cnn_models_present):
    case_id = case_record["id"]
    original = load_resized_rgb(resolve_existing_path(case_record["path"])).astype(np.uint8)
    hsi = load_rgb(os.path.join(HSI_ROOT, "hard_cases_rgb", f"{case_id}.png"))
    hsi = resize_to_match(hsi, original.shape)

    panels = [
        (original, f"{case_record['label']}: Original"),
        (hsi, f"{case_record['label']}: HSI HF + Tone"),
    ]
    for model_id, model_display in cnn_models_present:
        cnn_rgb = load_rgb(os.path.join(CNN_ROOT, model_id, "hard_cases_rgb", f"{case_id}.png"))
        cnn_rgb = resize_to_match(cnn_rgb, original.shape)
        panels.append((cnn_rgb, f"{case_record['label']}: {model_display}"))
    return panels


if __name__ == "__main__":
    cnn_models_path = os.path.join(CNN_ROOT, "cnn_models.json")
    if not os.path.exists(cnn_models_path):
        raise FileNotFoundError(
            f"{cnn_models_path} missing. Run scripts/26_pretrained_cnn_hard_cases.py first."
        )
    with open(cnn_models_path, "r", encoding="utf-8") as input_file:
        cnn_models = json.load(input_file)

    desired_order = ["zerodcepp", "retinexnet"]
    cnn_by_id = {model["id"]: model for model in cnn_models}
    cnn_models_present = [
        (model_id, cnn_by_id[model_id]["display_name"])
        for model_id in desired_order
        if model_id in cnn_by_id
    ]
    if not cnn_models_present:
        raise RuntimeError(
            "No CNN models found in cnn_models.json. Re-run scripts/26_pretrained_cnn_hard_cases.py."
        )

    manifest = load_hard_case_manifest()
    manifest_by_id = {case["id"]: case for case in manifest}

    representative_records = [
        manifest_by_id[case_id]
        for case_id in REPRESENTATIVE_CASE_IDS
        if case_id in manifest_by_id
    ]
    if not representative_records:
        raise RuntimeError(
            f"None of {REPRESENTATIVE_CASE_IDS} are in the hard-case manifest."
        )

    representative_rows = [build_row(record, cnn_models_present) for record in representative_records]
    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)
    save_comparison_grid(representative_rows, FINAL_PATH)
    print(f"Saved color comparison showcase to {FINAL_PATH}")

    all_rows = [build_row(case_record, cnn_models_present) for case_record in manifest]
    os.makedirs(os.path.dirname(ALL_PATH), exist_ok=True)
    save_comparison_grid(all_rows, ALL_PATH)
    print(f"Saved full-set color overview to {ALL_PATH}")
