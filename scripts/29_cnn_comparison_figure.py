"""
Script 29: Build the report-facing CNN comparison figure.

Produces results/final/cnn_comparison_showcase.png — a 3-case x 4-column grid
comparing Original, the accepted Homomorphic + Tone pipeline, Zero-DCE++, and
RetinexNet. Cases were chosen to span document, deep-shadow, and
specular-glare regimes.
"""
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.comparison_layout import save_comparison_grid
from utils.experimental_methods import (
    load_hard_case_manifest,
    load_resized_grayscale,
    resolve_existing_path,
)


CNN_ROOT = os.path.join("results", "experimental", "cnn")
LOCAL_ROOT = os.path.join("results", "experimental", "local_eq")
FINAL_PATH = os.path.join("results", "final", "cnn_comparison_showcase.png")
REPRESENTATIVE_CASE_IDS = ["page", "seat", "markers"]


def load_uint8(path):
    return np.array(Image.open(path).convert("L"), dtype=np.uint8)


if __name__ == "__main__":
    cnn_models_path = os.path.join(CNN_ROOT, "cnn_models.json")
    if not os.path.exists(cnn_models_path):
        raise FileNotFoundError(
            f"{cnn_models_path} missing. Run scripts/26_pretrained_cnn_hard_cases.py first."
        )
    with open(cnn_models_path, "r", encoding="utf-8") as input_file:
        cnn_models = json.load(input_file)

    expected_ids = {"zerodcepp", "retinexnet"}
    available_ids = {model["id"] for model in cnn_models}
    missing_ids = expected_ids - available_ids
    if missing_ids:
        raise RuntimeError(
            f"Missing CNN model outputs for: {sorted(missing_ids)}. "
            "Re-run scripts/26_pretrained_cnn_hard_cases.py after preparing both TorchScript files."
        )
    cnn_by_id = {model["id"]: model for model in cnn_models}

    manifest = load_hard_case_manifest()
    manifest_by_id = {case["id"]: case for case in manifest}
    case_ids = [case_id for case_id in REPRESENTATIVE_CASE_IDS if case_id in manifest_by_id]
    if not case_ids:
        raise RuntimeError(
            f"None of {REPRESENTATIVE_CASE_IDS} are in the hard-case manifest."
        )

    rows = []
    for case_id in case_ids:
        case_record = manifest_by_id[case_id]
        original = load_resized_grayscale(resolve_existing_path(case_record["path"]))
        hf_tone = load_uint8(os.path.join(LOCAL_ROOT, "hard_cases", "hf_tone_baseline", f"{case_id}.png"))
        zero_dce = load_uint8(os.path.join(CNN_ROOT, "zerodcepp", "hard_cases_gray", f"{case_id}.png"))
        retinex = load_uint8(os.path.join(CNN_ROOT, "retinexnet", "hard_cases_gray", f"{case_id}.png"))

        rows.append(
            [
                (original, f"{case_record['label']}: Original"),
                (hf_tone, f"{case_record['label']}: HF + Tone"),
                (zero_dce, f"{case_record['label']}: Zero-DCE++"),
                (retinex, f"{case_record['label']}: RetinexNet"),
            ]
        )

    os.makedirs(os.path.dirname(FINAL_PATH), exist_ok=True)
    save_comparison_grid(rows, FINAL_PATH)
    print(f"Saved CNN comparison showcase to {FINAL_PATH}")
