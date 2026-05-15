"""
Script 34: Prepare a deterministic Bentham handwriting sample.

Reads the Bentham R0 ground-truth archive, samples 30 non-empty test lines with
seed 490, extracts their line images/transcripts into data/bentham_sample/, and
rewrites configs/handwriting_lines_manifest.json with:

- 8 refined writing.jpeg line boxes
- 30 Bentham full-line samples with clean visible-text transcripts
"""
from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import tarfile
from pathlib import Path


ARCHIVE_PATH = Path("data/bentham_raw/BenthamDatasetR0-GT.tbz")
OUTPUT_ROOT = Path("data/bentham_sample")
MANIFEST_PATH = Path("configs/handwriting_lines_manifest.json")
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_SEED = 490

TEST_LINES_MEMBER = "BenthamDatasetR0-GT/Partitions/TestLines.lst"
TRANSCRIPT_PREFIX = "BenthamDatasetR0-GT/Transcriptions/"
IMAGE_PREFIX = "BenthamDatasetR0-GT/Images/Lines/"


WRITING_LINES = [
    {
        "id": "writing_l1",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [40, 235, 930, 345],
        "transcript": "7 key OS Services",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l2",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [35, 365, 800, 485],
        "transcript": "program development",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l3",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [75, 500, 760, 585],
        "transcript": "program exec",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l4",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [70, 600, 780, 680],
        "transcript": "I/O access",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l5",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [75, 700, 1060, 790],
        "transcript": "controlled file access",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l6",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [75, 805, 1110, 885],
        "transcript": "system access control",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l7",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [70, 895, 1190, 980],
        "transcript": "error detection and response",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
    {
        "id": "writing_l8",
        "dataset": "writing_jpeg",
        "source_image": "images/writing.jpeg",
        "bbox_abs": [80, 990, 710, 1085],
        "transcript": "accounting",
        "split": "manual",
        "source": "manual bad-scan writing.jpeg line crop",
    },
]


def ensure_clean_directory(path: Path) -> None:
    if path.exists():
        resolved = path.resolve()
        root = OUTPUT_ROOT.resolve()
        if root not in resolved.parents and resolved != root:
            raise RuntimeError(f"Refusing to clean unexpected path: {path}")
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def read_test_ids(archive_path: Path) -> list[str]:
    with tarfile.open(archive_path, "r:bz2") as archive:
        member = archive.extractfile(TEST_LINES_MEMBER)
        if member is None:
            raise FileNotFoundError(TEST_LINES_MEMBER)
        return [line.strip() for line in member.read().decode("utf-8").splitlines() if line.strip()]


def normalize_transcript(text: str) -> str:
    return " ".join(text.strip().split())


def is_usable_transcript(text: str) -> bool:
    if not text:
        return False
    if "<" in text or ">" in text:
        return False
    if "_" in text:
        return False
    if any(ord(char) > 127 for char in text):
        return False
    if len(text) < 12 or len(text) > 120:
        return False
    alpha_count = sum(char.isalpha() for char in text)
    return alpha_count >= 8


def collect_test_transcripts(archive_path: Path, test_ids: set[str]) -> dict[str, str]:
    transcripts: dict[str, str] = {}
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(TRANSCRIPT_PREFIX):
                continue
            line_id = Path(member.name).stem
            if line_id not in test_ids:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            text = normalize_transcript(extracted.read().decode("utf-8", errors="replace"))
            if is_usable_transcript(text):
                transcripts[line_id] = " ".join(text.split())
    return transcripts


def sample_line_ids(transcripts: dict[str, str], sample_size: int, seed: int) -> list[str]:
    if len(transcripts) < sample_size:
        raise RuntimeError(f"Only found {len(transcripts)} non-empty transcripts; need {sample_size}.")
    rng = random.Random(seed)
    return sorted(rng.sample(sorted(transcripts), sample_size))


def extract_selected_images(archive_path: Path, selected_ids: set[str], images_dir: Path) -> set[str]:
    extracted_ids: set[str] = set()
    with tarfile.open(archive_path, "r:bz2") as archive:
        for member in archive:
            if not member.isfile() or not member.name.startswith(IMAGE_PREFIX):
                continue
            line_id = Path(member.name).stem
            if line_id not in selected_ids:
                continue
            extracted = archive.extractfile(member)
            if extracted is None:
                continue
            output_path = images_dir / f"{line_id}.png"
            output_path.write_bytes(extracted.read())
            extracted_ids.add(line_id)
            if len(extracted_ids) == len(selected_ids):
                break
    return extracted_ids


def build_manifest(selected_ids: list[str], transcripts: dict[str, str]) -> list[dict[str, object]]:
    bentham_lines = []
    for line_id in selected_ids:
        bentham_lines.append(
            {
                "id": f"bentham_{line_id}",
                "dataset": "bentham",
                "source_image": str(Path("data/bentham_sample/images") / f"{line_id}.png").replace("\\", "/"),
                "bbox_abs": None,
                "transcript": transcripts[line_id],
                "split": "test",
                "source": "Bentham R0 ICFHR 2014 HTR competition test line",
            }
        )
    return WRITING_LINES + bentham_lines


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare deterministic Bentham OCR sample.")
    parser.add_argument("--archive", default=str(ARCHIVE_PATH))
    parser.add_argument("--output-root", default=str(OUTPUT_ROOT))
    parser.add_argument("--manifest", default=str(MANIFEST_PATH))
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--no-write-manifest", action="store_true")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest)

    if not archive_path.exists():
        raise FileNotFoundError(f"Missing Bentham archive: {archive_path}")

    images_dir = output_root / "images"
    transcripts_dir = output_root / "transcripts"
    ensure_clean_directory(images_dir)
    ensure_clean_directory(transcripts_dir)

    print(f"Reading Bentham test ids from {archive_path}...")
    test_ids = read_test_ids(archive_path)
    print(f"  test ids: {len(test_ids)}")

    print("Collecting usable test transcripts in one archive pass...")
    transcripts = collect_test_transcripts(archive_path, set(test_ids))
    selected_ids = sample_line_ids(transcripts, args.sample_size, args.seed)
    print(f"  usable transcripts: {len(transcripts)}")
    print(f"  selected ids: {len(selected_ids)}")

    print("Extracting selected line images...")
    extracted_ids = extract_selected_images(archive_path, set(selected_ids), images_dir)
    missing_ids = sorted(set(selected_ids) - extracted_ids)
    if missing_ids:
        raise RuntimeError(f"Missing selected images in archive: {missing_ids}")

    for line_id in selected_ids:
        (transcripts_dir / f"{line_id}.txt").write_text(transcripts[line_id] + "\n", encoding="utf-8")

    write_json(output_root / "selected_ids.json", selected_ids)
    sample_manifest = [
        {
            "id": line_id,
            "image": str(images_dir / f"{line_id}.png").replace("\\", "/"),
            "transcript": transcripts[line_id],
        }
        for line_id in selected_ids
    ]
    write_json(output_root / "sample_manifest.json", sample_manifest)

    if not args.no_write_manifest:
        manifest = build_manifest(selected_ids, transcripts)
        write_json(manifest_path, manifest)
        print(f"Wrote {manifest_path} with {len(manifest)} lines.")

    print(f"Prepared Bentham sample under {output_root}.")


if __name__ == "__main__":
    main()
