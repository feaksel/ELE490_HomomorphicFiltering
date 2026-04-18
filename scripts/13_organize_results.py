"""
Script 13: Organize the results folder.
This script sorts generated outputs for the current photo showcase into topic
folders, quarantines rejected CLAHE experiments under results/old/, and copies
the current regular-pipeline showcase figures into results/final.
"""
import os
import shutil


def ensure_directories(base_path, folder_names):
    for folder_name in folder_names:
        os.makedirs(os.path.join(base_path, folder_name), exist_ok=True)


def move_matching_files(source_folder, destination_folder, prefixes=None, names=None):
    prefixes = prefixes or []
    names = names or []

    if not os.path.exists(source_folder):
        return

    for entry_name in os.listdir(source_folder):
        entry_path = os.path.join(source_folder, entry_name)

        if not os.path.isfile(entry_path):
            continue

        if entry_name in {".gitkeep", "README.md"}:
            continue

        should_move = entry_name in names or any(entry_name.startswith(prefix) for prefix in prefixes)
        if not should_move:
            continue

        os.makedirs(destination_folder, exist_ok=True)
        destination_path = os.path.join(destination_folder, entry_name)
        if os.path.exists(destination_path):
            os.remove(destination_path)
        shutil.move(entry_path, destination_path)


def copy_if_exists(source_path, destination_path):
    if os.path.exists(source_path):
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)
        shutil.copy2(source_path, destination_path)


def remove_if_exists(path):
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


if __name__ == "__main__":
    results_path = "results"
    current_real_prefixes = [
        "cardboard_",
        "carboard_uniform_",
        "carpet_",
        "markers_",
        "markers_uniform_",
        "page_",
        "pillar_",
        "seat_",
    ]

    folders = [
        "final",
        os.path.join("old", "clahe_rejected"),
        "real_batch",
        "real_references",
    ]
    ensure_directories(results_path, folders)

    move_matching_files(
        results_path,
        os.path.join(results_path, "old", "clahe_rejected"),
        names=[
            "clahe_page_pipeline_comparison.png",
            "page_conservative_clahe_comparison.png",
        ],
    )
    move_matching_files(
        os.path.join(results_path, "real_batch"),
        os.path.join(results_path, "old", "clahe_rejected"),
        names=[
            "cardboard_hist_eq.png",
            "carpet_hist_eq.png",
            "markers_hist_eq.png",
            "page_clahe.png",
            "page_clahe_then_conservative_hf_bright.png",
            "page_clahe_then_hf_bright.png",
            "page_conservative_hf_bright_then_clahe.png",
            "page_hf_bright_then_clahe.png",
            "pillar_hist_eq.png",
            "seat_hist_eq.png",
        ],
    )
    move_matching_files(
        results_path,
        os.path.join(results_path, "final"),
        names=[
            "page_conservative_hf_comparison.png",
        ],
    )
    move_matching_files(results_path, os.path.join(results_path, "real_batch"), prefixes=current_real_prefixes)
    move_matching_files(
        os.path.join(results_path, "real_batch"),
        os.path.join(results_path, "final"),
        names=[
            "page_detail_comparison.png",
            "page_conservative_hf_comparison.png",
            "seat_pillar_detail_comparison.png",
        ],
    )
    move_matching_files(
        os.path.join(results_path, "real_batch"),
        os.path.join(results_path, "real_references"),
        names=[
            "cardboard_uniform_reference_comparison.png",
            "markers_uniform_reference_comparison.png",
        ],
    )
    move_matching_files(
        results_path,
        os.path.join(results_path, "real_references"),
        names=[
            "cardboard_uniform_reference_comparison.png",
            "markers_uniform_reference_comparison.png",
        ],
    )
    move_matching_files(results_path, os.path.join(results_path, "real_references"), names=["uniform_reference_comparison_overview.png"])
    move_matching_files(
        results_path,
        os.path.join(results_path, "final"),
        names=[
            "color_grayscale_standard_overview.png",
            "page_detail_comparison.png",
            "selected_real_images_hf_bright_overview.png",
            "seat_pillar_detail_comparison.png",
            "tone_adjusted_showcase_overview.png",
            "uniform_reference_comparison_overview.png",
        ],
    )

    stale_final_paths = [
        os.path.join(results_path, "final", "old"),
        os.path.join(results_path, "final", "blind_results_table.csv"),
        os.path.join(results_path, "final", "blind_results_table.md"),
        os.path.join(results_path, "final", "clahe_page_pipeline_comparison.png"),
        os.path.join(results_path, "final", "flashlight_standard_blind_comparison.png"),
        os.path.join(results_path, "final", "page_conservative_clahe_comparison.png"),
        os.path.join(results_path, "final", "page_grayscale_vs_hf_vs_tone.png"),
        os.path.join(results_path, "final", "page_grayscale_vs_standard.png"),
        os.path.join(results_path, "final", "rice_aggressive_comparison.png"),
        os.path.join(results_path, "final", "tun_final_2x2_comparison.png"),
        os.path.join(results_path, "final", "tun_homomorphic_restored_extreme_flattening.png"),
        os.path.join(results_path, "final", "xray_hf_vs_heq.png"),
    ]
    for stale_path in stale_final_paths:
        remove_if_exists(stale_path)

    curated_copies = [
        ("real_batch/cardboard_grayscale_vs_standard.png", "final/cardboard_grayscale_vs_standard.png"),
        ("real_batch/markers_grayscale_vs_standard.png", "final/markers_grayscale_vs_standard.png"),
        ("real_batch/page_conservative_hf_comparison.png", "final/page_conservative_hf_comparison.png"),
        ("real_batch/page_detail_comparison.png", "final/page_detail_comparison.png"),
        ("real_references/uniform_reference_comparison_overview.png", "final/uniform_reference_comparison_overview.png"),
        ("real_references/cardboard_uniform_reference_comparison.png", "final/cardboard_uniform_reference_comparison.png"),
        ("real_references/markers_uniform_reference_comparison.png", "final/markers_uniform_reference_comparison.png"),
    ]

    for source_relative, destination_relative in curated_copies:
        copy_if_exists(os.path.join(results_path, source_relative), os.path.join(results_path, destination_relative))

    print("Results folder organized.")
