"""
Script 06: Compute quantitative metrics for the synthetic test case.
Uses the original cameraman image as reference and compares it against
the synthetic corrupted image and the homomorphic filtering result.
"""
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity


def compute_psnr(reference, test_image):
    mse = np.mean((reference - test_image) ** 2)

    if mse < 1e-12:
        return mse, float("inf")

    psnr = 10.0 * np.log10((255.0 ** 2) / mse)
    return mse, psnr


def evaluate_case(reference, corrupted_path, restored_path):
    corrupted = np.array(Image.open(corrupted_path).convert("L"), dtype=np.float64)
    restored = np.array(Image.open(restored_path).convert("L"), dtype=np.float64)

    corrupted_mse, corrupted_psnr = compute_psnr(reference, corrupted)
    restored_mse, restored_psnr = compute_psnr(reference, restored)
    corrupted_ssim = structural_similarity(reference, corrupted, data_range=255)
    restored_ssim = structural_similarity(reference, restored, data_range=255)

    return {
        "corrupted_mse": corrupted_mse,
        "corrupted_psnr": corrupted_psnr,
        "corrupted_ssim": corrupted_ssim,
        "restored_mse": restored_mse,
        "restored_psnr": restored_psnr,
        "restored_ssim": restored_ssim,
    }


if __name__ == "__main__":
    reference_path = "images/cameraman.tif"
    corrupted_path = "images/synthetic_uneven.png"
    restored_path = "results/homomorphic_restored.png"

    print("Loading images for evaluation...")
    reference = np.array(Image.open(reference_path).convert("L"), dtype=np.float64)
    corrupted = np.array(Image.open(corrupted_path).convert("L"), dtype=np.float64)
    restored = np.array(Image.open(restored_path).convert("L"), dtype=np.float64)
    print(f"  Reference shape: {reference.shape}")
    print(f"  Corrupted shape: {corrupted.shape}")
    print(f"  Restored shape: {restored.shape}")

    print("Checking shape consistency...")
    if reference.shape != corrupted.shape or reference.shape != restored.shape:
        raise ValueError("Reference, corrupted, and restored images must have the same shape.")
    print("  Shapes are consistent.")

    print("Computing PSNR...")
    corrupted_mse, corrupted_psnr = compute_psnr(reference, corrupted)
    restored_mse, restored_psnr = compute_psnr(reference, restored)
    print(f"  Corrupted MSE: {corrupted_mse:.4f}")
    print(f"  Corrupted PSNR: {corrupted_psnr:.4f} dB")
    print(f"  Restored MSE: {restored_mse:.4f}")
    print(f"  Restored PSNR: {restored_psnr:.4f} dB")

    print("Computing SSIM...")
    corrupted_ssim = structural_similarity(reference, corrupted, data_range=255)
    restored_ssim = structural_similarity(reference, restored, data_range=255)
    print(f"  Corrupted SSIM: {corrupted_ssim:.4f}")
    print(f"  Restored SSIM: {restored_ssim:.4f}")

    print("Saving metric summary figure...")
    os.makedirs("results", exist_ok=True)

    comparison_figure, comparison_axes = plt.subplots(1, 3, figsize=(15, 5))

    comparison_axes[0].imshow(reference, cmap="gray", vmin=0, vmax=255)
    comparison_axes[0].set_title("Baseline Cameraman")
    comparison_axes[0].set_xlabel("Column")
    comparison_axes[0].set_ylabel("Row")

    comparison_axes[1].imshow(corrupted, cmap="gray", vmin=0, vmax=255)
    comparison_axes[1].set_title("Synthetic Uneven Image")
    comparison_axes[1].set_xlabel("Column")
    comparison_axes[1].set_ylabel("Row")

    comparison_axes[2].imshow(restored, cmap="gray", vmin=0, vmax=255)
    comparison_axes[2].set_title("Homomorphic Restored Image")
    comparison_axes[2].set_xlabel("Column")
    comparison_axes[2].set_ylabel("Row")

    comparison_figure.suptitle("Baseline vs Synthetic vs Restored")
    comparison_figure.tight_layout()
    comparison_figure.savefig("results/baseline_synthetic_restored.png", dpi=200)
    plt.close(comparison_figure)
    print("  Saved side-by-side comparison to results/baseline_synthetic_restored.png")

    print("Evaluating all blind synthetic cases...")
    case_evaluations = [
        ("Vertical", "images/synthetic_vertical.png", "results/homomorphic_restored_vertical.png"),
        ("Rotated", "images/synthetic_rotated.png", "results/homomorphic_restored_rotated.png"),
        ("Sine", "images/synthetic_sine.png", "results/homomorphic_restored_sine.png"),
    ]

    all_case_results = []
    for case_name, corrupted_case_path, restored_case_path in case_evaluations:
        case_result = evaluate_case(reference, corrupted_case_path, restored_case_path)
        all_case_results.append((case_name, case_result))
        print(f"  {case_name} corrupted PSNR/SSIM: {case_result['corrupted_psnr']:.4f} dB / {case_result['corrupted_ssim']:.4f}")
        print(f"  {case_name} restored  PSNR/SSIM: {case_result['restored_psnr']:.4f} dB / {case_result['restored_ssim']:.4f}")

    figure, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].imshow(reference, cmap="gray", vmin=0, vmax=255)
    axes[0, 0].set_title("Reference: Cameraman")
    axes[0, 0].set_xlabel("Column")
    axes[0, 0].set_ylabel("Row")

    axes[0, 1].imshow(corrupted, cmap="gray", vmin=0, vmax=255)
    axes[0, 1].set_title("Synthetic Uneven Image")
    axes[0, 1].set_xlabel("Column")
    axes[0, 1].set_ylabel("Row")

    axes[1, 0].imshow(restored, cmap="gray", vmin=0, vmax=255)
    axes[1, 0].set_title("Homomorphic Restored Image")
    axes[1, 0].set_xlabel("Column")
    axes[1, 0].set_ylabel("Row")

    axes[1, 1].axis("off")
    summary_text = (
        "Metric Summary\n\n"
        f"Corrupted vs Reference\n"
        f"MSE  = {corrupted_mse:.4f}\n"
        f"PSNR = {corrupted_psnr:.4f} dB\n"
        f"SSIM = {corrupted_ssim:.4f}\n\n"
        f"Restored vs Reference\n"
        f"MSE  = {restored_mse:.4f}\n"
        f"PSNR = {restored_psnr:.4f} dB\n"
        f"SSIM = {restored_ssim:.4f}"
    )
    axes[1, 1].text(0.05, 0.95, summary_text, va="top", ha="left", fontsize=11)

    figure.suptitle("Quantitative Evaluation on Synthetic Test Case")
    figure.tight_layout()
    figure.savefig("results/metrics_summary.png", dpi=200)
    plt.close(figure)
    print("  Saved metric summary to results/metrics_summary.png")

    multicase_figure, multicase_axes = plt.subplots(1, 1, figsize=(11, 6))
    multicase_axes.axis("off")

    multicase_lines = ["Blind Homomorphic Metrics by Illumination Case", ""]
    for case_name, case_result in all_case_results:
        multicase_lines.append(
            f"{case_name}: corrupted {case_result['corrupted_psnr']:.2f} dB / {case_result['corrupted_ssim']:.4f}"
        )
        multicase_lines.append(
            f"{case_name}: restored  {case_result['restored_psnr']:.2f} dB / {case_result['restored_ssim']:.4f}"
        )
        multicase_lines.append("")

    multicase_axes.text(
        0.03,
        0.97,
        "\n".join(multicase_lines),
        va="top",
        ha="left",
        fontsize=12,
        family="monospace",
    )
    multicase_figure.suptitle("Blind Multi-Case Metric Summary")
    multicase_figure.tight_layout()
    multicase_figure.savefig("results/blind_multicase_metrics.png", dpi=200)
    plt.close(multicase_figure)
    print("  Saved blind multi-case metric summary to results/blind_multicase_metrics.png")

    print("Writing blind results tables...")
    csv_lines = [
        "Case,Corrupted_PSNR_dB,Corrupted_SSIM,Restored_PSNR_dB,Restored_SSIM",
    ]
    markdown_lines = [
        "# Blind Results Table",
        "",
        "| Case | Corrupted PSNR (dB) | Corrupted SSIM | Restored PSNR (dB) | Restored SSIM |",
        "|------|---------------------|----------------|--------------------|---------------|",
    ]

    for case_name, case_result in all_case_results:
        csv_lines.append(
            f"{case_name},{case_result['corrupted_psnr']:.4f},{case_result['corrupted_ssim']:.4f},"
            f"{case_result['restored_psnr']:.4f},{case_result['restored_ssim']:.4f}"
        )
        markdown_lines.append(
            f"| {case_name} | {case_result['corrupted_psnr']:.4f} | {case_result['corrupted_ssim']:.4f} | "
            f"{case_result['restored_psnr']:.4f} | {case_result['restored_ssim']:.4f} |"
        )

    with open("results/blind_results_table.csv", "w", encoding="utf-8") as csv_file:
        csv_file.write("\n".join(csv_lines) + "\n")

    with open("results/blind_results_table.md", "w", encoding="utf-8") as markdown_file:
        markdown_file.write("\n".join(markdown_lines) + "\n")

    print("  Saved blind results table to results/blind_results_table.csv")
    print("  Saved blind results table to results/blind_results_table.md")

    print("Done! Metrics saved.")
