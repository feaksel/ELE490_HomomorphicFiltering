# Results Guide

## Main Folders

- `final/`: current presentation-ready figures and tables
- `real_batch/`: per-image grayscale, homomorphic, brightened, and histogram-equalized outputs for the active RGB photo showcase
- `real_references/`: side-by-side comparisons against the uniform-lighting reference photos
- `synthetic/`: synthetic illumination experiments and related figures
- `rice/`: rice benchmark comparisons
- `tunnel/`: tunnel tuning studies, sweeps, and candidate outputs
- `flashlight/`: flashlight demo figures
- `xray/`: chest X-ray comparisons
- `metrics/`: PSNR, SSIM, tables, and summary exports
- `archive/`: older summary figures and intermediate outputs that are still worth keeping but are not the main deliverables

## Start Here

If you only want the most useful current results, open:

- `final/color_grayscale_standard_overview.png`
- `final/selected_real_images_hf_overview.png`
- `final/seat_pillar_detail_comparison.png`
- `final/uniform_reference_comparison_overview.png`
- `final/cardboard_uniform_reference_comparison.png`
- `final/markers_uniform_reference_comparison.png`
- `final/rice_aggressive_comparison.png`
- `final/tun_final_2x2_comparison.png`
- `final/tun_homomorphic_restored_extreme_flattening.png`
- `final/flashlight_standard_blind_comparison.png`
- `final/blind_results_table.md`

## Reorganizing After New Runs

If an older script writes files back into the top-level `results/` folder, run:

- `py -3.9 scripts/13_organize_results.py`
