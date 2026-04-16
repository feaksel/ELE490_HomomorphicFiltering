# Results Guide

## Main Folders

- `final/`: current presentation-ready figures for the new photo showcase
- `real_batch/`: per-image grayscale, homomorphic, and brightened outputs for the active RGB photo showcase
- `real_references/`: side-by-side comparisons against the uniform-lighting reference photos

## Start Here

If you only want the most useful current results, open:

- `final/color_grayscale_standard_overview.png`
- `final/selected_real_images_hf_bright_overview.png`
- `final/seat_pillar_detail_comparison.png`
- `final/uniform_reference_comparison_overview.png`
- `final/cardboard_uniform_reference_comparison.png`
- `final/markers_uniform_reference_comparison.png`

## Reorganizing After New Runs

If an older script writes files back into the top-level `results/` folder, run:

- `py -3.9 scripts/13_organize_results.py`
