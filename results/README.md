# Results Guide

## Main Folders

- `final/`: current mid-project report figures we want to show the professor
- `real_batch/`: per-image grayscale, homomorphic, brightened, and tone-equalized outputs for the active RGB photo showcase
- `real_references/`: side-by-side comparisons against the uniform-lighting reference photos
- `old/`: rejected or retired experiments such as CLAHE outputs

The active showcase interpretation is now the regular pipeline:

- grayscale
- homomorphic filtering
- brightness lift
- tone equalization

## Start Here

If you only want the most useful current results, open:

- `final/color_grayscale_standard_overview.png`
- `final/selected_real_images_hf_bright_overview.png`
- `final/seat_pillar_detail_comparison.png`
- `final/page_conservative_hf_comparison.png`
- `final/page_detail_comparison.png`
- `final/uniform_reference_comparison_overview.png`
- `final/cardboard_uniform_reference_comparison.png`
- `final/markers_uniform_reference_comparison.png`

Note:
`final/selected_real_images_hf_bright_overview.png` keeps an older filename for
continuity, but it now represents the accepted regular pipeline ending in tone
equalization.

## Reorganizing After New Runs

If an older script writes files back into the top-level `results/` folder, run:

- `py -3.9 scripts/13_organize_results.py`
