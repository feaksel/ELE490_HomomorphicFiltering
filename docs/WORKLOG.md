# Worklog

## 2026-04-07

- Implemented the grayscale-first homomorphic filtering pipeline.
- Added synthetic illumination generation, restoration, parameter sweeps, and
  metric calculation scripts.
- Verified synthetic-case improvement:
  corrupted `13.35 dB / 0.8280 SSIM` to restored `17.89 dB / 0.9019 SSIM`.
- Retuned the synthetic default after stronger corruption was introduced.
- Added direct comparison figures so the synthetic restoration was easier to
  explain visually.

## 2026-04-08

- Implemented the HSI color-pipeline structure in `05_color_hsi.py`.
- Added report-oriented helper outputs including tables, final grayscale pack,
  and the padding/windowing experiment.

## 2026-04-10

- Added real-scene demos for `flashlight.jpeg` and `tun.jpg`.
- Ran tunnel-focused sweeps and adopted the strong blind real-scene direction:
  `Gaussian`, `gamma_L=0.06`, `gamma_H=1.00`, `D0=320`.
- Added RGB-to-grayscale batch processing for the earlier real-photo set.
- Added brightness lift after the homomorphic result because the blind-only
  output looked visually too dim in presentation figures.
- Increased brightness lift strength from `gamma=0.82` to `gamma=0.72`.
- Added HF vs HEQ comparisons on the older real-photo set.
- Promoted `writing.jpeg` as one of the stronger old real-scene examples.
- Added results-folder organization scripts and folder structure.

## 2026-04-12

- Added chest X-ray experiments and HF/HEQ comparison outputs.
- Recorded that X-ray gains may be subtle or visually misleading because the
  source images may already be heavily processed.

## 2026-04-16

- Replaced the active showcase image set with:
  `cardboard`, `carboard_uniform`, `carpet`, `markers`, `markers_uniform`,
  `pillar`, and `seat`.
- Moved old and weak examples out of the active workflow and cleaned the
  organizer so the final folder focused on the new showcase.
- Added uniform-lighting reference comparisons for `cardboard` and `markers`.
- Added `seat` and `pillar` detail-comparison figure for texture evaluation.
- Switched the active final comparisons to the brightened homomorphic outputs.
- Simplified `results/final/` so it only carried the current photo-showcase
  figures rather than flashlight/tunnel/rice/xray carryover files.

## 2026-04-17

- Added `page.jpeg` as a document-like real-life use case with directional
  lighting and shadow.
- Processed `page.jpeg` through the active grayscale homomorphic workflow and
  added `page_grayscale_vs_standard.png` as a reportable result.
- Tested histogram equalization before and after homomorphic filtering on the
  page case.
  Outcome: rejected. It produced harsher, noisier, less natural document
  results than plain homomorphic filtering.
- Removed the page-specific HEQ experiment script and generated traces after
  rejecting that direction.
- Tested CLAHE around the page case using the global real-scene setting.
  Outcome: rejected for reporting. CLAHE increased local sharpness but also
  exaggerated paper texture and background unevenness.
- Tested milder page-specific homomorphic settings:
  - standard: `gamma_L=0.06`, `D0=320`, bright gamma `0.72`
  - mild: `gamma_L=0.15`, `D0=220`, bright gamma `0.78`
  - conservative: `gamma_L=0.25`, `D0=160`, bright gamma `0.84`
- Result of page-specific tuning:
  conservative looked best overall for document readability; mild was a useful
  intermediate candidate; standard looked a bit over-processed for the page.
- Tested CLAHE again around the conservative page setting.
  Outcome: still rejected. Conservative HF + bright remained cleaner and more
  reportable than any CLAHE combination.
- Implemented a reproducible shadow/highlight style tone-adjustment stage on
  top of homomorphic filtering and tested it across the active showcase.
- Outcome of tone-adjustment test:
  - strongest benefit on the `page` document case
  - mild benefit on `cardboard`
  - limited or flattening effect on `markers`, `pillar`, and `seat`
  - initial conclusion that day: useful as optional post-processing, with final
    adoption still pending
- Added a professor-facing progress draft in `PROGRESS_UPDATE_PROF.md`.
- Backfilled a structured documentation system under `docs/`.

## 2026-04-18

- Retired the rejected CLAHE page experiments from the active workflow.
- Moved `17_page_clahe_pipeline_comparison.py` and
  `19_page_conservative_clahe_comparison.py` into `scripts/old/`.
- Moved the corresponding rejected outputs into `results/old/clahe_rejected/`
  and kept non-active exploratory post-processing outputs archived under
  `results/old/`.
- Renamed the active selected-image comparison script to
  `12_selected_hf_comparison.py` so the active pipeline no longer implies HEQ
  is part of the current story.
- Tightened `results/final/` to the reportable mid-project set centered on:
  `color_grayscale_standard_overview`, `selected_real_images_hf_bright_overview`,
  `seat_pillar_detail_comparison`, `page_conservative_hf_comparison`, and the
  uniform-reference comparisons.
- Switched the active showcase again so tone equalization is now the regular
  final stage after homomorphic filtering across the professor-facing figures.
- Added a shared regular-pipeline helper under `utils/showcase_pipeline.py` so
  the active scripts use the same homomorphic plus tone-equalization flow.
- Kept `20_tone_adjusted_showcase.py` active as an inspection script for the
  accepted regular pipeline rather than archiving it.
- Updated the selected-image, per-image, reference, detail, and page-specific
  comparison scripts to use the regular pipeline consistently.
- Added `results/final/page_detail_comparison.png` as a dedicated zoomed page
  comparison alongside the full-frame page figures.

## Current Active Result Files

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/page_conservative_hf_comparison.png`
- `results/final/page_detail_comparison.png`
- `results/final/cardboard_grayscale_vs_standard.png`
- `results/final/markers_grayscale_vs_standard.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
