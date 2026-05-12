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

## 2026-05-11

- Closed out the Phase-2 CNN comparison branch (`scripts/26`, `scripts/27`).
- Vendored Zero-DCE++ (`enhance_net_nopool`) and a PyTorch port of RetinexNet
  (`DecomNet`, `RelightNet`) under `utils/external/`, with attribution headers
  pointing at the upstream repositories.
- Acquired pretrained weights:
  - `models/zerodcepp/Epoch99.pth` from Li-Chongyi/Zero-DCE_extension
  - `models/retinexnet/Decom_9200.tar` and `Relight_9200.tar` from
    aasharma90/RetinexNet_PyTorch
- Added `scripts/28_prepare_cnn_models.py` that wraps each model so its
  `forward(rgb_01)` returns only the final enhanced RGB in `[0, 1]`, then
  exports a TorchScript `.ts` file the existing CNN scaffolding can load.
  The Zero-DCE++ wrapper reflect-pads inputs to multiples of `scale_factor=12`
  so it accepts arbitrary spatial sizes.
- Refactored `utils/cnn_baseline.py` with `find_all_available_model_specs` so
  multiple models can be discovered in a single run.
- Refactored `scripts/26_pretrained_cnn_hard_cases.py` to iterate over every
  discovered model, write per-model CSVs (`cnn_*_metrics_{model_id}.csv`),
  and emit a combined `cnn_models.json` manifest.
- Refactored `scripts/27_next_phase_evaluation.py` to consume the per-model
  CSVs, include each CNN as its own method row, add a column per CNN to the
  visual overview grids, and list each CNN in the summary file.
- Re-ran the pipeline. Headline numbers (synthetic average, four corruption
  patterns):
  - Homomorphic baseline: SSIM `0.9380`, PSNR `17.889` dB
  - CLAHE 16x16: SSIM `0.8655`, PSNR `16.150` dB
  - Zero-DCE++: SSIM `0.8665`, PSNR `15.546` dB, average runtime `13.78 ms`
  - RetinexNet: SSIM `0.7624`, PSNR `12.181` dB, average runtime `317.98 ms`
- Homomorphic filtering remains the synthetic-quality leader. Zero-DCE++ is
  the runtime leader and the tradeoff-rank winner; RetinexNet is both slower
  and lower quality than HF on this comparison.
- Added `scripts/29_cnn_comparison_figure.py` and promoted one new figure to
  `results/final/cnn_comparison_showcase.png` (page, seat, markers across
  Original / HF + Tone / Zero-DCE++ / RetinexNet) — this is the
  report-facing artifact for the CNN comparison.
- Marked the Phase-2 CLAHE branches (`CLAHE 16x16 clip=0.01` and its
  high-boost variant) as failed for the illumination-correction goal and
  removed them from `hard_case_visual_overview.png` and
  `hard_case_crop_overview.png` (`R-007`). They remain in the
  `synthetic_method_table` and `hard_case_method_table` CSVs / Markdown
  for honest quantitative reporting, but they are no longer in the
  professor-facing visual story.

## 2026-05-12

- Extended script 27 to also write `hard_case_all_overview.png` and
  `hard_case_all_crop_overview.png` covering every entry in the
  hard-case manifest. The previously-existing 4-case overview is kept
  for the report body; the 8-case version supports the appendix.
- Validated the HSI color pipeline. Extracted the HSI conversion into
  `utils/hsi_pipeline.py` so the project's accepted homomorphic +
  brightness lift + tone equalization pipeline can run on the intensity
  channel while hue and saturation are preserved. Closes `I-001`.
- Added `scripts/30_hsi_color_pipeline.py` which runs the HSI pipeline
  on every hard-case image and writes color outputs and proxy metrics
  to `results/experimental/hsi/`.
- Added `scripts/31_hsi_cnn_color_comparison.py` which builds two
  color comparison figures: a 3-case representative figure promoted to
  `results/final/hsi_cnn_color_comparison.png`, and an 8-case appendix
  figure under `results/experimental/evaluation/hsi_cnn_all_color_overview.png`.
- New active decision `D-010`: HSI color pipeline reuses the accepted
  grayscale settings (per-scene config + intensity-only processing).
- Visually, HSI clearly leads the color comparison: hue and saturation
  stay natural and shadows flatten without the bluish color cast that
  RetinexNet introduces, while Zero-DCE++ stays close to the original
  brightness.
- Added a quantitative downstream-task evaluation: handwriting OCR
  readability with `microsoft/trocr-large-handwritten` on
  `writing.jpeg` (real handwriting with severe flashlight beam + dark
  edges + yellow color cast). Manifest with manually-transcribed
  ground truth at `configs/handwriting_lines_manifest.json` (8 lines).
- New scripts: `scripts/32_ocr_handwriting_pipeline.py` (loads TrOCR,
  crops lines from each method's output, computes corpus CER and WER
  via jiwer) and `scripts/33_ocr_comparison_figure.py` (builds the
  report-facing comparison figure).
- Headline OCR numbers (corpus-level over 8 lines):
  - Original: CER `33.1 %`, WER `90.5 %`
  - HF + Tone (Gray): CER `26.6 %`, WER `76.2 %`
  - HSI HF + Tone: CER `25.2 %`, WER `76.2 %`
  - Zero-DCE++: CER `30.2 %`, WER `81.0 %`
  - RetinexNet: CER `28.8 %`, WER `85.7 %`
- HSI HF + Tone gives the largest reduction (-7.9 absolute CER
  points; ~24% relative). Both project pipelines outperform both
  pretrained CNN baselines on this downstream task. Promoted figure
  at `results/final/ocr_handwriting_comparison.png`.

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
- `results/final/cnn_comparison_showcase.png`
- `results/final/hsi_cnn_color_comparison.png`
- `results/final/ocr_handwriting_comparison.png`
