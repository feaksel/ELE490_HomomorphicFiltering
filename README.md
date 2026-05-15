# Homomorphic Filtering for Non-Uniform Illumination Correction

## Course

ELE490 - Fundamentals of Image Processing, Hacettepe University

## Project Summary

This project studies blind homomorphic filtering for non-uniform illumination correction. The implementation is intentionally script-based and easy to inspect, so each experiment can be run and explained independently.

The project now has three main parts:

- controlled grayscale experiments on synthetic illumination patterns
- benchmark-style grayscale experiments on `rice.png`
- real-scene grayscale experiments on tunnel, flashlight, chest X-rays, and the current RGB photo showcase converted to grayscale

The current real-scene standard uses a blind Gaussian homomorphic filter with:

- `gamma_L = 0.06`
- `gamma_H = 1.00`
- `D0 = 320`

followed by a gentle brightness lift and a tone-equalization post step for the
active showcase figures. For `page.jpeg`, the homomorphic stage uses a more
conservative setting before the same tone-equalization step.

## Current Status

- The grayscale pipeline is implemented end to end.
- Synthetic experiments, metrics, parameter sweeps, and histogram-equalization comparisons are implemented.
- Real-scene grayscale demos are implemented for tunnel, flashlight, and multiple RGB photos converted to grayscale.
- The Phase-2 comparison branch is complete: local-equalization (CLAHE/AHE) and two pretrained CNN baselines (Zero-DCE++ and a PyTorch port of RetinexNet) have been run against the homomorphic baseline. The homomorphic pipeline remains the synthetic-quality leader; see `results/experimental/evaluation/` and `results/final/cnn_comparison_showcase.png`.
- The HSI color pipeline is validated. The accepted homomorphic + tone-equalization pipeline now runs end to end on color hard-case images through `utils/hsi_pipeline.py` and `scripts/30_hsi_color_pipeline.py`. Color comparison against the two CNN baselines is at `results/final/hsi_cnn_color_comparison.png` (representative set) and `results/experimental/evaluation/hsi_cnn_all_color_overview.png` (full set).
- A downstream-task OCR benchmark is in place: TrOCR `microsoft/trocr-large-handwritten` runs on the severe bad-scan handwriting image (`writing.jpeg`, 8 manual lines) plus 30 deterministic Bentham historical handwriting lines. On `writing.jpeg`, HF + Tone reduces CER from 34.5 % to 20.9 %, while Sauvola gives 20.1 % and is the strongest classical OCR preprocessor for that sample. On Bentham, the original line crops are already strong, so preprocessing is not universally beneficial. See `results/final/ocr_handwriting_comparison.png` and `results/experimental/ocr/`.

## Setup

Install packages with:

```bash
pip install -r requirements.txt
```

On this machine the working interpreter was `py -3.9`, so the scripts were verified with that command style.

## Main Inputs

Expected or currently used active inputs in `images/`:

- `rice.png`
- `tun.jpg`
- `cardboard.jpg`
- `carboard_uniform.jpg`
- `carpet.jpg`
- `markers.jpg`
- `markers_uniform.jpg`
- `page.jpeg`
- `pillar.jpg`
- `seat.jpg`
- `writing.jpeg`

Older or retired examples can live under `images/old/` without affecting the current showcase scripts.

Script `01_create_synthetic.py` also generates:

- `synthetic_vertical.png`
- `synthetic_rotated.png`
- `synthetic_sine.png`
- `synthetic_uneven.png`

## Scripts

Run scripts from the project root:

```bash
py -3.9 scripts/01_create_synthetic.py
py -3.9 scripts/02_homomorphic_grayscale.py
py -3.9 scripts/03_parameter_sweep.py
py -3.9 scripts/04_compare_with_heq.py
py -3.9 scripts/05_color_hsi.py
py -3.9 scripts/06_metrics.py
py -3.9 scripts/07_padding_windowing_experiment.py
py -3.9 scripts/08_final_grayscale_pack.py
py -3.9 scripts/09_flashlight_demo.py
py -3.9 scripts/10_tunnel_demo.py
py -3.9 scripts/11_color_grayscale_standard_demo.py
py -3.9 scripts/12_selected_hf_comparison.py
py -3.9 scripts/13_organize_results.py
py -3.9 scripts/14_xray_demo.py
py -3.9 scripts/15_uniform_reference_comparison.py
py -3.9 scripts/16_seat_pillar_detail_comparison.py
py -3.9 scripts/18_page_conservative_hf_comparison.py
py -3.9 scripts/20_tone_adjusted_showcase.py
py -3.9 scripts/21_page_detail_comparison.py
py -3.9 scripts/22_frequency_analysis_showcase.py
py -3.9 scripts/23_page_pipeline_math.py
py -3.9 scripts/24_seat_pipeline_math.py
py -3.9 scripts/25_local_equalization_hard_cases.py
py -3.9 scripts/26_pretrained_cnn_hard_cases.py
py -3.9 scripts/27_next_phase_evaluation.py
py -3.9 scripts/28_prepare_cnn_models.py
py -3.9 scripts/29_cnn_comparison_figure.py
py -3.9 scripts/30_hsi_color_pipeline.py
py -3.9 scripts/31_hsi_cnn_color_comparison.py
py -3.9 scripts/32_ocr_handwriting_pipeline.py
py -3.9 scripts/33_ocr_comparison_figure.py
py -3.9 scripts/34_prepare_bentham_sample.py
```

Scripts 32-34 use the OCR-only dependencies in `requirements-ocr.txt`:

```bash
pip install -r requirements-ocr.txt
```

Script 34 expects the Bentham R0 ground-truth archive at
`data/bentham_raw/BenthamDatasetR0-GT.tbz` and regenerates the ignored
`data/bentham_sample/` folder plus the 38-line OCR manifest. Tesseract is not
required for the current OCR benchmark; on this machine the native Tesseract
binary and Turkish language pack were not available, so Tesseract remains a
future printed-document side experiment.

Script `28_prepare_cnn_models.py` reads raw `.pth` / `.tar` weights from
`models/zerodcepp/` and `models/retinexnet/` and writes TorchScript files
that script `26` can load. Run it before script `26` if the CNN branch is
new on this machine.

## Recommended Outputs

If you want the strongest current mid-project figures first, open these:

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/page_conservative_hf_comparison.png`
- `results/final/page_detail_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`
- `results/final/cnn_comparison_showcase.png`
- `results/final/hsi_cnn_color_comparison.png`
- `results/final/ocr_handwriting_comparison.png`
- `results/analysis/showcase_metric_deltas.png`
- `results/analysis/showcase_analysis_summary.md`
- `results/analysis/page_pipeline_math_figure.png`
- `results/analysis/seat_pipeline_math_figure.png`

## Results Folder

The `results/` folder is now organized to be easier to navigate:

- `results/final/` for the current professor-reportable showcase
- `results/real_batch/` for per-image grayscale, homomorphic, brightened, and tone-equalized outputs for the active photo showcase
- `results/real_references/` for side-by-side comparisons against the uniform-lighting reference photos
- `results/analysis/` for before/after histograms, FFT-domain visuals, and metric tables
- `results/experimental/` for the new local-HEQ, optional CNN, and hard-case evaluation branch
- `results/old/` for rejected or retired experiments such as CLAHE trials

## Optional CNN Setup

The CNN comparison branch is optional and expects local TorchScript weights.
Core project requirements stay in `requirements.txt`; the optional CNN branch
uses:

```bash
pip install -r requirements-cnn.txt
```

Script `26_pretrained_cnn_hard_cases.py` discovers every available
TorchScript model under `models/zerodcepp/` and `models/retinexnet/` (or
via the `ZERO_DCEPP_MODEL_PATH` / `RETINEXNET_MODEL_PATH` environment
variables) and runs each one. If no weights are present, it writes a
status note and skips inference. Use `scripts/28_prepare_cnn_models.py`
to convert raw `.pth` / `.tar` weights into the TorchScript wrappers the
inference script expects.

Expected raw weights:

- `models/zerodcepp/Epoch99.pth` from
  https://github.com/Li-Chongyi/Zero-DCE_extension
  (path: `Zero-DCE++/snapshots_Zero_DCE++/Epoch99.pth`)
- `models/retinexnet/Decom_9200.tar` and
  `models/retinexnet/Relight_9200.tar` from
  https://github.com/aasharma90/RetinexNet_PyTorch
  (paths: `ckpts/Decom/9200.tar` and `ckpts/Relight/9200.tar`)

After downloading, run `py -3 scripts/28_prepare_cnn_models.py` once to
produce `models/zerodcepp/zerodcepp.ts` and `models/retinexnet/retinexnet.ts`.

See `results/README.md` for a short guide.

## Documentation

Project history and reporting support now live under `docs/`:

- `docs/WORKLOG.md` for chronological changes and experiment history
- `docs/DECISIONS.md` for accepted and rejected choices
- `docs/ISSUES_AND_RISKS.md` for bugs, open issues, and reporting risks
- `docs/REPORTING_NOTES.md` for final-report talking points
- `docs/PROJECT_SUMMARY.md` for a clean full-project summary of what we built and what the active pipeline is
- `docs/README.md` for the maintenance rule going forward

If older scripts generate new files back into the root `results/` folder, run:

```bash
py -3.9 scripts/13_organize_results.py
```

## Project Story

- `rice.png` remains the clearest grayscale benchmark demo.
- The synthetic images are mainly for controlled validation and PSNR / SSIM reporting.
- The current showcase focuses on cardboard, markers, page, pillar, and seat, with uniform-lighting references for cardboard and markers.
- Tunnel experiments were used to tune a stronger blind-only flattening direction and to separate true blind filtering from extra display enhancement.
- The active final figures now use a regular two-stage pipeline: homomorphic filtering followed by tone equalization.
- For the page example, the report-facing recommendation is the conservative page-specific homomorphic setting before tone equalization.

## References

- Rafael C. Gonzalez and Richard E. Woods, _Digital Image Processing_, 2nd Edition, Section 4.5
- Alan V. Oppenheim, Ronald W. Schafer, and Thomas G. Stockham Jr., "Nonlinear Filtering of Multiplied and Convolved Signals," _Proceedings of the IEEE_, 1968
