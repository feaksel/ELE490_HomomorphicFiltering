# Project Summary

## Project Title

Homomorphic Filtering for Non-Uniform Illumination Correction

## One-Sentence Project Story

This project studies how well homomorphic filtering can reduce slow,
low-frequency illumination variation while preserving useful scene detail, then
extends that base correction with a regular report-facing post-processing
pipeline for clearer real-image presentation.

## What We Built

We implemented a full grayscale homomorphic-filtering pipeline based on the
standard multiplicative illumination-reflectance model:

- `f(x, y) = i(x, y) * r(x, y)`
- logarithm transform
- Fourier transform
- homomorphic frequency-domain filtering
- inverse transform
- exponential reconstruction

On top of that base pipeline, we added:

- synthetic uneven-illumination generation for controlled validation
- parameter sweeps for `gamma_L`, `gamma_H`, and `D0`
- PSNR and SSIM measurement on the synthetic case
- grayscale real-scene demos
- a structured results organization workflow
- a shared showcase helper that applies the accepted regular pipeline

## Current Accepted Showcase Pipeline

For the active professor-facing real-scene showcase, the project now uses this
regular flow:

1. convert RGB input to grayscale when needed
2. apply homomorphic filtering
3. apply a gentle brightness lift
4. apply tone equalization for the final report-facing output

Current global real-scene homomorphic setting:

- filter: `Gaussian`
- `gamma_L = 0.06`
- `gamma_H = 1.00`
- `D0 = 320`
- bright gamma: `0.72`

Current page-specific document setting:

- filter: `Gaussian`
- `gamma_L = 0.25`
- `gamma_H = 1.00`
- `D0 = 160`
- bright gamma: `0.84`

The page case uses the milder setting because the stronger global setting can
over-process fine paper texture.

## What We Did

### 1. Built and Validated the Core Method

- Implemented the grayscale homomorphic-filtering pipeline end to end.
- Generated synthetic uneven-illumination cases from `cameraman.tif`.
- Added multiple synthetic illumination styles instead of relying on one mild
  corruption pattern.
- Measured restoration quality with PSNR and SSIM on the synthetic case.
- Verified improvement from about `13.35 dB / 0.8280 SSIM` on the corrupted
  image to about `17.89 dB / 0.9019 SSIM` on the restored image.

### 2. Studied Parameter Behavior

- Compared Butterworth and Gaussian filter families.
- Swept `gamma_L`, `gamma_H`, and `D0` to understand how the method balances
  illumination suppression and detail preservation.
- Added a padding/windowing experiment to study processing choices around the
  base algorithm.

### 3. Expanded to Real Images

- Added grayscale real-scene demos using `flashlight.jpeg` and `tun.jpg`.
- Used tunnel experiments to tune the current global real-scene direction.
- Extended the workflow to RGB photos by converting them to grayscale first.
- Built batch-style real-image comparisons for multiple scene types.

### 4. Compared Against Simpler Alternatives

- Implemented histogram-equalization comparisons.
- Tested chest X-ray examples with both homomorphic filtering and histogram
  equalization.
- Tested document-page variants with histogram equalization before and after
  homomorphic filtering.
- Tested CLAHE-based document variants.

### 5. Selected the Strongest Reporting Direction

- Retired weaker or noisier enhancement branches from the active story.
- Narrowed the active showcase to the strongest reportable examples:
  `cardboard`, `markers`, `page`, `pillar`, and `seat`.
- Added uniform-lighting reference comparisons for `cardboard` and `markers`.
- Added dedicated detail comparisons for `seat` and `pillar`.
- Promoted the page example as the strongest practical document-readability
  use case.

### 6. Standardized the Final Showcase

- Added a shared helper in `utils/showcase_pipeline.py`.
- Unified the active scripts around the same regular pipeline.
- Adopted tone equalization as the accepted final report-facing stage.
- Kept rejected CLAHE branches archived under `scripts/old/` and `results/old/`
  instead of mixing them into the active showcase.

## Strongest Current Results

The strongest current figures are:

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/page_conservative_hf_comparison.png`
- `results/final/page_detail_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`

Note:
Some filenames still contain older wording such as `hf_bright`, but the active
showcase interpretation is now the full regular pipeline ending in tone
equalization.

## Strongest Examples To Talk About

- `cardboard`
  Strong uneven-illumination correction with easy-to-see improvement.
- `markers`
  Good reference-backed example for flattening uneven lighting.
- `page`
  Best practical document-readability case.
- `seat` and `pillar`
  Best examples for discussing texture/detail behavior.

## What We Rejected

- Keeping histogram equalization as part of the active final story.
- Using CLAHE around the document pipeline.
- Treating uniform-reference images as exact ground truth.
- Claiming real-image correction as objectively perfect restoration.

These branches remain useful as evidence of honest experimentation, but they
are not part of the accepted final showcase.

## Main Technical Conclusion

Homomorphic filtering is effective when the dominant problem is slow,
low-frequency uneven illumination. It is especially useful when paired with a
carefully controlled display-oriented post step. The method is not a universal
enhancement tool, and aggressive settings can amplify texture, noise, or make
some images look unnatural.

## Best Real-Life Framing Right Now

The cleanest practical framing for this project is:

- document enhancement under uneven lighting
- preprocessing before OCR or readability improvement
- general low-cost vision preprocessing for scenes with shadows or directional
  illumination

## Remaining Gaps

- the HSI color pipeline exists structurally but is not yet fully validated
- most real-image evaluation is still qualitative
- scene-specific parameter choices should be described honestly in the report

## Recommended Next Step

If the project needs one stronger final-phase identity, the best next step is
to measure a practical downstream task, especially document readability or OCR,
before and after the current pipeline.
