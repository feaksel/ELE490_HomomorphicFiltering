# Progress Update for Professor

## Short Mail Draft

Subject: ELE490 Project Progress Update - Homomorphic Filtering

Dear Professor,

I wanted to share a short progress update on our ELE490 project on homomorphic filtering for non-uniform illumination correction.

So far, we have implemented the full grayscale homomorphic filtering pipeline and tested it on synthetic images, benchmark-style grayscale data, and several real photographs converted to grayscale. We also completed parameter sweeps and visual comparisons to understand how filter design affects illumination correction and detail preservation.

On the synthetic test case, the tuned homomorphic filter improved the restoration quality from about `13.35 dB / 0.8280 SSIM` to `17.89 dB / 0.9019 SSIM`, which gave us a useful quantitative validation before moving to real images.

For real scenes, we tested several examples and selected a small showcase set. The current standard real-scene setting is a Gaussian homomorphic filter with:

- `gamma_L = 0.06`
- `gamma_H = 1.00`
- `D0 = 320`

Using this setting, we prepared presentation-ready comparisons on images such as `cardboard`, `markers`, `page`, `pillar`, and `seat`. In the active showcase, the homomorphic output is followed by a mild tone-equalization step to improve readability while keeping the illumination-correction effect visible. We also added uniform-lighting reference comparisons for `cardboard` and `markers` so that we can judge whether the final result moves closer to a more evenly illuminated version of the same scene.

We also started testing a book-page example with directional lighting and shadow. This is useful as a more practical real-life use case, because it suggests that homomorphic filtering may help document readability under uneven illumination and may be useful as a preprocessing step before OCR or document enhancement. For this page case, a more conservative homomorphic setting gave the best visual balance before the same tone-equalization step.

At this stage, our main conclusion is that the filter is effective for reducing uneven illumination and improving visibility of surface details, especially on scenes with strong low-frequency lighting variation. The strongest current examples in our dataset are `cardboard` and `markers`.

Our next step is to strengthen the project by either:

1. extending the work to a better color-image pipeline, or
2. adding a more structured quantitative evaluation for the real-image examples.

If you have time, we would appreciate your opinion on which direction would be more valuable for the final presentation/report.

Best regards,  
[Your Name]

## What We Can Report Right Now

- The core homomorphic filtering pipeline is implemented and working end to end.
- Synthetic-image validation is complete, including PSNR and SSIM evaluation.
- Parameter sweeps were used to tune the filter and understand the effect of `gamma_L`, `gamma_H`, and `D0`.
- Real-scene grayscale demos are working and organized into a clean showcase.
- The current final showcase focuses on:
  - `cardboard`
  - `markers`
  - `page`
  - `pillar`
  - `seat`
- Uniform-lighting references were added for `cardboard` and `markers`.
- A dedicated detail comparison was prepared for `seat` and `pillar`.

## Useful Files to Mention or Attach

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/page_conservative_hf_comparison.png`
- `results/final/page_detail_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`

## Good Questions to Ask the Professor

- Would it be better for the final project to emphasize theoretical analysis and parameter tuning, or practical application on real images?
- Should we keep the project grayscale-focused, or would adding a validated color pipeline make the project stronger?
- Would the final report benefit more from additional quantitative metrics, or from a stronger qualitative comparison section?
- Is it acceptable to treat uniform-lighting reference photos as a visual validation tool, even when exact ground truth is not available?
- Would you prefer us to compare homomorphic filtering with one stronger baseline method, or keep the scope focused on homomorphic filtering only?

## How We Can Improve the Project

- Validate the color pipeline properly instead of only converting RGB images to grayscale.
- Add real-image evaluation measures such as local contrast improvement, entropy change, or gradient-based sharpness measures.
- Show failure cases too, so the report is more honest and academically stronger.
- Study how robust the chosen parameter set is across different scenes instead of only optimizing per image.

## Can This Filter Be Used or Enhanced?

Yes, definitely, but with a clear scope.

### Good Use Cases

- correcting non-uniform lighting in scanned documents or printed surfaces
- improving visibility in scenes with shadows or strong directional lighting
- preprocessing before segmentation, OCR, document enhancement, or inspection tasks
- highlighting texture on surfaces where illumination hides detail

### Important Limitation

Homomorphic filtering is good at separating slow illumination changes from higher-frequency reflectance/detail, but it is not a universal enhancement tool. It can also amplify noise, flatten scenes too much, or make results look unnatural if the parameters are too aggressive.

### Reasonable Enhancement Directions

- adaptive parameter selection based on image statistics
- color-space implementation on intensity/value channel
- combining homomorphic filtering with denoising
- adding objective comparison against a small number of modern illumination-correction baselines
