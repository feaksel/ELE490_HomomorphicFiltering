# Reporting Notes

## Strongest Current Story

The project shows that homomorphic filtering can reduce low-frequency uneven
illumination while preserving useful scene detail, especially when the
homomorphic stage is followed by the project's regular tone-equalization step.

## Strongest Figures Right Now

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/page_conservative_hf_comparison.png`
- `results/final/page_detail_comparison.png`

## Best Examples To Talk About

- `cardboard`
  Strong uneven illumination correction and easy-to-read visual improvement.
- `markers`
  Good reference-backed example for visible illumination flattening.
- `page`
  Best real-life document/readability use case.
- `seat` and `pillar`
  Good for showing texture/detail behavior, not just overall brightness change.

## What To Say About The Page Example

- The page image is a real-life document scene with directional lighting.
- Homomorphic filtering reduces broad illumination variation across the page.
- A page-specific conservative setting followed by tone equalization gives the
  best readability-preserving result.
- This supports possible use as document enhancement or preprocessing before
  OCR, but we should not claim OCR improvement unless we measure it.

## What Not To Overclaim

- Do not say the method "recovers ground truth illumination" for real photos.
- Do not say CLAHE or histogram equalization improved the page result; they
  were tested and rejected for the reporting direction.
- Do not treat the uniform-lighting references as exact ground truth.

## Current Rejected Enhancements

- Histogram equalization before or after HF on the page.
- CLAHE around the global page HF setting.
- CLAHE around the conservative page HF setting.

## Good Questions For The Final Report / Professor

- Should the final scope stay grayscale-focused, or should we validate the
  color pipeline?
- Should we invest more effort in real-image metrics or in cleaner practical
  demonstrations?
- Is the document-readability use case valuable enough to highlight as a real
  application example?
