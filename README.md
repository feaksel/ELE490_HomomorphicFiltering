# Homomorphic Filtering for Non-Uniform Illumination Correction

## Course

ELE490 - Fundamentals of Image Processing, Hacettepe University

## Description

This project studies blind homomorphic filtering for correcting non-uniform illumination in grayscale and color images. The implementation is intentionally step by step and script-based so each stage of the textbook pipeline can be inspected and presented clearly.

The current focus is on grayscale experiments with synthetic illumination patterns and the `rice.png` benchmark. The color HSI pipeline is also prepared and only needs real uneven-lighting photos for final validation.

## Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

On this machine, the working interpreter was `py -3.9`, so the scripts were tested with that command style.

## Dataset

Place the input images inside `images/`.

Expected files:

- `cameraman.tif` - grayscale base image for synthetic experiments
- `rice.png` - grayscale benchmark with non-uniform background illumination
- `photo_1.jpg`
- `photo_2.jpg`
- `photo_3.jpg`

Script `01_create_synthetic.py` also generates:

- `synthetic_vertical.png`
- `synthetic_rotated.png`
- `synthetic_sine.png`
- `synthetic_uneven.png` as the default synthetic test case

## Usage

Run the scripts from the project root:

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
```

## Recommended Figures

For the strongest grayscale presentation, the most useful output figures are:

- `results/rice_aggressive_comparison.png`
- `results/homomorphic_multicase_overview.png`
- `results/synthetic_aggressive_demo.png`
- `results/synthetic_difference_maps.png`
- `results/padding_windowing_experiment.png`
- `results/final_grayscale_demo_pack.png`
- `results/flashlight_homomorphic_comparison.png`
- `results/tun_homomorphic_comparison.png`
- `results/blind_results_table.md`

## Current Project Story

- `rice.png` is the main visual demo because the illumination correction is easy to see there.
- The synthetic cases are mainly for controlled blind evaluation and PSNR / SSIM reporting.
- Blind-only results and tuning decisions are tracked in `DESIGN.md`.

## References

- Rafael C. Gonzalez and Richard E. Woods, _Digital Image Processing_, 2nd Edition, Section 4.5
- Alan V. Oppenheim, Ronald W. Schafer, and Thomas G. Stockham Jr., "Nonlinear Filtering of Multiplied and Convolved Signals," _Proceedings of the IEEE_, 1968
