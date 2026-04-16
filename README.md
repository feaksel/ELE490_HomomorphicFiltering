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

followed by a gentle brightness lift for presentation on difficult real scenes.

## Current Status

- The grayscale pipeline is implemented end to end.
- Synthetic experiments, metrics, parameter sweeps, and histogram-equalization comparisons are implemented.
- Real-scene grayscale demos are implemented for tunnel, flashlight, and multiple RGB photos converted to grayscale.
- The HSI color pipeline is structurally ready, but it still needs final validation on a real color-photo set.

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
py -3.9 scripts/12_selected_hf_vs_heq_comparison.py
py -3.9 scripts/13_organize_results.py
py -3.9 scripts/14_xray_demo.py
py -3.9 scripts/15_uniform_reference_comparison.py
py -3.9 scripts/16_seat_pillar_detail_comparison.py
```

## Recommended Outputs

If you want the strongest current figures first, open these:

- `results/final/color_grayscale_standard_overview.png`
- `results/final/selected_real_images_hf_bright_overview.png`
- `results/final/seat_pillar_detail_comparison.png`
- `results/final/uniform_reference_comparison_overview.png`
- `results/final/cardboard_uniform_reference_comparison.png`
- `results/final/markers_uniform_reference_comparison.png`

## Results Folder

The `results/` folder is now organized to be easier to navigate:

- `results/final/` for current presentation-ready figures from the new photo showcase
- `results/real_batch/` for per-image grayscale, homomorphic, and brightened outputs for the active photo showcase
- `results/real_references/` for side-by-side comparisons against the uniform-lighting reference photos

See `results/README.md` for a short guide.

If older scripts generate new files back into the root `results/` folder, run:

```bash
py -3.9 scripts/13_organize_results.py
```

## Project Story

- `rice.png` remains the clearest grayscale benchmark demo.
- The synthetic images are mainly for controlled validation and PSNR / SSIM reporting.
- The current showcase focuses on cardboard, carpet, markers, pillar, and seat, with uniform-lighting references for cardboard and markers.
- Tunnel experiments were used to tune a stronger blind-only flattening direction and to separate true blind filtering from extra display enhancement.

## References

- Rafael C. Gonzalez and Richard E. Woods, _Digital Image Processing_, 2nd Edition, Section 4.5
- Alan V. Oppenheim, Ronald W. Schafer, and Thomas G. Stockham Jr., "Nonlinear Filtering of Multiplied and Convolved Signals," _Proceedings of the IEEE_, 1968
