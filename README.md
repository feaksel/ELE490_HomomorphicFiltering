# Homomorphic Filtering Personal Project

This repository is my personal course project for `ELE490 - Fundamentals of Image Processing` at Hacettepe University.

## What I Am Exploring

The main idea behind the project is the classic illumination-reflectance model:

`f(x, y) = i(x, y) * r(x, y)`

I wanted to better understand how homomorphic filtering separates slow illumination changes from image detail in the frequency domain, and how parameter choices affect the final result. The project follows the textbook pipeline step by step so I can see each stage clearly instead of hiding everything inside one function.

## Why I Built It This Way

I split the work into small scripts because this is easier for me to study, debug, and present. Each script focuses on one part of the project:

- `scripts/01_create_synthetic.py` creates a synthetic uneven-lighting example from `cameraman.tif`
- `scripts/02_homomorphic_grayscale.py` is meant to hold the main grayscale homomorphic filtering pipeline
- `scripts/03_parameter_sweep.py` is for testing how `gamma_L`, `gamma_H`, and `D0` change the output
- `scripts/04_compare_with_heq.py` compares homomorphic filtering with histogram equalization
- `scripts/05_color_hsi.py` extends the idea to color images by filtering the intensity channel in HSI space
- `scripts/06_metrics.py` is for basic evaluation such as PSNR and SSIM on the synthetic example

## Current Status

This is still a work in progress. The repo already has the structure, script flow, and experiment plan, but many parts of the implementation are still marked with `TODO`s. I wanted the repository to reflect the actual development process rather than pretend everything is already complete.

## Project Layout

- `images/` stores the input images used for experiments
- `results/` stores generated figures and outputs
- `scripts/` contains the step-by-step experiment files
- `utils/filters.py` contains helper functions for filter construction
- `DESIGN.md` is my rough planning document for the project

## Input Images

I expect to place these files in `images/` before running the experiments:

- `cameraman.tif` for the synthetic test setup
- `rice.png` as a standard grayscale test image
- `photo_1.jpg`
- `photo_2.jpg`
- `photo_3.jpg`

The photo files are intended to be my own uneven-lighting examples for the color part of the project.

## Setup

Install the required packages with:

```bash
pip install -r requirements.txt
```

## Planned Script Order

When the implementations are filled in, the scripts are intended to be run from the project root in this order:

```bash
python scripts/01_create_synthetic.py
python scripts/02_homomorphic_grayscale.py
python scripts/03_parameter_sweep.py
python scripts/04_compare_with_heq.py
python scripts/05_color_hsi.py
python scripts/06_metrics.py
```

## References I Am Following

- Rafael C. Gonzalez and Richard E. Woods, _Digital Image Processing_, 2nd Edition, Section 4.5
- Alan V. Oppenheim, Ronald W. Schafer, and Thomas G. Stockham Jr., "Nonlinear Filtering of Multiplied and Convolved Signals," _Proceedings of the IEEE_, 1968
