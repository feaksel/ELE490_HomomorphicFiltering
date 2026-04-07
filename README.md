# Homomorphic Filtering for Non-Uniform Illumination Correction

## Course

ELE490 - Fundamentals of Image Processing, Hacettepe University, Spring 2025-2026

## Description

This project studies homomorphic filtering for correcting non-uniform illumination in grayscale and color images. The scripts follow the textbook pipeline in a simple, step-by-step style so the effect of filter type and parameter choices can be observed clearly.

The project includes a synthetic test case, grayscale experiments, parameter sweeps, comparison with histogram equalization, a color extension in HSI space, and basic quantitative evaluation.

## Setup

Install the required Python packages:

```bash
pip install -r requirements.txt
```

## Dataset

Place the input images inside the `images/` folder before running the scripts.

Expected files:

- `cameraman.tif` - from course homework
- `rice.png` - standard test image (download from MATLAB image processing examples or another standard source)
- `photo_1.jpg` - self-captured uneven lighting photo
- `photo_2.jpg` - self-captured uneven lighting photo
- `photo_3.jpg` - self-captured uneven lighting photo

Script `01_create_synthetic.py` will also create:

- `synthetic_uneven.png` - generated synthetic non-uniform illumination image

## Usage

Run the scripts from the project root in this order:

```bash
python scripts/01_create_synthetic.py
python scripts/02_homomorphic_grayscale.py
python scripts/03_parameter_sweep.py
python scripts/04_compare_with_heq.py
python scripts/05_color_hsi.py
python scripts/06_metrics.py
```

Each script saves figures and outputs to the `results/` folder.

## References

- Rafael C. Gonzalez and Richard E. Woods, Digital Image Processing, 2nd Edition, Section 4.5
- Alan V. Oppenheim, Ronald W. Schafer, and Thomas G. Stockham Jr., "Nonlinear Filtering of Multiplied and Convolved Signals," Proceedings of the IEEE, 1968
