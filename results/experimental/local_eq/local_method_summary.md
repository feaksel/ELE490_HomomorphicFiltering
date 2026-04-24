# Local Equalization Method Summary

| Method | Avg Synthetic PSNR | Avg Synthetic SSIM | Avg Hard Mean Gradient | Avg Hard Runtime (ms) |
| --- | --- | --- | --- | --- |
| CLAHE 16x16 clip=0.01 | 16.150 | 0.8655 | 23.300 | 2817.46 |
| CLAHE 64x64 clip=0.01 | 17.012 | 0.8574 | 27.459 | 760.64 |
| CLAHE 32x32 clip=0.01 | 16.990 | 0.8539 | 27.690 | 1021.89 |
| CLAHE 16x16 clip=0.01 + High Boost | 15.605 | 0.7823 | 34.327 | 2016.09 |
| CLAHE 64x64 clip=0.03 | 15.334 | 0.6385 | 44.041 | 846.55 |
| CLAHE 16x16 clip=0.03 | 17.018 | 0.6144 | 45.437 | 2475.00 |
| CLAHE 32x32 clip=0.03 | 15.702 | 0.6049 | 45.955 | 1037.52 |
| AHE 64x64 | 13.101 | 0.5372 | 56.244 | 846.63 |
| AHE 32x32 | 11.638 | 0.4231 | 63.854 | 1022.08 |
| AHE 16x16 | 10.984 | 0.3476 | 71.344 | 2010.80 |
