# Synthetic Method Table

| Method | Avg MSE | Avg PSNR | Avg SSIM | Avg Runtime (ms) |
| --- | --- | --- | --- | --- |
| Homomorphic Synthetic Baseline | 1100.199 | 17.889 | 0.9380 | 69.96 |
| CLAHE 16x16 clip=0.01 | 1597.519 | 16.150 | 0.8655 | 132.13 |
| CLAHE 16x16 clip=0.01 + High Boost | 1807.681 | 15.605 | 0.7823 | 64.74 |
| Zero-DCE++ | 1878.978 | 15.546 | 0.8665 | 13.78 |
| RetinexNet | 3936.978 | 12.181 | 0.7624 | 317.98 |
