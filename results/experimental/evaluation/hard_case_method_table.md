# Hard-Case Proxy Metric Table

| Method | Avg Entropy | Avg P99-P1 | Avg HF Share (%) | Avg Mean Gradient | Avg Clipped (%) | Avg Runtime (ms) |
| --- | --- | --- | --- | --- | --- | --- |
| HF + Tone Baseline | 7.301 | 230.25 | 50.67 | 52.398 | 3.52 | 1867.47 |
| CLAHE 16x16 clip=0.01 | 7.587 | 242.75 | 6.39 | 23.300 | 4.79 | 710.46 |
| CLAHE 16x16 clip=0.01 + High Boost | 7.509 | 249.38 | 12.15 | 34.327 | 8.82 | 782.77 |
| Zero-DCE++ | 7.207 | 193.75 | 5.45 | 15.192 | 1.84 | 262.72 |
| RetinexNet | 6.860 | 166.25 | 15.08 | 19.614 | 1.55 | 9798.34 |
