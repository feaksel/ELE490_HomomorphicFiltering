# Next-Phase Evaluation Summary

- Synthetic quality leader: `Homomorphic Synthetic Baseline`
- Hard-case runtime leader: `CLAHE 16x16 clip=0.01 + High Boost`
- Balanced quality/runtime winner: `CLAHE 16x16 clip=0.01 + High Boost` (ranked by synthetic SSIM, aggregate hard-case proxy quality, and CPU runtime)
- Selected best local method from script 25: `CLAHE 16x16 clip=0.01`
- Selected boosted local branch: `CLAHE 16x16 clip=0.01 + High Boost`
- CNN comparator not included: no local TorchScript weights were available.
