# Next-Phase Evaluation Summary

- Synthetic quality leader: `Homomorphic Synthetic Baseline`
- Hard-case runtime leader: `Zero-DCE++`
- Balanced quality/runtime winner: `Zero-DCE++` (ranked by synthetic SSIM, aggregate hard-case proxy quality, and CPU runtime)
- Selected best local method from script 25: `CLAHE 16x16 clip=0.01` (rejected for the visual story, see `D-009`/`R-007`: emphasizes texture / noise rather than recovering uniform illumination)
- Selected boosted local branch: `CLAHE 16x16 clip=0.01 + High Boost` (same rejection rationale; kept only in the metric tables)
- CNN comparators included: `Zero-DCE++`, `RetinexNet`
