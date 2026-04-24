# Best Local Method

- Selected method: `CLAHE 16x16 clip=0.01`
- Boosted branch: `CLAHE 16x16 clip=0.01 + High Boost`
- Selection rule: highest average synthetic SSIM, then PSNR, then lowest hard-case runtime
- High-boost sigma: `1.2`
- High-boost amount: `0.65`
