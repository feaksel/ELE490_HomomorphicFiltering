# Decisions

## Active Decisions

### D-001 Synthetic Validation First

- Status: accepted
- Decision:
  Build and validate the grayscale pipeline on synthetic illumination before
  relying on real-image visual judgment.
- Why:
  Synthetic data gives known corruption and lets us measure PSNR and SSIM.

### D-002 Global Real-Scene Standard

- Status: accepted
- Decision:
  Use a Gaussian homomorphic filter with:
  `gamma_L=0.06`, `gamma_H=1.00`, `D0=320`
  plus a gentle brightness lift before the final showcase tone-equalization
  stage.
- Why:
  This setting came out of tunnel-driven real-scene tuning and generalized
  reasonably well across the active photo showcase.

### D-003 Brightened Outputs for Final Comparisons

- Status: accepted
- Decision:
  Use the brightened homomorphic outputs as the intermediate stage of the
  active showcase pipeline rather than the raw non-bright versions, then apply
  tone equalization for the final report-facing result.
- Why:
  The raw outputs often revealed detail but still read too dim in report-facing
  side-by-side figures.

### D-004 Current Active Showcase Set

- Status: accepted
- Decision:
  Keep the active final showcase focused on:
  `cardboard`, `markers`, `page`, `pillar`, and `seat`,
  with `carboard_uniform` and `markers_uniform` as reference-only images.
- Why:
  These examples better support the current report story than the older mixed
  image set.

### D-005 Uniform-Lighting References Are Validation Aids

- Status: accepted
- Decision:
  Use `carboard_uniform` and `markers_uniform` as visual reference images, not
  as strict ground truth.
- Why:
  They help judge whether homomorphic filtering moves the scene closer to more
  even illumination without pretending pixel-perfect correspondence exists.

### D-006 Page Uses a Different Recommended Setting

- Status: accepted
- Decision:
  For the book-page document example, prefer the conservative page-specific
  setting over the global real-scene standard:
  `gamma_L=0.25`, `gamma_H=1.00`, `D0=160`, bright gamma `0.84`.
- Why:
  The global standard looked somewhat over-processed on fine paper texture.
  The conservative setting gave a better readability-preserving balance.

### D-007 Tone Equalization Is the Active Final Stage

- Status: accepted
- Decision:
  Use the shadow/highlight style tone-equalization step as the default final
  stage for the active showcase after homomorphic filtering.
- Why:
  The user chose this as the regular showcase pipeline, and it gives a more
  readable final presentation across the professor-facing figures.

### D-008 Mid-Project Final Showcase Must Stay Narrow

- Status: accepted
- Decision:
  Keep `results/final/` restricted to the figures we would actually attach in a
  professor progress update rather than every experiment that produced a nice
  image.
- Why:
  A smaller set is easier to explain and avoids mixing accepted results with
  rejected post-processing branches.

### D-011 Downstream OCR Task Confirms The Pipeline Improves Real-Image Readability

- Status: accepted
- Decision:
  Use handwriting OCR (TrOCR `microsoft/trocr-large-handwritten`) on
  `writing.jpeg` as the project's quantitative real-image
  validation. Each line is processed through (Original, HF + Tone
  Gray, HSI HF + Tone, Zero-DCE++, RetinexNet); corpus CER and WER
  are computed against manually-transcribed ground truth.
- Why:
  Synthetic PSNR / SSIM already validates the method in a controlled
  setting, but `R-001` flagged that real-image success was only
  qualitative. A downstream task gives an honest quantitative
  improvement claim that does not rely on a "clean" reference for
  real photos. On `writing.jpeg`, HSI HF + Tone reduces corpus CER
  from `33.1 %` to `25.2 %` (-7.9 abs / ~24 % relative); HF + Tone
  grayscale reduces CER to `26.6 %`. Both project pipelines
  outperform Zero-DCE++ (`30.2 %`) and RetinexNet (`28.8 %`) on this
  downstream consumer.

### D-010 HSI Color Pipeline Reuses The Accepted Grayscale Settings

- Status: accepted
- Decision:
  When the homomorphic + brightness lift + tone equalization pipeline is run
  on a color image, convert RGB to HSI, process only the intensity channel
  with `utils.showcase_pipeline.apply_regular_showcase_pipeline`, and
  recombine. Use the same per-scene config rules: standard global settings
  by default, the conservative override for `page`.
- Why:
  Treating illumination as an intensity-channel phenomenon and leaving hue /
  saturation untouched keeps colors natural and avoids the color shifts the
  pretrained CNN baselines introduce (especially RetinexNet, which casts the
  output bluish-warm depending on case). Reusing the validated grayscale
  configuration means no new parameters need separate validation.

### D-009 Pretrained CNN Comparison Is Phase-2 Evidence, Not A Replacement Baseline

- Status: accepted
- Decision:
  Include pretrained Zero-DCE++ and RetinexNet as comparison baselines in the
  Phase-2 evaluation (`results/experimental/cnn/` and the
  `results/experimental/evaluation/` tables) and in the report-facing figure
  `results/final/cnn_comparison_showcase.png`. Do not promote either CNN over
  the homomorphic baseline in the project's main pipeline.
- Why:
  Homomorphic filtering remains the synthetic-quality leader (SSIM `0.9380`
  versus `0.8665` for Zero-DCE++ and `0.7624` for RetinexNet on the four
  controlled corruption patterns). Zero-DCE++ wins on CPU runtime but trades
  away detail recovery. RetinexNet is both slower and lower quality than the
  classical baseline on this evaluation, since the pretrained checkpoint was
  trained on LOL-style low-light pairs rather than non-uniform illumination.
  Presenting these models as comparison baselines strengthens the report
  without overstating their performance.

## Rejected Decisions

### R-001 Histogram Equalization as Part of the Current Showcase

- Status: rejected
- Rejected idea:
  Keep histogram equalization in the active current-showcase comparisons.
- Why rejected:
  It distracts from the main story, is not the best comparison for our current
  goal, and usually looked harsher or less aligned with illumination
  correction.

### R-002 Page HEQ Before or After HF

- Status: rejected
- Rejected idea:
  Apply global histogram equalization before or after homomorphic filtering on
  the page image.
- Why rejected:
  Both `HE -> HF` and `HF -> HE` increased harshness, texture, and background
  irregularity on the page.

### R-003 CLAHE Around the Global Page HF Setting

- Status: rejected
- Rejected idea:
  Use CLAHE by itself or around the global page homomorphic result.
- Why rejected:
  It made the paper texture and local noise too strong even when text contrast
  increased.

### R-004 CLAHE Around the Conservative Page HF Setting

- Status: rejected
- Rejected idea:
  Add CLAHE before or after the conservative page-specific homomorphic result.
- Why rejected:
  Even with the milder page setting, CLAHE still made the page harsher than the
  conservative HF + bright result alone.

### R-005 Tone Adjustment As a Universal Final Stage

- Status: superseded
- Rejected idea:
  Apply the same tone-adjustment post-processing as the default final step to
  all active showcase images.
- Why rejected:
  This entry records the earlier evaluation stage where tone adjustment was
  still treated as provisional. It was later accepted as the active default
  under `D-007` after the project standardized the regular showcase pipeline.

### R-006 Keep CLAHE / Tone Experiments in the Active Mid-Project Showcase

- Status: rejected
- Rejected idea:
  Leave the CLAHE and tone-adjustment branches in the active scripts and
  `results/final/` even after deciding not to present them.
- Why rejected:
  They clutter the report path, confuse the professor-facing narrative, and are
  better kept under `scripts/old/` and `results/old/` as archived decisions.

### R-007 Promote Phase-2 CLAHE Branch (Selected and Boosted) Into The Visual Story

- Status: rejected / failed
- Rejected idea:
  Show the Phase-2 winner `CLAHE 16x16 clip=0.01` and its high-boost variant
  in the report-facing visual overviews next to the homomorphic baseline.
- Why rejected:
  These CLAHE branches do not improve the image in the direction this project
  cares about. They emphasize local texture, noise, and fabric / paper grain
  rather than restoring uniform illumination — the same failure mode that
  rejected earlier page-specific CLAHE branches (`R-003`, `R-004`). The
  Phase-2 sweep selected them only on lightweight proxy metrics
  (entropy, runtime); homomorphic filtering remains stronger on the
  illumination-correction goal and is the synthetic SSIM leader by a wide
  margin. CLAHE rows are kept in the quantitative tables for honest
  reporting but omitted from `hard_case_visual_overview.png` and
  `hard_case_crop_overview.png`.

## Pending Decisions

### P-001 Final Reporting Direction

- Status: resolved
  (CNN comparison closed under `D-009`, HSI color pipeline validated
  under `D-010`, downstream OCR readability validated under `D-011`).
- Remaining optional follow-ups for future work:
  - extending the handwriting OCR benchmark to additional bad-scan
    samples for stronger n
  - broader real-life application examples (X-ray, document
    binarization, etc.)

### P-002 Final Presentation Scope After Mid-Project Update

- Status: pending
- Decision to make:
  After the mid-project progress update, decide whether the final phase should
  prioritize validated color work, real-image metrics, or one stronger practical
  application track such as documents.
