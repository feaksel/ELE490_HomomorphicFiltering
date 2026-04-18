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

## Pending Decisions

### P-001 Final Reporting Direction

- Status: pending
- Decision to make:
  Whether to spend the remaining effort on:
  - validated color-pipeline extension
  - stronger real-image quantitative analysis
  - broader real-life application examples

### P-002 Final Presentation Scope After Mid-Project Update

- Status: pending
- Decision to make:
  After the mid-project progress update, decide whether the final phase should
  prioritize validated color work, real-image metrics, or one stronger practical
  application track such as documents.
