# Results And Attachment Guide

## Purpose

This file explains what the result folders contain, which outputs are active,
which ones are older or retired, which ones were rejected, and how they should
be used in a report or presentation.

## How To Read The Result Categories

Use these meanings consistently in the report:

- `active / accepted`
  Results that support the current project story and can be attached directly.
- `supporting`
  Results that are useful as evidence, appendix material, or internal backup,
  but are not the main figures.
- `archived / old`
  Results from earlier stages that are still valid historically, but are no
  longer central to the final narrative.
- `rejected`
  Results from experiments we intentionally decided not to use because they
  looked harsher, less natural, or less aligned with the illumination-
  correction goal.
- `failed`
  In this project, "failed" usually means an explored branch that did not
  improve the image in a report-worthy way. It does not mean the code crashed.

## Recommended Report Attachment Strategy

### Main Attachments

These are the best files to attach in the main report or presentation:

- `results/final/color_grayscale_standard_overview.png`
  Shows the active grayscale-to-regular-pipeline behavior across the current
  showcase set. Good overview figure.
- `results/final/selected_real_images_hf_bright_overview.png`
  Main selected-image summary for the accepted real-scene pipeline.
  Important note: the filename still says `hf_bright`, but the figure now
  represents the accepted regular pipeline ending in tone equalization.
- `results/final/uniform_reference_comparison_overview.png`
  Shows whether the processed outputs move visually closer to more even
  lighting on the reference-backed scenes.
- `results/final/cardboard_uniform_reference_comparison.png`
  Strong reference-backed example for visible illumination flattening.
- `results/final/markers_uniform_reference_comparison.png`
  Another strong reference-backed example.
- `results/final/seat_pillar_detail_comparison.png`
  Best figure for discussing texture/detail behavior rather than only overall
  brightness.
- `results/final/page_conservative_hf_comparison.png`
  Best figure for the practical document-readability story.
- `results/final/page_detail_comparison.png`
  Best close-up figure for showing why the page example matters.

### Good Appendix Attachments

These are useful in an appendix or backup section:

- `results/metrics/blind_multicase_metrics.png`
  Compact synthetic-case metric summary.
- `results/metrics/metrics_summary.png`
  Quantitative evidence for the controlled synthetic setup.
- `results/metrics/blind_results_table.md`
  Good for a written appendix if you want parameter/metric traces.
- `results/synthetic/*`
  Strong appendix material because these show controlled validation.
- `results/rice/*`
  Good benchmark-style appendix material.
- `results/tunnel/*`
  Good tuning-history appendix material.
- `results/flashlight/*`
  Good early real-scene appendix material.
- `results/xray/*`
  Good exploratory appendix material, but not ideal as the main project claim.

### Do Not Attach In The Main Report Unless Asked

These should usually stay out of the main report body:

- `results/archive/*`
  Historically useful, but not part of the current polished story.
- `results/old/*`
  Explicitly rejected or retired branches.
- most raw per-image intermediates in `results/real_batch/*`
  Useful for evidence and debugging, but too detailed for the main narrative.

## Folder-By-Folder Explanation

### `results/final/`

This is the current report-facing folder.

Meaning:

- Contains the strongest accepted figures.
- These are the ones we would show first to a professor.
- They reflect the accepted regular pipeline:
  `grayscale -> homomorphic filtering -> brightness lift -> tone equalization`

How to use in the report:

- Use most of your actual attachments from here.
- If you want a compact report, choose 4 to 6 figures from this folder.

### `results/real_batch/`

This is the evidence folder for the active real-image pipeline.

Meaning:

- Contains per-image intermediates and final outputs.
- Useful for proving what each processing stage did.
- Not all files here should be attached directly.

How to read the names:

- `*_grayscale.png`
  Input after RGB-to-grayscale conversion.
- `*_homomorphic_standard.png`
  Base homomorphic result before extra presentation-oriented steps.
- `*_homomorphic_standard_bright.png`
  Brightened homomorphic result.
- `*_homomorphic_tone_adjusted.png`
  Final accepted report-facing result for most active images.
- `*_grayscale_vs_standard.png`
  Simple side-by-side summary.
- `*_grayscale_vs_hf_vs_tone.png`
  Three-stage explanation figure showing grayscale, HF + bright, and final
  tone-equalized output.

How to use in the report:

- Use as backup evidence.
- Pull from here if you want to explain the pipeline step by step.
- Especially useful for the page case, because it shows why the conservative
  setting was selected.

### `results/real_references/`

This folder supports the visual validation story.

Meaning:

- Compares processed non-uniform scenes with more evenly illuminated reference
  captures for `cardboard` and `markers`.
- These are not strict ground truth, but they help judge whether illumination
  becomes more even.

How to use in the report:

- Strong material for the main body or appendix.
- Good when explaining that the method moves the scene toward more uniform
  lighting without claiming exact recovery.

### `results/metrics/`

This folder is the quantitative validation block.

Meaning:

- Synthetic-only measurement support.
- Good place to justify that the base method works in a controlled setting.

How to use in the report:

- Include at least one metric figure or table if you want stronger academic
  support.
- Use this especially if someone asks whether the method was validated beyond
  visual inspection.

### `results/synthetic/`

This folder contains the cleanest controlled experiments.

Meaning:

- Synthetic illumination generation
- multiple corruption patterns
- parameter sweeps
- restoration examples
- difference maps and histogram views

How to use in the report:

- Best place to explain the theory and the basic method.
- Good appendix material.
- `baseline_synthetic_restored.png`,
  `homomorphic_pipeline_overview.png`,
  and the sweep figures are the strongest here.

### `results/rice/`

This folder contains a benchmark-style grayscale example.

Meaning:

- Useful for a standard image-processing demonstration.
- Shows homomorphic filtering on a classic uneven-illumination style case.

How to use in the report:

- Good supplementary figure.
- Helpful if you want one example that is not a self-collected photo.

### `results/tunnel/`

This folder is mostly the tuning-history lab for the global real-scene setting.

Meaning:

- Strong blind-flattening experiments
- parameter sweeps
- comparisons between milder and stronger settings
- early post-enhancement comparisons

How to use in the report:

- Use in appendix or discussion.
- Very useful if you want to explain where the global setting
  `gamma_L=0.06`, `gamma_H=1.00`, `D0=320` came from.
- Usually not one of the first figures to attach unless your story emphasizes
  tunnel or shadowed-scene inspection.

### `results/flashlight/`

This folder contains an early real-scene demonstration.

Meaning:

- Shows that the method can reveal detail in a strong localized-light scene.
- Important historically, but no longer one of the strongest final examples.

How to use in the report:

- Keep as optional appendix material.
- Mention only if you want to show the project's development path.

### `results/xray/`

This folder contains exploratory medical-image comparisons.

Meaning:

- Homomorphic filtering versus histogram equalization on chest X-rays.
- Useful as an exploratory application example, not as the main project story.

Why it is not a core final attachment:

- X-rays may already include acquisition or post-processing adjustments.
- Gains can be subtle.
- Overprocessing risk is harder to judge responsibly.

How to use in the report:

- Optional appendix or future-work example.
- Do not present it as strong proof of medical improvement.

### `results/archive/`

This folder keeps older-but-still-valid history.

Meaning:

- older real-batch image set
- obsolete final outputs from a previous reporting phase
- padding/windowing experiment outputs

How to use in the report:

- Mainly historical support.
- Useful if you want to show the project evolved through several image sets and
  figure-selection phases.
- Usually not worth attaching unless your professor asks for the full process.

### `results/old/`

This folder contains explicitly retired or rejected outputs.

Meaning:

- These were explored honestly.
- They were not chosen for the accepted final story.
- Keeping them is academically useful because it shows what was tested and why
  it was dropped.

How to use in the report:

- Mention them briefly as rejected branches.
- Do not attach them as if they are successful results.

## What Counts As Rejected Results

### 1. CLAHE-Related Rejections

Location:

- `results/old/clahe_rejected/`

Includes:

- page CLAHE outputs
- CLAHE before HF
- CLAHE after HF
- conservative page HF plus CLAHE
- leftover HEQ/CLAHE comparison outputs on some real images

Why rejected:

- increased paper texture too much
- increased harshness or local noise
- made document background irregularity more visible
- looked less aligned with illumination correction than the accepted pipeline

Report wording:

- "CLAHE-based variants were tested but rejected because they produced harsher
  and less reportable document results than the chosen homomorphic pipeline."

### 2. Earlier Post-Processing Rejections

Location:

- `results/old/postprocessing_rejected/`

Includes:

- older tone-adjusted showcase outputs
- exploratory stage-by-stage post-processing figures from an earlier decision
  phase

Why rejected or retired:

- some were exploratory rather than polished
- some flattened texture too much on `markers`, `pillar`, and `seat`
- some belonged to an earlier decision stage before the final regular pipeline
  was standardized

Important nuance:

- Tone equalization itself is not rejected anymore.
- What is rejected here is the older exploratory branch or older archived
  output set, not the final accepted regular pipeline now used in `results/final/`.

Report wording:

- "Earlier post-processing branches were explored and archived. After
  refinement, a controlled tone-equalization stage was adopted as part of the
  final showcase pipeline."

## Old Results That Are Not Exactly Rejected

Some results are old but not "bad." They are simply no longer central.

Examples:

- `results/archive/old_real_batch/*`
  Earlier RGB-to-grayscale batch set using images like `bath`, `bed`,
  `jackets`, `knob`, and `writing`.
- `results/flashlight/*`
  Early real-scene demonstration.
- `results/tunnel/*`
  Strong tuning history but not part of the narrowed final story.
- `results/xray/*`
  Exploratory extension rather than final identity.

How to describe these:

- "These experiments were useful during project development and remain valid as
  supporting or exploratory material, but they were not selected as the main
  report attachments."

## Suggested Attachment Set For A Clean Report

If you want a concise, strong attachment pack, use:

1. `results/final/selected_real_images_hf_bright_overview.png`
2. `results/final/uniform_reference_comparison_overview.png`
3. `results/final/seat_pillar_detail_comparison.png`
4. `results/final/page_conservative_hf_comparison.png`
5. `results/final/page_detail_comparison.png`
6. `results/metrics/blind_multicase_metrics.png`

If you want a larger pack, add:

1. `results/final/color_grayscale_standard_overview.png`
2. `results/final/cardboard_uniform_reference_comparison.png`
3. `results/final/markers_uniform_reference_comparison.png`
4. one synthetic sweep figure such as `results/synthetic/sweep_gamma_l.png`

## Suggested One-Paragraph Report Explanation

The active report attachments were selected from `results/final/` because they
represent the strongest accepted outputs of the current pipeline. Supporting
evidence from `results/metrics/`, `results/synthetic/`, and `results/real_batch/`
documents how the method was validated and how intermediate stages behave.
Older folders such as `results/archive/` contain earlier but historically useful
experiments, while `results/old/` stores rejected branches such as CLAHE-heavy
document variants and earlier exploratory post-processing outputs that were not
used in the final reporting direction.
