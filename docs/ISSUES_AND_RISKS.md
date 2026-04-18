# Issues And Risks

## Open Issues

### I-001 HSI Color Pipeline Not Yet Validated

- Status: open
- Scope:
  `scripts/05_color_hsi.py`
- Problem:
  The HSI extension exists structurally, but it is not yet part of the active
  validated showcase.
- Risk:
  If presented too strongly, it would overstate project maturity.

### I-002 Some Documentation Still Reflects Older Workflow

- Status: closed
- Problem:
  `DESIGN.md` still emphasizes older image sets, HEQ-heavy comparisons, and an
  earlier project structure.
- Risk:
  Reporting may become inconsistent if we treat `README.md`, `DESIGN.md`, and
  the new docs as equally current without cleanup.
- Resolution:
  Updated the project docs so the active pipeline, script list, and reporting
  story consistently reflect the accepted homomorphic plus tone-equalization
  workflow.

### I-003 Organizer Handles Page-Specific Files Carefully

- Status: open
- Problem:
  Files beginning with `page_` can be swept into `results/real_batch/` unless
  explicitly exempted.
- Risk:
  New page-specific comparison figures may not automatically land in
  `results/final/`.

### I-004 DESIGN.md Still Mentions Retired Experiment Naming

- Status: closed
- Problem:
  `DESIGN.md` still refers to older active names such as
  `12_selected_hf_vs_heq_comparison.py` and broader experiment branches that
  are no longer part of the active mid-project showcase.
- Risk:
  It can make the project look less settled than it really is during reporting.
- Resolution:
  Refreshed `DESIGN.md` to describe the current active scripts and archived
  older branches more clearly.

## Closed Issues

### C-001 Results Folder Not Tracked By Git

- Status: closed
- Problem:
  `.gitignore` ignored `results/*`, so important generated outputs were not
  trackable.
- Resolution:
  Removed the ignore rule and staged the results tree.

### C-002 Final Folder Contained Old Non-Showcase Outputs

- Status: closed
- Problem:
  `results/final/` still carried flashlight, tunnel, rice, xray, and other old
  outputs after the showcase shifted to the new photo set.
- Resolution:
  Simplified `13_organize_results.py` so the active final folder only carries
  the current showcase outputs.

### C-003 Page HEQ Experiment Left Residual Traces

- Status: closed
- Problem:
  The rejected page HEQ experiment briefly left scripts and generated files in
  the repo.
- Resolution:
  Removed the page-specific HEQ script and deleted its generated result traces.

### C-004 Rejected CLAHE / Tone Branches Were Mixed Into Active Final Results

- Status: closed
- Problem:
  `results/final/` and the active script list still exposed rejected CLAHE
  branches and exploratory post-processing outputs even after the reporting
  direction became narrower and more structured.
- Resolution:
  Moved the rejected CLAHE scripts into `scripts/old/`, archived non-active
  experimental outputs under `results/old/`, and simplified the active
  organizer and report-facing file lists around the accepted regular pipeline.

## Scientific / Reporting Risks

### R-001 Overclaiming Real-Image Success

- Risk:
  Real-image success is mostly qualitative. We do not have true ground truth
  for most scenes.
- Mitigation:
  Phrase conclusions as visual improvement under uneven illumination, not as
  guaranteed correction.

### R-002 Overprocessing Fine Texture

- Risk:
  Strong homomorphic settings, HEQ, or CLAHE can overemphasize paper texture,
  noise, or fabric weave.
- Mitigation:
  Use scene-specific caution, especially for document-like images such as
  `page.jpeg`.

### R-003 Mixing Global and Per-Scene Settings Without Explanation

- Risk:
  If we use the global real-scene standard for some images and a conservative
  page-specific setting for the page, we need to explain why.
- Mitigation:
  Explicitly document that the page is a special readability-focused document
  case rather than a general real-scene texture case.

### R-004 Too Many Side Experiments Can Blur the Story

- Risk:
  Showing every rejected enhancement may confuse the final presentation.
- Mitigation:
  Keep rejected experiments documented, but present only the strongest accepted
  results in the final report.
