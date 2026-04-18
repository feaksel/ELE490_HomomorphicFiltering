# Documentation System

This folder is the project memory for reporting, debugging, and explaining why
we made certain choices.

## Files

- `WORKLOG.md`
  Chronological project history. Use this for "what changed and when".
- `DECISIONS.md`
  Accepted, active, and rejected decisions. Use this for "why we chose it".
- `ISSUES_AND_RISKS.md`
  Bugs, open risks, known weaknesses, and resolved issues.
- `REPORTING_NOTES.md`
  Short report-facing summary of what is worth showing, what to say, and what
  to avoid over-claiming.
- `PROJECT_SUMMARY.md`
  Clean full-project summary of what we built, what we tested, what worked,
  and how the current pipeline should be described.
- `RESULTS_ATTACHMENT_GUIDE.md`
  Explains what each results folder means, which figures are active, old, or
  rejected, and which files should be attached in the report.

## Update Rule Going Forward

For every meaningful work session, update at least one of these:

1. `WORKLOG.md`
   Add what changed, what was tested, and what files/results were produced.
2. `DECISIONS.md`
   Add new accepted or rejected decisions when parameters, workflows, or
   reporting choices change.
3. `ISSUES_AND_RISKS.md`
   Add new bugs, limitations, or risks when we discover them.
4. `REPORTING_NOTES.md`
   Refresh only when the project story, strongest examples, or reporting
   recommendations change.

## Practical Policy

- Every new test image should be mentioned in `WORKLOG.md`.
- Every parameter change that affects the final story should be mentioned in
  both `WORKLOG.md` and `DECISIONS.md`.
- Every failed or rejected enhancement idea should be recorded as rejected
  rather than silently forgotten.
- Every reporting-facing recommendation should be backed by a result file.
- If a result is no longer part of the active showcase, keep the history in the
  docs even if the file is removed from `results/final/`.

## Current Standard

- Global real-scene standard:
  `Gaussian`, `gamma_L=0.06`, `gamma_H=1.00`, `D0=320`, plus gentle brightness
  lift and tone equalization for the active showcase.
- Page-specific recommendation:
  use the conservative page setting rather than the global standard when the
  goal is document readability under directional lighting, then apply the same
  tone-equalization stage.
