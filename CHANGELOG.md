# Changelog

Dated notes on what changed on this page and why. Newest first. Versions
v1 and v2 are described in the README.

## 2026-11-11

- New subsection "The pullet network, session by session"
  (`src/session_calls.py`, `results/session_calls.json`), read from the
  stored out-of-fold predictions. The control cage did learn the day, as
  finding 2 claimed: 7 of its 8 hours are scored like the other half of
  their day, every "after" hour ranks below every "before" hour, and the
  pooled 0.19 is the lowest of the 16 within-day relabellings (p = 1/16).
  The stressed cages' 0.47 is not that signature but a coin: 16 of the 30
  hours with their partner in training are scored like the partner, 6 of
  16 cage-days have "after" above "before", and the observed AUC sits in
  the middle of the 65 536 within-day relabellings (0.24 to 0.77).
- On a held-out week "after" scores above "before" in 12 of 16 cage-days
  (one tie; sign test p = 0.018 one-sided), by +0.016 in mean probability
  [−0.044, +0.070] — and the control cage moves the same way by the same
  amount (3 of 4, +0.030) with no stressor. Finding 2 amended.
- `results/cv/` published: the 111 per-fold prediction files (label,
  probability, session for every clip under every protocol, fold and
  seed). The README's "Reproducing" section had promised them since v2
  while `.gitignore` kept them out. `tests/test_session_calls.py` (eight
  tests) recomputes the four pooled pullet AUCs from them exactly as
  `train.summarise` does and pins the new subsection to `results/`.
- "Apple silicon (MPS)" in the run-time notes replaced by "a laptop GPU",
  as in the other repositories.

## 2026-10-19

- LICENSE now holds the standard MIT text and LICENSE-CONTENT the standard CC BY 4.0 text, so GitHub recognises the licence (it showed "Other" before). The terms are unchanged; the note on what each licence covers moved to `NOTICE.md`.

## 2026-10-01

- New subsection "The same trend with the recording day as the unit"
  (`src/growth_units_check.py`, `results/growth_units_check.json`). The 42
  sessions behind the age trend of f₀ were recorded on 11 days, and days
  carry 40 % of the variance of log f₀ (design effect 1.8, about 24
  independent sessions). Resampling whole days turns the published interval
  [−0.65, −0.01] into [−0.71, +0.17]; permuting ages between days gives
  p = 0.12; the 11 day medians give −0.55, p = 0.083.
- The trend is the first recorded week. Without days 14 and 15 the
  correlation over the remaining 35 sessions is +0.07 (p = 0.69). The
  curve with the published growth parameters predicts a 19 % fall from day
  21 to day 42 (1019 to 820 Hz); the weekly medians are 800, 949, 885 and
  884 Hz.
- Only one of the three cages was recorded on all 11 days; stressed cage 1
  stops at day 29 and the control cage starts at day 35.
- Correction: "the sign is established; the size is not" now reads "at the
  session level". With the day as the unit the sign is likely, not
  established. Finding 4 carries the same qualification.
- `tests/test_growth_units.py`: eight tests pin the subsection to
  `results/` and recompute its non-random parts.
- Four macOS `._*` files removed from the repository; `._*` ignored.
