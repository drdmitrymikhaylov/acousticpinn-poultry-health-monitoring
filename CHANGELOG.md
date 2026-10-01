# Changelog

Dated notes on what changed on this page and why. Newest first. Versions
v1 and v2 are described in the README.

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
