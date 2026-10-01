"""Pins "The same trend with the recording day as the unit" to the results.

Reads README.md, results/growth_pinn.json and results/growth_units_check.json
(written by src/growth_units_check.py).  No audio is needed.  The parts of
the check that involve no random numbers are recomputed here from the 42
session points; the bootstrap and permutation figures are compared with the
stored file.
"""

import json
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
# minus signs unified and line breaks collapsed, so a sentence can be matched
# wherever the page wraps it
README = " ".join((ROOT / "README.md").read_text(encoding="utf-8")
                  .replace("−", "-").split())
G = json.loads((ROOT / "results" / "growth_pinn.json").read_text())
C = json.loads((ROOT / "results" / "growth_units_check.json").read_text())

AGE = np.array(G["points"]["age_days"], float)
F0 = np.array(G["points"]["f0_median_hz"], float)
COND = np.array(G["points"]["cond"])
CAGE = {"trt1": "stressed 1", "trt2": "stressed 2", "ctrl": "control"}


def r(x, nd=2, sign=False):
    q = Decimal(1).scaleb(-nd)
    s = str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))
    if sign and float(x) > 0:
        s = "+" + s
    return s


def week_row(i):
    w = C["week_medians"]
    days = C["days_per_week"][str(w["week"][i])]
    span = f"{days[0]}-{days[-1]}" if len(days) > 1 else f"{days[0]}"
    cages = ", ".join(CAGE[c] for c in ("trt1", "trt2", "ctrl")
                      if c in w["cages"][i])
    out = C["leave_one_week_out"][str(w["week"][i])]["spearman"]
    return (f"| {span} | {w['n_days'][i]} | {w['n_sessions'][i]} | {cages} | "
            f"{r(w['f0_median_hz'][i], 0)} | {r(out, 2, sign=True)} |")


def test_the_points_are_42_sessions_on_11_days():
    assert len(AGE) == 42 == C["n_sessions"]
    assert len(np.unique(AGE)) == 11 == C["n_days"]
    assert C["n_weeks"] == 5 and C["n_cages"] == 3
    assert "recorded on 11 days" in README


def test_session_level_numbers_are_the_published_ones():
    rho = spearmanr(AGE, F0)
    assert abs(rho.statistic - G["spearman_f0_vs_age"]) < 1e-9
    assert abs(rho.statistic - C["spearman_sessions"]) < 1e-9
    lo, hi = G["bootstrap"]["spearman_ci95"]
    row = (f"| session, sessions resampled (as published) | 42 | "
           f"{r(rho.statistic)} | [{r(lo)}, {r(hi)}] | {r(rho.pvalue, 3)} |")
    assert row in README, row


def test_day_level_interval_includes_zero():
    lo, hi = C["ci95_day_bootstrap"]
    assert lo < 0 < hi
    row = (f"| session, whole days resampled | 11 days | "
           f"{r(C['spearman_sessions'])} | [{r(lo)}, {r(hi, 2, sign=True)}] | "
           f"{r(C['p_day_permutation'])} |")
    assert row in README, row
    assert C["p_day_permutation"] > 0.05
    assert r(100 * C["day_bootstrap_share_negative"], 0) == "90"
    assert "90 % of the day-level resamples are negative" in README
    slo, shi = C["ci95_session_bootstrap"]
    assert f"[{r(slo)}, {r(shi)}]" in README


def test_day_medians_recomputed():
    days = np.unique(AGE)
    med = [np.median(F0[AGE == d]) for d in days]
    rho = spearmanr(days, med)
    assert abs(rho.statistic - C["day_medians"]["spearman"]) < 1e-9
    row = f"| day median | 11 | {r(rho.statistic)} | — | {r(rho.pvalue, 3)} |"
    assert row in README, row


def test_week_table_rows_and_the_first_week():
    for i in range(5):
        assert week_row(i) in README, week_row(i)
    late = AGE >= 21
    rho = spearmanr(AGE[late], F0[late])
    assert late.sum() == 35
    assert abs(rho.statistic - C["leave_one_week_out"]["1"]["spearman"]) < 1e-9
    assert rho.statistic > 0
    assert f"correlation is {r(rho.statistic, 2, sign=True)} (p = {r(rho.pvalue)})" in README
    lodo = C["leave_one_day_out"]
    assert f"day 14 alone gives {r(lodo['14'])}" in README
    assert f"day 15 alone {r(lodo['15'])}" in README


def test_published_curve_against_the_late_weeks():
    c = C["published_curve_f0_hz"]
    assert abs(c["14"] - G["physics_one_free"]["curve_f0"][0]) < 1e-6
    assert f"{r(c['21'], 0)} Hz at day 21 to {r(c['42'], 0)} Hz at day 42" in README
    assert r(-100 * C["published_curve_change_21_to_42"], 0) == "19"
    w = C["week_medians"]["f0_median_hz"]
    assert [r(x, 0) for x in w[1:]] == ["800", "949", "885", "884"]
    assert "800, 949, 885 and 884 Hz" in README


def test_variance_between_days():
    assert r(100 * C["icc_day_log_f0"], 0) == "40"
    assert r(100 * C["icc_day_log_f0_after_linear_trend"], 0) == "27"
    assert r(C["mean_sessions_per_day_n0"], 1) == "3.8"
    assert r(C["design_effect"], 1) == "1.8"
    assert r(C["effective_sessions"], 0) == "24"
    assert "carry 40 % of the variance" in README
    assert "about 24 independent ones" in README


def test_cages_do_not_cover_the_same_ages():
    b = C["by_cage"]
    assert (b["trt2"]["n"], b["trt2"]["n_days"]) == (21, 11)
    assert (b["trt1"]["n"], b["trt1"]["last_day"]) == (13, 29)
    assert (b["ctrl"]["n"], b["ctrl"]["first_day"]) == (8, 35)
    for k, n in (("trt2", 21), ("trt1", 13), ("ctrl", 8)):
        m = COND == k
        rho = spearmanr(AGE[m], F0[m])
        assert abs(rho.statistic - b[k]["spearman"]) < 1e-9
        assert f"{n} sessions, {r(rho.statistic)}, p = {r(rho.pvalue, 3 if rho.pvalue < 0.01 else 2)}" in README


if __name__ == "__main__":
    for i in range(5):
        print(week_row(i))
