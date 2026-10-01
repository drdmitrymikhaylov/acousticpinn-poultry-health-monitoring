"""Is the age trend of the flock's fundamental replicated at the level it
is tested at?

growth_pinn.py reports Spearman(f0, age) = -0.35 over 42 sessions, with a
bootstrap interval that resamples sessions.  But a session is a cage-hour,
and the 42 of them sit on 11 recording days in 4 protocol weeks and 3
cages.  Sessions of one day share the day: the same age to the day, the
same weather, the same state of the house.  The page itself shows that the
day is audible (the control cage's CNN scores 0.19 on held-out sessions
because it has learned which day a clip is from).  If the day is the unit,
the trend has 11 observations, not 42.

This script re-reads results/growth_pinn.json (the 42 session points; no
audio needed) and reports

  - the trend on the 11 day medians and on the 4 week medians
  - a bootstrap that resamples whole days, next to the one that resamples
    sessions
  - a permutation test that shuffles ages between days, keeping each day's
    sessions together
  - the share of the variance of log f0 that lies between days
    (one-way intraclass correlation) and the design effect it implies
  - the trend inside each cage, inside each phase, and with each day and
    each week left out in turn

Writes results/growth_units_check.json.  Run from the repository root:

    python src/growth_units_check.py
"""

import json

import numpy as np
from scipy.stats import spearmanr

N_BOOT = 10000
N_PERM = 20000
SEED = 0


def rho(x, y):
    if len(set(x)) < 2 or len(set(y)) < 2:
        return float("nan")
    return float(spearmanr(x, y).statistic)


def icc_oneway(groups):
    """ICC(1) for unbalanced groups: between-group share of the variance."""
    k = len(groups)
    n = np.array([len(g) for g in groups], float)
    N = n.sum()
    grand = np.concatenate(groups).mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    n0 = (N - (n ** 2).sum() / N) / (k - 1)
    return float((msb - msw) / (msb + (n0 - 1) * msw)), float(n0)


def main():
    g = json.load(open("results/growth_pinn.json"))
    p = g["points"]
    age = np.array(p["age_days"], float)
    f0 = np.array(p["f0_median_hz"], float)
    lf = np.log(f0)
    phase = np.array(p["phase"])
    cond = np.array(p["cond"])
    n = len(age)
    days = np.unique(age)
    week = np.floor((age - age.min()) / 7).astype(int) + 1
    rng = np.random.default_rng(SEED)

    r_sessions = rho(age, f0)
    assert abs(r_sessions - g["spearman_f0_vs_age"]) < 1e-9

    # --- the day as the unit -------------------------------------------
    day_med = np.array([np.median(f0[age == d]) for d in days])
    r_days = spearmanr(days, day_med)
    weeks = np.unique(week)
    week_med = np.array([np.median(f0[week == w]) for w in weeks])
    week_age = np.array([age[week == w].mean() for w in weeks])

    # --- two bootstraps ------------------------------------------------
    b_sess, b_day = [], []
    idx_by_day = [np.flatnonzero(age == d) for d in days]
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        b_sess.append(rho(age[i], f0[i]))
        pick = rng.integers(0, len(days), len(days))
        j = np.concatenate([idx_by_day[q] for q in pick])
        b_day.append(rho(age[j], f0[j]))
    b_sess = np.array(b_sess)
    b_day = np.array(b_day)
    b_day = b_day[~np.isnan(b_day)]

    # --- permutation of ages between days --------------------------------
    hits = 0
    for _ in range(N_PERM):
        perm = rng.permutation(days)
        a = age.copy()
        for d, new in zip(days, perm):
            a[age == d] = new
        if abs(rho(a, f0)) >= abs(r_sessions) - 1e-12:
            hits += 1
    p_day_perm = (hits + 1) / (N_PERM + 1)

    # --- how much of the variance is the day -----------------------------
    icc_raw, n0 = icc_oneway([lf[i] for i in idx_by_day])
    slope, icpt = np.polyfit(age, lf, 1)
    resid = lf - (slope * age + icpt)
    icc_res, _ = icc_oneway([resid[i] for i in idx_by_day])
    deff = 1 + (n0 - 1) * max(icc_res, 0.0)

    # --- inside cages and phases; leave one day / week out ---------------
    by_cond = {str(c): {"n": int((cond == c).sum()),
                        "n_days": int(len(np.unique(age[cond == c]))),
                        "first_day": int(age[cond == c].min()),
                        "last_day": int(age[cond == c].max()),
                        "spearman": rho(age[cond == c], f0[cond == c]),
                        "p": float(spearmanr(age[cond == c],
                                             f0[cond == c]).pvalue)}
               for c in sorted(set(cond))}
    by_phase = {str(c): {"n": int((phase == c).sum()),
                         "spearman": rho(age[phase == c], f0[phase == c]),
                         "p": float(spearmanr(age[phase == c],
                                              f0[phase == c]).pvalue)}
                for c in sorted(set(phase))}
    lodo = {str(int(d)): rho(age[age != d], f0[age != d]) for d in days}
    lowo = {str(int(w)): {"spearman": rho(age[week != w], f0[week != w]),
                          "p": float(spearmanr(age[week != w],
                                               f0[week != w]).pvalue),
                          "n": int((week != w).sum())} for w in weeks}

    # --- what the published-parameter curve predicts after week one ------
    one = g["physics_one_free"]

    def curve(t):
        return float(one["f0_inf_hz"] * np.exp(
            one["alpha"] * np.exp(-one["k_per_day"] * (t - one["t_i_days"]))))

    late = age >= 21
    late_r = spearmanr(age[late], f0[late])

    out = {
        "published_curve_f0_hz": {"14": curve(14.0), "21": curve(21.0),
                                  "42": curve(42.0)},
        "published_curve_change_21_to_42": curve(42.0) / curve(21.0) - 1,
        "from_day_21": {"n": int(late.sum()),
                        "spearman": float(late_r.statistic),
                        "p": float(late_r.pvalue),
                        "f0_min_week_median_hz": float(week_med[1:].min()),
                        "f0_max_week_median_hz": float(week_med[1:].max())},
        "source": "results/growth_pinn.json (points)",
        "n_sessions": int(n), "n_days": int(len(days)),
        "n_weeks": int(len(weeks)), "n_cages": int(len(set(cond))),
        "sessions_per_day": {str(int(d)): int((age == d).sum()) for d in days},
        "days_per_week": {str(int(w)): sorted(int(d) for d in
                                               np.unique(age[week == w]))
                          for w in weeks},
        "spearman_sessions": r_sessions,
        "p_sessions": float(spearmanr(age, f0).pvalue),
        "ci95_session_bootstrap": [float(np.percentile(b_sess, 2.5)),
                                   float(np.percentile(b_sess, 97.5))],
        "ci95_day_bootstrap": [float(np.percentile(b_day, 2.5)),
                               float(np.percentile(b_day, 97.5))],
        "day_bootstrap_share_negative": float((b_day < 0).mean()),
        "session_bootstrap_share_negative": float((b_sess < 0).mean()),
        "p_day_permutation": p_day_perm,
        "day_medians": {"age_days": days.tolist(),
                        "f0_median_hz": day_med.tolist(),
                        "spearman": float(r_days.statistic),
                        "p": float(r_days.pvalue)},
        "week_medians": {"week": weeks.tolist(),
                         "mean_age_days": week_age.tolist(),
                         "f0_median_hz": week_med.tolist(),
                         "n_sessions": [int((week == w).sum()) for w in weeks],
                         "n_days": [int(len(np.unique(age[week == w])))
                                    for w in weeks],
                         "cages": [sorted(str(c) for c in set(cond[week == w]))
                                   for w in weeks]},
        "icc_day_log_f0": icc_raw,
        "icc_day_log_f0_after_linear_trend": icc_res,
        "mean_sessions_per_day_n0": n0,
        "design_effect": deff,
        "effective_sessions": n / deff,
        "by_cage": by_cond,
        "by_phase": by_phase,
        "leave_one_day_out": lodo,
        "leave_one_day_out_range": [min(lodo.values()), max(lodo.values())],
        "leave_one_week_out": lowo,
        "n_boot": N_BOOT, "n_perm": N_PERM, "seed": SEED,
    }
    with open("results/growth_units_check.json", "w") as fh:
        json.dump(out, fh, indent=1)
    for k, v in out.items():
        if k not in ("day_medians", "week_medians"):
            print(k, v)
    print("day medians", [(int(d), round(m)) for d, m in zip(days, day_med)],
          round(float(r_days.statistic), 3), round(float(r_days.pvalue), 3))
    print("week medians", [(int(w), round(a, 1), round(m)) for w, a, m in
                           zip(weeks, week_age, week_med)])


if __name__ == "__main__":
    main()
