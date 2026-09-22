"""A growth curve read by microphone.

Across birds the fundamental frequency of a call falls with body mass as a
power law, f0 ~ M^(-alpha): geometric similarity (every length ~ M^(1/3))
gives alpha = 1/3, and Fletcher's optimal-communication rule (JASA 2004)
gives alpha = 0.4.  Both are cross-species laws; applying either within one
growing flock is an assumption this file carries and reports.  A growing
flock's mass follows a Gompertz law,

    M(t) = M_inf * exp(-exp(-k (t - t_i))).

Put together, the flock's fundamental over age is

    f0(t) = f0_inf * exp( alpha * exp(-k (t - t_i)) ),

where f0_inf = a * M_inf^(-1/3) folds the allometric constant and the mature
mass into one number -- the only one the microphone can identify on its own.
The rate k and the inflection age t_i are the growth parameters of the
flock, and those a hatchery publishes.  So the test is: take k and t_i from
a published Gompertz fit for white-egg layer pullets, leave ONE free number,
and see whether the flock's fundamental over nine weeks follows the curve.
Then free all three and see whether the data ask for a different k and t_i.
The comparison model is a straight line in log f0 -- the same number of free
parameters as the physics with one free constant, and no biology.

Reference growth curve: Hy-Line white-egg pullets, Gompertz k = 0.02 /day,
inflection at 53 days (Oliveira et al., Semina: Ciências Agrárias 39(3), 2018,
doi:10.5433/1679-0359.2018v39n3p1327).  The recorded birds are Super Nick,
another white-egg layer; the curves of the two are close but not identical,
and the gap is part of what the fit is allowed to absorb.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import spearmanr, wilcoxon

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
K_REF, TI_REF = 0.02, 53.0          # per day, days
ALPHA = 1.0 / 3.0                   # isometry; ALPHA_FLETCHER is run as a sensitivity
ALPHA_FLETCHER = 0.4


def f0_curve(t, f0_inf, k, t_i, alpha=ALPHA):
    return f0_inf * np.exp(alpha * np.exp(-k * (t - t_i)))


def fit_curve(t, f0, free_all, alpha=ALPHA):
    lf = np.log(f0)

    def resid(q):
        if free_all:
            f0_inf, k, t_i = np.exp(q[0]), q[1], q[2]
        else:
            f0_inf, k, t_i = np.exp(q[0]), K_REF, TI_REF
        return np.log(f0_curve(t, f0_inf, k, t_i, alpha)) - lf

    q0 = [np.log(np.median(f0)) - alpha * np.exp(-K_REF * (np.median(t) - TI_REF))]
    if free_all:
        q0 += [K_REF, TI_REF]
    r = least_squares(resid, q0)
    q = r.x
    out = {"f0_inf_hz": float(np.exp(q[0])), "k_per_day": float(q[1] if free_all else K_REF),
           "t_i_days": float(q[2] if free_all else TI_REF), "rmse_log": float(np.sqrt(np.mean(r.fun ** 2))),
           "n_free": 3 if free_all else 1, "alpha": alpha}
    return out


def main():
    d = json.loads((RESULTS / "syrinx_pinn.json").read_text())
    rows = [r for ds in ("pullet", "pullet_control") for r in d["datasets"].get(ds, {}).get("recordings", [])]
    if not rows:
        print("no pullet recording table yet"); return
    # only sessions with enough voiced clips enter: a fundamental read off an unvoiced
    # decomposition is not a fundamental
    all_n = len(rows)
    voiced_rows = [r for r in rows if r.get("f0_median_voiced_hz") is not None]
    relaxed = len(voiced_rows) < 20
    if relaxed:                       # too few voiced sessions: fall back to all clips and say so
        key_f0 = "f0_median_hz"
    else:
        rows, key_f0 = voiced_rows, "f0_median_voiced_hz"
    t = np.array([r["age_days"] for r in rows], float)
    f0 = np.array([r[key_f0] for r in rows], float)
    phase = np.array([r["phase"] for r in rows])
    cond = np.array([r["cond"] for r in rows])
    rho, pv = spearmanr(t, f0)
    out = {"n_sessions": int(len(t)), "n_sessions_all": all_n, "n_sessions_voiced": len(voiced_rows),
           "voiced_rule": "median f0 over clips with voiced share > 0.2, at least 5 such clips",
           "voiced_rule_relaxed": relaxed,
           "spearman_f0_vs_age": float(rho), "p": float(pv),
           "reference": {"k_per_day": K_REF, "t_i_days": TI_REF,
                         "source": "Oliveira et al. 2018, Semina 39(3):1327, Hy-Line white pullets"},
           "physics_one_free": fit_curve(t, f0, False),
           "physics_three_free": fit_curve(t, f0, True)}
    A = np.stack([np.ones_like(t), t], 1)
    coef, *_ = np.linalg.lstsq(A, np.log(f0), rcond=None)
    res = np.log(f0) - A @ coef
    out["line_in_log_f0"] = {"slope_per_day": float(coef[1]), "rmse_log": float(np.sqrt(np.mean(res ** 2))), "n_free": 2}
    # model selection and uncertainty: AIC/BIC on the Gaussian log-residual, bootstrap over sessions
    n = len(t)
    for key in ("physics_one_free", "physics_three_free", "line_in_log_f0"):
        q = out[key]; rss = n * q["rmse_log"] ** 2; kf = q["n_free"] + 1
        q["aic"] = float(n * np.log(rss / n) + 2 * kf); q["bic"] = float(n * np.log(rss / n) + kf * np.log(n))
    rng = np.random.default_rng(0)
    ks, tis, rhos = [], [], []
    for _ in range(500):
        b = rng.choice(n, n, replace=True)
        if len(set(t[b])) < 4:
            continue
        q = fit_curve(t[b], f0[b], True); ks.append(q["k_per_day"]); tis.append(q["t_i_days"])
        rhos.append(spearmanr(t[b], f0[b])[0])
    out["bootstrap"] = {"k_ci95": np.percentile(ks, [2.5, 97.5]).tolist(), "t_i_ci95": np.percentile(tis, [2.5, 97.5]).tolist(),
                        "spearman_ci95": np.percentile(rhos, [2.5, 97.5]).tolist()}
    # the exponent is an assumption too: Fletcher's 0.4 instead of isometric 1/3
    out["alpha_fletcher"] = {"one_free": fit_curve(t, f0, False, ALPHA_FLETCHER), "three_free": fit_curve(t, f0, True, ALPHA_FLETCHER)}
    # the age offset is an assumption; how much do the conclusions move with it?
    out["offset_sensitivity"] = []
    for off in (0, 7, 13, 21):
        tt_ = t - 13 + off
        q1 = fit_curve(tt_, f0, False); q3 = fit_curve(tt_, f0, True)
        out["offset_sensitivity"].append({"offset_days": off, "rmse_one_free": q1["rmse_log"], "rmse_three_free": q3["rmse_log"],
                                          "k_three_free": q3["k_per_day"], "t_i_three_free": q3["t_i_days"]})
    # per-age summary and the stress contrast at matched age
    ages = sorted(set(t))
    out["by_age"] = [{"age_days": a, "n": int((t == a).sum()), "f0_median_hz": float(np.median(f0[t == a]))} for a in ages]
    contrast = []
    for a in ages:
        for c in sorted(set(cond)):
            pre = f0[(t == a) & (cond == c) & (phase == "pre")]; post = f0[(t == a) & (cond == c) & (phase == "post")]
            if len(pre) and len(post):
                contrast.append({"age_days": a, "cond": c, "n_pre": int(len(pre)), "n_post": int(len(post)),
                                 "f0_pre_hz": float(np.median(pre)), "f0_post_hz": float(np.median(post)),
                                 "ratio_post_over_pre": float(np.median(post) / np.median(pre))})
    out["pre_post_contrast"] = contrast
    # paired test on the sessions of the stressed cages: same cage, same day, before vs after
    pairs = [c["ratio_post_over_pre"] for c in contrast if not c["cond"].startswith("c")]
    if len(pairs) >= 6:
        w = wilcoxon(np.log(pairs))
        out["wilcoxon_stressed_pairs"] = {"n_pairs": len(pairs), "statistic": float(w.statistic), "p": float(w.pvalue),
                                          "median_ratio": float(np.median(pairs))}
    cp = [c["ratio_post_over_pre"] for c in contrast if c["cond"].startswith("c")]
    out["control_pairs"] = {"n_pairs": len(cp), "median_ratio": float(np.median(cp)) if cp else None}
    tt = np.linspace(t.min(), t.max(), 100)
    out["curve_t"] = tt.tolist()
    for key in ("physics_one_free", "physics_three_free"):
        q = out[key]
        out[key]["curve_f0"] = f0_curve(tt, q["f0_inf_hz"], q["k_per_day"], q["t_i_days"]).tolist()
    out["line_in_log_f0"]["curve_f0"] = np.exp(coef[0] + coef[1] * tt).tolist()
    out["points"] = {"age_days": t.tolist(), "f0_median_hz": f0.tolist(), "phase": phase.tolist(), "cond": cond.tolist()}
    (RESULTS / "growth_pinn.json").write_text(json.dumps(out, indent=1))
    print(f"n={len(t)} of {all_n} sessions voiced enough; Spearman(f0, age) = {rho:.2f} (p={pv:.2g}) "
          f"CI {np.round(out['bootstrap']['spearman_ci95'], 2).tolist()}")
    print("  offset sensitivity:", out["offset_sensitivity"])
    print("  wilcoxon:", out.get("wilcoxon_stressed_pairs"), " control:", out["control_pairs"])
    for key in ("physics_one_free", "physics_three_free", "line_in_log_f0"):
        print(f"  {key:20s} rmse(log f0) {out[key]['rmse_log']:.3f}  " +
              " ".join(f"{k}={v:.4g}" for k, v in out[key].items() if isinstance(v, float) and k not in ("rmse_log",)))
    for c in contrast:
        print(f"  age {c['age_days']:.0f} d {c['cond']:10s} f0 pre {c['f0_pre_hz']:.0f} -> post {c['f0_post_hz']:.0f} Hz")


if __name__ == "__main__":
    main()
