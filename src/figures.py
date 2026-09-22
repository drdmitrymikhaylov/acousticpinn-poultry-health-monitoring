"""Figures for the README, from results/*.json only."""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402
import numpy as np                                   # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
R = ROOT / "results"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"figure.dpi": 130, "savefig.dpi": 130, "font.size": 9, "axes.grid": True,
                     "grid.alpha": 0.25, "axes.spines.top": False, "axes.spines.right": False,
                     "legend.frameon": False})
COL = {"healthy": "#2c6fb0", "unhealthy": "#b0442c", "noise": "#6a6a6a", "pre": "#2c6fb0", "post": "#b0442c",
       "control": "#6a6a6a", "treatment": "#b0442c"}
NAMES = {"broiler": "broiler chicks (Mendeley zp4nf2dxbh)", "pullet": "layer pullets, stressed cages (Zenodo 10433023)",
         "pullet_control": "layer pullets, control cage"}
READERS = [("cnn", "log-mel CNN"), ("band_energies", "8 band energies"), ("loudness_only", "loudness only"),
           ("f0_distribution", "f0 histogram (80 numbers)"), ("physics_7_named", "physics, 7 named numbers"),
           ("free_net_no_physics", "free net, no physics"), ("free_net_physics_loss", "free net, physics in the loss"),
           ("transients_only", "burst count only")]


def fig1_spectra():
    d = json.loads((R / "spectra.json").read_text())
    n = len(d)
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 3.8), squeeze=False)
    for ax, (ds, s) in zip(axes[0], d.items()):
        f = np.array(s["f"])
        for nm, c in s["classes"].items():
            m, lo, hi = (np.array(c[k]) for k in ("mean_spectrum", "p10", "p90"))
            ax.fill_between(f, lo, hi, color=COL.get(nm, "k"), alpha=0.12, lw=0)
            ax.loglog(f, m, color=COL.get(nm, "k"), lw=1.5, label=f"{nm} ({c['n_recordings']} recordings)")
        ax.set_xlabel("frequency, Hz"); ax.set_ylabel("share of clip power per bin")
        ax.set_title(NAMES.get(ds, ds) + "\nmean over recordings, band = 10th-90th percentile of recordings")
        ax.legend(fontsize=7, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "01_class_spectra.png"); plt.close(fig)


def fig2_protocols():
    s = json.loads((R / "syrinx_pinn.json").read_text())
    cnn = s.get("cnn_pooled_auc", {})
    dss = [ds for ds in ("broiler", "pullet", "pullet_control") if ds in s["datasets"]]
    fig, axes = plt.subplots(1, len(dss), figsize=(5.0 * len(dss), 3.9), squeeze=False)
    for ax, ds in zip(axes[0], dss):
        pr = s["datasets"][ds]["protocols"]
        prots = list(pr)
        readers = [(k, lab) for k, lab in READERS if k == "cnn" or k in pr[prots[0]]]
        w = 0.8 / len(readers)
        cci = s.get("cnn_ci95", {})
        for j, (k, lab) in enumerate(readers):
            vals = [cnn.get(ds, {}).get(p, np.nan) if k == "cnn" else pr[p][k]["pooled_auc"] for p in prots]
            cis = [(cci.get(ds, {}).get(p) or [np.nan, np.nan]) if k == "cnn" else pr[p][k].get("ci95", [np.nan, np.nan]) for p in prots]
            x = np.arange(len(prots)) + (j - len(readers) / 2 + 0.5) * w
            err = np.array([[v - c[0], c[1] - v] for v, c in zip(vals, cis)]).T
            ax.bar(x, vals, w, label=lab, color=plt.cm.viridis(j / max(len(readers) - 1, 1)),
                   yerr=np.where(np.isfinite(err), err, 0), error_kw={"lw": 0.6, "capsize": 1.5})
            for xi, v in zip(x, vals):
                if np.isfinite(v):
                    ax.text(xi, 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5, rotation=90, color="w")
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(np.arange(len(prots))); ax.set_xticklabels([f"leave-{p}-out" if p != "random" else "random over clips" for p in prots], fontsize=7.5)
        ax.set_ylim(0, 1.12); ax.set_ylabel("pooled AUC, 95 % CI over sessions"); ax.set_title(NAMES.get(ds, ds), fontsize=8.5)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, fontsize=7, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.09, 1, 1)); fig.savefig(FIG / "02_protocols.png"); plt.close(fig)


def fig3_source_filter():
    s = json.loads((R / "syrinx_pinn.json").read_text())
    f = np.array(s["shared"]["f"]); f0 = np.array(s["f0_grid"])
    fig, ax = plt.subplots(1, 3, figsize=(14.5, 3.9))
    a = ax[0]
    a.semilogy(f, s["shared"]["comb_basis_example"], color="#2c6fb0", lw=1.2, label=f"one voice at f0 = {f0[40]:.0f} Hz (harmonic comb)")
    a.semilogy(f, s["shared"]["tube_filter_example"], color="#b0442c", lw=1.2, label="tube filter |H|² of one clip")
    a.set_xlabel("frequency, Hz"); a.set_ylabel("relative power"); a.set_xlim(200, 7000)
    a.set_title(f"the parts: comb (line width {100 * s['shared']['eps']:.1f} % of f), tube (Q = {s['shared']['Q']:.1f})")
    a.legend(fontsize=7)
    b = ax[1]
    prof = np.array(s["shared"]["harmonic_profile"])
    for h in range(prof.shape[0]):
        b.semilogx(f0, prof[h] / prof.max(), lw=1.2, label=f"harmonic {h + 1}")
    b.set_xlabel("fundamental f0, Hz"); b.set_ylabel("relative amplitude"); b.set_title("the learned harmonic profile a(h, f0)")
    b.legend(fontsize=7)
    c = ax[2]
    ds = "broiler"
    ex = s["datasets"][ds]["examples"]
    labels = {"0": "healthy", "1": "unhealthy"}
    for cls, e in ex.items():
        col = COL[labels[cls]]
        c.semilogy(f, e["observed"], color=col, lw=0.7, alpha=0.5)
        c.semilogy(f, e["model"], color=col, lw=1.4, label=f"{labels[cls]}: f0 {e['features'][0]:.0f} Hz, tract {e['features'][3]:.1f} cm, voiced {e['features'][2]:.2f}")
    c.set_xlabel("frequency, Hz"); c.set_ylabel("power (normalised)"); c.set_xlim(200, 7000)
    c.set_title("the most voiced clip of each class: spectrum (thin) and model (thick)")
    c.legend(fontsize=6.5)
    fig.tight_layout(); fig.savefig(FIG / "03_source_filter.png"); plt.close(fig)


def fig4_f0_survey():
    s = json.loads((R / "syrinx_pinn.json").read_text())
    f0 = np.array(s["f0_grid"])
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 3.9))
    a = ax[0]
    for t, v in s["survey"].items():
        if t == "calls_by_type":
            continue
        lab = {"broiler": "broiler chicks, 0-65 d", "pullet": "layer pullets, stressed cages", "pullet_control": "layer pullets, control cage",
               "calls": "single calls of backyard hens"}.get(t, t)
        a.semilogx(f0, v["p_f0"], lw=1.5, label=f"{lab} (median {v['f0_median_hz']:.0f} Hz, n={v['n']})")
    a.set_xlabel("fundamental f0, Hz"); a.set_ylabel("mean p(f0) over clips"); a.set_title("where the model puts the voices, per source")
    a.legend(fontsize=7)
    if "survey_wide_grid" in s:
        wg = s["survey_wide_grid"]; f0w = np.array(wg["f0_grid"])
        for t in ("broiler", "calls", "pullet"):
            if t in wg:
                a.semilogx(f0w, wg[t]["p_f0"], lw=0.8, ls="--", color="k", alpha=0.5)
        a.plot([], [], ls="--", color="k", alpha=0.5, label="same, on a 150 Hz - 6 kHz grid (grid-sensitivity control)")
        a.legend(fontsize=6.5)
    c = ax[2]
    if "identifiability" in s:
        idf = s["identifiability"]
        for prof in idf["loss_profiles"]:
            c.plot(idf["L_cm_grid"], prof, lw=1.2)
        c.set_xlabel("tract length L, cm"); c.set_ylabel("log-spectral loss of the clip")
        c.set_title(f"is the tract length identified? loss over L for three clips\n"
                    f"{100 * idf['fraction_at_bounds']:.0f} % of clips sit at a bound of the allowed range")
    b = ax[1]
    d = s["datasets"]["broiler"]
    for cls, nm in (("0", "healthy"), ("1", "unhealthy")):
        if cls in d.get("index_trend", {}):
            tr = d["index_trend"][cls]
            b.semilogy(tr["index"], tr["f0_median_hz"], "o", ms=3, color=COL[nm], alpha=0.7,
                       label=f"{nm}: Spearman(f0, file number) = {tr['spearman_f0_vs_index']:.2f}")
    b.set_xlabel("file number in the dataset folder"); b.set_ylabel("recording's median f0, Hz")
    b.set_title("broiler chicks: does the file number carry the age?\n(the dataset has no day labels)")
    b.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(FIG / "04_f0_survey.png"); plt.close(fig)


def fig5_growth():
    p = R / "growth_pinn.json"
    if not p.exists():
        return
    g = json.loads(p.read_text())
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.9))
    a = ax[0]
    pts = g["points"]
    t = np.array(pts["age_days"]); f0 = np.array(pts["f0_median_hz"]); ph = np.array(pts["phase"]); cd = np.array(pts["cond"])
    for phase, mk in (("pre", "o"), ("post", "^")):
        for cond in sorted(set(cd)):
            k = (ph == phase) & (cd == cond)
            a.plot(t[k] + (0.0 if phase == "pre" else 0.6), f0[k], mk, ms=4, alpha=0.75,
                   color=COL["control"] if cond.startswith("c") else COL[phase], label=f"{cond}, {phase}-stressor")
    tt = np.array(g["curve_t"])
    a.plot(tt, g["physics_one_free"]["curve_f0"], color="#1baf7a", lw=2,
           label=f"Gompertz+allometry, published k, t_i; one free number (rmse {g['physics_one_free']['rmse_log']:.3f})")
    a.plot(tt, g["physics_three_free"]["curve_f0"], color="#1baf7a", lw=1.2, ls="--",
           label=f"same, k and t_i free: k={g['physics_three_free']['k_per_day']:.3f}/d, t_i={g['physics_three_free']['t_i_days']:.0f} d (rmse {g['physics_three_free']['rmse_log']:.3f})")
    a.plot(tt, g["line_in_log_f0"]["curve_f0"], color="k", lw=1, ls=":", label=f"straight line in log f0 (rmse {g['line_in_log_f0']['rmse_log']:.3f})")
    a.set_xlabel("age, days"); a.set_ylabel("recording's median f0, Hz")
    ci = g.get("bootstrap", {}).get("spearman_ci95", [np.nan, np.nan])
    a.set_title(f"the flock's fundamental over age, {g['n_sessions']} sessions"
                f"{' (voiced clips only)' if not g.get('voiced_rule_relaxed') else ''}\n"
                f"Spearman(f0, age) = {g['spearman_f0_vs_age']:.2f}, 95 % CI [{ci[0]:.2f}, {ci[1]:.2f}]")
    a.legend(fontsize=5.8, loc="upper right")
    b = ax[1]
    c = g["pre_post_contrast"]
    if c:
        x = np.arange(len(c))
        b.bar(x, [r["ratio_post_over_pre"] for r in c], color=[COL["control"] if r["cond"].startswith("c") else COL["post"] for r in c])
        b.axhline(1, color="k", lw=0.8)
        b.set_xticks(x); b.set_xticklabels([f"{r['cond']}\n{r['age_days']:.0f} d" for r in c], fontsize=6.5)
        wt = g.get("wilcoxon_stressed_pairs", {})
        b.set_ylabel("median f0 after / before the stressor")
        b.set_title(f"the stress contrast at matched age (grey: control cage)\n"
                    f"stressed pairs: Wilcoxon p = {wt.get('p', float('nan')):.2f}, n = {wt.get('n_pairs', 0)}")
    fig.tight_layout(); fig.savefig(FIG / "05_growth.png"); plt.close(fig)


if __name__ == "__main__":
    fig1_spectra(); fig2_protocols(); fig3_source_filter(); fig4_f0_survey(); fig5_growth()
    print("ok")
