"""A spectral mixture in which a flock is many source-filter voices at once.

Terminology, first: this is a physics-based spectral mixture model fitted by
gradient descent, with a small network for one shape function.  It is not a
PDE-residual PINN.  The control that IS "physics as a loss term" -- a free
network whose spectrum is penalised towards the mixture model's fit -- lives
in FreeSpectrum/fit_free below and is scored next to everything else.

A chicken is a source-filter system.  The syrinx sets a fundamental f0 and
its harmonics; the trachea and open beak are a tube, closed at the syrinx
and open at the beak, whose resonances sit at odd quarter-wavelengths,

    F_m = (2m - 1) c / (4 L),      m = 1, 2, 3,

with L the effective tract length.  A clip of flock sound is written as a
population of such voices over a background, and nothing else:

    S(f) = g * [ ( sum_k p(f0_k) sum_h a(h, f0_k) L(f; h f0_k, h eps f0_k) ) * |H_L(f)|^2  +  B(f) ]

p(f0) is the distribution of fundamentals in the clip (an 80-point histogram
on a log grid from 250 Hz to 4 kHz: the day-old peep sits near the top, the
adult cluck near the bottom, and nothing a chicken does sits below 250 Hz --
what is periodic down there is the building, and it is left to the
background); a(h, f0)
is the harmonic profile of one voice, a small network shared by every clip;
the line width is a fixed *fraction* eps of the harmonic's frequency, because
pitch jitter is relative; |H_L|^2 is the three-formant tube filter with one
length L per clip and one quality factor shared by all; B(f) is a power-law
background per clip for fans, litter and the microphone.

The physics is in the model class -- two shared numbers (eps, Q) and one
tract length per clip -- not in a loss term.  The model is fitted to the
spectra alone by log-spectral error; the health label never enters the fit.
What comes out per clip has names: the flock's fundamental and its spread,
the share of sound that is voiced at all, the tract length, the background
slope and level.  Whether those numbers carry health -- and carry it to a
recording the model has never heard, where a spectrogram network is free to
learn the recording instead -- is the question this file asks.
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
from scipy.signal import welch
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import PROTOCOLS, folds_for, load_dataset, record_meta, CLIPS, SR    # noqa: E402
from transients import transient_features                         # noqa: E402
from stats import permutation_p_auc, session_bootstrap_auc         # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DEVICE = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
DTYPE = torch.float32 if DEVICE.type == "mps" else torch.float64
torch.set_default_dtype(DTYPE)

F_LO, F_HI = 200.0, 7000.0
F0_GRID = np.geomspace(250.0, 4000.0, 80)
N_HARM = 6
C_AIR = 343.0
L_MIN, L_MAX = 0.02, 0.25                     # tract length bounds, m
BANDS = [(100, 250), (250, 500), (500, 1000), (1000, 2000), (2000, 3000),
         (3000, 4000), (4000, 5500), (5500, 7000)]
FEATURE_NAMES = ["f0_mean_hz", "f0_spread_oct", "voiced_share", "tract_length_cm",
                 "bg_slope", "bg_level", "bursts_per_s", "burst_share"]
NAMED = [0, 1, 2, 4, 5, 6, 7]          # the seven numbers used for prediction: tract length is
                                        # not identifiable from these spectra (see identifiability())
F0_GRID_WIDE = np.geomspace(150.0, 6000.0, 100)   # the grid-sensitivity control
VOICED_MIN = 0.2                        # a clip counts as voiced for the growth curve above this share
SUBSAMPLE = 1500
DECOMP_STEPS = 300      # per-clip decomposition with the shared physics frozen
FLOOR = 1e-4            # -40 dB below the clip mean.  The released pullet files are hard high-passed
                        # at 2 kHz; bins below the floor are excluded from the fit, not fitted as zeros


def spectra(X, sr=SR, nperseg=2048):
    f, P = welch(X, fs=sr, nperseg=nperseg, noverlap=nperseg // 2, axis=-1)
    m = (f >= F_LO) & (f <= F_HI)
    return f[m], P[:, m]


def mlp(n_in, width=32):
    return torch.nn.Sequential(torch.nn.Linear(n_in, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, width), torch.nn.Tanh(),
                               torch.nn.Linear(width, 1))


class SourceFilterMixture(torch.nn.Module):
    def __init__(self, n_clips, f, f0_grid=F0_GRID, n_harm=N_HARM):
        super().__init__()
        self.register_buffer("f", torch.tensor(np.asarray(f), dtype=DTYPE))
        self.register_buffer("f0", torch.tensor(np.asarray(f0_grid), dtype=DTYPE))
        self.register_buffer("h", torch.arange(1, n_harm + 1, dtype=DTYPE))
        self.p_logit = torch.nn.Parameter(torch.zeros(n_clips, len(f0_grid)))
        self.log_eps = torch.nn.Parameter(torch.tensor(float(np.log(0.02))))   # relative line width
        self.log_q = torch.nn.Parameter(torch.tensor(float(np.log(4.0))))      # formant quality factor
        self.profile = mlp(2)                                            # a(h, f0)
        self.length_raw = torch.nn.Parameter(torch.zeros(n_clips, 1))   # tract length, via sigmoid
        self.bg = torch.nn.Parameter(torch.tensor([[-2.0, 1.0, -4.0]] * n_clips))
        self.log_gain = torch.nn.Parameter(torch.zeros(n_clips, 1))

    # ---- physics -------------------------------------------------------
    def harmonic_profile(self):
        H, K = len(self.h), len(self.f0)
        hh = self.h.reshape(H, 1).expand(H, K)
        ff = torch.log(self.f0 / 800.0).reshape(1, K).expand(H, K)
        inp = torch.stack([(hh - 3.5) / 2.0, ff], -1)
        return torch.nn.functional.softplus(self.profile(inp).squeeze(-1))     # (H, K)

    def comb_basis(self):
        """(K, F): the spectrum of one voice at fundamental f0_k."""
        centre = self.h.reshape(-1, 1) * self.f0.reshape(1, -1)                 # (H, K)
        gamma = torch.exp(self.log_eps) * centre                                # (H, K)
        d = self.f.reshape(1, 1, -1) - centre.unsqueeze(-1)                     # (H, K, F)
        g = gamma.unsqueeze(-1)
        L = (g / np.pi) / (d ** 2 + g ** 2)
        a = self.harmonic_profile().unsqueeze(-1)
        return (a * L).sum(0)                                                   # (K, F)

    def tract_length(self):
        return L_MIN + (L_MAX - L_MIN) * torch.sigmoid(self.length_raw)         # (N, 1)

    def tube_filter(self):
        """(N, F): |H|^2 of a closed-open tube, three formants, one Q."""
        Lt = self.tract_length()
        Q = torch.exp(self.log_q)
        out = torch.ones(Lt.shape[0], len(self.f), dtype=DTYPE, device=Lt.device)
        for m in (1, 2, 3):
            Fm = (2 * m - 1) * C_AIR / (4 * Lt)                                 # (N, 1)
            r = self.f.reshape(1, -1) / Fm
            out = out * (1.0 + 1.0 / Q ** 2) / ((1 - r ** 2) ** 2 + (r / Q) ** 2)
        return out / out.mean(1, keepdim=True)

    def parts(self):
        p = torch.softmax(self.p_logit, 1)                                      # (N, K)
        voiced = (p @ self.comb_basis()) * self.tube_filter()                   # (N, F)
        lf = torch.log(self.f / 1000.0).reshape(1, -1)
        b0, b1, c = self.bg[:, :1], self.bg[:, 1:2], self.bg[:, 2:]
        bg = torch.exp(b0 - torch.nn.functional.softplus(b1) * lf) + torch.exp(c)
        return p, voiced, bg

    def forward(self):
        p, voiced, bg = self.parts()
        return torch.exp(self.log_gain) * (voiced + bg)


def fit(P, f, steps=1500, lr=2e-2, model=None, freeze_shared=False, log=print, f0_grid=F0_GRID):
    Y = torch.tensor(np.asarray(P, dtype=np.float64) / np.asarray(P, dtype=np.float64).mean(1, keepdims=True), dtype=DTYPE, device=DEVICE)
    m = SourceFilterMixture(Y.shape[0], f, f0_grid=f0_grid).to(DEVICE)
    if model is not None:
        m.log_eps.data.copy_(model.log_eps.data)
        m.log_q.data.copy_(model.log_q.data)
        m.profile.load_state_dict(model.profile.state_dict())
    per_clip = [m.p_logit, m.bg, m.log_gain, m.length_raw]
    shared = [m.log_eps, m.log_q] + list(m.profile.parameters())
    if freeze_shared:
        for q in shared:
            q.requires_grad_(False)
        params = per_clip
    else:
        params = per_clip + shared
    opt = torch.optim.Adam(params, lr=lr)
    logY = torch.log(Y + FLOOR)
    mask = (Y > FLOOR).to(DTYPE)                 # bins the recording actually contains
    for it in range(steps):
        opt.zero_grad()
        loss = torch.sum(mask * (torch.log(m() + FLOOR) - logY) ** 2) / mask.sum()
        loss.backward()
        opt.step()
        if it % 300 == 0 or it == steps - 1:
            log(f"    step {it:5d}  log-spectral loss {float(loss):.4f}  "
                f"eps {float(torch.exp(m.log_eps)):.4f}  Q {float(torch.exp(m.log_q)):.2f}")
    return m, float(loss)


def features(m, X=None):
    """Named per-clip quantities from the fitted mixture (+ transient counts)."""
    with torch.no_grad():
        p, voiced, bg = m.parts()
        lf0 = torch.log2(m.f0)
        mean_l = (p * lf0).sum(1)
        spread = torch.sqrt((p * (lf0 - mean_l[:, None]) ** 2).sum(1))
        f0_mean = 2 ** mean_l
        share = voiced.sum(1) / (voiced.sum(1) + bg.sum(1))
        slope = torch.nn.functional.softplus(m.bg[:, 1])
        level = m.bg[:, 0] - m.bg[:, 2]
        Lcm = 100 * m.tract_length().squeeze(1)
        F = torch.stack([f0_mean, spread, share, Lcm, slope, level], 1).cpu().numpy()
    if X is not None:
        F = np.concatenate([F, transient_features(X)], 1)
    return F, p.cpu().numpy()


def identifiability(m, P, f, n_clips=3, n_grid=40):
    """Profile of the log-spectral loss over tract length for a few clips, everything
    else held at its fitted value.  A flat profile means the number is not identified."""
    with torch.no_grad():
        Y = torch.tensor(np.asarray(P, dtype=np.float64) / np.asarray(P, dtype=np.float64).mean(1, keepdims=True),
                         dtype=DTYPE, device=DEVICE)
        logY = torch.log(Y + FLOOR); mask = (Y > FLOOR).to(DTYPE)
        Ls = np.linspace(L_MIN + 1e-3, L_MAX - 1e-3, n_grid)
        raw0 = m.length_raw.data.clone()
        prof = np.zeros((min(n_clips, Y.shape[0]), n_grid))
        for j, L in enumerate(Ls):
            r = float(np.log((L - L_MIN) / (L_MAX - L)))
            m.length_raw.data.fill_(r)
            S = torch.log(m() + FLOOR)
            per = (mask * (S - logY) ** 2).sum(1) / mask.sum(1)
            prof[:, j] = per[: prof.shape[0]].cpu().numpy()
        m.length_raw.data.copy_(raw0)
        Lcm = 100 * m.tract_length().squeeze(1).cpu().numpy()
    at_bounds = float(np.mean((Lcm < 100 * L_MIN + 0.5) | (Lcm > 100 * L_MAX - 0.5)))
    return {"L_cm_grid": (100 * Ls).tolist(), "loss_profiles": prof.tolist(),
            "fraction_at_bounds": at_bounds, "L_cm_percentiles": np.percentile(Lcm, [5, 25, 50, 75, 95]).tolist()}


class FreeSpectrum(torch.nn.Module):
    """The control the series keeps talking about: a free network, physics only as a loss term.
    Each clip has an 8-number embedding; a shared MLP maps (embedding, log f) to log power."""
    def __init__(self, n_clips, f, dim=8, width=64):
        super().__init__()
        self.register_buffer("lf", torch.tensor(np.log(np.asarray(f) / 1000.0), dtype=DTYPE))
        self.z = torch.nn.Parameter(0.1 * torch.randn(n_clips, dim))
        self.net = torch.nn.Sequential(torch.nn.Linear(dim + 1, width), torch.nn.Tanh(),
                                       torch.nn.Linear(width, width), torch.nn.Tanh(), torch.nn.Linear(width, 1))

    def forward(self):
        N, F = self.z.shape[0], self.lf.shape[0]
        zz = self.z.unsqueeze(1).expand(N, F, self.z.shape[1])
        ff = self.lf.reshape(1, F, 1).expand(N, F, 1)
        return self.net(torch.cat([zz, ff], -1)).squeeze(-1)                     # log S, (N, F)


def fit_free(P, f, phys_logS=None, lam=1.0, steps=300, lr=1e-2, net=None):
    """Free spectrum with an optional physics penalty towards the mixture model's fit."""
    Y = torch.tensor(np.asarray(P, dtype=np.float64) / np.asarray(P, dtype=np.float64).mean(1, keepdims=True), dtype=DTYPE, device=DEVICE)
    logY = torch.log(Y + FLOOR); mask = (Y > FLOOR).to(DTYPE)
    m = FreeSpectrum(Y.shape[0], f).to(DEVICE)
    if net is not None:
        m.net.load_state_dict(net.state_dict())
    params = [m.z] + ([] if net is not None else list(m.net.parameters()))
    opt = torch.optim.Adam(params, lr=lr)
    target = None if phys_logS is None else torch.tensor(phys_logS, dtype=DTYPE, device=DEVICE)
    for _ in range(steps):
        opt.zero_grad()
        S = m()
        loss = torch.sum(mask * (S - logY) ** 2) / mask.sum()
        if target is not None and lam > 0:
            loss = loss + lam * torch.mean((S - target) ** 2)
        loss.backward(); opt.step()
    return m, m.z.detach().cpu().numpy()


def band_features(P, f):
    out = []
    for lo, hi in BANDS:
        k = (f >= lo) & (f < hi)
        out.append(np.log(P[:, k].mean(1) + 1e-12))
    return np.stack(out, 1)


def logreg_auc(F_train, y_train, F_test):
    sc = StandardScaler().fit(F_train)
    clf = LogisticRegression(max_iter=3000, class_weight="balanced", C=0.5)
    clf.fit(sc.transform(F_train), y_train)
    return clf.predict_proba(sc.transform(F_test))[:, 1]


def protocol_test(ds, X, y, rec, week, f, P, log):
    """Shared physics from the training clips only; held-out clips decomposed with it
    frozen.  rec is the SESSION of each clip (a cage-hour for the pullets); every
    interval below is a bootstrap over sessions, every p-value a session-level
    permutation."""
    F_band = band_features(P, f)
    T = transient_features(X)
    level = np.log(np.sqrt((X ** 2).mean(1)) + 1e-9)[:, None]      # the dumbest reading: how loud
    res = {}
    keys = ["physics_7_named", "physics_8_with_tract", "physics_5_spectral", "f0_distribution",
            "band_energies", "transients_only", "loudness_only", "free_net_no_physics", "free_net_physics_loss"]
    for protocol in PROTOCOLS[ds]:
        yy, gg = [], []
        store = {k: [] for k in keys}
        for name, tr, te in folds_for(protocol, y, rec, week):
            rng = np.random.default_rng(0)
            sub = tr if len(tr) <= SUBSAMPLE else np.sort(rng.choice(tr, SUBSAMPLE, replace=False))
            m_sub, _ = fit(P[sub], f, steps=600, log=lambda s: None)          # shared physics: training clips only
            m_tr, _ = fit(P[sub], f, steps=DECOMP_STEPS, model=m_sub, freeze_shared=True, log=lambda s: None)
            m_te, _ = fit(P[te], f, steps=DECOMP_STEPS, model=m_sub, freeze_shared=True, log=lambda s: None)
            Ftr, ptr = features(m_tr); Fte, pte = features(m_te)
            with torch.no_grad():
                phys_tr = torch.log(m_tr() + FLOOR).cpu().numpy(); phys_te = torch.log(m_te() + FLOOR).cpu().numpy()
            tr = sub                                                            # the regression sees the decomposed training clips
            Ftr8 = np.concatenate([Ftr, T[tr]], 1); Fte8 = np.concatenate([Fte, T[te]], 1)
            # the two free-network controls: no physics, and physics as a penalty
            n0_tr, z0_tr = fit_free(P[tr], f, None, 0.0); _, z0_te = fit_free(P[te], f, None, 0.0, net=n0_tr.net)
            n1_tr, z1_tr = fit_free(P[tr], f, phys_tr, 1.0); _, z1_te = fit_free(P[te], f, phys_te, 1.0, net=n1_tr.net)
            for key, (A, B) in {"physics_7_named": (Ftr8[:, NAMED], Fte8[:, NAMED]),
                                "physics_8_with_tract": (Ftr8, Fte8),
                                "physics_5_spectral": (Ftr[:, [0, 1, 2, 4, 5]], Fte[:, [0, 1, 2, 4, 5]]),
                                "f0_distribution": (ptr, pte),
                                "band_energies": (F_band[tr], F_band[te]),
                                "transients_only": (T[tr], T[te]),
                                "loudness_only": (level[tr], level[te]),
                                "free_net_no_physics": (z0_tr, z0_te),
                                "free_net_physics_loss": (z1_tr, z1_te)}.items():
                store[key].append(logreg_auc(A, y[tr], B))
            yy.append(y[te]); gg.append(rec[te])
            log(f"  {ds} {protocol:10s} {name:8s} n_test={len(te):5d}")
        yy = np.concatenate(yy); gg = np.concatenate(gg)
        res[protocol] = {}
        for k, v in store.items():
            pp = np.concatenate(v)
            res[protocol][k] = session_bootstrap_auc(yy, pp, gg)
            res[protocol][k]["perm_p"] = permutation_p_auc(yy, pp, gg, n_perm=300)
        log(f"  {ds} {protocol}: " + "  ".join(f"{k} {v['pooled_auc']:.3f} [{v['ci95'][0]:.2f},{v['ci95'][1]:.2f}]"
                                            for k, v in res[protocol].items()))
    return res


def load_calls():
    index = json.loads((ROOT / "data" / "index.json").read_text())
    X, call = [], []
    for r in index:
        if r["dataset"] != "calls" or r["cls"] != "single_vocalizations" or r["n_clips"] == 0:
            continue
        A = np.load(CLIPS / (r["file"] + ".npz"))["X"]
        X.append(A); call += [r["call"]] * len(A)
    return (np.concatenate(X), np.array(call)) if X else (np.zeros((0, 2 * SR), np.float32), np.array([]))


def main():
    RESULTS.mkdir(exist_ok=True)
    log = lambda s: print(s, flush=True)                   # noqa: E731
    out = {"f0_grid": F0_GRID.tolist(), "feature_names": FEATURE_NAMES, "datasets": {}}

    # ---- one shared physics for all three sources, fitted on a subsample of each
    sets = {}
    for ds in ("broiler", "pullet", "pullet_control"):
        try:
            X, y, rec, week, idx = load_dataset(ds)
        except Exception:                                   # noqa: BLE001
            continue
        if len(y):
            sets[ds] = (X, y, rec, week, idx)
    Xc, calls = load_calls()
    rng = np.random.default_rng(0)
    pool, tag = [], []
    for ds, (X, y, rec, week, idx) in sets.items():
        k = rng.choice(len(y), min(SUBSAMPLE, len(y)), replace=False)
        pool.append(X[k]); tag += [ds] * len(k)
    if len(Xc):
        pool.append(Xc); tag += ["calls"] * len(Xc)
    Xp = np.concatenate(pool); tag = np.array(tag)
    f, Pp = spectra(Xp)
    log(f"global fit on {len(Xp)} clips ({dict(zip(*np.unique(tag, return_counts=True)))}), "
        f"{len(f)} bins {f[0]:.0f}-{f[-1]:.0f} Hz")
    m_all, loss_all = fit(Pp, f, steps=1000, log=log)
    with torch.no_grad():
        out["shared"] = {"eps": float(torch.exp(m_all.log_eps)), "Q": float(torch.exp(m_all.log_q)),
                         "harmonic_profile": m_all.harmonic_profile().cpu().numpy().tolist(),
                         "comb_basis_example": m_all.comb_basis().cpu().numpy()[40].tolist(),
                         "tube_filter_example": m_all.tube_filter().cpu().numpy()[0].tolist(),
                         "f": f.tolist(), "global_log_rmse": float(np.sqrt(loss_all))}
    Fp, pp = features(m_all)

    # ---- the f0 survey: chicks, adult layers, backyard hens on one grid
    survey = {}
    for t in sorted(set(tag)):
        k = tag == t
        survey[t] = {"n": int(k.sum()), "p_f0": pp[k].mean(0).tolist(),
                     "f0_median_hz": float(np.median(Fp[k, 0])),
                     "tract_length_cm_median": float(np.median(Fp[k, 3])),
                     "voiced_share_median": float(np.median(Fp[k, 2]))}
        log(f"  {t:8s} n={k.sum():5d}  f0 median {survey[t]['f0_median_hz']:.0f} Hz  "
            f"tract {survey[t]['tract_length_cm_median']:.1f} cm  voiced {survey[t]['voiced_share_median']:.2f}")
    if len(calls):
        kc = tag == "calls"
        by_call = {}
        for c in sorted(set(calls)):
            j = calls == c
            if j.sum() >= 3:
                by_call[c] = {"n": int(j.sum()), "f0_median_hz": float(np.median(Fp[kc][j, 0])),
                              "tract_length_cm_median": float(np.median(Fp[kc][j, 3]))}
        survey["calls_by_type"] = by_call
    out["survey"] = survey
    # the same pool on a wider grid: does the survey depend on where the grid ends?
    log("grid-sensitivity fit (150 Hz - 6 kHz)")
    m_wide, _ = fit(Pp, f, steps=600, log=lambda s: None, f0_grid=F0_GRID_WIDE)
    Fw, pw = features(m_wide)
    out["survey_wide_grid"] = {"f0_grid": F0_GRID_WIDE.tolist(),
                               **{t: {"f0_median_hz": float(np.median(Fw[tag == t, 0])), "p_f0": pw[tag == t].mean(0).tolist()}
                                  for t in sorted(set(tag))}}
    for t in sorted(set(tag)):
        log(f"  wide grid {t:8s} f0 median {out['survey_wide_grid'][t]['f0_median_hz']:.0f} Hz "
            f"(narrow grid {survey[t]['f0_median_hz']:.0f} Hz)")
    out["identifiability"] = identifiability(m_all, Pp, f)
    log(f"  tract length: {100 * out['identifiability']['fraction_at_bounds']:.0f} % of clips at a bound; "
        f"percentiles {np.round(out['identifiability']['L_cm_percentiles'], 1).tolist()} cm")

    # ---- per dataset: decomposition of every clip, class summaries, index trend, protocols
    for ds, (X, y, rec, week, idx) in sets.items():
        f, P = spectra(X)
        m_ds, _ = fit(P, f, steps=DECOMP_STEPS, model=m_all, freeze_shared=True, log=lambda s: None)
        F, p = features(m_ds, X)
        d = {"n_clips": int(len(y)), "n_sessions": int(len(set(rec))), "classes": {}, "examples": {}}
        with torch.no_grad():
            model_spec = m_ds().cpu().numpy(); _, voiced, bg = m_ds.parts()
            gain = np.exp(m_ds.log_gain.cpu().numpy())
        Pn = P / P.mean(1, keepdims=True)
        for c in sorted(set(y)):
            k = y == c
            j = int(np.flatnonzero(k)[np.argmax(F[k, 2])])          # the most voiced clip of the class
            d["examples"][int(c)] = {"observed": Pn[j].tolist(), "model": model_spec[j].tolist(),
                                     "voiced": (gain[j] * voiced[j].cpu().numpy()).tolist(),
                                     "background": (gain[j] * bg[j].cpu().numpy()).tolist(),
                                     "features": F[j].tolist(), "p_f0": p[j].tolist()}
            d["classes"][int(c)] = {"n": int(k.sum()), "p_f0": p[k].mean(0).tolist(),
                                    "feature_median": np.median(F[k], 0).tolist(),
                                    "feature_iqr": (np.percentile(F[k], 75, 0) - np.percentile(F[k], 25, 0)).tolist()}
        # session-level: the unit of replication
        recs = sorted(set(rec))
        R = np.array([np.median(F[rec == r], 0) for r in recs]); yr = np.array([y[rec == r][0] for r in recs])
        d["session_level"] = {"n": len(recs), "auc_per_feature": {}}
        for j, nm in enumerate(FEATURE_NAMES):
            d["session_level"]["auc_per_feature"][nm] = float(roc_auc_score(yr, R[:, j]))
        if ds == "broiler":
            ir = np.array([idx[rec == r][0] for r in recs])
            d["index_trend"] = {}
            for c in (0, 1):
                k = yr == c
                rho, pv = spearmanr(ir[k], R[k, 0])
                d["index_trend"][int(c)] = {"n": int(k.sum()), "spearman_f0_vs_index": float(rho), "p": float(pv),
                                            "index": ir[k].tolist(), "f0_median_hz": R[k, 0].tolist(),
                                            "tract_cm": R[k, 3].tolist()}
                log(f"  broiler class {c}: Spearman(f0, file index) = {rho:.2f} (p={pv:.3f}, n={k.sum()})")
        if ds.startswith("pullet"):
            meta = {}
            for mr in record_meta().values():
                meta.setdefault(mr.get("session"), mr)
            d["recordings"] = []
            for r_ in recs:
                k = rec == r_; mr = meta[r_]
                kv = k & (F[:, 2] > VOICED_MIN)
                d["recordings"].append({"session": r_, "age_days": mr["age_days"], "week": mr["week"], "day": mr["day"],
                                        "phase": mr["phase"], "cond": mr["cond"],
                                        "n_clips": int(k.sum()), "n_voiced": int(kv.sum()),
                                        "f0_median_hz": float(np.median(F[k, 0])),
                                        "f0_median_voiced_hz": float(np.median(F[kv, 0])) if kv.sum() >= 5 else None,
                                        "feature_median": np.median(F[k], 0).tolist()})
        log(f"{ds}: protocols (physics refitted per fold, labels only in the regression)")
        d["protocols"] = protocol_test(ds, X, y, rec, week, f, P, log)
        out["datasets"][ds] = d
        (RESULTS / "syrinx_pinn.json").write_text(json.dumps(out, indent=1))

    cnn = {}
    ps = RESULTS / "protocol_summary.json"
    if ps.exists():
        summ = json.loads(ps.read_text())
        cnn = {ds: {k: v["pooled_auc"] for k, v in s.items()} for ds, s in summ.items()}
        out["cnn_ci95"] = {ds: {k: v.get("ci95") for k, v in s.items()} for ds, s in summ.items()}
    out["cnn_pooled_auc"] = cnn
    (RESULTS / "syrinx_pinn.json").write_text(json.dumps(out, indent=1))
    log("saved")


if __name__ == "__main__":
    main()
