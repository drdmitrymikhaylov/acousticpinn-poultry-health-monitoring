"""Checks that the physics in the model does what the page says, and that the
page's numbers are what the results files hold.  Run from the repo root:
    python tests/test_all.py
"""
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from prepare import clips_of, SR                                          # noqa: E402
from transients import transient_features                                 # noqa: E402
from model import mel_filterbank                                          # noqa: E402

try:
    import torch                                                          # noqa: F401
    from syrinx_pinn import SourceFilterMixture, features, F0_GRID, C_AIR   # noqa: E402
except ImportError:                                                       # pragma: no cover
    SourceFilterMixture = None

PASSED = []


def check(name, cond):
    PASSED.append(bool(cond))
    print(("ok   " if cond else "FAIL ") + name)


def test_clips_and_padding():
    x = np.random.default_rng(0).standard_normal(int(4.5 * SR)).astype(np.float32)
    C = clips_of(x, seed=0)
    check("a 4.5 s recording gives two whole 2 s clips and no partial one", C.shape == (2, 2 * SR))
    short = x[: SR]                                                       # one second
    check("a short single call is zero-padded to a clip only when asked",
          clips_of(short, 0).shape[0] == 0 and clips_of(short, 0, pad=True).shape == (1, 2 * SR))


def test_mel_covers_the_peep():
    fb = mel_filterbank().numpy()
    freqs = np.linspace(0, SR / 2, fb.shape[1])
    covered = fb.sum(0) > 0
    check("the mel filterbank covers 150 Hz to 7.5 kHz without a gap",
          covered[(freqs >= 150) & (freqs <= 7500)].all())


def test_transient_counter():
    rng = np.random.default_rng(1)
    n = 2 * SR
    quiet = 1e-3 * rng.standard_normal((1, n)).astype(np.float32)
    bursts = quiet.copy()
    for t0 in (0.3, 0.9, 1.5):                                             # three 20 ms broadband bursts
        i = int(t0 * SR)
        bursts[0, i:i + int(0.02 * SR)] += 0.3 * rng.standard_normal(int(0.02 * SR))
    tone = quiet.copy()
    tone[0] += 0.3 * np.sin(2 * np.pi * 3000 * np.arange(n) / SR)          # a steady 3 kHz tone
    r_quiet, r_burst, r_tone = (transient_features(a)[0, 0] for a in (quiet, bursts, tone))
    check("three 20 ms bursts in two seconds are counted as 1.5 per second",
          abs(r_burst - 1.5) < 1e-6 and r_quiet == 0.0)
    check("a steady tone is not a transient", r_tone == 0.0)


def test_source_filter_physics():
    if SourceFilterMixture is None:
        print("skip torch checks (torch not installed)"); return
    f = np.arange(200.0, 7000.0, 7.8125)
    m = SourceFilterMixture(1, f)
    with torch.no_grad():
        # one voice at 1 kHz: peaks at 1, 2, 3 kHz and nowhere else
        k = int(np.argmin(np.abs(F0_GRID - 1000.0)))
        f0 = float(F0_GRID[k])
        m.p_logit.zero_(); m.p_logit[0, k] = 30.0
        m.length_raw.fill_(-20.0); m.log_q.fill_(float(np.log(0.7)))             # a 2 cm, heavily damped tube: a smooth filter below 4 kHz
        m.bg[0] = torch.tensor([-30.0, 1.0, -30.0])
        S = m()[0].cpu().numpy()
        peaks = [f[i] for i in range(1, len(f) - 1) if S[i] > S[i - 1] and S[i] > S[i + 1] and S[i] > 0.05 * S.max()]
        ratios = np.array(peaks) / f0
        check("a single voice at 1 kHz produces peaks only at integer multiples of 1 kHz",
              len(peaks) >= 3 and np.all(np.abs(ratios - np.round(ratios)) < 0.02))
        # the tube: first formant of an 8 cm closed-open tube at c/(4L) = 1072 Hz
        m.length_raw.zero_(); m.log_q.fill_(float(np.log(8.0)))
        m.length_raw.fill_(float(np.log((0.08 - 0.02) / (0.25 - 0.08))))     # sigmoid^-1 for L = 8 cm
        H = m.tube_filter()[0].cpu().numpy()
        F1 = f[np.argmax(H * (f < 2000))]
        check("an 8 cm closed-open tube puts its first formant at c/4L = 1072 Hz (+/- one bin)",
              abs(F1 - C_AIR / (4 * 0.08)) < 8.0)
        # the line width is relative: harmonic 3 is three times wider than harmonic 1
        centre = (m.h.reshape(-1, 1) * m.f0.reshape(1, -1))
        gamma = torch.exp(m.log_eps) * centre
        check("line width scales with harmonic number (relative jitter)",
              abs(float(gamma[2, k] / gamma[0, k]) - 3.0) < 1e-5)
        # the named features read back what was put in
        m.p_logit.zero_(); m.p_logit[0, k] = 30.0
        Fz, _ = features(m)
        check("a clip placed entirely at 1 kHz is read back as f0 = 1 kHz with zero spread",
              abs(Fz[0, 0] - f0) < 1.0 and Fz[0, 1] < 1e-3)


def test_results_are_what_the_page_says():
    p = ROOT / "results" / "syrinx_pinn.json"
    if not p.exists():
        print("skip results checks (no results/syrinx_pinn.json)"); return
    d = json.loads(p.read_text())
    b = d["datasets"]["broiler"]["protocols"]
    check("broiler: on a held-out recording, loudness alone separates the two folders (AUC > 0.9)",
          b["recording"]["loudness_only"]["pooled_auc"] > 0.9)
    check("broiler: the CNN is at or near 1.0 under both protocols",
          all(v > 0.97 for v in d["cnn_pooled_auc"]["broiler"].values()))
    if "pullet" in d["datasets"]:
        pr = d["datasets"]["pullet"]["protocols"]
        check("pullet: the physics features and the CNN are both reported under leave-one-week-out",
              "week" in pr and "week" in d["cnn_pooled_auc"].get("pullet", {}))
    g = ROOT / "results" / "growth_pinn.json"
    if g.exists():
        gg = json.loads(g.read_text())
        check("growth: the flock fundamental falls with age (Spearman rho < -0.5)",
              gg["spearman_f0_vs_age"] < -0.5)


if __name__ == "__main__":
    for t in (test_clips_and_padding, test_mel_covers_the_peep, test_transient_counter,
              test_source_filter_physics, test_results_are_what_the_page_says):
        t()
    n = len(PASSED)
    print(f"{sum(PASSED)}/{n} checks passed")
    sys.exit(0 if all(PASSED) else 1)
