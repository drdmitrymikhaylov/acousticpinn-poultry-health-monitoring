"""What the recordings sound like before any model: class-mean spectra,
recording-level band shares, and how much the recordings of one class
differ from each other.  The unit of replication is the recording."""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import load_dataset, CLIPS, ROOT                       # noqa: E402
from syrinx_pinn import spectra, BANDS                            # noqa: E402

RESULTS = ROOT / "results"


def noise_clips():
    index = json.loads((ROOT / "data" / "index.json").read_text())
    X = [np.load(CLIPS / (r["file"] + ".npz"))["X"] for r in index
         if r["dataset"] == "broiler" and r["cls"] == "noise" and r["n_clips"]]
    return np.concatenate(X) if X else None


def summarise(X, y, rec, names):
    f, P = spectra(X)
    Pn = P / P.sum(1, keepdims=True)
    out = {"f": f.tolist(), "classes": {}, "bands": BANDS}
    recs = sorted(set(rec))
    R = np.array([Pn[rec == r].mean(0) for r in recs]); yr = np.array([y[rec == r][0] for r in recs])
    for c, nm in names.items():
        k = yr == c
        shares = np.array([[R[k][:, (f >= lo) & (f < hi)].sum(1)] for lo, hi in BANDS]).squeeze(1)
        out["classes"][nm] = {"n_recordings": int(k.sum()), "n_clips": int((y == c).sum()),
                              "mean_spectrum": R[k].mean(0).tolist(),
                              "p10": np.percentile(R[k], 10, 0).tolist(),
                              "p90": np.percentile(R[k], 90, 0).tolist(),
                              "band_share_mean": shares.mean(1).tolist(),
                              "band_share_sd": shares.std(1).tolist()}
    return out


def main():
    RESULTS.mkdir(exist_ok=True)
    res = {}
    X, y, rec, week, _ = load_dataset("broiler")
    Xn = noise_clips()
    if Xn is not None:
        X = np.concatenate([X, Xn]); y = np.concatenate([y, np.full(len(Xn), 2)])
        rec = np.concatenate([rec, np.array([f"noise{i // 60}" for i in range(len(Xn))])])
    res["broiler"] = summarise(X, y, rec, {0: "healthy", 1: "unhealthy", 2: "noise"})
    for ds in ("pullet", "pullet_control"):
        try:
            X, y, rec, week, _ = load_dataset(ds)
            if len(y):
                res[ds] = summarise(X, y, rec, {0: "pre", 1: "post"})
        except Exception as e:                                      # noqa: BLE001
            print(ds, "skipped:", e)
    (RESULTS / "spectra.json").write_text(json.dumps(res))
    for ds, d in res.items():
        for nm, c in d["classes"].items():
            print(ds, nm, c["n_recordings"], "recordings;",
                  " ".join(f"{lo}-{hi}:{m:.2f}±{s:.2f}" for (lo, hi), m, s in
                           zip(BANDS, c["band_share_mean"], c["band_share_sd"])))


if __name__ == "__main__":
    main()
