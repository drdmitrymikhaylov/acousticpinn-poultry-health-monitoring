"""Uncertainty for pooled AUCs, with the recording session — not the clip — as
the unit of replication.  A clip-level bootstrap would give intervals a
hundred times too narrow: clips from one session are one observation."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score


def session_bootstrap_auc(y, p, groups, n_boot=1000, seed=0):
    """Pooled AUC with a 95 % interval from resampling whole groups (sessions)."""
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    G = np.array(sorted(set(groups)))
    idx = {g: np.flatnonzero(groups == g) for g in G}
    point = float(roc_auc_score(y, p))
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(G, len(G), replace=True)
        ii = np.concatenate([idx[g] for g in pick])
        if len(set(y[ii])) < 2:
            continue
        vals.append(roc_auc_score(y[ii], p[ii]))
    lo, hi = (np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan))
    return {"pooled_auc": point, "ci95": [float(lo), float(hi)], "n_groups": int(len(G)),
            "n": int(len(y)), "n_boot_valid": int(len(vals))}


def permutation_p_auc(y, p, groups, n_perm=1000, seed=0):
    """One-sided p-value for AUC > 0.5 when labels are permuted at the group level
    (every clip of a session keeps one label, as in the real data)."""
    y = np.asarray(y); p = np.asarray(p); groups = np.asarray(groups)
    rng = np.random.default_rng(seed)
    G = np.array(sorted(set(groups)))
    lab = np.array([y[groups == g][0] for g in G])
    point = roc_auc_score(y, p)
    count = 0; valid = 0
    for _ in range(n_perm):
        perm = rng.permutation(lab)
        yp = np.empty_like(y)
        for g, l in zip(G, perm):
            yp[groups == g] = l
        if len(set(yp)) < 2:
            continue
        valid += 1
        if roc_auc_score(yp, p) >= point:
            count += 1
    return float((count + 1) / (valid + 1))
