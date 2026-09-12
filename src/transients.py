"""Respiratory-type transients: broadband, tens of milliseconds, 1-8 kHz.

A rale, snick or sneeze is a flow-induced burst in the airway; a beak on a
feeder pan is an impact.  Both are short and broadband, and both are counted
here by one rule: the 1-7.5 kHz envelope rises at least ONSET_RATIO above its
own recent median and falls back within MAX_MS.  The count per second and
the share of clip energy in those bursts are two numbers with units, computed
the same way for every recording.  Nothing here knows the label.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import butter, sosfiltfilt

SR = 16000
BAND = (1000.0, 7500.0)
FRAME = 160                    # 10 ms
ONSET_RATIO = 6.0
MAX_MS = 80.0
BACK_FRAMES = 25               # 250 ms median window


def _sos(sr=SR):
    return butter(4, [BAND[0] / (sr / 2), BAND[1] / (sr / 2)], btype="band", output="sos")


def envelope(X, sr=SR):
    Y = sosfiltfilt(_sos(sr), X, axis=-1)
    n = Y.shape[-1] // FRAME
    E = (Y[..., : n * FRAME] ** 2).reshape(*Y.shape[:-1], n, FRAME).mean(-1)
    return E


def transient_features(X, sr=SR):
    """(N, 2): bursts per second, share of band energy inside bursts."""
    E = envelope(X, sr)
    N, T = E.shape
    back = median_filter(E, size=(1, BACK_FRAMES), mode="nearest")
    active = E > ONSET_RATIO * (back + 1e-10)
    rate = np.zeros(N); share = np.zeros(N)
    max_frames = int(MAX_MS / 1000 * sr / FRAME)
    for i in range(N):
        a = active[i]
        edges = np.flatnonzero(np.diff(np.concatenate([[0], a.astype(int), [0]])))
        starts, ends = edges[::2], edges[1::2]
        keep = (ends - starts) <= max_frames
        rate[i] = keep.sum() / (T * FRAME / sr)
        if keep.any():
            inside = np.zeros(T, bool)
            for s, e in zip(starts[keep], ends[keep]):
                inside[s:e] = True
            share[i] = E[i, inside].sum() / (E[i].sum() + 1e-12)
    return np.stack([rate, share], 1)
