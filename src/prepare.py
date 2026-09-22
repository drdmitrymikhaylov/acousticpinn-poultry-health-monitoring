"""Recordings to labelled clips, with provenance.

Three open sources, none redistributed here (see data/SOURCE.md):

  broiler   Mendeley zp4nf2dxbh -- 346 recordings of chicks from day-old to
            65 days, folders Healthy / Unhealthy / Noise, 48 kHz mono.  There is
            no day, pen or bird identity in the file names; the only group a
            clip belongs to is its recording.
  pullet    Zenodo 10433023 -- white-egg layer pullets, 3 days to 9 weeks,
            recorded for an hour before and after an acute stressor each
            week; file names carry condition, week, day, phase and position.
            The control cage (no stressor) is kept as 'pullet_control'.
  calls     github.com/zebular13/ChickenLanguageDataset -- single
            vocalisations of backyard hens, named by what the recordist thought
            the bird meant.  Used only for the fundamental-frequency survey.

Every recording is resampled to 16 kHz, cut into two-second clips, and each
clip is stored with the recording AND the session it came from.  For the
pullets a session is one cage-hour: the files _1.._4 are four microphones
recording the same birds at the same time, and a split that put position 1
in training and position 3 in test would be a leak.  The protocols in
train.py group by session.
"""
from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

ROOT = pathlib.Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CLIPS = DATA / "clips"
SR = 16000
CLIP_S = 2.0
MAX_CLIPS_PER_FILE = 60


def load_16k(path):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(1)
    if sr != SR:
        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def clips_of(x, seed, pad=False):
    n = int(CLIP_S * SR)
    if pad and SR // 4 <= len(x) < n:                  # a single call shorter than a clip
        x = np.pad(x, (0, n - len(x)))
    k = len(x) // n
    if k == 0:
        return np.zeros((0, n), np.float32)
    C = x[: k * n].reshape(k, n)
    C = C[np.abs(C).max(1) > 1e-4]                     # drop digital silence
    if len(C) > MAX_CLIPS_PER_FILE:
        rng = np.random.default_rng(seed)
        C = C[np.sort(rng.choice(len(C), MAX_CLIPS_PER_FILE, replace=False))]
    return C


def broiler_records():
    root = DATA / "mendeley_zp4nf2dxbh"
    label = {"Healthy": 0, "Unhealthy": 1, "Noise": -1}
    for cls, y in label.items():
        for p in sorted((root / cls).glob("*.wav"), key=lambda q: int(q.stem)):
            yield {"dataset": "broiler", "path": str(p.relative_to(DATA)), "y": y,
                   "cls": cls.lower(), "group": f"broiler/{cls}/{p.stem}", "session": f"broiler/{cls}/{p.stem}",
                   "index": int(p.stem)}


PULLET_RE = re.compile(r"^([A-Za-z0-9]+)_W(\d+)_D(\d+)_([A-Za-z]+)(?:_(\d+))?$")
AGE_OFFSET = 13          # the preprint says stressors ran from day 14: protocol week 1, day 1 = 14 days old
PHASE = {"prs": "pre", "pos": "post"}


def pullet_records():
    """Zenodo 10433023.  Label: after the stressor (1) or before it (0).  The
    control cage, which had no stressor, is kept as its own dataset so that
    'before vs after' can be tested where nothing happened in between."""
    root = DATA / "zenodo_10433023"
    for p in sorted(root.rglob("*.mp3")):
        m = PULLET_RE.match(p.stem)
        if not m:
            print("unparsed", p.name); continue
        cond, week, day, phase = m.group(1).lower(), int(m.group(2)), int(m.group(3)), m.group(4).lower()
        pos = int(m.group(5)) if m.group(5) else 0
        phase = PHASE.get(phase, phase)
        if phase not in ("pre", "post"):
            continue
        ds = "pullet_control" if cond.startswith("c") else "pullet"
        yield {"dataset": ds, "path": str(p.relative_to(DATA)), "y": int(phase == "post"),
               "cls": phase, "group": f"{ds}/{cond}/W{week}/D{day}/{phase}/{pos}",
               "session": f"{ds}/{cond}/W{week}/D{day}/{phase}",       # the four microphones of one hour are ONE observation
               "week": week, "day": day,
               "age_days": AGE_OFFSET + 7 * (week - 1) + day, "phase": phase, "cond": cond, "position": pos}


def call_records():
    root = DATA / "chicken_language"
    for sub in ("single_vocalizations", "longer_segments", "noise"):
        for p in sorted((root / sub).rglob("*.wav")):
            name = re.sub(r"^\d+\.\d+secs", "", p.stem)
            name = re.sub(r"\d+$", "", name).strip("_ -").lower()
            yield {"dataset": "calls", "path": str(p.relative_to(DATA)),
                   "y": -1 if sub == "noise" else -2, "cls": sub,
                   "group": f"calls/{sub}/{p.stem}", "call": name}


def main():
    CLIPS.mkdir(parents=True, exist_ok=True)
    index = []
    for i, r in enumerate(list(broiler_records()) + list(pullet_records()) + list(call_records())):
        out = CLIPS / (r["group"].replace("/", "__") + ".npz")
        if not out.exists():
            try:
                x = load_16k(DATA / r["path"])
            except Exception as e:                           # noqa: BLE001
                print("skip", r["path"], e, flush=True)
                continue
            C = clips_of(x, seed=i, pad=(r["dataset"] == "calls"))
            np.savez_compressed(out, X=C)
            r["seconds"] = round(len(x) / SR, 2)
            r["n_clips"] = int(len(C))
        else:
            r["n_clips"] = int(np.load(out)["X"].shape[0])
        r["file"] = out.stem
        index.append(r)
        if i % 50 == 0:
            print(i, r["group"], r["n_clips"], flush=True)
    (DATA / "index.json").write_text(json.dumps(index, indent=0))
    for ds in ("broiler", "pullet", "pullet_control", "calls"):
        rows = [r for r in index if r["dataset"] == ds]
        by = {}
        for r in rows:
            by.setdefault(r["cls"], [0, 0]); by[r["cls"]][0] += 1; by[r["cls"]][1] += r["n_clips"]
        print(ds, {k: f"{v[0]} recordings, {v[1]} clips" for k, v in by.items()}, flush=True)


if __name__ == "__main__":
    main()
