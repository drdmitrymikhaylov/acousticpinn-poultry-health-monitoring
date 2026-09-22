"""Does the burst counter count the codec?  The pullet files are MP3; the broiler
files are WAV.  Take broiler clips, run them through MP3 at the pullet bit
rate and back, and count bursts before and after.  If the count rises, part of
what the counter sees in the pullet files is the codec, not the bird."""
from __future__ import annotations

import json
import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import soundfile as sf

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from train import load_dataset, SR                 # noqa: E402
from transients import transient_features          # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def through_mp3(x, sr, kbps=128):
    ff = shutil.which("ffmpeg")
    if ff is None:
        return None
    with tempfile.TemporaryDirectory() as td:
        w = pathlib.Path(td) / "in.wav"; m = pathlib.Path(td) / "x.mp3"; o = pathlib.Path(td) / "out.wav"
        sf.write(w, x, sr)
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(w), "-b:a", f"{kbps}k", str(m)], check=True)
        subprocess.run([ff, "-loglevel", "error", "-y", "-i", str(m), "-ar", str(sr), str(o)], check=True)
        y, _ = sf.read(o, dtype="float32")
    y = y[: len(x)]
    if len(y) < len(x):
        y = np.pad(y, (0, len(x) - len(y)))
    return y.astype(np.float32)


def main():
    X, y, rec, week, _ = load_dataset("broiler")
    rng = np.random.default_rng(0)
    k = rng.choice(len(X), 300, replace=False)
    A = X[k]
    B = [through_mp3(a, SR) for a in A]
    if B[0] is None:
        out = {"ffmpeg": False, "note": "ffmpeg not found; the codec check did not run"}
    else:
        B = np.stack(B)
        ta, tb = transient_features(A), transient_features(B)
        out = {"ffmpeg": True, "n_clips": int(len(A)), "kbps": 128,
               "bursts_per_s_wav_median": float(np.median(ta[:, 0])), "bursts_per_s_mp3_median": float(np.median(tb[:, 0])),
               "bursts_per_s_wav_mean": float(ta[:, 0].mean()), "bursts_per_s_mp3_mean": float(tb[:, 0].mean()),
               "burst_share_wav_mean": float(ta[:, 1].mean()), "burst_share_mp3_mean": float(tb[:, 1].mean()),
               "clips_with_more_bursts_after_mp3": float(np.mean(tb[:, 0] > ta[:, 0])),
               "clips_with_fewer_bursts_after_mp3": float(np.mean(tb[:, 0] < ta[:, 0]))}
    (RESULTS / "transients_mp3.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
