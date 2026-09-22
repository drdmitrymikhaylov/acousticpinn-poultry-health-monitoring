"""Write results/environment.json and data/manifest.sha256 so that a reader can
check what ran, on what, and on which bytes."""
from __future__ import annotations

import hashlib
import json
import pathlib
import platform
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]


def sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def main():
    env = {"python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine()}
    for mod in ("numpy", "scipy", "torch", "sklearn", "soundfile", "matplotlib"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:                                   # noqa: BLE001
            env[mod] = None
    try:
        import torch
        env["torch_device"] = "mps" if torch.backends.mps.is_available() else "cpu"
    except Exception:                                       # noqa: BLE001
        pass
    try:
        env["ffmpeg"] = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.split("\n")[0]
    except Exception:                                       # noqa: BLE001
        env["ffmpeg"] = None
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "environment.json").write_text(json.dumps(env, indent=1))
    lines = []
    for p in sorted((ROOT / "data").glob("*.zip")) + sorted((ROOT / "data" / "zenodo_10433023").glob("*.zip")):
        lines.append(f"{sha256(p)}  {p.relative_to(ROOT / 'data')}  {p.stat().st_size}")
    (ROOT / "data" / "manifest.sha256").write_text("\n".join(lines) + "\n")
    print(json.dumps(env, indent=1)); print("\n".join(lines))


if __name__ == "__main__":
    main()
