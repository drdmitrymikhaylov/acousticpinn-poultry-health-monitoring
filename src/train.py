"""Healthy or not, from sound, under splitting protocols that get harder.

  random     -- five stratified folds over clips.  Clips from the same
                recording land on both sides.  This is what most published
                poultry-audio results are.
  recording  -- five folds over whole recordings (GroupKFold).  A clip is
                scored by a model that never heard its recording.
  week       -- pullets only: leave one experimental week out.

Per-fold AUC is undefined when a fold holds one class, so out-of-fold scores
are pooled and scored once.  Resumable: one JSON per (dataset, protocol, fold).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GroupKFold, StratifiedKFold

sys.path.insert(0, str(pathlib.Path(__file__).parent))
from model import FlockNet, SR                     # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
CLIPS = ROOT / "data" / "clips"
RESULTS = ROOT / "results" / "cv"
EPOCHS = 12
BATCH = 32
PROTOCOLS = {"broiler": ("random", "recording"), "pullet": ("random", "recording", "week"),
             "pullet_control": ("random", "recording", "week")}


def device():
    return torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")


def record_meta():
    index = json.loads((ROOT / "data" / "index.json").read_text())
    return {r["group"]: r for r in index}


def load_dataset(name):
    """Clips of one dataset with a label, plus the group keys the protocols use."""
    index = json.loads((ROOT / "data" / "index.json").read_text())
    X, y, rec, week, idx = [], [], [], [], []
    for r in index:
        if r["dataset"] != name or r["y"] < 0 or r["n_clips"] == 0:
            continue
        A = np.load(CLIPS / (r["file"] + ".npz"))["X"]
        X.append(A)
        y += [r["y"]] * len(A)
        rec += [r["group"]] * len(A)
        week += [r.get("week", -1)] * len(A)
        idx += [r.get("index", -1)] * len(A)
    return (np.concatenate(X), np.array(y, dtype=np.int64), np.array(rec),
            np.array(week), np.array(idx))


def folds_for(protocol, y, rec, week):
    if protocol == "random":
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        return [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(skf.split(y, y))]
    if protocol == "recording":
        gkf = GroupKFold(n_splits=5)
        return [(f"fold{i}", tr, te) for i, (tr, te) in enumerate(gkf.split(y, y, rec))]
    out = []
    for w in sorted(set(week)):
        te = np.flatnonzero(week == w)
        tr = np.flatnonzero(week != w)
        if len(set(y[tr])) < 2:
            continue
        out.append((f"W{w}", tr, te))
    return out


def fit(model, X, y, tr, dev, seed):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    counts = np.array([(y[tr] == c).sum() for c in (0, 1)], dtype=float)
    w = torch.tensor(counts.sum() / np.maximum(counts, 1), dtype=torch.float32)
    lossf = nn.CrossEntropyLoss(weight=(w / w.sum() * 2).to(dev))
    Xtr = torch.from_numpy(X[tr]); ytr = torch.from_numpy(y[tr])
    for _ in range(EPOCHS):
        model.train()
        perm = torch.randperm(Xtr.size(0))
        for i in range(0, perm.numel(), BATCH):
            b = perm[i:i + BATCH]
            xb = torch.from_numpy(
                np.roll(Xtr[b].numpy(), int(rng.integers(-SR, SR)), axis=-1)).to(dev)
            opt.zero_grad()
            lossf(model(xb), ytr[b].to(dev)).backward()
            opt.step()
        sched.step()


def predict(model, X, te, dev):
    model.eval()
    Xte = torch.from_numpy(X[te])
    out = []
    with torch.no_grad():
        for i in range(0, Xte.size(0), 64):
            lg = model(Xte[i:i + 64].to(dev))
            out.append(torch.softmax(lg, 1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def summarise(dataset):
    summary = {}
    for protocol in PROTOCOLS[dataset]:
        rows = [json.loads(f.read_text())
                for f in sorted(RESULTS.glob(f"{dataset}__{protocol}__*.json"))]
        if not rows:
            continue
        yy = np.concatenate([r["y"] for r in rows])
        pp = np.concatenate([r["p"] for r in rows])
        summary[protocol] = {
            "folds": len(rows), "n": int(yy.size),
            "pooled_auc": float(roc_auc_score(yy, pp)),
            "pooled_bal_acc": float(balanced_accuracy_score(yy, (pp >= 0.5).astype(int))),
        }
        per = [r["auc"] for r in rows if "auc" in r]
        if per:
            summary[protocol]["per_fold_auc_mean"] = float(np.mean(per))
            summary[protocol]["per_fold_auc_sd"] = float(np.std(per))
    return summary


def main(datasets=("broiler", "pullet", "pullet_control")):
    RESULTS.mkdir(parents=True, exist_ok=True)
    dev = device()
    all_summary = {}
    for ds in datasets:
        X, y, rec, week, _ = load_dataset(ds)
        if len(y) == 0:
            continue
        print(f"{ds}: clips {len(y)}  positive {(y == 1).sum()}  negative {(y == 0).sum()}  "
              f"recordings {len(set(rec))}  weeks {sorted(set(week))}", flush=True)
        for protocol in PROTOCOLS[ds]:
            for name, tr, te in folds_for(protocol, y, rec, week):
                out = RESULTS / f"{ds}__{protocol}__{name}.json"
                if out.exists():
                    continue
                model = FlockNet().to(dev)
                fit(model, X, y, tr, dev, seed=abs(hash((ds, protocol, name))) % 10_000)
                p = predict(model, X, te, dev)
                rec_ = {"dataset": ds, "protocol": protocol, "fold": name,
                        "n_train": int(len(tr)), "n_test": int(len(te)),
                        "test_classes": sorted(set(int(v) for v in y[te])),
                        "y": [int(v) for v in y[te]], "p": [float(v) for v in p]}
                if len(rec_["test_classes"]) == 2:
                    rec_["auc"] = float(roc_auc_score(y[te], p))
                out.write_text(json.dumps(rec_))
                print("%-8s %-10s %-8s n_test=%5d %s" %
                      (ds, protocol, name, len(te),
                       ("auc %.3f" % rec_["auc"]) if "auc" in rec_ else "single-class fold"),
                      flush=True)
        all_summary[ds] = summarise(ds)
    (ROOT / "results" / "protocol_summary.json").write_text(json.dumps(all_summary, indent=2))
    print(json.dumps(all_summary, indent=2))


if __name__ == "__main__":
    main(tuple(sys.argv[1:]) or ("broiler", "pullet", "pullet_control"))
