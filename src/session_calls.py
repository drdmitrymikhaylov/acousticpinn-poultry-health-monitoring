"""The pullet network read session by session, and the stressor tested
against the day it happened on.

Finding 2 says the control cage scores below chance on held-out sessions
(0.19) "because the network learns which day a clip is from": a held-out
hour is scored by its nearest training hour, which is the other half of
the same day with the opposite label.  That is a mechanism stated from one
pooled AUC.  This script reads it off the stored out-of-fold predictions
(results/cv/, the same files train.summarise pools):

  1. per session (cage-hour): the seed-averaged mean probability, the
     majority call over its clips, whether the call is right, and whether
     the session's partner -- the other half of the same cage-day -- was in
     the training set of the fold that scored it;
  2. the session-level AUC (sessions ranked by mean probability) next to
     the clip-level one the page reports;
  3. an exact permutation test that respects the day: every "before" /
     "after" pair is swapped or not, all 2^(number of pairs) relabellings
     enumerated, and the AUC recomputed under each.  The page's permutation
     shuffles labels across sessions and so also across days; this one asks
     only whether the stressor adds anything to the day.

Reproduces the pooled AUCs of results/protocol_summary.json to the digit
before anything else is computed.  Writes results/session_calls.json.
Needs no audio.  Run from the repository root:

    python src/session_calls.py
"""

import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import binom, rankdata, wilcoxon
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
CV = ROOT / "results" / "cv"
DATASETS = {"pullet": ("session", "week"), "pullet_control": ("session", "week")}


def partner(session):
    """The other half of the same cage-day: .../pre <-> .../post."""
    head, tail = session.rsplit("/", 1)
    return f"{head}/{'post' if tail == 'pre' else 'pre'}"


def pooled(dataset, protocol):
    """Seed-averaged out-of-fold predictions, fold by fold, as train.summarise."""
    rows = [json.loads(f.read_text())
            for f in sorted(CV.glob(f"{dataset}__{protocol}__*.json"))]
    by_fold = {}
    for r in rows:
        by_fold.setdefault(r["fold"], []).append(r)
    folds = []
    for fold in sorted(by_fold):
        rs = sorted(by_fold[fold], key=lambda r: r["seed"])
        y = np.asarray(rs[0]["y"])
        g = np.asarray(rs[0]["groups"])
        for r in rs[1:]:
            assert r["y"] == rs[0]["y"] and r["groups"] == rs[0]["groups"], fold
        p = np.mean([np.asarray(r["p"]) for r in rs], axis=0)
        folds.append((fold, y, p, g, rs[0]["n_train"]))
    return folds


def rank_auc_parts(p, groups, sessions):
    """Per-session rank sums, so the AUC of any relabelling is a dot product."""
    r = rankdata(p)
    S = np.array([r[groups == s].sum() for s in sessions])
    n = np.array([(groups == s).sum() for s in sessions])
    return S, n


def exact_pair_test(p, groups, label_of, pairs, singles, observed):
    """All 2^k swaps of before/after within cage-days; AUC under each.  A
    session without a partner (its other half was not released) keeps its
    label in every relabelling."""
    groups = np.asarray(groups)
    sessions = [s for pr in pairs for s in pr]
    S, n = rank_auc_parts(p, groups, sessions)
    S = S.reshape(-1, 2); n = n.reshape(-1, 2)          # [pair, (pre, post)]
    S_fix, n_fix = rank_auc_parts(p, groups, singles) if singles else (np.zeros(0), np.zeros(0, int))
    pos_fix = np.array([label_of[s] == 1 for s in singles], dtype=bool)
    k = len(pairs)
    bits = np.array(list(itertools.product([0, 1], repeat=k)), dtype=bool)
    # bit 0: post is positive (the real labels); bit 1: pre is positive
    sum_r = np.where(bits, S[:, 0], S[:, 1]).sum(axis=1) + S_fix[pos_fix].sum()
    n1 = np.where(bits, n[:, 0], n[:, 1]).sum(axis=1) + n_fix[pos_fix].sum()
    n0 = len(p) - n1
    auc = (sum_r - n1 * (n1 + 1) / 2) / (n1 * n0)
    real = auc[0]
    assert abs(real - observed) < 1e-9, (real, observed)
    return {
        "n_pairs": k, "n_relabellings": int(len(auc)),
        "auc_observed": float(real),
        "p_auc_at_least_observed": float((auc >= real - 1e-12).mean()),
        "p_auc_at_most_observed": float((auc <= real + 1e-12).mean()),
        "auc_relabelled_min": float(auc.min()), "auc_relabelled_max": float(auc.max()),
        "auc_relabelled_median": float(np.median(auc)),
        "rank_of_observed_from_top": int((auc > real + 1e-12).sum() + 1),
    }


def session_level_auc(rows):
    y = np.array([r["label"] for r in rows]); m = np.array([r["mean_p"] for r in rows])
    return float(roc_auc_score(y, m)) if len(set(y)) > 1 else None


def main():
    summary = json.loads((ROOT / "results" / "protocol_summary.json").read_text())
    out = {"source": "results/cv", "datasets": {}}
    for dataset, protocols in DATASETS.items():
        out["datasets"][dataset] = {}
        for protocol in protocols:
            folds = pooled(dataset, protocol)
            yy = np.concatenate([f[1] for f in folds])
            pp = np.concatenate([f[2] for f in folds])
            gg = np.concatenate([f[3] for f in folds])
            auc = float(roc_auc_score(yy, pp))
            stored = summary[dataset][protocol]["pooled_auc"]
            assert abs(auc - stored) < 1e-9, (dataset, protocol, auc, stored)

            label_of = {}
            for s in sorted(set(gg)):
                labs = set(yy[gg == s].tolist())
                assert len(labs) == 1, s
                label_of[s] = int(labs.pop())
            all_sessions = sorted(label_of)
            singles = [s for s in all_sessions if partner(s) not in label_of]
            pairs = sorted({tuple(sorted((s, partner(s)), key=lambda x: x.endswith("post")))
                            for s in all_sessions if partner(s) in label_of})
            assert all(label_of[a] == 0 and label_of[b] == 1 for a, b in pairs)
            assert 2 * len(pairs) + len(singles) == len(all_sessions)

            n_post_total = int(yy.sum())
            rows = []
            fold_rows = []
            for fold, y, p, g, n_train in folds:
                test_sessions = sorted(set(g))
                n_post_test = int(y.sum())
                prior_train = (n_post_total - n_post_test) / n_train
                fold_rows.append({"fold": fold, "n_train": int(n_train), "n_test": int(len(y)),
                                  "test_sessions": test_sessions,
                                  "train_share_post": float(prior_train),
                                  "test_share_post": float(y.mean())})
                for s in test_sessions:
                    m = g == s
                    ps = p[m]
                    lab = label_of[s]
                    has_partner = partner(s) in label_of
                    partner_held_out = partner(s) in test_sessions
                    rows.append({
                        "session": s, "fold": fold, "label": lab, "n_clips": int(m.sum()),
                        "mean_p": float(ps.mean()), "median_p": float(np.median(ps)),
                        "share_called_after": float((ps >= 0.5).mean()),
                        "majority_call": int((ps >= 0.5).mean() >= 0.5),
                        "majority_right": bool(int((ps >= 0.5).mean() >= 0.5) == lab),
                        "mean_right": bool((ps.mean() >= 0.5) == bool(lab)),
                        "partner": partner(s) if has_partner else None,
                        "partner_in_training": (not partner_held_out) if has_partner else None,
                        "train_share_post": float(prior_train),
                    })
            rows.sort(key=lambda r: r["session"])

            n = len(rows)
            right = sum(r["majority_right"] for r in rows)
            with_partner = [r for r in rows if r["partner_in_training"] is True]
            without = [r for r in rows if r["partner_in_training"] is False]
            no_partner = [r for r in rows if r["partner_in_training"] is None]
            # how firmly: share of clips on the wrong side
            wrong_share = [1 - r["share_called_after"] if r["label"] else r["share_called_after"]
                           for r in rows]
            # when the partner was in training: is the held-out hour scored
            # like its partner's label (i.e. the wrong way)?
            like_partner = sum((r["mean_p"] >= 0.5) != bool(r["label"]) for r in with_partner)

            res = {
                "n_sessions": n, "n_pairs": len(pairs), "n_clips": int(len(yy)),
                "sessions_without_partner": [r["session"] for r in no_partner],
                "of_those_called_right__": int(sum(r["majority_right"] for r in no_partner)),
                "pooled_auc_clips": auc,
                "pooled_auc_stored": stored,
                "session_level_auc": session_level_auc(rows),
                "sessions_called_right_by_majority": int(right),
                "sessions_called_right_by_mean": int(sum(r["mean_right"] for r in rows)),
                "sessions_with_partner_in_training": len(with_partner),
                "of_those_called_right": int(sum(r["majority_right"] for r in with_partner)),
                "of_those_scored_like_partner_label": int(like_partner),
                "sessions_with_partner_held_out_too": len(without),
                "of_those_called_right_": int(sum(r["majority_right"] for r in without)),
                "wrong_side_share_median": float(np.median(wrong_share)),
                "wrong_side_share_min": float(min(wrong_share)),
                "wrong_side_share_max": float(max(wrong_share)),
                "sessions_wrong_on_over_80pct": int(sum(w > 0.8 for w in wrong_share)),
                "train_share_post_min": float(min(f["train_share_post"] for f in fold_rows)),
                "train_share_post_max": float(max(f["train_share_post"] for f in fold_rows)),
                "folds": fold_rows,
                "sessions": rows,
            }
            if len(pairs) <= 20:
                res["exact_within_pair_test"] = exact_pair_test(pp, gg, label_of, pairs, singles, auc)
                # the same test on the session-level AUC: one score per session
                ms = np.array([r["mean_p"] for r in rows]); gs = np.array([r["session"] for r in rows])
                sl = session_level_auc(rows)
                res["exact_within_pair_test_session_level"] = exact_pair_test(ms, gs, label_of, pairs, singles, sl)
            # per-pair view: did the network put "after" above "before" on the same day?
            mp = {r["session"]: r["mean_p"] for r in rows}
            diffs = np.array([mp[b] - mp[a] for a, b in pairs])
            TIE = 1e-3                      # a difference below 0.001 in mean probability is a tie
            k_up = int((diffs > TIE).sum())
            k_down = int((diffs < -TIE).sum())
            n_eff = k_up + k_down
            rng = np.random.default_rng(0)
            boot = [diffs[rng.integers(0, len(diffs), len(diffs))].mean() for _ in range(10000)]
            w = wilcoxon(diffs) if len(diffs) >= 4 else None
            res["pairs_after_above_before"] = k_up
            res["paired"] = {
                "n_pairs": len(pairs),
                "after_above_before": k_up, "after_below_before": k_down,
                "ties": int(len(pairs) - n_eff), "tie_threshold": TIE,
                # exact sign test on the untied pairs: P(at least k_up of n_eff the "after" way | coin)
                "sign_test_p_one_sided": float(binom.sf(k_up - 1, n_eff, 0.5)),
                "sign_test_p_two_sided": float(min(1.0, 2 * min(binom.sf(k_up - 1, n_eff, 0.5),
                                                                binom.cdf(k_up, n_eff, 0.5)))),
                "mean_diff": float(diffs.mean()), "median_diff": float(np.median(diffs)),
                "diff_min": float(diffs.min()), "diff_max": float(diffs.max()),
                "pair_bootstrap_ci95_mean_diff": [float(np.percentile(boot, 2.5)),
                                                  float(np.percentile(boot, 97.5))],
                "wilcoxon_p_two_sided": float(w.pvalue) if w else None,
            }
            res["pair_differences_after_minus_before"] = [
                {"cage_day": a.rsplit("/", 1)[0], "diff": float(mp[b] - mp[a]),
                 "before_mean_p": float(mp[a]), "after_mean_p": float(mp[b])} for a, b in pairs]
            out["datasets"][dataset][protocol] = res

    (ROOT / "results" / "session_calls.json").write_text(json.dumps(out, indent=1))

    for dataset, protos in out["datasets"].items():
        for protocol, r in protos.items():
            print(f"\n{dataset} / {protocol}: {r['n_sessions']} sessions, {r['n_pairs']} pairs, "
                  f"clip AUC {r['pooled_auc_clips']:.4f} (stored {r['pooled_auc_stored']:.4f}), "
                  f"session-level AUC {r['session_level_auc']:.3f}")
            print(f"  right by majority {r['sessions_called_right_by_majority']}/{r['n_sessions']}; "
                  f"partner in training {r['sessions_with_partner_in_training']} "
                  f"(right {r['of_those_called_right']}, scored like partner {r['of_those_scored_like_partner_label']}); "
                  f"partner held out too {r['sessions_with_partner_held_out_too']} (right {r['of_those_called_right_']})")
            print(f"  wrong-side share median {r['wrong_side_share_median']:.2f} "
                  f"[{r['wrong_side_share_min']:.2f}, {r['wrong_side_share_max']:.2f}], "
                  f">80% wrong on {r['sessions_wrong_on_over_80pct']}; train share post "
                  f"{r['train_share_post_min']:.3f}-{r['train_share_post_max']:.3f}; "
                  f"after above before in {r['pairs_after_above_before']}/{r['n_pairs']} pairs")
            q = r["paired"]
            print(f"  paired: up {q['after_above_before']} down {q['after_below_before']} ties {q['ties']}; sign p1 {q['sign_test_p_one_sided']:.4f} p2 {q['sign_test_p_two_sided']:.4f}, "
                  f"mean diff {q['mean_diff']:+.3f} [{q['pair_bootstrap_ci95_mean_diff'][0]:+.3f}, "
                  f"{q['pair_bootstrap_ci95_mean_diff'][1]:+.3f}], median {q['median_diff']:+.3f}, "
                  f"range [{q['diff_min']:+.3f}, {q['diff_max']:+.3f}], wilcoxon p {q['wilcoxon_p_two_sided']}")
            for d in r["pair_differences_after_minus_before"]:
                print(f"      {d['cage_day']:28s} before {d['before_mean_p']:.3f} after {d['after_mean_p']:.3f} diff {d['diff']:+.3f}")
            if "exact_within_pair_test" in r:
                e = r["exact_within_pair_test"]; s = r["exact_within_pair_test_session_level"]
                print(f"  exact within-pair: {e['n_relabellings']} relabellings, p(>=) {e['p_auc_at_least_observed']:.4f}, "
                      f"p(<=) {e['p_auc_at_most_observed']:.4f}, range [{e['auc_relabelled_min']:.3f}, {e['auc_relabelled_max']:.3f}], "
                      f"median {e['auc_relabelled_median']:.3f}; session-level p(>=) {s['p_auc_at_least_observed']:.4f} "
                      f"p(<=) {s['p_auc_at_most_observed']:.4f}")
            for row in r["sessions"]:
                print(f"    {row['session']:34s} y={row['label']} n={row['n_clips']:3d} mean_p={row['mean_p']:.3f} "
                      f"after={row['share_called_after']:.2f} {'OK ' if row['majority_right'] else 'WRONG'} "
                      f"partner_in_train={row['partner_in_training']} fold={row['fold']}")


if __name__ == "__main__":
    main()
