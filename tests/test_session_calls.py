"""Pins the README section "The pullet network, session by session" to
results/session_calls.json (written by src/session_calls.py from the
out-of-fold predictions in results/cv/) and results/protocol_summary.json.
Recomputes what is cheap from results/cv/ itself.  No audio is needed.
"""

import json
import os
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
README = open(os.path.join(ROOT, "README.md"), encoding="utf-8").read()
FLAT = " ".join(README.split())
SC = json.load(open(os.path.join(ROOT, "results", "session_calls.json")))["datasets"]
PS = json.load(open(os.path.join(ROOT, "results", "protocol_summary.json")))

ROWS = [("pullet", "session"), ("pullet", "week"),
        ("pullet_control", "session"), ("pullet_control", "week")]
LABEL = {"pullet": "stressed cages", "pullet_control": "control cage"}


def r(x, nd=2):
    q = Decimal(1).scaleb(-nd)
    return str(Decimal(repr(float(x))).quantize(q, rounding=ROUND_HALF_UP))


def signed(x, nd=3):
    s = r(x, nd)
    return s if s.startswith("-") else "+" + s


def summary_row(ds, pr):
    c = SC[ds][pr]
    e = c["exact_within_pair_test"]
    like = (f"{c['of_those_scored_like_partner_label']} of {c['sessions_with_partner_in_training']}"
            if c["sessions_with_partner_in_training"] else "—")
    p = c["paired"]
    up = f"{p['after_above_before']} of {p['n_pairs']}" + (f" ({p['ties']} tie)" if p["ties"] else "")
    pexact = e["p_auc_at_most_observed"] if c["pooled_auc_clips"] < 0.5 else e["p_auc_at_least_observed"]
    side = "below" if c["pooled_auc_clips"] < 0.5 else "above"
    return (f"| {LABEL[ds]} · {pr} | {c['n_sessions']} | {r(c['pooled_auc_clips'])} | "
            f"{r(c['session_level_auc'])} | {c['sessions_called_right_by_majority']} of {c['n_sessions']} | "
            f"{like} | {up} | {r(pexact, 3)} ({side}) |")


def paired_row(ds, pr):
    p = SC[ds][pr]["paired"]
    lo, hi = p["pair_bootstrap_ci95_mean_diff"]
    return (f"| {LABEL[ds]} · {pr} | {p['after_above_before']} / {p['after_below_before']} / {p['ties']} | "
            f"{r(p['sign_test_p_one_sided'], 3)} | {signed(p['mean_diff'])} [{signed(lo)}, {signed(hi)}] | "
            f"{signed(p['median_diff'])} | {r(p['wilcoxon_p_two_sided'])} |")


def control_row(s):
    name = s["session"].split("/", 2)[2]           # W4/D1/post
    return (f"| {name} | {'after' if s['label'] else 'before'} | {s['n_clips']} | {r(s['mean_p'], 3)} | "
            f"{r(100 * s['share_called_after'], 0)} % | {'right' if s['majority_right'] else 'wrong'} |")


def pair_row(d):
    name = d["cage_day"].split("/", 1)[1]           # trt1/W1/D2
    return (f"| {name} | {r(d['before_mean_p'], 3)} | {r(d['after_mean_p'], 3)} | "
            f"{signed(d['diff'])} |")


def test_pooled_aucs_are_the_published_ones():
    for ds, pr in ROWS:
        c = SC[ds][pr]
        assert abs(c["pooled_auc_clips"] - PS[ds][pr]["pooled_auc"]) < 1e-9
        assert abs(c["pooled_auc_stored"] - PS[ds][pr]["pooled_auc"]) < 1e-12
        assert c["n_clips"] == PS[ds][pr]["n"]
        assert c["n_sessions"] == PS[ds][pr]["n_sessions"]


def test_pooled_auc_recomputed_from_the_cv_files():
    """Seed-averaged, fold-pooled, exactly as train.summarise does it."""
    import glob
    from sklearn.metrics import roc_auc_score
    for ds, pr in ROWS:
        files = sorted(glob.glob(os.path.join(ROOT, "results", "cv", f"{ds}__{pr}__*.json")))
        assert files, (ds, pr)
        by_fold = {}
        for f in files:
            d = json.load(open(f))
            by_fold.setdefault(d["fold"], []).append(d)
        yy, pp = [], []
        for fold in sorted(by_fold):
            rs = by_fold[fold]
            assert len(rs) == 3
            yy.append(np.asarray(rs[0]["y"]))
            pp.append(np.mean([np.asarray(x["p"]) for x in rs], axis=0))
        auc = roc_auc_score(np.concatenate(yy), np.concatenate(pp))
        assert abs(auc - PS[ds][pr]["pooled_auc"]) < 1e-9, (ds, pr, auc)


def test_summary_table_rows():
    for ds, pr in ROWS:
        assert summary_row(ds, pr) in README, summary_row(ds, pr)


def test_paired_table_rows():
    for ds, pr in ROWS:
        assert paired_row(ds, pr) in README, paired_row(ds, pr)


def test_control_cage_learned_the_day():
    c = SC["pullet_control"]["session"]
    assert c["n_sessions"] == 8 and c["n_pairs"] == 4
    assert c["sessions_with_partner_in_training"] == 8
    assert c["of_those_scored_like_partner_label"] == 7
    assert c["sessions_called_right_by_majority"] == 1
    assert c["session_level_auc"] == 0.0
    assert c["pairs_after_above_before"] == 0
    e = c["exact_within_pair_test"]
    assert e["n_relabellings"] == 16
    assert abs(e["auc_relabelled_min"] - c["pooled_auc_clips"]) < 1e-12   # the floor
    assert e["p_auc_at_most_observed"] == 1 / 16
    assert c["sessions_wrong_on_over_80pct"] == 4
    for s in c["sessions"]:
        assert control_row(s) in README, control_row(s)
    p = c["paired"]
    assert r(p["mean_diff"], 3) == "-0.218" and p["pair_bootstrap_ci95_mean_diff"][1] < 0
    assert "7 of the 8" in FLAT and "every \"after\" hour below every \"before\" hour" in FLAT


def test_stressed_cages_are_a_coin_under_the_session_protocol():
    c = SC["pullet"]["session"]
    assert c["n_sessions"] == 34 and c["n_pairs"] == 16
    assert c["sessions_without_partner"] == ["pullet/trt1/W1/D1/pre", "pullet/trt2/W2/D2/post"]
    assert c["sessions_with_partner_in_training"] == 30
    assert c["of_those_scored_like_partner_label"] == 16
    assert c["of_those_called_right"] == 14
    assert c["sessions_called_right_by_majority"] == 16
    assert r(c["session_level_auc"]) == "0.48"
    assert c["pairs_after_above_before"] == 6
    e = c["exact_within_pair_test"]
    assert e["n_relabellings"] == 65536
    assert r(e["p_auc_at_least_observed"]) == "0.66"
    assert r(e["auc_relabelled_min"]) == "0.24" and r(e["auc_relabelled_max"]) == "0.77"
    assert "16 of the 30" in FLAT and "0.24 to 0.77" in FLAT
    # the training prior barely moves between folds: not the beehive mechanism
    assert c["train_share_post_min"] > 0.45 and c["train_share_post_max"] < 0.55


def test_held_out_week_direction_and_its_size():
    c = SC["pullet"]["week"]
    p = c["paired"]
    assert c["sessions_with_partner_in_training"] == 0        # whole weeks held out
    assert (p["after_above_before"], p["after_below_before"], p["ties"]) == (12, 3, 1)
    assert r(p["sign_test_p_one_sided"], 3) == "0.018"
    assert r(p["median_diff"], 3) == "0.031" and r(p["mean_diff"], 3) == "0.016"
    lo, hi = p["pair_bootstrap_ci95_mean_diff"]
    assert lo < 0 < hi
    assert r(c["exact_within_pair_test"]["p_auc_at_least_observed"], 3) == "0.118"
    assert r(c["session_level_auc"]) == "0.59"
    # the control cage moves the same way by the same amount with no stressor
    q = SC["pullet_control"]["week"]["paired"]
    assert (q["after_above_before"], q["after_below_before"]) == (3, 1)
    assert r(q["mean_diff"], 3) == "0.030" and r(q["median_diff"], 3) == "0.038"
    assert q["pair_bootstrap_ci95_mean_diff"][0] < 0
    for d in c["pair_differences_after_minus_before"]:
        assert pair_row(d) in README, pair_row(d)
    assert "12 of the 16" in FLAT and "p = 0.018" in FLAT and "+0.016" in FLAT


def test_cv_files_are_published():
    import glob
    files = glob.glob(os.path.join(ROOT, "results", "cv", "*.json"))
    assert len(files) == 111
    assert "111 files" in FLAT


if __name__ == "__main__":
    for ds, pr in ROWS:
        print(summary_row(ds, pr))
    print()
    for ds, pr in ROWS:
        print(paired_row(ds, pr))
    print()
    for s in SC["pullet_control"]["session"]["sessions"]:
        print(control_row(s))
    print()
    for d in SC["pullet"]["week"]["pair_differences_after_minus_before"]:
        print(pair_row(d))
    c = SC["pullet"]["session"]
    print(c["train_share_post_min"], c["train_share_post_max"], c["wrong_side_share_median"])
    print(SC["pullet_control"]["session"]["paired"], SC["pullet"]["week"]["exact_within_pair_test"])
