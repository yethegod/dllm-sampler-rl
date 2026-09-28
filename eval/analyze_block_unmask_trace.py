"""What does the block_unmask policy actually do? Reads --record_policy_trace generations.

Usage:
    python -m eval.analyze_block_unmask_trace <generations.json> [<more.json> ...] [--thres 0.7]

Two questions, one section each:

Block head -- which block size is chosen, where in the answer, how sure the head is, and
whether questions that end up correct used different blocks.

Unmask head -- is the committed set just "the most confident masked positions"? Per forward
it compares the DPLS picks with (a) the same number of positions taken by dLLM confidence and
(b) the Fast-dLLM rule at --thres, and fits pick ~ confidence, pick ~ position and both,
reporting held-out AUC. If confidence alone reaches the AUC of the full fit, the unmask
head is an adaptive-count confidence ranker; any gain from position is left-to-right bias.
"""

import argparse
import json
import os
import sys
from collections import Counter
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train.reward_func import _process_answers_gsm8k  # noqa: E402
from train.reward_func import extract_gsm_answer  # noqa: E402

CANDIDATES = (8, 16, 32, 48, 64, 96, 128)


def _correct(g) -> int:
    try:
        return int(
            _process_answers_gsm8k(
                [extract_gsm_answer(g["generations"])], [str(g["ground_truth"])], 1.0
            )[0]
            > 0
        )
    except Exception:
        return 0


def _softmax(x):
    # Candidates that overrun the sequence end are masked and stored as None.
    x = np.asarray([v for v in x if v is not None], dtype=np.float64)
    e = np.exp(x - x.max())
    return e / e.sum()


def _auc(score, label):
    """Mann-Whitney AUC with average ranks for ties."""
    score = np.asarray(score, dtype=np.float64)
    label = np.asarray(label, dtype=bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    s = score[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return (ranks[label].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def _q(a, qs=(10, 50, 90)):
    a = np.asarray(a, dtype=np.float64)
    return "/".join(f"{np.percentile(a, q):.2f}" for q in qs) if len(a) else "-"


def block_section(gens, correct):
    print("\n## Block head")
    chosen, ent, top_p, pos_b = [], [], [], defaultdict(list)
    free = []  # decisions where every candidate still fit before the sequence end
    per_q_mean = []
    for g in gens:
        bs = g.get("block_sizes") or []
        per_q_mean.append(np.mean(bs) if bs else np.nan)
        start = 0
        for k, b in enumerate(bs):
            chosen.append(b)
            pos_b[min(start // 32, 7)].append(b)
            start += b
            al = g.get("action_logits") or []
            if k < len(al) and al[k] and "block" in al[k]:
                if None not in al[k]["block"]:
                    free.append(b)
                p = _softmax(al[k]["block"])
                ent.append(float(-(p * np.log(p + 1e-12)).sum()))
                top_p.append(float(p.max()))
    c = Counter(chosen)
    n = len(chosen)
    print(f"decisions: {n} ({n / len(gens):.1f} per question)")
    print("chosen b:  " + "  ".join(f"{b}:{c.get(b, 0) / n:.3f}" for b in CANDIDATES))
    if ent:
        print(
            f"head entropy (nats, ln7=1.95) p10/p50/p90: {_q(ent)};  max prob p10/p50/p90: {_q(top_p)}"
        )
    cf = Counter(free)
    print(f"  unconstrained decisions only (n={len(free)}): "
          + "  ".join(f"{b}:{cf.get(b, 0) / max(len(free), 1):.3f}" for b in CANDIDATES))
    print("mean b by block start (32-token bins):")
    for k in sorted(pos_b):
        v = pos_b[k]
        print(f"  start {k * 32:3d}-{k * 32 + 31:3d}: n={len(v):5d} mean b={np.mean(v):5.1f}  "
              + " ".join(f"{b}:{v.count(b) / len(v):.2f}" for b in CANDIDATES))
    m = np.array(per_q_mean)
    y = np.array(correct, dtype=bool)
    ok = ~np.isnan(m)
    print(
        f"per-question mean b: correct {np.nanmean(m[y]):.1f} vs wrong {np.nanmean(m[~y]):.1f};"
        f" AUC(mean b -> correct) {_auc(m[ok], y[ok]):.3f}"
    )
    # Does the first decision (made before any answer text exists) vary by question?
    first = [g["block_sizes"][0] for g in gens if g.get("block_sizes")]
    print("first decision: " + "  ".join(f"{b}:{first.count(b) / len(first):.2f}" for b in CANDIDATES))


def unmask_section(gens, correct, thres):
    print("\n## Unmask head")
    X, Y, grp = [], [], []  # features per (forward, masked position), pick label, question id
    n_pick, n_cand, jac_conf, jac_fd, n_fd, logit_conf_rho = [], [], [], [], [], []
    left_to_right = []
    for qi, g in enumerate(gens):
        tr = g.get("policy_trace")
        if not tr:
            continue
        for r in tr:
            pos = np.array(r["pos"])
            conf = np.array(r["conf"])
            logit = np.array(r["logit"])
            pick = set(r["pick"])
            k = len(pick)
            if len(pos) == 0 or k == 0:
                continue
            n_pick.append(k)
            n_cand.append(len(pos))
            by_conf = set(pos[np.argsort(-conf, kind="stable")[:k]].tolist())
            jac_conf.append(len(by_conf & pick) / len(by_conf | pick))
            fd = set(pos[conf > thres].tolist()) or {int(pos[np.argmax(conf)])}
            n_fd.append(len(fd))
            jac_fd.append(len(fd & pick) / len(fd | pick))
            if len(pos) >= 3:
                rc = np.argsort(np.argsort(conf))
                rl = np.argsort(np.argsort(logit))
                logit_conf_rho.append(np.corrcoef(rc, rl)[0, 1])
            # left-to-right: are the picks exactly the leftmost k masked positions?
            left_to_right.append(set(np.sort(pos)[:k].tolist()) == pick)
            width = max(r["be"] - r["bs"], 1)
            first_masked = pos.min()
            for p, cf in zip(pos, conf):
                X.append([cf, np.log(cf + 1e-6), (p - r["bs"]) / width, p - first_masked, p / 256.0])
                Y.append(p in pick)
                grp.append(qi)
    if not n_pick:
        print("no policy_trace found (run eval with --record_policy_trace)")
        return
    n_pick, n_cand = np.array(n_pick), np.array(n_cand)
    print(f"forwards with a pick: {len(n_pick)}")
    print(f"committed per forward p10/p50/p90: {_q(n_pick)} (mean {n_pick.mean():.2f});"
          f" masked in window: {_q(n_cand)}")
    print(f"Fast-dLLM t{thres} on the same confidences would commit: {_q(n_fd)} (mean {np.mean(n_fd):.2f})")
    print(f"Jaccard(picks, top-k by confidence, same k): mean {np.mean(jac_conf):.3f}, "
          f"exact match {np.mean(np.array(jac_conf) == 1):.3f}")
    print(f"Jaccard(picks, Fast-dLLM t{thres} set):      mean {np.mean(jac_fd):.3f}, "
          f"exact match {np.mean(np.array(jac_fd) == 1):.3f}")
    print(f"picks == leftmost k masked positions: {np.mean(left_to_right):.3f}")
    print(f"within-forward Spearman(logit, conf), windows >=3 masked: p10/p50/p90 {_q(logit_conf_rho)}")

    X, Y, grp = np.array(X), np.array(Y), np.array(grp)
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("(sklearn missing: skipping the pick ~ features fits)")
        return
    rng = np.random.default_rng(0)
    qs = np.unique(grp)
    test_q = set(rng.choice(qs, size=max(1, len(qs) // 4), replace=False).tolist())
    te = np.array([q in test_q for q in grp])
    feats = {
        "confidence": [0, 1],
        "position (rel. in window, dist. from first masked, abs.)": [2, 3, 4],
        "confidence + position": [0, 1, 2, 3, 4],
    }
    print(f"held-out AUC for pick ~ features ({(~te).sum()} train / {te.sum()} test positions, split by question):")
    for name, cols in feats.items():
        m = LogisticRegression(max_iter=2000).fit(X[~te][:, cols], Y[~te])
        print(f"  {name:58s} {_auc(m.predict_proba(X[te][:, cols])[:, 1], Y[te]):.3f}")

    # Accuracy split: do wrong answers commit more per forward?
    per_q = defaultdict(list)
    for qi, g in enumerate(gens):
        tr = g.get("policy_trace") or []
        ks = [len(r["pick"]) for r in tr if r["pick"]]
        if ks:
            per_q[bool(correct[qi])].append(np.mean(ks))
    print(f"mean committed/forward per question: correct {np.mean(per_q[True]):.2f} "
          f"vs wrong {np.mean(per_q[False]):.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--thres", type=float, default=0.7)
    args = ap.parse_args()
    for f in args.files:
        d = json.load(open(f))
        gens = d["generations"]
        correct = [_correct(g) for g in gens]
        steps = [g["steps"] for g in gens]
        print(f"# {f}")
        print(f"block_sampling_mode={d.get('block_sampling_mode')} n={len(gens)} acc={100 * np.mean(correct):.2f} NFE={np.mean(steps):.1f}")
        block_section(gens, correct)
        unmask_section(gens, correct, args.thres)
        print()


if __name__ == "__main__":
    main()
