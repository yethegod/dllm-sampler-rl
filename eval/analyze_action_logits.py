"""Is the joint (block size, threshold) policy adaptive to the *problem*, or only to
the *generation progress*?

Realized actions cannot answer that. Eval samples from the policy
(``sampling_mode: categorical``), so two problems drawing different actions is equally
consistent with "the policy conditions on the problem" and with "one distribution drawn
twice". The action *distributions* can answer it, and ``eval/eval.py`` now records them
per decision (``action_logits``).

Group the decisions by their index within the rollout -- every problem's 1st block
together, every problem's 2nd block together, and so on -- and within each group
measure how much the distribution moves from problem to problem:

    I_k = H(mean_i p_ik) - mean_i H(p_ik)          [nats]

which is exactly the mutual information between the action at decision k and the
problem identity. Both terms are computed from exact distributions rather than
samples, so this is not an estimate and carries no finite-sample bias.

    I_k ~ 0    the policy runs the same distribution on every problem at decision k.
               Whatever it learned is a content-independent schedule, reproducible by
               a fixed action list (see ``--remasking block_schedule``).
    I_k >> 0   the policy reads the problem state.

The scale to read I_k against is ln(15) = 2.708 nats, the information in a uniform
choice over the 15-action space; the report gives the fraction.

Also reports, per run, the per-example E[R] under the training reward
``R = 1{correct} * ((L - NFE + 1)/L)^alpha``, so a schedule control can be ranked
directly against the learned policy and the best constant.

Usage:
    python -m eval.analyze_action_logits <results_dir> [<results_dir> ...] \
        [--alpha 0.5] [--gen_length 256] [--max_decisions 8]
"""
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from common.parsing.parse_and_get_acc import check_gsm_correct
from common.parsing.parse_and_get_acc import check_math_correct
from common.parsing.parse_and_get_acc import extract_gsm_answer
from common.parsing.parse_and_get_acc import extract_math_answer


def _softmax(logits):
    """Softmax over a list that may contain None for infeasible actions (-> 0)."""
    finite = [(i, v) for i, v in enumerate(logits) if v is not None]
    if not finite:
        raise ValueError("every action is infeasible")
    m = max(v for _, v in finite)
    exp = {i: math.exp(v - m) for i, v in finite}
    z = sum(exp.values())
    return np.array([exp.get(i, 0.0) / z for i in range(len(logits))])


def joint_distribution(entry):
    """(K_b * K_t,) joint action distribution for one decision.

    The two heads are conditionally independent given the state (policy.py), so the
    joint is their outer product.
    """
    pb = _softmax(entry["block"])
    pt = _softmax(entry["thres"])
    return np.outer(pb, pt).reshape(-1)


def entropy(p):
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())


def analyze_run(gen_file, alpha, gen_length, max_decisions):
    with open(gen_file) as f:
        data = json.load(f)

    dataset = Path(gen_file).name.split("_generations")[0]
    if dataset == "math":
        extract_fn, check_fn = extract_math_answer, check_math_correct
    else:
        extract_fn, check_fn = extract_gsm_answer, check_gsm_correct

    by_decision = defaultdict(list)
    rewards, accs, nfes = [], [], []
    n_with_logits = 0

    for item in data.get("generations", []):
        nfe = item.get("steps", 0)
        correct = bool(check_fn(extract_fn(item.get("generations", "")),
                                item.get("ground_truth")))
        compute = max(gen_length - nfe + 1, 0) / gen_length
        rewards.append((1.0 if correct else 0.0) * compute**alpha)
        accs.append(1.0 if correct else 0.0)
        nfes.append(nfe)

        logits = item.get("action_logits")
        if not logits:
            continue
        n_with_logits += 1
        for k, entry in enumerate(logits[:max_decisions]):
            by_decision[k].append(joint_distribution(entry))

    summary = {
        "run": str(Path(gen_file).parent.name),
        "dataset": dataset,
        "n": len(rewards),
        "n_with_logits": n_with_logits,
        "accuracy": 100 * float(np.mean(accs)) if accs else float("nan"),
        "nfe": float(np.mean(nfes)) if nfes else float("nan"),
        "E[R]": float(np.mean(rewards)) if rewards else float("nan"),
        "block_length": str(data.get("block_length")),
        "remasking": data.get("remasking"),
        "decisions": [],
    }

    for k in sorted(by_decision):
        P = np.stack(by_decision[k])  # (N, A)
        mean_p = P.mean(axis=0)
        h_mean = entropy(mean_p)
        mean_h = float(np.mean([entropy(p) for p in P]))
        summary["decisions"].append(
            {
                "k": k,
                "n": len(P),
                "H(mean p)": h_mean,
                "mean H(p)": mean_h,
                "I (nats)": h_mean - mean_h,
                "argmax": int(mean_p.argmax()),
                "top mass": float(mean_p.max()),
            }
        )
    return summary


def fmt(summaries, action_labels, ln_a):
    lines = []
    lines.append(
        f"{'run':<52} {'n':>5} {'acc%':>6} {'NFE':>6} {'E[R]':>7}  block_length"
    )
    lines.append("-" * 100)
    for s in summaries:
        lines.append(
            f"{s['run'][:52]:<52} {s['n']:>5} {s['accuracy']:>6.2f} "
            f"{s['nfe']:>6.1f} {s['E[R]']:>7.4f}  {s['block_length']}"
        )

    for s in summaries:
        if not s["decisions"]:
            continue
        lines.append("")
        lines.append(f"== {s['run']}  ({s['n_with_logits']} rollouts with logits) ==")
        lines.append(
            f"{'dec':>4} {'n':>6} {'H(mean p)':>10} {'mean H(p)':>10} "
            f"{'I (nats)':>9} {'I/ln|A|':>8}  most likely action"
        )
        for d in s["decisions"]:
            lines.append(
                f"{d['k']:>4} {d['n']:>6} {d['H(mean p)']:>10.3f} "
                f"{d['mean H(p)']:>10.3f} {d['I (nats)']:>9.4f} "
                f"{d['I (nats)'] / ln_a:>8.3f}  "
                f"{action_labels[d['argmax']]} p={d['top mass']:.2f}"
            )
        total = sum(d["I (nats)"] * d["n"] for d in s["decisions"])
        n_tot = sum(d["n"] for d in s["decisions"])
        lines.append(
            f"  decision-weighted mean I = {total / n_tot:.4f} nats "
            f"({total / n_tot / ln_a:.3f} of ln|A|)"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results_dirs", nargs="+")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--max_decisions", type=int, default=8)
    ap.add_argument("--block_size_candidates", default="8,16,32,64,128")
    ap.add_argument("--threshold_candidates", default="0.5,0.7,0.9")
    ap.add_argument("--output", default=None, help="also write the report here")
    args = ap.parse_args()

    cand_b = [int(b) for b in args.block_size_candidates.split(",")]
    cand_t = [float(t) for t in args.threshold_candidates.split(",")]
    action_labels = [f"(b={b}, t={t})" for b in cand_b for t in cand_t]
    ln_a = math.log(len(action_labels))

    files = []
    for d in args.results_dirs:
        files.extend(sorted(Path(d).glob("**/*_generations.json")))
    if not files:
        raise SystemExit(f"no *_generations.json under {args.results_dirs}")

    summaries = [
        analyze_run(f, args.alpha, args.gen_length, args.max_decisions) for f in files
    ]
    summaries.sort(key=lambda s: -s["E[R]"])

    report = (
        f"E[R] = 1{{correct}} * ((L - NFE + 1)/L)^alpha, "
        f"L={args.gen_length}, alpha={args.alpha}, per-example mean\n"
        f"|A| = {len(action_labels)}, ln|A| = {ln_a:.3f} nats\n\n"
        + fmt(summaries, action_labels, ln_a)
    )
    print(report)
    if args.output:
        Path(args.output).write_text(report + "\n")
        json.dump(
            summaries, open(Path(args.output).with_suffix(".json"), "w"), indent=2
        )
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
