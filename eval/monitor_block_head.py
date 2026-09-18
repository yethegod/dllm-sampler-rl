#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
"""Watch the block head of a block_unmask_policy run straight from TensorBoard.

Reads the trainer's scalars (no checkpoint is loaded) and prints, per bucket of
global steps, the correctness reward next to the block-size marginal, so the
question "does the block head move, and does accuracy move with it?" can be
answered while the job is still running. Written for the alpha=0 run
(llada8b_block_unmask_cond_v2_a0), where rewards/mixed_correctness_reward_func
IS the training accuracy, but works on any run that logs block_size/frac_*.

    python -m eval.monitor_block_head /work/hdd/bhta/zsun9/checkpoints/llada8b_block_unmask_cond_v2_a0
    python -m eval.monitor_block_head <run_dir> --bucket 100 --last 600

Two block-size-vs-accuracy numbers are printed, both over per-step scalars:
  - level corr: Pearson between block_size/mean and acc across steps. Both drift
    with training time (the unmask head is learning too), so a large value here
    can be pure co-trending.
  - diff corr: the same on first differences (step-to-step changes), which
    removes the shared slow trend. This is the one to read. Near 0 means the
    rollouts that happened to draw larger blocks were no more or less correct.
Neither is a within-question test (the trainer does not dump per-rollout
(b, correct) pairs); the group advantage the block head trains on is that
within-question signal, and its integral is block_head/bias_*, printed last.
"""

import argparse
import glob
import os

import numpy as np


def load_scalars(run_dir):
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,
    )

    event_dirs = sorted(glob.glob(os.path.join(run_dir, "runs", "*")))
    if not event_dirs:
        event_dirs = [run_dir]
    series = {}
    # Several event dirs exist after a resume; later ones override earlier steps.
    for d in event_dirs:
        ea = EventAccumulator(d, size_guidance={"scalars": 0})
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for ev in ea.Scalars(tag):
                series.setdefault(tag, {})[ev.step] = ev.value
    out = {}
    for tag, by_step in series.items():
        steps = np.array(sorted(by_step))
        out[tag] = (steps, np.array([by_step[s] for s in steps]))
    return out


def aligned(scalars, tags):
    """Steps present in every tag, and each tag's values at those steps."""
    common = None
    for t in tags:
        if t not in scalars:
            return None, {}
        s = set(scalars[t][0].tolist())
        common = s if common is None else common & s
    steps = np.array(sorted(common))
    vals = {}
    for t in tags:
        st, v = scalars[t]
        idx = {int(a): i for i, a in enumerate(st)}
        vals[t] = np.array([v[idx[int(s)]] for s in steps])
    return steps, vals


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", help="output_dir of the run (contains runs/)")
    ap.add_argument("--bucket", type=int, default=50, help="global steps per row")
    ap.add_argument("--last", type=int, default=None, help="only the last N steps")
    args = ap.parse_args()

    sc = load_scalars(args.run_dir)
    p = "train/"
    frac_tags = sorted(
        (t for t in sc if t.startswith(p + "block_size/frac_")),
        key=lambda t: int(t.rsplit("_", 1)[1]),
    )
    if not frac_tags:
        raise SystemExit(f"no block_size/frac_* scalars under {args.run_dir}")
    sizes = [int(t.rsplit("_", 1)[1]) for t in frac_tags]
    acc_tag = p + "rewards/mixed_correctness_reward_func"
    core = [acc_tag, p + "block_size/mean", p + "block_entropy", p + "num_steps_mean",
            p + "mean_unmask_prob"] + frac_tags
    steps, v = aligned(sc, core)
    if steps is None:
        raise SystemExit(f"missing one of {core}")
    if args.last:
        keep = steps >= steps[-1] - args.last
        steps = steps[keep]
        v = {t: x[keep] for t, x in v.items()}
    acc, bmean, hb = v[acc_tag], v[p + "block_size/mean"], v[p + "block_entropy"]
    nfe, up = v[p + "num_steps_mean"], v[p + "mean_unmask_prob"]

    print(f"run: {args.run_dir}")
    print(f"steps {steps[0]}..{steps[-1]}  ({len(steps)} logged points, "
          f"uniform block entropy = {np.log(len(sizes)):.3f} nats)\n")
    head = f"{'steps':>11} {'acc':>6} {'b_mean':>7} {'H_b':>5} {'NFE':>6} {'p_unm':>6} | " + \
        " ".join(f"{b:>4}" for b in sizes)
    print(head)
    print("-" * len(head))
    lo = int(steps[0]) // args.bucket * args.bucket
    while lo <= steps[-1]:
        m = (steps >= lo) & (steps < lo + args.bucket)
        if m.any():
            fr = " ".join(f"{v[t][m].mean():4.2f}" for t in frac_tags)
            print(f"{lo:>5}-{lo + args.bucket - 1:<5} {acc[m].mean():6.3f} {bmean[m].mean():7.1f} "
                  f"{hb[m].mean():5.2f} {nfe[m].mean():6.1f} {up[m].mean():6.3f} | {fr}")
        lo += args.bucket

    print("\nblock size vs accuracy (per-step scalars):")
    print(f"  level corr(b_mean, acc) = {corr(bmean, acc):+.3f}   (co-trends with training time)")
    print(f"  diff  corr(db_mean, dacc) = {corr(np.diff(bmean), np.diff(acc)):+.3f}   (read this one)")
    # Which sizes' share moves with accuracy, again on differences.
    parts = []
    for t, b in zip(frac_tags, sizes):
        parts.append(f"{b}:{corr(np.diff(v[t]), np.diff(acc)):+.2f}")
    print("  diff corr(dfrac_b, dacc) by size: " + "  ".join(parts))

    bias_tags = [p + f"block_head/bias_{b}" for b in sizes]
    if all(t in sc for t in bias_tags):
        last = {b: sc[t][1][-1] for t, b in zip(bias_tags, sizes)}
        first = {b: sc[t][1][0] for t, b in zip(bias_tags, sizes)}
        print("\nblock_head/bias (prior logits) at first / latest step:")
        print("  " + "  ".join(f"{b}:{first[b]:+.3f}->{last[b]:+.3f}" for b in sizes))
    gn = p + "grad_norm/block_head"
    if gn in sc:
        g = sc[gn][1]
        print(f"grad_norm/block_head: mean {g.mean():.3e}, last {g[-1]:.3e}")


if __name__ == "__main__":
    main()
