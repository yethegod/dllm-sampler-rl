"""Analyse a fixed-block sweep of a block_unmask_policy checkpoint.

Reads the generations written by eval.pipeline with --block_unmask_fixed_schedule
(and optionally --block_unmask_cond_block / --record_unmask_order) and answers:

  1. With the SAME frozen unmask head, how do accuracy, NFE and E[R] move with the
     block size b?  (table per (b, told-b) cell, averaged over seeds)
  2. Does the head's *behaviour* depend on b beyond the sampling-mask restriction:
     tokens per step, distance of unmasked tokens from the frontier, left-to-right-ness
     (from unmask_order).
  3. Per-question oracle over b vs the best constant b -- the ceiling a stage-2 block
     head could reach on this decoder -- and whether a question's best b is stable
     across seeds (structure) or not (decoding-path lottery).
  4. Conditioning probes: same real b, different b told to the window-conditioned
     head. Identical traces mean the head never learned p(u | s, b).

Usage:
    python -m eval.analyze_fixed_block_sweep <results_dir> \
        [--learned_dir <dir with the learned-policy evals>] [--out report.txt]
"""

import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from train.reward_func import _process_answers_gsm8k  # noqa: E402
from train.reward_func import extract_gsm_answer  # noqa: E402

ALPHAS = (0.0, 1.0, 3.0)


def _correct(row) -> float:
    try:
        return float(
            _process_answers_gsm8k(
                [extract_gsm_answer(row["generations"])], [str(row["ground_truth"])], 1.0
            )[0]
        )
    except Exception:
        return 0.0


def _seed_of(path: str) -> int:
    m = re.search(r"_seed_(\d+)_", path)
    return int(m.group(1)) if m else -1


def _reward(correct: float, steps: int, L: int, alpha: float) -> float:
    return correct * ((L - min(steps, L) + 1) / L) ** alpha


def _order_stats(order: list[int], block: int | None):
    """Behavioural stats of one decode from its (L,) unmask step indices.

    frontier = lowest still-masked position when a step is applied; offset = position
    minus frontier for every token unmasked at that step. Returns (tokens/step,
    mean offset, frac offset > 8, spearman(position, step)).
    """
    o = np.asarray(order)
    done = o >= 0
    if done.sum() < 2:
        return (float("nan"),) * 4
    steps = int(o[done].max()) + 1
    tokens_per_step = done.sum() / steps
    offsets = []
    masked = np.ones(len(o), dtype=bool)
    for s in range(steps):
        idx = np.nonzero(done & (o == s))[0]
        if len(idx) == 0:
            continue
        frontier = int(np.nonzero(masked)[0][0])
        offsets.extend((idx - frontier).tolist())
        masked[idx] = False
    offsets = np.asarray(offsets)
    pos = np.nonzero(done)[0].astype(float)
    st = o[done].astype(float)
    if st.std() == 0 or pos.std() == 0:
        rho = float("nan")
    else:
        rho = float(np.corrcoef(np.argsort(np.argsort(pos)), np.argsort(np.argsort(st)))[0, 1])
    return (
        float(tokens_per_step),
        float(offsets.mean()),
        float((offsets > 8).mean()),
        rho,
    )


def load_cells(results_dir: str, checkpoint=None):
    """{(b_label, cond, seed): {qid: record}} plus gen_length."""
    cells = {}
    L = None
    for f in sorted(glob.glob(os.path.join(results_dir, "**", "gsm8k*generations.json"), recursive=True)):
        data = json.load(open(f))
        if data.get("remasking") != "block_unmask_policy":
            continue
        sched = data.get("block_unmask_fixed_schedule")
        cond = data.get("block_unmask_cond_block")
        mode = data.get("block_sampling_mode")
        m = re.search(r"checkpoint-([^_/]+)_", f)
        ckpt = m.group(1) if m else "?"
        if checkpoint is not None and ckpt != str(checkpoint):
            continue
        # Fixed cells are always one checkpoint; learned rows are labelled with theirs
        # so a ckpt-50 control never averages into the trained checkpoint's seeds.
        b_label = "-".join(str(b) for b in sched) if sched else f"learned:{mode}@{ckpt}"
        seed = _seed_of(f)
        L = L or int(data.get("gen_length", 256))
        recs = {}
        for i, row in enumerate(data["generations"]):
            recs[row["question"]] = {
                "correct": _correct(row),
                "steps": int(row["steps"]),
                "order": row.get("unmask_order"),
                "blocks": row.get("block_sizes"),
            }
        cells[(b_label, cond, seed)] = recs
    return cells, L or 256


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--learned_dir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--checkpoint", default=None, help="only this checkpoint (default all)")
    args = ap.parse_args()

    cells, L = load_cells(args.results_dir, args.checkpoint)
    if args.learned_dir:
        more, _ = load_cells(args.learned_dir, args.checkpoint)
        cells.update(more)
    if not cells:
        sys.exit(f"no block_unmask_policy generations under {args.results_dir}")
    lines = []
    say = lines.append

    # ---------------- 1 + 2: per-cell table ----------------
    say(f"L={L}; E[R]_a = correct * ((L-min(NFE,L)+1)/L)^a")
    say("")
    say("cell (b | told) seeds   n     acc    NFE   E[R]a1  E[R]a3  tok/step  offset  frac>8   rho_LR")
    by_cell = defaultdict(list)
    for (b, cond, seed), recs in cells.items():
        by_cell[(b, cond)].append((seed, recs))
    def _key(k):
        b, cond = k
        return (0 if b[0].isdigit() else 1, int(b.split("-")[0]) if b[0].isdigit() else 0, cond or 0, b)
    for (b, cond) in sorted(by_cell, key=_key):
        seeds = by_cell[(b, cond)]
        accs, nfes, r1, r3, tps, offs, fr8, rhos = ([] for _ in range(8))
        n = 0
        for _, recs in seeds:
            c = np.array([r["correct"] for r in recs.values()])
            s = np.array([r["steps"] for r in recs.values()])
            n = len(c)
            accs.append(c.mean()); nfes.append(s.mean())
            r1.append(np.mean([_reward(a, b_, L, 1.0) for a, b_ in zip(c, s)]))
            r3.append(np.mean([_reward(a, b_, L, 3.0) for a, b_ in zip(c, s)]))
            st = [
                _order_stats(r["order"], None)
                for r in recs.values()
                if r["order"] is not None
            ]
            if st:
                st = np.array(st, dtype=float)
                tps.append(np.nanmean(st[:, 0])); offs.append(np.nanmean(st[:, 1]))
                fr8.append(np.nanmean(st[:, 2])); rhos.append(np.nanmean(st[:, 3]))
        told = "-" if cond is None else str(cond)
        fmt = lambda v: f"{np.mean(v):7.3f}" if v else "    n/a"
        say(
            f"{b:>14s} | {told:>4s}  {len(seeds):2d}  {n:5d}  {100*np.mean(accs):6.2f}  {np.mean(nfes):5.1f}  "
            f"{np.mean(r1):6.3f}  {np.mean(r3):6.3f}  {fmt(tps)} {fmt(offs)} {fmt(fr8)} {fmt(rhos)}"
        )
    say("")

    # ---------------- 3: per-question oracle over constant b ----------------
    const = {
        (b, seed): recs
        for (b, cond, seed), recs in cells.items()
        if cond is None and b[0].isdigit() and "-" not in b
    }
    seeds = sorted({seed for _, seed in const})
    bs = sorted({int(b) for b, _ in const})
    if const:
        say(f"constant-b cells: b={bs}, seeds={seeds}")
        oracle_b_by_seed = {}
        for alpha in ALPHAS:
            say(f"-- alpha={alpha}")
            for seed in seeds:
                avail = [b for b in bs if (str(b), seed) in const]
                qs = set.intersection(*[set(const[(str(b), seed)]) for b in avail])
                if not qs:
                    continue
                qs = sorted(qs)
                R = np.array(
                    [
                        [
                            _reward(const[(str(b), seed)][q]["correct"], const[(str(b), seed)][q]["steps"], L, alpha)
                            for b in avail
                        ]
                        for q in qs
                    ]
                )  # (Q, B)
                mean_per_b = R.mean(0)
                best_i = int(mean_per_b.argmax())
                oracle = R.max(1).mean()
                arg = R.argmax(1)
                # ties -> count each b that attains the row max
                ties = (R == R.max(1, keepdims=True))
                share = ties.mean(0)
                say(
                    f"seed {seed}: best constant b={avail[best_i]} E[R]={mean_per_b[best_i]:.4f}; "
                    f"per-question oracle {oracle:.4f} (+{100*(oracle/mean_per_b[best_i]-1):.1f}%); "
                    f"any-b-correct {100*(R.max(1)>0).mean():.1f}%"
                )
                say("   share of questions where b attains the max: " + ", ".join(f"{b}:{100*s:.0f}%" for b, s in zip(avail, share)))
                if alpha == 1.0:
                    oracle_b_by_seed[seed] = {q: set(np.array(avail)[ties[i]].tolist()) for i, q in enumerate(qs)}
        # cross-seed stability of the oracle b (alpha=1), ties as sets
        if len(oracle_b_by_seed) >= 2:
            s0 = seeds[0]
            say("-- cross-seed stability of the alpha=1 oracle b (a question's argmax set at one seed vs another)")
            for s1 in seeds[1:]:
                qs = sorted(set(oracle_b_by_seed[s0]) & set(oracle_b_by_seed[s1]))
                inter = np.array([len(oracle_b_by_seed[s0][q] & oracle_b_by_seed[s1][q]) > 0 for q in qs])
                # chance level: draw the seed-s1 argmax set from a random other question
                rng = np.random.default_rng(0)
                perm = rng.permutation(len(qs))
                chance = np.array([len(oracle_b_by_seed[s0][qs[i]] & oracle_b_by_seed[s1][qs[j]]) > 0 for i, j in enumerate(perm)])
                say(f"seed {s0} vs {s1}: argmax sets overlap on {100*inter.mean():.1f}% of questions (shuffled-question chance {100*chance.mean():.1f}%)")
        say("")

    # ---------------- 4: conditioning probes ----------------
    probes = [(b, cond, seed) for (b, cond, seed) in cells if cond is not None]
    if probes:
        say("conditioning probes: same real b, head told a different b (seed-matched)")
        say("real b  told   acc(real)  acc(told)  NFE(real)  NFE(told)  same-correct  same-NFE  same-order  same-step0  mean|dNFE|")
        say("  (same-order = whole trace identical; same-step0 = the first step's unmask set identical, i.e. before any")
        say("   Bernoulli-draw divergence -- with bit-identical logits both would be 100%; a tiny logit shift flips a draw")
        say("   somewhere and the rest of the trace diverges chaotically, so same-step0 is the cleaner sensitivity probe)")
        for (b, cond, seed) in sorted(probes, key=lambda k: (int(k[0].split('-')[0]), k[1], k[2])):
            base = cells.get((b, None, seed))
            if base is None:
                say(f"{b:>6s}  {cond:>4d}  (no un-lied cell at seed {seed})")
                continue
            lied = cells[(b, cond, seed)]
            qs = sorted(set(base) & set(lied))
            cb = np.array([base[q]["correct"] for q in qs]); cl = np.array([lied[q]["correct"] for q in qs])
            sb = np.array([base[q]["steps"] for q in qs]); sl = np.array([lied[q]["steps"] for q in qs])
            same_order = [
                base[q]["order"] == lied[q]["order"]
                for q in qs
                if base[q]["order"] is not None and lied[q]["order"] is not None
            ]
            so = f"{100*np.mean(same_order):7.1f}%" if same_order else "    n/a"
            same_step0 = [
                [i for i, v in enumerate(base[q]["order"]) if v == 0]
                == [i for i, v in enumerate(lied[q]["order"]) if v == 0]
                for q in qs
                if base[q]["order"] is not None and lied[q]["order"] is not None
            ]
            s0 = f"{100*np.mean(same_step0):7.1f}%" if same_step0 else "    n/a"
            say(
                f"{b:>6s}  {cond:>4d}  {100*cb.mean():8.2f}  {100*cl.mean():8.2f}  {sb.mean():8.1f}  {sl.mean():8.1f}  "
                f"{100*(cb==cl).mean():10.1f}%  {100*(sb==sl).mean():7.1f}%  {so}  {s0}  {np.abs(sb-sl).mean():8.2f}"
            )
        say("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")
        print(f"\nwritten {args.out}")


if __name__ == "__main__":
    main()
