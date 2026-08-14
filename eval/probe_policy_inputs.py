"""Is the *input* to the block-size policy problem-dependent at all?

``eval/analyze_action_logits.py`` shows the learned joint policy carries only
~0.1 of ln|A| in I(action; problem), and the block_start-conditioned version of
that measurement shows half of even that is explained by generation progress
rather than problem content. Two explanations survive:

  (a) the policy is fine but its observation is nearly constant across problems,
  (b) the observation carries signal and the policy failed to extract it.

This script separates them, on the state where the ambiguity is sharpest: the
very first block decision. There, the policy's three input channels are

    m         all-ones (the whole generation region is masked)   -> constant
    timestep  0                                                  -> constant
    c         top-1 softmax prob at each of the L generation
              positions, from one forward pass on [prompt, MASK*L]

so *c is the only channel that can differ between problems at all*, and whatever
adaptivity is achievable at decision 0 must be a function of it.

``--mode dump`` runs that single forward pass per problem and saves c (N, L).
``--mode analyze`` asks how much task-relevant signal a probe can pull out of c,
against labels harvested from the Phase-0 constant-block sweep:

    nfe      NFE of a reference constant policy      (difficulty proxy)
    correct  whether that policy got the problem right
    oracle_b per-example argmax over the sweep's block sizes of the training
             reward -- the label the block-size head would need to predict

Probes are ridge / multinomial-logistic on c with k-fold CV, reported against the
constant predictor. A probe that cannot beat the constant means (a): no policy of
this observation could have been adaptive at decision 0, and the ceiling is a
schedule. A probe that clearly beats it means (b), and the fix is in the policy
or the optimization, not the interface.

Usage:
    python -m eval.probe_policy_inputs --mode dump \
        --config configs/experiment_configs/llada_8b_instruct_dit_blocksize_joint.yaml \
        --out /work/hdd/bhta/zsun9/eval_results/input_probe/c_decision0.npz

    python -m eval.probe_policy_inputs --mode analyze \
        --npz /work/hdd/bhta/zsun9/eval_results/input_probe/c_decision0.npz \
        --sweep_dir /work/hdd/bhta/zsun9/eval_results/blocksweep \
        --ref_run t0.7_bl64
"""
#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


# --------------------------------------------------------------------------- dump


def dump(args):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from transformers import AutoModel
    from transformers import AutoTokenizer
    from trl import TrlParser

    from common.config import Config
    from eval.eval import DATASET_MAP
    from eval.eval import FEW_SHOT_DEFAULTS
    from eval.eval import MASK_TOKENS_MAP
    from eval.eval import init_seed

    init_seed(args.seed)
    (cfg,) = TrlParser((Config,)).parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    gen_length = args.gen_length or cfg.max_completion_length
    few_shot = FEW_SHOT_DEFAULTS[args.dataset] if args.few_shot == -1 else args.few_shot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    mask_id = MASK_TOKENS_MAP["LLaDA" if "LLaDA" in cfg.model_path else "Dream"]

    dataset = DATASET_MAP[args.dataset](
        tokenizer=tokenizer, subsample=-1, num_examples=few_shot, add_reasoning=True
    )
    collate_fn = dataset.collate_fn
    if args.n_test is not None and args.n_test < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    all_c, all_q, all_plen = [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            prompt = batch["input_ids"].to(device)
            attn = batch["attention_mask"].bool().to(device)
            B, prompt_L = prompt.shape

            # Exactly the decision-0 state of generate_unified: prompt followed by a
            # fully masked generation region, same attention-mask construction.
            x = torch.full(
                (B, prompt_L + gen_length), mask_id, dtype=torch.long, device=device
            )
            x[:, :prompt_L] = prompt
            _attn = torch.ones((B, prompt_L + gen_length), dtype=torch.float, device=device)
            _attn[:, :prompt_L] = attn.float()
            _attn = _attn.to(model.dtype)

            logits = model(x, attention_mask=_attn).logits[:, prompt_L:]
            c = F.softmax(logits.float(), dim=-1).max(dim=-1).values  # (B, L)

            all_c.append(c.cpu().numpy().astype(np.float32))
            all_q.extend(batch["questions"])
            all_plen.extend(attn.sum(-1).cpu().numpy().tolist())
            print(f"batch {bi + 1}: {sum(len(a) for a in all_c)} problems", flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        c=np.concatenate(all_c, axis=0),
        questions=np.array(all_q, dtype=object),
        prompt_len=np.array(all_plen, dtype=np.int32),
    )
    print(f"wrote {out}")


# ------------------------------------------------------------------------ analyze


def _question_key(rec):
    """Join key shared by the npz dump and the sweep JSONs."""
    q = rec if isinstance(rec, str) else rec.get("question", "")
    return " ".join(q.split())[-200:]


def _load_run(gen_path):
    """{question_key: (nfe, correct)} for one generations JSON."""
    from common.parsing.parse_and_get_acc import check_gsm_correct
    from common.parsing.parse_and_get_acc import extract_gsm_answer

    d = json.load(open(gen_path))
    recs = d
    if isinstance(d, dict):
        for v in d.values():
            if isinstance(v, list) and v and isinstance(v[0], dict) and "generations" in v[0]:
                recs = v
                break
    out = {}
    for r in recs:
        ok = check_gsm_correct(extract_gsm_answer(r["generations"]), r.get("ground_truth"))
        out[_question_key(r)] = (float(r.get("steps", 0)), bool(ok))
    return out


def _reward(nfe, correct, L, alpha):
    return float(correct) * ((L - nfe + 1) / L) ** alpha


def _kfold_ridge(X, y, k=5, lam=10.0, seed=0):
    """CV R^2 of ridge against the constant predictor (R^2 <= 0 means no signal)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    preds = np.zeros_like(y, dtype=np.float64)
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xt, yt = X[tr], y[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = (Xt - mu) / sd
        Xv = (X[f] - mu) / sd
        ybar = yt.mean()
        A = Xt.T @ Xt + lam * np.eye(X.shape[1])
        w = np.linalg.solve(A, Xt.T @ (yt - ybar))
        preds[f] = Xv @ w + ybar
    ss_res = ((y - preds) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    return 1 - ss_res / ss_tot


def _kfold_logistic(X, y, k=5, lam=10.0, iters=200, seed=0):
    """CV AUC of L2 logistic regression (0.5 = chance)."""
    rng = np.random.RandomState(seed)
    idx = rng.permutation(len(y))
    folds = np.array_split(idx, k)
    scores = np.zeros(len(y))
    for f in folds:
        tr = np.setdiff1d(idx, f)
        Xt = X[tr]
        mu, sd = Xt.mean(0), Xt.std(0) + 1e-6
        Xt = np.hstack([(Xt - mu) / sd, np.ones((len(tr), 1))])
        Xv = np.hstack([(X[f] - mu) / sd, np.ones((len(f), 1))])
        yt = y[tr].astype(np.float64)
        w = np.zeros(Xt.shape[1])
        for _ in range(iters):  # Newton / IRLS
            p = 1 / (1 + np.exp(-Xt @ w))
            g = Xt.T @ (p - yt) + lam * w
            W = p * (1 - p) + 1e-6
            Hm = Xt.T @ (Xt * W[:, None]) + lam * np.eye(Xt.shape[1])
            step = np.linalg.solve(Hm, g)
            w -= step
            if np.abs(step).max() < 1e-6:
                break
        scores[f] = Xv @ w
    pos, neg = scores[y == 1], scores[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[y == 1].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def analyze(args):
    z = np.load(args.npz, allow_pickle=True)
    C = z["c"].astype(np.float64)
    keys = [_question_key(q) for q in z["questions"]]
    plen = z["prompt_len"].astype(np.float64)
    N, L = C.shape
    print(f"loaded c: {C.shape}")

    # --- how much does the observation itself move from problem to problem? ---
    print("\n== variability of c across problems (decision 0) ==")
    print(f"{'pos':>6}{'mean':>9}{'std':>9}{'p5':>9}{'p95':>9}")
    for p in [0, 7, 15, 31, 63, 127, 255]:
        if p < L:
            col = C[:, p]
            print(f"{p:>6}{col.mean():>9.4f}{col.std():>9.4f}"
                  f"{np.percentile(col, 5):>9.4f}{np.percentile(col, 95):>9.4f}")
    across = C.mean(1).std()
    within = C.std(1).mean()
    print(f"std of per-problem mean(c) across problems: {across:.4f}")
    print(f"mean of within-problem std over positions:  {within:.4f}")

    # --- labels from the constant-block sweep ---
    sweep = Path(args.sweep_dir)
    cells = {}
    for d in sorted(sweep.glob("t*_bl*")):
        gens = list(d.rglob("*generations.json"))
        if gens:
            cells[d.name] = _load_run(gens[0])
    if not cells:
        raise SystemExit(f"no generations under {sweep}")
    print(f"\nsweep cells: {len(cells)} ({', '.join(sorted(cells))})")

    ref = cells.get(args.ref_run)
    if ref is None:
        raise SystemExit(f"--ref_run {args.ref_run} not among {sorted(cells)}")

    common = [i for i, k in enumerate(keys) if k in ref]
    print(f"joined {len(common)}/{N} problems on question text")
    if len(common) < 50:
        raise SystemExit("join failed -- question keys do not match")
    X = C[common]
    nfe = np.array([ref[keys[i]][0] for i in common])
    correct = np.array([ref[keys[i]][1] for i in common]).astype(int)

    # oracle block size per example: argmax of the training reward over the sweep
    # cells that share the reference threshold (block size is the free variable).
    by_b = {}
    for name, run in cells.items():
        t, b = name.split("_bl")
        if t == args.ref_run.split("_bl")[0]:
            by_b[int(b)] = run
    bs = sorted(by_b)
    oracle_b = None
    if len(bs) >= 3:
        R = np.full((len(common), len(bs)), np.nan)
        for j, b in enumerate(bs):
            run = by_b[b]
            for r, i in enumerate(common):
                if keys[i] in run:
                    n_, c_ = run[keys[i]]
                    R[r, j] = _reward(n_, c_, args.gen_length, args.alpha)
        ok = ~np.isnan(R).any(1)
        oracle_b = np.argmax(R[ok], axis=1)
        print(f"\n== oracle block size (thres={args.ref_run.split('_bl')[0]}, "
              f"alpha={args.alpha}, {ok.sum()} problems) ==")
        for j, b in enumerate(bs):
            print(f"  b={b:<4} best for {100 * (oracle_b == j).mean():5.1f}% of problems"
                  f"   mean R = {np.nanmean(R[ok][:, j]):.4f}")
        print(f"  best single b       : R = {np.nanmean(R[ok], axis=0).max():.4f}")
        print(f"  per-example oracle b: R = {R[ok].max(1).mean():.4f}"
              f"   (headroom {R[ok].max(1).mean() - np.nanmean(R[ok], axis=0).max():+.4f})")

    # --- can a probe read any of it out of c? ---
    print("\n== probes on c (5-fold CV; R^2<=0 / AUC~0.5 means no usable signal) ==")
    r2_nfe = _kfold_ridge(X, nfe, lam=args.lam)
    r2_nfe_len = _kfold_ridge(plen[common][:, None], nfe, lam=args.lam)
    auc_corr = _kfold_logistic(X, correct, lam=args.lam)
    auc_corr_len = _kfold_logistic(plen[common][:, None], correct, lam=args.lam)
    print(f"  c        -> NFE      : R^2 = {r2_nfe:+.4f}")
    print(f"  promptlen-> NFE      : R^2 = {r2_nfe_len:+.4f}")
    print(f"  c        -> correct  : AUC = {auc_corr:.4f}")
    print(f"  promptlen-> correct  : AUC = {auc_corr_len:.4f}")

    if oracle_b is not None and len(set(oracle_b.tolist())) > 1:
        Xo = X[ok]
        top = np.bincount(oracle_b).argmax()
        y = (oracle_b == top).astype(int)
        auc_b = _kfold_logistic(Xo, y, lam=args.lam)
        print(f"  c        -> oracle b == {bs[top]} : AUC = {auc_b:.4f} "
              f"(base rate {y.mean():.3f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["dump", "analyze"], required=True)
    # dump
    ap.add_argument("--config", type=str)
    ap.add_argument("--out", type=str)
    ap.add_argument("--dataset", type=str, default="gsm8k")
    ap.add_argument("--gen_length", type=int, default=256)
    ap.add_argument("--n_test", type=int, default=None)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--few_shot", type=int, default=-1)
    ap.add_argument("--seed", type=int, default=42)
    # analyze
    ap.add_argument("--npz", type=str)
    ap.add_argument("--sweep_dir", type=str)
    ap.add_argument("--ref_run", type=str, default="t0.7_bl64")
    ap.add_argument("--alpha", type=float, default=0.5)
    ap.add_argument("--lam", type=float, default=10.0)
    args = ap.parse_args()

    if args.mode == "dump":
        assert args.config and args.out, "--mode dump needs --config and --out"
        dump(args)
    else:
        assert args.npz and args.sweep_dir, "--mode analyze needs --npz and --sweep_dir"
        analyze(args)


if __name__ == "__main__":
    main()
