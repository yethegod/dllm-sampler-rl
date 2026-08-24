"""Analyze the *order* in which a decoding strategy unmasks positions.

Runs `generate_unified` with `record_unmask_order=True` on a slice of a dataset and
reports how autoregressive (left-to-right) the resulting unmasking schedule is, so
that policies trained at different block lengths can be compared against each other
and against the confidence heuristics.

Example:
    python -m eval.analyze_unmask_order \
        --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --policy_path /path/checkpoint-1870/model.safetensors \
        --label bl32-policy --n_test 64 --out /path/order_stats
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.models.policy import DiTConfidencePolicy
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTHiddenProjPolicy
from common.models.policy import PolicyHFWrapper
from eval.eval import DATASET_MAP
from eval.eval import FEW_SHOT_DEFAULTS
from eval.eval import MASK_TOKENS_MAP
from eval.eval import init_seed


def _rankdata(a: np.ndarray) -> np.ndarray:
    """Average-tie ranks (scipy.stats.rankdata equivalent, avoids the dependency)."""
    order = np.argsort(a, kind="stable")
    ranks = np.empty(len(a), dtype=np.float64)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sorted_a[j + 1] == sorted_a[i]:
            j += 1
        ranks[order[i : j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3:
        return float("nan")
    rx, ry = _rankdata(x), _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx**2).sum() * (ry**2).sum())
    if denom == 0:
        return float("nan")
    return float((rx * ry).sum() / denom)


def _windowed_spearman(pos: np.ndarray, step: np.ndarray, width: int) -> float:
    """Mean Spearman(position, unmask step) inside fixed-width position windows.

    Removes the left-to-right ordering that the block structure imposes for free,
    so BL32 and BL128 can be compared on the same footing (use width=32 for both).
    """
    rhos = []
    for w0 in range(0, 256, width):
        sel = (pos >= w0) & (pos < w0 + width)
        if sel.sum() >= 5:
            r = _spearman(pos[sel], step[sel])
            if not np.isnan(r):
                rhos.append(r)
    return float(np.mean(rhos)) if rhos else float("nan")


def _frontier_stats(order: np.ndarray, block_length: int) -> tuple[float, float]:
    """(left_done_rate, mean tokens ahead of the left frontier) for one schedule.

    `left_done_rate` = fraction of revealed tokens whose whole left context inside
    the block was already revealed; a strictly left-to-right schedule gives 1.0.
    """
    pos = np.nonzero(order >= 0)[0]
    left_done, ahead = [], []
    for p in pos:
        s = order[p]
        blk = (p // block_length) * block_length
        left = order[blk:p]
        still_masked_left = np.nonzero((left < 0) | (left > s))[0]
        left_done.append(1.0 if len(still_masked_left) == 0 else 0.0)
        ahead.append(
            0.0 if len(still_masked_left) == 0 else float(p - blk - still_masked_left[0])
        )
    if not left_done:
        return float("nan"), float("nan")
    return float(np.mean(left_done)), float(np.mean(ahead))


def _shuffled_null(order: np.ndarray, block_length: int, n_rep: int = 5) -> tuple[float, float]:
    """Same per-block, per-step reveal *counts*, but positions chosen at random.

    Both frontier statistics depend on how many tokens are revealed per step: a
    slower schedule looks more left-to-right for free. Permuting the step labels
    within each block keeps the speed identical and destroys only the ordering,
    which gives the value to compare a real schedule against.
    """
    rng = np.random.RandomState(0)
    L = len(order)
    lds, aheads = [], []
    for _ in range(n_rep):
        null = order.copy()
        for blk in range(0, L, block_length):
            sel = np.nonzero(null[blk : blk + block_length] >= 0)[0]
            if len(sel) > 1:
                null[blk + sel] = null[blk + rng.permutation(sel)]
        ld, ah = _frontier_stats(null, block_length)
        lds.append(ld)
        aheads.append(ah)
    return float(np.mean(lds)), float(np.mean(aheads))


def analyze_sample(order: np.ndarray, valid_len: int, block_length: int) -> dict:
    """Order stats for one generation.

    :param order: (L,) step index at which each position was unmasked (-1 = never)
    :param valid_len: only positions [0, valid_len) are considered (pre-EOS content)
    :param block_length: semi-AR block length used during generation
    """
    L = len(order)
    valid = (order >= 0) & (np.arange(L) < valid_len)
    pos = np.nonzero(valid)[0]
    if len(pos) < 5:
        return {}
    step = order[pos]

    # Restrict to the pre-EOS content for the frontier statistics too
    trimmed = np.where(np.arange(L) < valid_len, order, -1)
    left_done_rate, ahead = _frontier_stats(trimmed, block_length)
    null_left_done, null_ahead = _shuffled_null(trimmed, block_length)

    n_steps = int(order.max()) + 1
    per_step = np.bincount(step, minlength=n_steps)

    return {
        "n_valid_tokens": int(len(pos)),
        "n_steps": n_steps,
        "tokens_per_step": float(len(pos) / max(len(np.unique(step)), 1)),
        "spearman_global": _spearman(pos.astype(float), step.astype(float)),
        "spearman_within_block": _windowed_spearman(pos, step, block_length),
        "spearman_within_w32": _windowed_spearman(pos, step, 32),
        "left_done_rate": left_done_rate,
        "left_done_rate_null": null_left_done,
        # >0 means more left-to-right than a same-speed random schedule
        "left_done_excess": left_done_rate - null_left_done,
        "mean_tokens_ahead_of_frontier": ahead,
        "mean_tokens_ahead_null": null_ahead,
        "max_tokens_per_step": int(per_step.max()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--label", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--remasking", type=str, default="policy")
    parser.add_argument("--thres", type=float, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature_policy", type=float, default=1.0)
    parser.add_argument("--sampling_mode", type=str, default=None)
    parser.add_argument("--n_test", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot", type=int, default=-1)
    args = parser.parse_args()

    init_seed(args.seed)

    (cfg,) = TrlParser((Config,)).parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    block_length = args.block_length or cfg.block_length
    gen_length = args.gen_length or cfg.max_completion_length
    sampling_mode = args.sampling_mode or cfg.sampling_mode
    few_shot = FEW_SHOT_DEFAULTS[args.dataset] if args.few_shot == -1 else args.few_shot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    mask_id = MASK_TOKENS_MAP["LLaDA" if "LLaDA" in cfg.model_path else "Dream"]
    model_type = "LLaDA" if "LLaDA" in cfg.model_path else "Dream"

    policy = None
    if args.remasking == "policy":
        if cfg.policy_type == "dit_confidence":
            hidden_dim = cfg.policy_hidden_dim or 128
            core = DiTConfidencePolicy(
                hidden_dim=hidden_dim,
                feedforward_dim=cfg.policy_feedforward_dim or (4 * hidden_dim),
                num_heads=cfg.policy_num_heads,
                dropout=cfg.policy_dropout,
                time_embed_dim=cfg.policy_time_embed_dim,
                smart_init=cfg.policy_smart_init,
                confidences_top_p=cfg.confidences_top_p,
                num_blocks=cfg.policy_num_blocks,
                time_period=cfg.policy_time_period,
            ).to(device)
        elif cfg.policy_type == "dit_hidden_proj":
            assert model_type == "LLaDA", (
                "dit_hidden_proj policy is only supported with LLaDA models, not Dream"
            )
            hidden_dim = cfg.policy_hidden_dim or 128
            core = DiTHiddenProjPolicy(
                dllm=model,
                hidden_dim=hidden_dim,
                feedforward_dim=cfg.policy_feedforward_dim or (4 * hidden_dim),
                num_heads=cfg.policy_num_heads,
                dropout=cfg.policy_dropout,
                time_embed_dim=cfg.policy_time_embed_dim,
                smart_init=cfg.policy_smart_init,
                num_blocks=cfg.policy_num_blocks,
                time_period=cfg.policy_time_period,
            ).to(device)
        elif cfg.policy_type == "dit_hidden":
            core = DiTHiddenStatePolicy(
                dllm=model,
                time_embed_dim=cfg.policy_time_embed_dim,
                num_blocks=cfg.policy_num_blocks,
                smart_init=cfg.policy_smart_init,
                time_period=cfg.policy_time_period,
            ).to(device)
        else:
            # Was a bare `else` building DiTHiddenStatePolicy, which silently loaded
            # the wrong module for any policy_type added later.
            raise ValueError(f"Unsupported policy_type {cfg.policy_type}")
        policy = PolicyHFWrapper(core, cfg.policy_type)
        assert args.policy_path, "--policy_path is required for remasking=policy"
        policy.load_state_dict(load_file(args.policy_path))
        policy.eval()

    dataset = DATASET_MAP[args.dataset](
        tokenizer=tokenizer, subsample=-1, num_examples=few_shot, add_reasoning=True
    )
    collate_fn = dataset.collate_fn
    if args.n_test < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    per_sample, orders, texts, valid_lens = [], [], [], []
    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].bool().to(device)
            gen_kwargs = dict(
                model=model,
                prompt=input_ids,
                remasking=args.remasking,
                gen_length=gen_length,
                block_length=block_length,
                temperature=args.temperature,
                mask_id=mask_id,
                model_type=model_type,
                attention_mask=attn,
                record_unmask_order=True,
            )
            if args.remasking == "policy":
                gen_kwargs.update(
                    policy=policy,
                    sampling_mode=sampling_mode,
                    dpls_stop_logit=cfg.dpls_stop_logit,
                    temperature_policy=args.temperature_policy,
                    full_context=cfg.policy_full_context,
                    confidences_top_p=cfg.confidences_top_p,
                )
            elif args.remasking == "fastdllm":
                gen_kwargs["thres"] = args.thres
            else:
                gen_kwargs["steps"] = args.diffusion_steps

            result = generate_unified(**gen_kwargs)
            completions = result.sequences[:, input_ids.shape[1] :]
            order = result.unmask_order.cpu().numpy()

            for j in range(completions.shape[0]):
                toks = completions[j]
                eos = (toks == tokenizer.eos_token_id).nonzero()
                valid_len = int(eos[0].item()) if len(eos) else gen_length
                stats = analyze_sample(order[j], valid_len, block_length)
                if stats:
                    per_sample.append(stats)
                # keep every schedule (tiny) so the order can be re-analyzed offline
                orders.append(order[j])
                valid_lens.append(valid_len)
                if len(texts) < 8:
                    texts.append(
                        tokenizer.decode(toks[:valid_len], skip_special_tokens=True)
                    )
            print(f"[{args.label}] batch {bi + 1}: {len(per_sample)} samples done", flush=True)

    keys = [k for k in per_sample[0] if k != "n_valid_tokens"]
    summary = {
        k: float(np.nanmean([s[k] for s in per_sample])) for k in keys
    } | {
        f"{k}_std": float(np.nanstd([s[k] for s in per_sample])) for k in keys
    }
    summary |= {
        "label": args.label,
        "block_length": block_length,
        "remasking": args.remasking,
        "temperature_policy": args.temperature_policy,
        "n_samples": len(per_sample),
        "mean_valid_tokens": float(np.mean([s["n_valid_tokens"] for s in per_sample])),
    }

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.label}_order_stats.json", "w") as f:
        json.dump({"summary": summary, "per_sample": per_sample}, f, indent=2)
    np.savez_compressed(
        out_dir / f"{args.label}_orders.npz",
        orders=np.stack(orders),
        valid_lens=np.array(valid_lens),
        texts=np.array(texts, dtype=object),
        block_length=block_length,
    )

    print(f"\n=== {args.label} (BL={block_length}, {args.remasking}) ===")
    for k in keys:
        print(f"  {k:32s} {summary[k]:8.3f} ± {summary[k + '_std']:.3f}")


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
