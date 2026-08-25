"""Collect the Stage 0 probe dataset: does the hidden state carry signal beyond top-k?

The question this answers, before any policy is trained on hidden states: is the
observation the bottleneck at all? Every policy in this repo sees one number per
position (the top-1 confidence). Feeding it the dLLM's 4096-d hidden state instead
costs a projector, a replay path, and days of GPU -- worth it only if that state
separates good unmasking decisions better than the confidences already do.

The label is free and dense. `generate_unified(record_probe_data=True)` records, per
decoding step, the token state the decision was made from and the dLLM's argmax x0 at
every position. Comparing x0 against the *final* sequence gives

    label(t, i) = [ x0_t[i] == final[i] ]   for positions still masked at step t

read as "had we unmasked position i at step t, would it have committed the token that
eventually won?" -- 1 means safe to commit now, 0 means wait. Unlike the confidences
this uses future information, so it is a privileged teacher rather than a restatement
of the input. Restricting to masked positions matters: an already-unmasked position
trivially matches the final sequence and would be pure label noise.

Decoding uses a FIXED baseline (Fast-dLLM at --thres), never a policy under training,
so the label does not depend on the thing being evaluated.

Features are stored as the *normalised* hidden state -- LayerNorm without affine, which
is exactly what HiddenProjInputMixin applies before its projection -- so the probe sees
the representation the policy would actually get, and the values stay in fp16 range
(raw LLaDA hidden states have outlier channels that do not).

Example:
    python -m eval.collect_probe_data \
        --config configs/experiment_configs/llada_8b_instruct_dit_confidence_BL32_mixture.yaml \
        --dataset gsm8k --n_test 64 --thres 0.9 \
        --out /work/hdd/bhta/zsun9/probe_data/gsm8k_t09.npz
"""

import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from eval.eval import DATASET_MAP
from eval.eval import FEW_SHOT_DEFAULTS
from eval.eval import MASK_TOKENS_MAP
from eval.eval import init_seed

TOP_K = 16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--dataset", type=str, default="gsm8k")
    parser.add_argument("--n_test", type=int, default=64)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--thres", type=float, default=0.9)
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--few_shot", type=int, default=-1)
    parser.add_argument(
        "--rows_per_step",
        type=int,
        default=192,
        help="Masked positions sampled per (batch, decoding step). Uniform within the "
        "step, so late steps -- which have few masks left -- contribute everything they "
        "have and early steps are subsampled.",
    )
    parser.add_argument(
        "--step_stride",
        type=int,
        default=1,
        help="Replay only every k-th recorded step. Decoding still costs T forwards but "
        "the replay pass drops to T/k, which is what buys the example count that "
        "actually determines the probe's effective sample size. Adjacent steps differ "
        "by a handful of tokens anyway.",
    )
    parser.add_argument("--max_rows", type=int, default=400_000)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    init_seed(args.seed)
    (cfg,) = TrlParser((Config,)).parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    gen_length = args.gen_length or cfg.max_completion_length
    block_length = args.block_length or cfg.block_length
    few_shot = FEW_SHOT_DEFAULTS[args.dataset] if args.few_shot == -1 else args.few_shot

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(
        cfg.model_path, trust_remote_code=True, torch_dtype=torch.bfloat16
    ).to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_path, trust_remote_code=True)
    is_llada = "LLaDA" in cfg.model_path
    mask_id = MASK_TOKENS_MAP["LLaDA" if is_llada else "Dream"]
    model_type = "LLaDA" if is_llada else "Dream"
    assert is_llada, "hidden-state features are only wired up for LLaDA"

    dataset = DATASET_MAP[args.dataset](
        tokenizer=tokenizer, subsample=-1, num_examples=few_shot, add_reasoning=True
    )
    collate_fn = dataset.collate_fn
    if args.n_test < len(dataset):
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn
    )

    d_model = model.config.hidden_size
    feats_h, feats_c, labels, ex_ids, steps, poss = [], [], [], [], [], []
    n_rows = 0
    example_offset = 0
    rng = np.random.default_rng(args.seed)

    with torch.no_grad():
        for bi, batch in enumerate(dataloader):
            if n_rows >= args.max_rows:
                break
            input_ids = batch["input_ids"].to(device)
            attn = batch["attention_mask"].bool().to(device)
            P = input_ids.shape[1]

            result = generate_unified(
                model=model,
                prompt=input_ids,
                remasking="fastdllm",
                thres=args.thres,
                gen_length=gen_length,
                block_length=block_length,
                temperature=0.0,
                mask_id=mask_id,
                model_type=model_type,
                attention_mask=attn,
                record_probe_data=True,
            )
            final_gen = result.sequences[:, P:]  # (B, L)
            state_hist = result.state_history  # (B, T, L)
            x0_hist = result.x0_history  # (B, T, L)
            B, T, L = state_hist.shape

            # Replay each recorded step to get the features the policy would have seen.
            # This is the same reconstruction compute_loss uses, and it reproduces the
            # rollout's hidden states exactly because the dLLM is deterministic.
            attn_full = torch.ones((B, P + L), dtype=torch.float, device=device)
            attn_full[:, :P] = attn.float()
            attn_full = attn_full.to(model.dtype)

            # A position the decoder never got to is still mask_id in the final
            # sequence, so its label would degenerate to (x0 == mask_id) = always 0 --
            # "ran out of steps", not "unsafe". Drop those columns entirely.
            decided = final_gen != mask_id  # (B, L)

            for t in range(0, T, args.step_stride):
                masked = (state_hist[:, t] == mask_id) & decided  # (B, L)
                if not masked.any():
                    continue
                x_t = torch.cat([input_ids, state_hist[:, t]], dim=1)
                out = model(
                    x_t, attention_mask=attn_full, output_hidden_states=True
                )
                h = out.hidden_states[-1][:, P:, :]  # (B, L, d)
                probs = F.softmax(out.logits[:, P:], dim=-1)
                conf = probs.topk(TOP_K, dim=-1).values  # (B, L, K)
                # Same normalisation HiddenProjInputMixin applies before projecting.
                h = F.layer_norm(h.float(), (d_model,))

                lab = (x0_hist[:, t] == final_gen) & masked  # (B, L)
                bidx, pidx = masked.nonzero(as_tuple=True)
                take = np.arange(bidx.numel())
                if bidx.numel() > args.rows_per_step:
                    take = rng.choice(bidx.numel(), args.rows_per_step, replace=False)
                take = torch.as_tensor(take, device=device)
                bsel, psel = bidx[take], pidx[take]

                feats_h.append(h[bsel, psel].to(torch.float16).cpu().numpy())
                feats_c.append(conf[bsel, psel].to(torch.float16).cpu().numpy())
                labels.append(lab[bsel, psel].to(torch.uint8).cpu().numpy())
                ex_ids.append((bsel + example_offset).cpu().numpy().astype(np.int32))
                steps.append(np.full(bsel.numel(), t, dtype=np.int16))
                poss.append(psel.cpu().numpy().astype(np.int16))
                n_rows += bsel.numel()

            example_offset += B
            print(
                f"batch {bi}: T={T} examples {example_offset} rows so far {n_rows}",
                flush=True,
            )

    out = dict(
        h=np.concatenate(feats_h),
        conf=np.concatenate(feats_c),
        label=np.concatenate(labels),
        example_id=np.concatenate(ex_ids),
        step=np.concatenate(steps),
        pos=np.concatenate(poss),
    )
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    np.savez(args.out, **out)
    pos_rate = float(out["label"].mean())
    print(
        f"wrote {args.out}: {len(out['label'])} rows, h={out['h'].shape}, "
        f"conf={out['conf'].shape}, positive rate {pos_rate:.4f}, "
        f"{len(np.unique(out['example_id']))} examples"
    )


if __name__ == "__main__":
    main()
