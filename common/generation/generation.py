#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import bernoulli_sample
from common.generation.sampling import categorical_sample
from common.generation.sampling import dpls_sample


class GenerationResult(NamedTuple):
    sequences: torch.Tensor  # Generated sequences (B, prompt_L + gen_L)
    steps_taken: torch.Tensor  # Steps taken per batch item

    # Policy training data (None for non-policy modes)
    sampling_inputs: torch.Tensor | None = None  # (B, T, BL)
    samples: torch.Tensor | None = None  # (B, T, BL)
    sampling_masks: torch.Tensor | None = None  # (B, T, BL)
    policy_inputs: tuple[torch.Tensor, ...] | None = None
    still_masked: torch.Tensor | None = None  # (B,)
    block_sizes: list[int] | None = None  # Adaptive block sizes (adaptive_block only)
    # (B, L) step index at which each generated position was unmasked, -1 if never
    # (only populated when record_unmask_order=True)
    unmask_order: torch.Tensor | None = None

    # Realized joint actions, one row per rollout (remasking='block_policy' only).
    # Padded to T with sampling_masks marking the real decisions.
    block_sizes_chosen: torch.Tensor | None = None  # (B, T) chosen block length
    thresholds_chosen: torch.Tensor | None = None  # (B, T) chosen Fast-dLLM threshold


def add_gumbel_noise(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature == 0.0:
        return logits
    logits = logits.to(torch.float32)
    noise = torch.rand_like(logits)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


@torch.no_grad()
def generate_unified(
    model,
    prompt: torch.Tensor,
    remasking: str,
    policy=None,
    thres: float | torch.Tensor | None = None,
    steps: int | None = None,
    gen_length: int = 128,
    block_length: int = 32,
    temperature: float = 0.0,
    mask_id: int = 126336,
    sampling_mode: str = "bernoulli",
    dpls_stop_logit: float = 0.0,
    model_type: str | None = None,
    attention_mask: torch.Tensor | None = None,
    temperature_policy: float = 1.0,
    full_context: bool = False,
    confidences_top_p: int = 1,
    adaptive_block: bool = False,
    delimiter_ids: tuple[int, ...] = (198,),
    delimiter_threshold: float = 0.3,
    record_unmask_order: bool = False,
    block_size_candidates: tuple[int, ...] = (8, 16, 32, 64, 128),
    threshold_candidates: tuple[float, ...] = (0.5, 0.7, 0.9),
    block_schedule: tuple[tuple[int, float], ...] | None = None,
) -> GenerationResult:
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
    elif remasking in ("block_policy", "block_schedule"):
        if remasking == "block_policy":
            if policy is None:
                raise ValueError("policy must be provided for remasking='block_policy'")
            # Silently ignoring an unsupported mode here would make an eval look like
            # it honoured --sampling_mode when it did not.
            if sampling_mode not in ("categorical", "categorical-argmax"):
                raise ValueError(
                    "remasking='block_policy' supports sampling_mode 'categorical' or "
                    f"'categorical-argmax', got {sampling_mode!r}"
                )
        elif not block_schedule:
            raise ValueError(
                "block_schedule must be provided for remasking='block_schedule'"
            )
        if gen_length % min(block_size_candidates) != 0:
            raise ValueError(
                f"gen_length ({gen_length}) must be divisible by the smallest block "
                f"candidate ({min(block_size_candidates)}) so blocks tile exactly"
            )
    elif remasking == "fastdllm":
        if thres is None:
            raise ValueError("thres must be provided for remasking='fastdllm'")
    elif remasking in ["low_confidence", "random"]:
        if steps is None:
            raise ValueError(f"steps must be provided for remasking='{remasking}'")
    else:
        raise ValueError(f"Unknown remasking strategy: {remasking}")

    if adaptive_block:
        if remasking not in ["policy", "fastdllm"]:
            raise ValueError(
                f"adaptive_block is not supported with remasking='{remasking}'"
            )
        if prompt.shape[0] != 1:
            raise ValueError("adaptive_block requires batch size 1")
        if remasking == "policy" and not full_context:
            raise ValueError(
                "adaptive_block with remasking='policy' requires full_context=True "
                "(block-local policy inputs have variable shapes across blocks)"
            )

    B, prompt_L = prompt.shape
    L = gen_length
    x = torch.full((B, L + prompt_L), mask_id, dtype=torch.long, device=prompt.device)
    x[:, :prompt_L] = prompt
    steps_taken = torch.zeros((B,), dtype=torch.int32, device=x.device)
    num_blocks = L // block_length

    if attention_mask is not None:
        _attn_mask = torch.ones((B, L + prompt_L), dtype=torch.float, device=x.device)
        _attn_mask[:, :prompt_L] = attention_mask.float()
        if model_type == "Dream":
            _attn_mask = _attn_mask.unsqueeze(1).unsqueeze(-2) * _attn_mask.unsqueeze(
                1
            ).unsqueeze(-1)
        # Handle DDP-wrapped models
        model_dtype = model.module.dtype if hasattr(model, "module") else model.dtype
        _attn_mask = _attn_mask.to(model_dtype)
    else:
        _attn_mask = None

    # Strategy-specific state. block_schedule carries no policy but still produces the
    # per-decision record that eval unpacks into block/threshold schedules.
    record_policy_data = policy is not None or remasking == "block_schedule"
    sampling_history = [] if record_policy_data else None

    # (B, L) step at which each position was unmasked; -1 while still masked
    unmask_order = (
        torch.full((B, L), -1, dtype=torch.int32, device=x.device)
        if record_unmask_order
        else None
    )

    def _record_order(unmask):
        # steps_taken is incremented after the decision, so it is the 0-based
        # index of the step that is being applied right now
        nonlocal unmask_order
        unmask_order = torch.where(
            unmask & (unmask_order < 0),
            steps_taken.unsqueeze(-1).to(unmask_order.dtype),
            unmask_order,
        )

    max_steps = L
    if remasking in ["low_confidence", "random"]:
        assert steps is not None and steps <= L
        tokens_per_step = L // steps
        max_steps = steps

    policy_type = None
    if policy is not None:
        policy_type = (
            policy.module.policy_type
            if hasattr(policy, "module")
            else policy.policy_type
        )

    # The hidden-state policies read model_output.hidden_states[-1]; the confidence
    # policies never touch it. Both decode paths below branch on this, so a new hidden
    # policy missing from this tuple fails as `NoneType` far from the config.
    wants_hidden = policy_type in (
        "dit_hidden",
        "dit_hidden_proj",
        "dit_block_size_hidden_proj",
    )

    def _forward_logits():
        hidden_kwargs = {}
        if wants_hidden:
            # LLaDA-only: the kwarg keeps LLaDA from materialising all 33 layers when
            # only the last is ever read. Dream's forward does not accept it, but every
            # hidden policy asserts LLaDA at construction.
            hidden_kwargs["final_hidden_state_only"] = True
        model_output = model(
            x,
            attention_mask=_attn_mask,
            output_hidden_states=wants_hidden,
            **hidden_kwargs,
        )

        # Handle Dream model logit shifting
        # Dream: logits at position i predict token i+1
        # For generated tokens at [P, P+1, ..., P+L-1], we need logits at [P-1, P, ..., P+L-2]
        if model_type == "Dream":
            logits = model_output.logits[
                :, prompt_L - 1 : -1
            ]  # Include last prompt pos, exclude last gen pos
        else:
            logits = model_output.logits[
                :, prompt_L:
            ]  # Just slice to generation portion

        # Apply Gumbel noise
        logits_with_noise = add_gumbel_noise(logits, temperature)
        x0 = torch.argmax(logits_with_noise, dim=-1)

        # Compute softmax once (needed by all strategies)
        probs = F.softmax(logits, dim=-1)

        return model_output, probs, x0

    def _step_decision(
        mask_index, block_mask_index, block_slice, model_output, probs, x0
    ):
        # Get unmask decisions based on strategy
        if remasking == "policy":
            unmask, sampling_data = _policy_unmask_decisions(
                mask_index,
                block_mask_index,
                probs,
                x0,
                steps_taken,
                block_slice,
                L,
                policy,
                policy_type,
                sampling_mode,
                full_context,
                confidences_top_p,
                model_output,
                prompt_L,
                dpls_stop_logit,
                temperature_policy,
            )
            sampling_history.append(sampling_data)

        elif remasking == "fastdllm":
            unmask = _confidence_threshold_unmask(
                block_mask_index, probs, block_slice, thres
            )
            if policy is not None:
                sampling_data = _record_policy_data(
                    mask_index,
                    block_mask_index,
                    probs,
                    steps_taken,
                    block_slice,
                    L,
                    policy,
                    policy_type,
                    full_context,
                    confidences_top_p,
                    model_output,
                    prompt_L,
                    temperature_policy,
                    unmask,
                )
                sampling_history.append(sampling_data)

        else:  # low_confidence / random
            unmask = _fixed_step_unmask_decisions(
                block_mask_index,
                probs,
                x0,
                block_slice,
                tokens_per_step,
                remasking,
            )

        return unmask

    block_sizes = None
    block_policy_data = None
    if remasking in ("block_policy", "block_schedule"):
        # Learned joint (block size, threshold) schedule; see formulation.md.
        #
        # remasking='block_schedule' reuses this loop verbatim with the policy call
        # replaced by a lookup into a fixed per-decision (block size, threshold) list.
        # It is the zero-parameter control for "did the policy learn anything beyond a
        # content-independent schedule?" -- everything else about the rollout, down to
        # the feasibility clamp and the NFE accounting, stays identical.
        #
        # Unlike the two paths below, block boundaries are PER ROW: group members
        # necessarily sample different block sizes, and training rolls out 128
        # sequences at once, so a batch-shared `block_slice` cannot represent the
        # state. Every row carries its own [block_start, block_end) and its own
        # threshold, and rows advance to their next block independently.
        #
        # The outer iteration is one forward pass, not one block. A row makes a
        # decision only on the step where it enters a new block, which is also the
        # step that performs that block's first unmasking -- so the decision costs
        # no extra NFE.
        cand_b = torch.tensor(block_size_candidates, dtype=torch.long, device=x.device)
        cand_t = torch.tensor(
            threshold_candidates, dtype=torch.float32, device=x.device
        )
        # Worst case is every block taking the smallest candidate.
        T_cap = L // int(cand_b.min().item())

        sched_b = sched_t = None
        if remasking == "block_schedule":
            # Resolve the schedule to candidate indices once, so the loop stays a
            # gather and an out-of-set entry fails here rather than silently.
            try:
                sched_b = torch.tensor(
                    [tuple(block_size_candidates).index(b) for b, _ in block_schedule],
                    dtype=torch.long,
                    device=x.device,
                )
                sched_t = torch.tensor(
                    [tuple(threshold_candidates).index(t) for _, t in block_schedule],
                    dtype=torch.long,
                    device=x.device,
                )
            except ValueError as e:
                raise ValueError(
                    f"block_schedule {block_schedule} contains an action outside the "
                    f"candidate sets {block_size_candidates} x {threshold_candidates}"
                ) from e

        block_start = torch.zeros(B, dtype=torch.long, device=x.device)
        block_end = torch.zeros(B, dtype=torch.long, device=x.device)
        dec_count = torch.zeros(B, dtype=torch.long, device=x.device)
        row_thres = torch.zeros((B, 1), dtype=torch.float32, device=x.device)

        # Decision-indexed buffers: slot t holds a row's t-th decision, NOT the t-th
        # forward pass. Rows decide on different steps, so packing by decision keeps
        # T at the number of blocks (<=32) instead of the number of forwards.
        def _buf(*shape, dtype=torch.float32):
            return torch.zeros((B, T_cap, *shape), dtype=dtype, device=x.device)

        rec_block_logits = _buf(len(cand_b))
        rec_thres_logits = _buf(len(cand_t))
        rec_actions = _buf(2, dtype=torch.long)
        rec_mask = _buf(1, dtype=torch.bool)
        rec_m = _buf(L, dtype=torch.bool)
        # The policy's input stream: top-p confidences, or the dLLM's last-layer
        # hidden state for the hidden-proj variant. Kept in the dLLM's own dtype
        # there -- upcasting 4096 channels to fp32 would double a buffer that is
        # already preallocated at full T_cap, and the policy recasts on use anyway.
        if wants_hidden:
            rec_c = _buf(L, model.config.hidden_size, dtype=model.dtype)
        else:
            rec_c = _buf(L, confidences_top_p)
        rec_t = _buf(1)
        rec_start = _buf(1, dtype=torch.long)
        rec_block_size = _buf(dtype=torch.long)
        rec_thres = _buf()

        positions = torch.arange(L, device=x.device)

        for _ in range(L):
            generation_part = x[:, prompt_L:]
            mask_index = (generation_part == mask_id) & (
                steps_taken < max_steps
            ).unsqueeze(-1)
            active = mask_index.any(dim=-1)
            if not active.any():
                break

            model_output, probs, x0 = _forward_logits()
            # The policy's per-position input stream, sliced to the generation region
            # so it lines up with mask_index -- the same slice _compute_policy_logits
            # takes on the per-step path. The confidence branch's topk over the full
            # vocab is skipped outright when the policy reads hidden states.
            policy_stream = (
                model_output.hidden_states[-1][:, prompt_L:, :]  # (B, L, D)
                if wants_hidden
                else probs.topk(confidences_top_p, dim=-1).values  # (B, L, P)
            )

            # A row needs a new action exactly when it has no live block.
            needs = active & (block_start == block_end)
            if needs.any():
                per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
                if remasking == "block_schedule":
                    # Decision index -> scheduled action; the last entry repeats.
                    slot_s = dec_count.clamp(max=len(sched_b) - 1)  # (B,)
                    b_idx = sched_b[slot_s]
                    t_idx = sched_t[slot_s]
                    # Same feasibility rule the policy head applies (policy.py: a
                    # candidate is selectable only if the block fits in what is left).
                    # An infeasible scheduled size falls back to the largest that fits,
                    # which is also what the masked policy would have been forced into.
                    feasible = (block_start.unsqueeze(-1) + cand_b) <= L  # (B, K)
                    feasible[:, 0] = True  # never leave a row with no action
                    largest_fit = torch.where(
                        feasible, cand_b.expand(B, -1), torch.zeros_like(cand_b)
                    ).argmax(dim=-1)
                    b_idx = torch.where(
                        feasible.gather(-1, b_idx.unsqueeze(-1)).squeeze(-1),
                        b_idx,
                        largest_fit,
                    )
                    # No policy distribution exists here; the recorded logits are only
                    # carried so the returned record has the same shape as block_policy.
                    block_logits = torch.zeros(
                        (B, len(cand_b)), dtype=torch.float32, device=x.device
                    ).masked_fill(~feasible, float("-inf"))
                    thres_logits = torch.zeros(
                        (B, len(cand_t)), dtype=torch.float32, device=x.device
                    )
                else:
                    block_logits, thres_logits = policy(
                        mask_index,
                        policy_stream,
                        per_batch_timestep,
                        block_start.unsqueeze(-1),
                    )
                    if temperature_policy != 1.0:
                        block_logits = block_logits / temperature_policy
                        thres_logits = thres_logits / temperature_policy

                    feasible = torch.isfinite(block_logits)
                    if sampling_mode == "categorical-argmax":
                        # Greedy deployment, as opposed to the E_{a~pi}[R] that GRPO
                        # optimises and that plain 'categorical' measures.
                        b_idx = block_logits.masked_fill(
                            ~feasible, float("-inf")
                        ).argmax(dim=-1)
                        t_idx = thres_logits.argmax(dim=-1)
                    else:
                        b_idx = categorical_sample(block_logits, feasible)  # (B,)
                        t_idx = categorical_sample(thres_logits)  # (B,)
                chosen_b = cand_b[b_idx]
                chosen_t = cand_t[t_idx]

                block_end = torch.where(needs, block_start + chosen_b, block_end)
                row_thres = torch.where(
                    needs.unsqueeze(-1), chosen_t.unsqueeze(-1), row_thres
                )

                rows = needs.nonzero(as_tuple=True)[0]
                # The tiling invariant caps decisions at T_cap; clamp defensively so a
                # violation corrupts one slot instead of raising an indexing error.
                slot = dec_count[rows].clamp(max=T_cap - 1)
                rec_block_logits[rows, slot] = block_logits[rows].float()
                rec_thres_logits[rows, slot] = thres_logits[rows].float()
                rec_actions[rows, slot, 0] = b_idx[rows]
                rec_actions[rows, slot, 1] = t_idx[rows]
                rec_mask[rows, slot, 0] = True
                rec_m[rows, slot] = mask_index[rows]
                rec_c[rows, slot] = policy_stream[rows].to(rec_c.dtype)
                rec_t[rows, slot, 0] = per_batch_timestep[rows, 0].float()
                rec_start[rows, slot, 0] = block_start[rows]
                rec_block_size[rows, slot] = chosen_b[rows]
                rec_thres[rows, slot] = chosen_t[rows]
                dec_count = dec_count + needs.long()

            block_index = (positions >= block_start.unsqueeze(-1)) & (
                positions < block_end.unsqueeze(-1)
            )  # (B, L)
            block_mask_index = mask_index & block_index
            unmask = _confidence_threshold_unmask_rowwise(
                block_mask_index, probs, row_thres
            )

            x[:, prompt_L:] = torch.where(unmask, x0, generation_part)
            if record_unmask_order:
                _record_order(unmask)
            steps_taken += block_mask_index.any(dim=-1).int()

            # Rows whose current block is now full advance; start == end then triggers
            # a fresh decision on the next iteration.
            block_done = ~((x[:, prompt_L:] == mask_id) & block_index).any(dim=-1)
            block_start = torch.where(active & block_done, block_end, block_start)

        assert (dec_count <= T_cap).all(), (
            f"decision count exceeded T_cap={T_cap}: {dec_count.max().item()}"
        )
        T_max = max(int(dec_count.max().item()), 1)
        block_policy_data = {
            # Both heads' logits share one tensor so the trainer's generic time-axis
            # slicing works unchanged; it splits them back at len(cand_b).
            "sampling_inputs": torch.cat(
                [rec_block_logits[:, :T_max], rec_thres_logits[:, :T_max]], dim=-1
            ),
            "samples": rec_actions[:, :T_max],
            "sampling_masks": rec_mask[:, :T_max],
            "policy_inputs": (
                rec_m[:, :T_max],
                rec_c[:, :T_max],
                rec_t[:, :T_max],
                rec_start[:, :T_max],
            ),
            "block_sizes_chosen": rec_block_size[:, :T_max],
            "thresholds_chosen": rec_thres[:, :T_max],
        }
    elif not adaptive_block:
        for num_block in range(num_blocks):
            start_idx = num_block * block_length
            end_idx = start_idx + block_length
            block_slice = slice(start_idx, end_idx)
            block_index = torch.zeros(L, dtype=torch.bool, device=x.device)
            block_index[start_idx:end_idx] = True

            for _ in range(block_length):
                generation_part = x[:, prompt_L:]
                mask_index = (generation_part == mask_id) & (
                    steps_taken < max_steps
                ).unsqueeze(-1)
                block_mask_index = mask_index[:, block_index]  # (B, BL)

                if (~block_mask_index).all():
                    break

                model_output, probs, x0 = _forward_logits()
                unmask = _step_decision(
                    mask_index, block_mask_index, block_slice, model_output, probs, x0
                )

                # Apply unmasking
                x[:, prompt_L:] = torch.where(unmask, x0, generation_part)
                if record_unmask_order:
                    _record_order(unmask)

                # Update steps taken: only count steps for batch elements that had work to do
                steps_taken += block_mask_index.any(dim=-1).int()
    else:
        # AdaBlock-style adaptive block boundaries (arXiv:2509.26432): the first
        # forward of each block doubles as the boundary decision, so it costs no
        # extra NFE compared to fixed-block decoding.
        block_sizes = []
        start_idx = 0
        while start_idx < L:
            generation_part = x[:, prompt_L:]
            mask_index = (generation_part == mask_id) & (
                steps_taken < max_steps
            ).unsqueeze(-1)
            if not mask_index[:, start_idx:].any():
                break

            model_output, probs, x0 = _forward_logits()
            block_len = _compute_adaptive_block_length(
                x0,
                probs,
                start_idx,
                L,
                block_length,
                delimiter_ids,
                delimiter_threshold,
            )
            end_idx = start_idx + block_len
            block_slice = slice(start_idx, end_idx)
            block_index = torch.zeros(L, dtype=torch.bool, device=x.device)
            block_index[start_idx:end_idx] = True
            block_sizes.append(block_len)

            block_mask_index = mask_index[:, block_index]
            unmask = _step_decision(
                mask_index, block_mask_index, block_slice, model_output, probs, x0
            )
            x[:, prompt_L:] = torch.where(unmask, x0, generation_part)
            if record_unmask_order:
                _record_order(unmask)
            steps_taken += block_mask_index.any(dim=-1).int()

            for _ in range(block_len - 1):
                generation_part = x[:, prompt_L:]
                mask_index = (generation_part == mask_id) & (
                    steps_taken < max_steps
                ).unsqueeze(-1)
                block_mask_index = mask_index[:, block_index]

                if (~block_mask_index).all():
                    break

                model_output, probs, x0 = _forward_logits()
                unmask = _step_decision(
                    mask_index, block_mask_index, block_slice, model_output, probs, x0
                )
                x[:, prompt_L:] = torch.where(unmask, x0, generation_part)
                if record_unmask_order:
                    _record_order(unmask)
                steps_taken += block_mask_index.any(dim=-1).int()

            start_idx = end_idx

    # Prepare metadata for gradient steps/loss computation
    if record_policy_data:
        generation_part = x[:, prompt_L:]
        still_masked = (generation_part == mask_id).any(dim=-1)

        if block_policy_data is not None:
            return GenerationResult(
                sequences=x,
                steps_taken=steps_taken,
                still_masked=still_masked,
                unmask_order=unmask_order,
                **block_policy_data,
            )

        if sampling_history:
            # Stack all sampling data for training
            sampling_inputs = torch.stack(
                [h["sampling_inputs"] for h in sampling_history], dim=1
            )
            samples = torch.stack([h["samples"] for h in sampling_history], dim=1)
            sampling_masks = torch.stack(
                [h["sampling_masks"] for h in sampling_history], dim=1
            )

            # Stack policy inputs
            policy_input_columns = zip(*[h["policy_inputs"] for h in sampling_history])
            policy_inputs_result = tuple(
                torch.stack(col, dim=1) for col in policy_input_columns
            )
        else:
            sampling_inputs = samples = sampling_masks = None
            policy_inputs_result = None

        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            sampling_inputs=sampling_inputs,
            samples=samples,
            sampling_masks=sampling_masks,
            policy_inputs=policy_inputs_result,
            still_masked=still_masked,
            block_sizes=block_sizes,
            unmask_order=unmask_order,
        )
    else:
        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            block_sizes=block_sizes,
            unmask_order=unmask_order,
        )


def _compute_adaptive_block_length(
    x0: torch.Tensor,
    probs: torch.Tensor,
    gen_offset: int,
    L: int,
    default_block_length: int,
    delimiter_ids: tuple[int, ...],
    delimiter_threshold: float,
) -> int:
    """Compute the next block length following AdaBlock-dLLM (arXiv:2509.26432).

    Looks at the argmax predictions in a window ahead of the generation frontier
    and ends the block at the highest-confidence delimiter token if its confidence
    exceeds delimiter_threshold; otherwise falls back to default_block_length.

    :param x0: (1, L) argmax token predictions over the generation region
    :param probs: (1, L, V) softmax probabilities over the generation region
    :param gen_offset: start of the next block, relative to the generation region
    :return: block length in [1, L - gen_offset]
    """
    remaining = L - gen_offset
    window_size = min(int(0.25 * L), remaining)
    window_tokens = x0[0, gen_offset : gen_offset + window_size]

    delimiter_mask = torch.zeros_like(window_tokens, dtype=torch.bool)
    for token_id in delimiter_ids:
        delimiter_mask |= window_tokens == token_id

    if not torch.any(delimiter_mask):
        return min(default_block_length, remaining)

    delimiter_pos = gen_offset + torch.nonzero(delimiter_mask).squeeze(-1)
    delimiter_confidences = probs[0, delimiter_pos, x0[0, delimiter_pos]]
    max_confidence, best_idx = torch.max(delimiter_confidences, dim=0)

    if max_confidence.item() >= delimiter_threshold:
        return int(delimiter_pos[best_idx].item()) - gen_offset + 1
    return min(default_block_length, remaining)


def _get_masks(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    block_slice: slice,
    full_context: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy_mask = mask_index if full_context else block_mask_index

    if full_context:
        # Policy sees full sequence (B, L), but we only sample in current block
        sampling_mask = torch.zeros_like(mask_index)
        sampling_mask[:, block_slice] = block_mask_index
    else:
        # Policy sees only block (B, BL), sample from same positions
        sampling_mask = policy_mask

    return policy_mask, sampling_mask


def _compute_policy_logits(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple]:
    """Compute policy logits and masks.

    :return: (policy_logits, policy_mask, sampling_mask, policy_inputs)
    """
    per_batch_timestep = steps_taken.unsqueeze(-1) * (1 / L)
    policy_mask, sampling_mask = _get_masks(
        mask_index, block_mask_index, block_slice, full_context
    )

    topk_result = probs.topk(confidences_top_p, dim=-1)
    c_max_input = (
        topk_result.values if full_context else topk_result.values[:, block_slice]
    )

    if policy_type in ("dit_hidden", "dit_hidden_proj"):
        hidden_states = model_output.hidden_states[-1]
        hidden_states_input = (
            hidden_states[:, prompt_L:, :]
            if full_context
            else hidden_states[
                :, prompt_L + block_slice.start : prompt_L + block_slice.stop, :
            ]
        )
        policy_inputs = (policy_mask, hidden_states_input, per_batch_timestep)
    elif policy_type == "dit_confidence":
        policy_inputs = (policy_mask, c_max_input, per_batch_timestep)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    policy_logits = policy(*policy_inputs)

    # Apply temperature scaling
    if temperature_policy != 1.0:
        policy_logits = policy_logits / temperature_policy

    return policy_logits, policy_mask, sampling_mask, policy_inputs


def _policy_unmask_decisions(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    sampling_mode: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    dpls_stop_logit: float = 0.0,
    temperature_policy: float = 1.0,
) -> tuple[torch.Tensor, dict]:
    policy_logits, _, sampling_mask, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
    )

    # Sample based on mode (using sampling_mask which is gated to current block)
    if sampling_mode == "bernoulli":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        samples_for_loglik = b
    elif sampling_mode == "bernoulli-argmax":
        b = bernoulli_sample(utilities=policy_logits, mask_index=sampling_mask)
        # For batch items where nothing was selected, force unmask at argmax
        no_selection = b.sum(dim=-1) == 0
        if no_selection.any():
            masked_logits = policy_logits.clone()
            masked_logits[~sampling_mask] = -torch.inf
            force_idx = torch.argmax(masked_logits, dim=-1)
            batch_indices = torch.arange(b.shape[0], device=b.device)[no_selection]
            b[batch_indices, force_idx[no_selection]] = True
        samples_for_loglik = b
    elif sampling_mode == "dpls":
        dpls_sequences, b = dpls_sample(
            utilities=policy_logits,
            stop_logit=dpls_stop_logit,
            mask_index=sampling_mask,
        )
        samples_for_loglik = dpls_sequences
    else:
        raise ValueError(f"Unknown sampling mode: {sampling_mode}")

    # Convert to sequence-level (always gate to current block)
    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    if full_context:
        unmask[:, block_slice] = b[:, block_slice]
    else:
        unmask[:, block_slice] = b

    sampling_data = {
        "sampling_inputs": policy_logits.detach(),
        "samples": samples_for_loglik.detach(),
        "sampling_masks": sampling_mask.detach(),
        "policy_inputs": tuple(
            pi.detach() if isinstance(pi, torch.Tensor) else pi for pi in policy_inputs
        ),
    }

    return unmask, sampling_data


def _confidence_threshold_unmask(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    block_slice: slice,
    thres: float | torch.Tensor,
) -> torch.Tensor:
    confidence = probs.max(dim=-1).values

    # Only consider masked positions in current block
    confidence_masked = confidence[:, block_slice].clone()
    confidence_masked[~block_mask_index] = -torch.inf

    unmask_local = confidence_masked > thres
    if not unmask_local.any():
        force_idx = torch.argmax(confidence_masked, dim=-1)
        unmask_local.scatter_(1, force_idx.unsqueeze(-1), True)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask


def _confidence_threshold_unmask_rowwise(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    thres: torch.Tensor,
) -> torch.Tensor:
    """Row-wise Fast-dLLM thresholding for per-row block boundaries and thresholds.

    Sibling of _confidence_threshold_unmask, which takes a batch-shared block slice and
    a single threshold. Two deliberate differences:

    - the block is a (B, L) mask, since every row may sit in a different block;
    - the "nothing cleared the threshold" fallback is applied **per row**. The shared
      version tests `if not unmask_local.any()` across the whole batch, so with B > 1 a
      stalled row can be starved by another row that did clear. Here a stalled row would
      never finish its block, so the fallback must be row-wise.

    The existing function is left untouched so previously collected baselines stay
    reproducible (all of them ran at batch size 1, where the two agree).

    :param block_mask_index: (B, L) still-masked positions inside each row's own block
    :param probs: (B, L, V) next-token probabilities
    :param thres: (B, 1) per-row confidence threshold
    :return: (B, L) boolean mask of positions to unmask
    """
    confidence = probs.max(dim=-1).values  # (B, L)
    confidence = confidence.masked_fill(~block_mask_index, -torch.inf)

    unmask = confidence > thres  # (B, L), broadcasts (B,1) over positions

    # A row with masked positions left in its block but nothing above threshold must
    # still make progress, otherwise the block never completes and the loop spins.
    stalled = block_mask_index.any(dim=-1) & ~unmask.any(dim=-1)  # (B,)
    if stalled.any():
        force_idx = confidence.argmax(dim=-1)  # (B,)
        rows = stalled.nonzero(as_tuple=True)[0]
        unmask[rows, force_idx[rows]] = True

    return unmask


def _record_policy_data(
    mask_index: torch.Tensor,
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    steps_taken: torch.Tensor,
    block_slice: slice,
    L: int,
    policy,
    policy_type: str,
    full_context: bool,
    confidences_top_p: int,
    model_output,
    prompt_L: int,
    temperature_policy: float,
    unmask: torch.Tensor,
) -> dict:
    policy_logits, policy_mask, _, policy_inputs = _compute_policy_logits(
        mask_index,
        block_mask_index,
        probs,
        steps_taken,
        block_slice,
        L,
        policy,
        policy_type,
        full_context,
        confidences_top_p,
        model_output,
        prompt_L,
        temperature_policy,
    )

    samples = unmask if full_context else unmask[:, block_slice]

    # ES (Expert Steering) special behavior: save policy_mask (not sampling_mask) so the
    # model learns to mimic the confidence thresholding in a block-agnostic way
    return {
        "sampling_inputs": policy_logits.detach().clone(),
        "samples": samples.detach().clone(),
        "sampling_masks": policy_mask.detach().clone(),
        "policy_inputs": tuple(
            pi.detach().clone() if isinstance(pi, torch.Tensor) else pi
            for pi in policy_inputs
        ),
    }


def _fixed_step_unmask_decisions(
    block_mask_index: torch.Tensor,
    probs: torch.Tensor,
    x0: torch.Tensor,
    block_slice: slice,
    tokens_per_step: int,
    mode: str,
) -> torch.Tensor:
    B = block_mask_index.shape[0]
    block_size = block_slice.stop - block_slice.start

    if mode == "low_confidence":
        confidence_block = torch.gather(
            probs[:, block_slice], dim=-1, index=x0[:, block_slice].unsqueeze(-1)
        ).squeeze(-1)
    elif mode == "random":
        confidence_block = torch.rand((B, block_size), device=x0.device)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    confidence_masked = torch.where(block_mask_index, confidence_block, -torch.inf)

    num_masked_block = block_mask_index.sum(dim=-1)
    k = torch.clamp(num_masked_block, max=tokens_per_step)
    max_k = k.max().item()

    if max_k == 0:
        unmask = torch.zeros(
            (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
        )
        return unmask

    _, topk_indices = torch.topk(confidence_masked, k=max_k, dim=-1)

    positions = torch.arange(max_k, device=x0.device).unsqueeze(0).expand(B, -1)
    valid_mask = positions < k.unsqueeze(-1)

    unmask_local = torch.zeros_like(block_mask_index, dtype=torch.bool)
    unmask_local.scatter_(1, topk_indices, valid_mask)

    unmask = torch.zeros(
        (probs.shape[0], probs.shape[1]), dtype=torch.bool, device=probs.device
    )
    unmask[:, block_slice] = unmask_local
    return unmask
