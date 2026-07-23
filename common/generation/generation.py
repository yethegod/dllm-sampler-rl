#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
from typing import NamedTuple

import torch
import torch.nn.functional as F

from common.generation.sampling import bernoulli_sample
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
) -> GenerationResult:
    if remasking == "policy":
        if policy is None:
            raise ValueError("policy must be provided for remasking='policy'")
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

    # Strategy-specific state
    record_policy_data = policy is not None
    sampling_history = [] if record_policy_data else None

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

    def _forward_logits():
        model_output = model(
            x,
            attention_mask=_attn_mask,
            output_hidden_states=(policy_type == "dit_hidden"),
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
    if not adaptive_block:
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
                steps_taken += block_mask_index.any(dim=-1).int()

            start_idx = end_idx

    # Prepare metadata for gradient steps/loss computation
    if record_policy_data:
        generation_part = x[:, prompt_L:]
        still_masked = (generation_part == mask_id).any(dim=-1)

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
        )
    else:
        return GenerationResult(
            sequences=x,
            steps_taken=steps_taken,
            block_sizes=block_sizes,
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

    if policy_type == "dit_hidden":
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
