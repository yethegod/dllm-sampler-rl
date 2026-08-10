#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import torch

# Construct 0-1 gumbel dist at module level to avoid realloc
gumbel_dist = torch.distributions.Gumbel(0, 1)


def bernoulli_sample(
    utilities: torch.Tensor,
    mask_index: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sample from Bernoulli distribution given utilities.

    :param utilities: (B, BL) logit values for each position
    :param mask_index: (B, BL) available positions to sample
    :param dtype: if not None, sampling will be done in this dtype
    :return: (B, BL) boolean mask of sampled positions
    """
    if dtype is not None:
        utilities = utilities.to(dtype=dtype)
    probs = torch.sigmoid(utilities) * mask_index
    return torch.bernoulli(probs).bool()


def plackett_luce_batch_loglik(
    selected_indices: torch.Tensor,
    utilities: torch.Tensor,
    mask_index: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Memory-efficient and numerically stable Plackett-Luce log-likelihood computation.

    Supports padding in selected_indices for upstream DPLS. Expected padding format:
    [2, 4, -4, -4] where -4 indicates padding (repetition of last selected element with negative sign).

    :param selected_indices: (*B, k) selected indices with negative values as padding
    :param utilities: (*B, BL) utility values
    :param mask_index: (*B, BL) available positions, or None if utilities already masked
    :param dtype: if not None, computation and result will be in this dtype
    :return: (*B,) log-likelihoods
    """
    if mask_index is not None:
        utilities = utilities.masked_fill(~mask_index, float("-inf"))
    if dtype is not None:
        utilities = utilities.to(dtype=dtype)

    # Separate padding and actual selected tokens
    padding_mask = selected_indices < 0  # (*B, k)
    actual_indices = selected_indices.abs()
    # Note: since padding is done by repeating the last element
    # this will not lead to any dangerous doubling of selected values

    ### 1. Get log-space unnormalized weights of selected items
    selected_log_weights = torch.gather(utilities, -1, actual_indices)  # (*B, k)
    # Mask out padding entries so they don't contribute to the sum
    selected_log_weights = selected_log_weights.masked_fill(padding_mask, float("-inf"))

    ### 2. Compute the log-normalizers for each step
    # The normalizer is basically: logsumexp of those items
    # that are never selected, plus those which have not been selected
    # *yet* but will be later (double-reversed cumsum trick)

    # Compute log sum of never-selected utilities
    never_selected_utilities = utilities.scatter(-1, actual_indices, float("-inf"))
    log_sum_never_selected = torch.logsumexp(
        never_selected_utilities, dim=-1, keepdim=True
    )  # (*B, 1)  - same across all steps

    # Reverse cumsum of selected utilities
    # Note: padding entries are -inf -> don't affect the cumsum
    # TODO: this could be made more efficient if needed, see discussion
    # here: https://github.com/pytorch/pytorch/issues/33520
    # For now, using double flip since it should have better numerics.
    reverse_cumsum_selected = torch.flip(
        torch.logcumsumexp(torch.flip(selected_log_weights, dims=[-1]), dim=-1),
        dims=[-1],
    )  # (*B, k)

    # Compute log-space normalizers by adding never selected
    # and "not selected thus far"
    log_normalizers = torch.logaddexp(
        log_sum_never_selected.expand(*selected_indices.shape),
        reverse_cumsum_selected,
    )  # (*B, k)

    ### 3. Compute log probabilities
    log_probs = selected_log_weights - log_normalizers  # (*B, k)
    # Zero-out padding tokens' final contribution
    log_probs = log_probs.masked_fill(padding_mask, 0.0)

    #### 4. Sum over k to get total log-likelihood (padding entries contribute 0)
    return log_probs.sum(dim=-1)  # (*B,)


def bernoulli_batch_loglik(
    samples: torch.Tensor,
    utilities: torch.Tensor,
    mask_index: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Per-timestep log-likelihood of independent Bernoulli outcomes.

    :param samples: (*B, L) Bernoulli samples (0/1 or bool)
    :param utilities: (*B, L) Bernoulli logits (ungated)
    :param mask_index: (*B, L) valid positions to include in likelihood
    :param dtype: if not None, computation will be in this dtype
    :return: (*B,) log-likelihood per batch item
    """
    if samples.shape != utilities.shape:
        raise ValueError(
            f"Shape mismatch: samples {samples.shape} vs utilities {utilities.shape}"
        )
    if mask_index.shape != utilities.shape:
        raise ValueError(
            f"Shape mismatch: mask_index {mask_index.shape} vs utilities {utilities.shape}"
        )

    probs = torch.sigmoid(utilities) * mask_index
    p = probs.to(dtype=dtype)
    y = samples.to(dtype=p.dtype)

    eps = torch.finfo(p.dtype).eps
    p = p.clamp(min=eps, max=1.0 - eps)

    logp1 = torch.log(p)
    logp0 = torch.log1p(-p)
    ll = y * logp1 + (1.0 - y) * logp0
    ll = ll * mask_index  # already-unmasked positions should not contribute to the loss

    return ll.sum(dim=-1)


def _build_dpls_stage2_logits(
    stage1_utilities: torch.Tensor,
    stop_logit: float,
    first_choices: torch.Tensor,
    first_indices_logit: float,
) -> torch.Tensor:
    """Build DPLS Stage 2 logits for sampling or likelihood calculation.

    :param stage1_utilities: (*B, BL) policy utilities (already masked)
    :param stop_logit: stop logit value
    :param first_choices: (*B,) indices of first choices to exclude
    :param first_indices_logit: value to set valid first_choices to (e.g. +inf or -inf)
    :return: (*B, BL+1) masked utilities + stop_logit with first choice set to first_indices_logit
    """
    *B, _ = stage1_utilities.shape
    device = stage1_utilities.device
    dtype = stage1_utilities.dtype

    # Add STOP token to utilities
    stop_shape = (*B, 1)
    stop_logit_tensor = torch.full(stop_shape, stop_logit, device=device, dtype=dtype)
    stage2_logits = torch.cat([stage1_utilities, stop_logit_tensor], dim=-1)

    # Mask out first choice by setting logit to first_indices_logit and mask to False
    # Only process valid first choices (not -1 padding)
    valid_first_choices = first_choices >= 0
    if valid_first_choices.any():
        first_indices = torch.nonzero(valid_first_choices, as_tuple=True) + (
            first_choices[valid_first_choices],
        )
        stage2_logits[first_indices] = first_indices_logit

    # Handle done elements (first_choice == -1) - force STOP to be first action
    done_elements = first_choices == -1
    if done_elements.any():
        # For done elements, set ALL utilities (not STOP) to -inf using broadcasting
        # stage2_logits has shape (*B, BL+1) where last dim is STOP
        stage2_logits[..., :-1][done_elements] = float("-inf")

    return stage2_logits


def dpls_sample(
    utilities: torch.Tensor,
    stop_logit: float,
    mask_index: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample from the DPLS model using fully vectorized operations.

    Two-stage process:
    1. Forced first choice using standard multinomial sampling
    2. Sequential addition with STOP using one-shot Gumbel-Max ranking

    :param utilities: (B, BL) utility values for each position
    :param stop_logit: scalar stop utility
    :param mask_index: (B, BL) available positions to sample
    :param dtype: if not None, calculations are done in this dtype
    :return: (sequences, chosen_sets) where sequences is (B, max_K) padded tensor of chosen item indices in order
             and chosen_sets is (B, BL) boolean mask of chosen items
    """
    _, BL = utilities.shape
    device = utilities.device

    # Check which batch elements have valid choices (not "done")
    has_valid_choice = mask_index.any(dim=-1)  # (B,)

    ### Stage 1: Forced first choice (no STOP option)
    stage1_utilities = torch.where(
        has_valid_choice.unsqueeze(-1),
        utilities.masked_fill(~mask_index, float("-inf")),
        torch.zeros_like(utilities),  # Prevent multinomial crash for "done" elements
    )
    first_choice_probs = torch.softmax(
        stage1_utilities,
        dim=-1,
        dtype=dtype,
    )
    s1_indices = torch.multinomial(first_choice_probs, 1).squeeze(-1)  # (B,)
    # For "done" elements (no valid choices), override sampled index with -1 (padding)
    s1_indices = torch.where(has_valid_choice, s1_indices, -1)

    ### Stage 2: One-shot variable length sampling by Gumbel ranking with STOP
    # The intuition here is that we can simulate "sequential draws until STOP"
    # by adding gumbel noise, argsorting the list, and then cutting off at the STOP
    # token.
    second_stage_logits = _build_dpls_stage2_logits(
        stage1_utilities, stop_logit, s1_indices, float("inf")
    )
    gumbel_noise = gumbel_dist.sample(second_stage_logits.shape).to(device)
    perturbed_logits = second_stage_logits + gumbel_noise
    ranking = torch.argsort(perturbed_logits, dim=1, descending=True)  # (B, BL+1)

    # Find STOP token positions in ranking
    stop_token_id = BL  # by assumption the stop action is always the LAST action
    stop_positions = (ranking == stop_token_id).nonzero(as_tuple=True)[1]  # (B,)
    assert (stop_positions[has_valid_choice] > 0).all(), (
        "If has_valid_choice, should have at least 1 sample before STOP"
    )
    assert (stop_positions[~has_valid_choice] == 0).all(), (
        "If not has_valid_choice, STOP should be first action"
    )

    # Create mask for items before STOP token
    arange_mask = torch.arange(BL + 1, device=device).unsqueeze(0)  # (1, BL+1)
    k_mask = arange_mask < stop_positions.unsqueeze(1)  # (B, BL+1)
    # Get indices of tokens sampled before STOP
    samples = torch.where(k_mask, ranking, -1)  # (B, BL+1) -- padded with -1
    assert (samples < BL).all(), "samples should not contain BL === [STOP]"
    # Trim the last dimension since (BL+1)-th position is always -1 (padding)
    # The STOP token gets dynamically reconstructed in dpls_batch_loglik
    samples = samples[:, :BL]  # (B, BL)

    # Create chosen sets boolean mask
    # Since samples vector may contain padding (-1), we clamp to valid indices
    # but then dynamically set the value to True/False - making the scatter
    # a no-op at the padded indices
    chosen_sets = (
        torch.zeros_like(utilities, dtype=torch.int)
        .scatter_add(-1, samples.clamp(min=0), (samples >= 0).int())
        .bool()
    )

    return samples, chosen_sets


def dpls_batch_loglik(
    samples: torch.Tensor,
    utilities: torch.Tensor,
    stop_logit: float,
    mask_index: torch.Tensor,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute log-likelihood of DPLS sample sequences.

    :param samples: (*B, max_K) chosen sequences (padded with -1)
    :param utilities: (*B, BL) utility values
    :param stop_logit: scalar stop utility
    :param mask_index: (*B, BL) available positions
    :param dtype: if not None, calculations and result are in this dtype
    :return: (*B,) log-likelihoods
    """

    *B, max_K = samples.shape
    BL = utilities.shape[-1]
    device = utilities.device
    sequence_lengths = (samples != -1).sum(dim=-1)  # (*B,)
    # Initialize log-likelihood tensor with zeros by default (ie probability 1)
    loglik_stage1 = torch.zeros(*B, device=device, dtype=dtype or utilities.dtype)

    ### Stage 1: Log-likelihood of forced first choice
    first_choices = samples[..., 0]  # (*B,)
    valid_first_choices_mask = first_choices >= 0
    stage1_utilities = torch.where(
        valid_first_choices_mask.unsqueeze(-1),
        utilities.masked_fill(~mask_index, float("-inf")),
        torch.zeros_like(utilities),  # prevent error in log_softmax
    ).to(dtype=dtype)
    log_probs_stage1 = torch.log_softmax(
        stage1_utilities,
        dim=-1,
    )  # (*B, BL)
    # Gather logliks of only the valid (non padding) first choices
    loglik_stage1[valid_first_choices_mask] = torch.gather(
        log_probs_stage1, -1, first_choices.clamp(min=0).unsqueeze(-1)
    ).squeeze(-1)[valid_first_choices_mask]

    ### Stage 2: Log-likelihood for remaining choices using Plackett-Luce
    stage2_logits = _build_dpls_stage2_logits(
        stage1_utilities,
        stop_logit,
        first_choices,
        float("-inf"),
    )

    # Build stage 2 sequences: (s2, s3, ..., sK, STOP)
    stage2_samples = torch.cat(
        [
            samples[..., 1:],
            torch.full((*B, 1), -1, dtype=samples.dtype, device=device),
        ],
        dim=-1,
    )  # (*B, max_K)

    # Add STOP token at the end of each stage2 sequence (dynamic location)
    # Stage 2 length = original sequence length - 1 (we removed the first choice)
    stage2_lengths = sequence_lengths - 1  # (*B,)
    # Clamp to ensure STOP goes at position 0 for fully masked sequences
    stage2_lengths = stage2_lengths.clamp(min=0)  # (*B,)
    stage2_samples.scatter_(-1, stage2_lengths.unsqueeze(-1), BL)
    # NOTE: For fully masked sequences (sequence_lengths=0), STOP goes at position 0

    ### PADDING: Convert -1 padding to -BL (negative STOP token)
    # Makes downstream PL calculation significantly easier
    stage2_samples.masked_fill_(stage2_samples == -1, -BL)

    loglik_stage2 = plackett_luce_batch_loglik(
        selected_indices=stage2_samples,
        utilities=stage2_logits,
        mask_index=None,  # Logits already contain -inf for masked positions
        dtype=dtype,
    )  # (*B,)

    # Total log-likelihood is simply sum of both stages
    total_loglik = loglik_stage1 + loglik_stage2  # (*B,)

    return total_loglik


def categorical_sample(
    logits: torch.Tensor,
    feasible_mask: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Sample one action index per batch item from a categorical distribution.

    Used by the joint (block size, threshold) policy, where each decision picks one
    element of a small candidate set rather than a subset of positions.

    :param logits: (*B, K) action logits
    :param feasible_mask: (*B, K) bool, True where the action is selectable; None = all
    :param dtype: if not None, sampling will be done in this dtype
    :return: (*B,) long action indices
    """
    if dtype is not None:
        logits = logits.to(dtype=dtype)
    if feasible_mask is not None:
        logits = logits.masked_fill(~feasible_mask, float("-inf"))

    # multinomial only accepts 1D/2D, so flatten the leading batch dims
    probs = torch.softmax(logits.float(), dim=-1)
    samples = torch.multinomial(probs.reshape(-1, probs.shape[-1]), 1).squeeze(-1)
    return samples.view(logits.shape[:-1])


def categorical_batch_loglik(
    samples: torch.Tensor,
    logits: torch.Tensor,
    action_mask: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Log-likelihood of a categorical draw.

    :param samples: (*B,) long action indices
    :param logits: (*B, K) action logits
    :param action_mask: (*B,) bool, False for padded timesteps (contribute 0)
    :param dtype: if not None, computation will be in this dtype
    :return: (*B,) log-likelihood per batch item
    """
    if samples.shape != logits.shape[:-1]:
        raise ValueError(
            f"Shape mismatch: samples {samples.shape} vs logits {logits.shape}"
        )
    if dtype is not None:
        logits = logits.to(dtype=dtype)

    log_probs = torch.log_softmax(logits, dim=-1)  # (*B, K)
    ll = log_probs.gather(-1, samples.long().unsqueeze(-1)).squeeze(-1)  # (*B,)

    if action_mask is not None:
        # Padded timesteps must contribute exactly 0, so zero them *before* they can
        # poison the sum (a -inf gathered from an infeasible padded slot would give NaN).
        ll = torch.where(action_mask.bool(), ll, torch.zeros_like(ll))

    return ll


def categorical_entropy(
    logits: torch.Tensor,
    feasible_mask: torch.Tensor | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Entropy of a categorical distribution, for collapse monitoring.

    :param logits: (*B, K) action logits
    :param feasible_mask: (*B, K) bool, True where the action is selectable; None = all
    :param dtype: if not None, computation will be in this dtype
    :return: (*B,) entropy in nats
    """
    if dtype is not None:
        logits = logits.to(dtype=dtype)
    if feasible_mask is not None:
        logits = logits.masked_fill(~feasible_mask, float("-inf"))

    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    # xlogy gives 0 for p=0 instead of the 0 * -inf = NaN that p * log_p would produce
    return -torch.xlogy(probs, probs).sum(dim=-1)
