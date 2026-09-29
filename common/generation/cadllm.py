"""CadLLM decoding (arXiv:2512.07173), ported for in-harness comparison.

Source: github.com/juchengshen/CadLLM, llada/cadllm_generate.py:generate_cadllm, with
the settings of its llada/eval_gsm8k.sh and eval_math.sh (identical for both tasks):
initial_block_length 24, initial_steps 24, max_steps 90, block size in [4, 64],
softmax confidence, adaptive block / steps / vocab / threshold all on, factor and
prophet off, temperature 0.

We follow the released CODE where it disagrees with the paper:
- the threshold is a within-block sawtooth, max(0.85 - 0.45 * i / S_t, 0.4) with i the
  forward index inside the block, reset at every block (paper Eq. 5 decays it once
  per block on global progress);
- S_t never caps the step count: a block is decoded until it has no mask left, and
  S_t only sets the slope of that sawtooth (the per-step quota it computes is dead
  code once a threshold is set);
- a step commits its top-1 masked position plus every other masked position whose
  confidence reaches the threshold.

One deliberate difference: CadLLM runs with a dual KV cache, so every forward after a
block's first sees stale keys/values outside the block. We run a full forward every
step (no cache), same as every other method in this harness. The decoding rule is
unchanged; only the cache approximation is gone, and NFE counts the same forwards.
"""

import numpy as np
import torch
import torch.nn.functional as F

INITIAL_BLOCK_LENGTH = 24
INITIAL_STEPS = 24
MAX_STEPS = 90
MAX_BLOCK = 64
MIN_BLOCK = 4
INITIAL_THRESHOLD = 0.85
MIN_THRESHOLD = 0.4
REPETITION_WINDOW = 50  # recently committed tokens kept for repetition detection


def detect_repetition(generated_tokens, window_size=8, min_repeat_length=2):
    """Repetition score in [0, 1] over the last window_size committed tokens (verbatim)."""
    if len(generated_tokens) < window_size:
        return 0.0

    recent_tokens = generated_tokens[-window_size:]

    if len(set(recent_tokens[-4:])) == 1:
        return 1.0

    max_repetition = 0.0
    for repeat_len in range(min_repeat_length, window_size // 2 + 1):
        if len(recent_tokens) >= repeat_len * 2:
            pattern = recent_tokens[-repeat_len:]
            prev_pattern = recent_tokens[-repeat_len * 2 : -repeat_len]
            if pattern == prev_pattern:
                max_repetition = max(max_repetition, repeat_len / window_size)

    return max_repetition


def adaptive_vocab_size(
    generation_progress,
    confidence_history,
    generated_tokens,
    initial_vocab_size=100,
    min_vocab_size=35,
    max_vocab_size=1000,
):
    """Top-V size for the confidence softmax (calculate_adaptive_vocab_size, verbatim).

    The defaults are the ones the threshold path passes; the factor path uses 15/5/35.
    """
    if generation_progress < 0.2:
        phase_vocab_size = max_vocab_size * 0.8
    elif generation_progress < 0.7:
        phase_vocab_size = initial_vocab_size * 0.7
    else:
        phase_vocab_size = initial_vocab_size * 0.9

    if confidence_history:
        recent_confidence = np.mean(confidence_history[-5:])
        confidence_trend = 0
        if len(confidence_history) >= 3:
            confidence_trend = confidence_history[-1] - confidence_history[-3]

        if recent_confidence < 0.3:
            confidence_factor = 1.5
        elif recent_confidence < 0.6:
            confidence_factor = 1.2
        elif recent_confidence > 0.8:
            confidence_factor = 0.8
        else:
            confidence_factor = 1.0

        if confidence_trend < -0.1:
            confidence_factor *= 1.3

        phase_vocab_size *= confidence_factor

    repetition_score = detect_repetition(generated_tokens)
    if repetition_score > 0.5:
        phase_vocab_size *= 1.0 + (repetition_score * 2.0)

    return int(np.clip(phase_vocab_size, min_vocab_size, max_vocab_size))


def topv_confidence(logits, x0, vocab_size):
    """p(x0) under a softmax restricted to the top-V logits (softmax_confidence_adaptive).

    :param logits: (..., V_full) raw logits
    :param x0: (...) committed-token candidates
    :return: (...) float64 confidence, 0 where x0 is outside the top V
    """
    vals, idx = torch.topk(logits, k=vocab_size, dim=-1)
    p = F.softmax(vals.to(torch.float64), dim=-1)
    return (p * (idx == x0.unsqueeze(-1))).sum(dim=-1)


def block_threshold(i, block_steps):
    """Sawtooth threshold for the i-th forward inside a block (calculate_adaptive_threshold)."""
    progress = i / max(block_steps, 1)
    threshold = INITIAL_THRESHOLD * (1 - progress) + MIN_THRESHOLD * progress
    return max(threshold, MIN_THRESHOLD)


def _recent_confidence(confidence_history):
    # torch float32 mean, as in the original, so int() truncation lands identically
    return torch.tensor(confidence_history[-2:]).mean().item()


def next_block_length(confidence_history, remaining):
    """calculate_adaptive_block_size with adaptive_blocks=True."""
    if confidence_history:
        block_length = MIN_BLOCK + int(
            (MAX_BLOCK - MIN_BLOCK) * _recent_confidence(confidence_history)
        )
    else:
        block_length = INITIAL_BLOCK_LENGTH
    block_length = max(MIN_BLOCK, min(block_length, MAX_BLOCK))
    return min(block_length, remaining)


def next_block_steps(confidence_history, block_length):
    """calculate_adaptive_step with adaptive_steps=True."""
    if not confidence_history:
        return INITIAL_STEPS
    avg_confidence = max(0.0, min(1.0, float(_recent_confidence(confidence_history))))
    steps_for_conf = INITIAL_STEPS + int((MAX_STEPS - INITIAL_STEPS) * (1.0 - avg_confidence))
    steps_for_conf = max(INITIAL_STEPS, min(steps_for_conf, MAX_STEPS))
    return max(1, int(steps_for_conf * block_length / INITIAL_BLOCK_LENGTH))


def threshold_commit(confidence, mask, threshold):
    """Top-1 masked position always, plus every other masked one at >= threshold.

    :param confidence: (N,) confidence over the block
    :param mask: (N,) bool, still-masked positions (at least one True)
    :return: (N,) bool positions to commit
    """
    conf = torch.where(mask, confidence, torch.full_like(confidence, -np.inf))
    commit = mask & (conf >= threshold)
    commit[torch.topk(conf, k=1).indices] = True
    return commit


def cadllm_loop(x, prompt_L, L, mask_id, gen_logits, steps_taken, record_order=None):
    """Decode x[:, prompt_L:] in place with CadLLM's rule. Batch size 1 only.

    :param gen_logits: callable returning (1, L, V) logits over the generation region
        for the current x
    :param steps_taken: (1,) int counter, incremented once per forward (NFE)
    :param record_order: optional callback taking the (1, L) bool commit mask, called
        before steps_taken is incremented
    :return: list of the block lengths used, in order
    """
    confidence_history = []
    generated_tokens = []
    block_sizes = []
    pos = 0
    while pos < L:
        progress = pos / L
        block_length = next_block_length(confidence_history, L - pos)
        block_steps = next_block_steps(confidence_history, block_length)
        end = pos + block_length
        region = slice(prompt_L + pos, prompt_L + end)

        i = 0
        while (x[0, region] == mask_id).any():
            logits = gen_logits()[0, pos:end]  # (b, V)
            mask = x[0, region] == mask_id

            step_progress = progress + (i / block_steps) * (block_length / L)
            vocab_size = adaptive_vocab_size(
                step_progress, confidence_history, generated_tokens
            )
            x0 = torch.argmax(logits, dim=-1)
            confidence = topv_confidence(logits, x0, vocab_size)
            commit = threshold_commit(confidence, mask, block_threshold(i, block_steps))

            confidence_history.append(confidence[mask].mean().item())
            generated_tokens.extend(x0[commit].tolist())
            generated_tokens = generated_tokens[-REPETITION_WINDOW:]

            x[0, region] = torch.where(commit, x0, x[0, region])
            if record_order is not None:
                full = torch.zeros((1, L), dtype=torch.bool, device=x.device)
                full[0, pos:end] = commit
                record_order(full)
            steps_taken += 1
            i += 1

        block_sizes.append(block_length)
        pos = end
    return block_sizes
