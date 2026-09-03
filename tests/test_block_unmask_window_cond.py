"""Tests for the window-conditioned block_unmask_policy (block_unmask_window_cond).

The policy factorises p(b, u | s) = p(b | s) p(u | s, b): the block head reads a trunk
pass without the window term, the unmask head a second pass with it. Three things
have to hold for the GRPO loss to be right, none of which needs a GPU:

  1. With the zero-initialised window embedding the conditioned policy is numerically
     the conditionally independent one, so a run with the flag starts where the
     control (job 3076743) started.
  2. Once the window embedding is non-zero, block_end moves the unmask logits and
     nothing else -- the block head must not see the window it is about to choose.
  3. What the decode loop records is exactly what the trainer replays: the 5-tuple of
     policy inputs fed back through `policy(*inputs)` reproduces the rollout's logits
     and hence the old log-probs, so the ratio is 1 on the first inner iteration.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.

Run with:  python tests/test_block_unmask_window_cond.py
"""

import sys
import unittest
from pathlib import Path

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402 - needs the sys.path line above
    _block_unmask_policy_loop,
)
from common.generation.sampling import (  # noqa: E402
    bernoulli_batch_loglik,
    categorical_batch_loglik,
)
from common.models.policy import DiTBlockUnmaskPolicy, PolicyHFWrapper  # noqa: E402

L = 32
P = 2  # confidences_top_p
CANDIDATES = (8, 16, 32)
MASK_ID = 0
POLICY_KW = dict(
    block_size_candidates=CANDIDATES,
    hidden_dim=16,
    feedforward_dim=32,
    num_heads=2,
    time_embed_dim=16,
    confidences_top_p=P,
    smart_init=-1.0,
)


def _make_policy(window_cond: bool, seed: int = 0) -> DiTBlockUnmaskPolicy:
    torch.manual_seed(seed)
    return DiTBlockUnmaskPolicy(window_cond=window_cond, **POLICY_KW).eval()


def _random_inputs(B: int, seed: int = 1):
    g = torch.Generator().manual_seed(seed)
    m = torch.rand((B, L), generator=g) < 0.6
    c = torch.rand((B, L, P), generator=g)
    t = torch.rand((B, 1), generator=g)
    start = torch.tensor([[0], [8]][:B], dtype=torch.long)
    end = start + torch.tensor([[16], [8]][:B], dtype=torch.long)
    return m, c, t, start, end


def _perturb(policy: DiTBlockUnmaskPolicy, seed: int = 2):
    """Give every zero-initialised head real weights so the tests are not vacuous."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, p in policy.named_parameters():
            if name.startswith(("window_embedding", "boundary_proj")) or (
                "ada_conditioning" in name
            ):
                p.add_(0.5 * torch.randn(p.shape, generator=g))


# --------------------------------------------------------------------------------
# 1. Zero init: the conditioned policy starts as the independent one
# --------------------------------------------------------------------------------


class TestZeroInit(unittest.TestCase):
    def test_window_cond_matches_independent_at_init(self):
        indep = _make_policy(window_cond=False)
        cond = _make_policy(window_cond=True)
        missing, unexpected = cond.load_state_dict(indep.state_dict(), strict=False)
        self.assertEqual(unexpected, [])
        self.assertEqual(missing, ["window_embedding.weight"])
        self.assertTrue(torch.all(cond.window_embedding.weight == 0))

        m, c, t, start, end = _random_inputs(B=2)
        u0, b0 = indep(m, c, t, start, end)
        u1, b1 = cond(m, c, t, start, end)
        torch.testing.assert_close(u1, u0)
        torch.testing.assert_close(b1, b0)

    def test_independent_policy_ignores_block_end(self):
        indep = _make_policy(window_cond=False)
        m, c, t, start, end = _random_inputs(B=2)
        u_none, b_none = indep(m, c, t, start, None)
        u_end, b_end = indep(m, c, t, start, end)
        torch.testing.assert_close(u_none, u_end)
        torch.testing.assert_close(b_none, b_end)


# --------------------------------------------------------------------------------
# 2. Conditioning goes to the unmask head only
# --------------------------------------------------------------------------------


class TestConditioning(unittest.TestCase):
    def test_block_end_moves_unmask_logits_not_block_logits(self):
        cond = _make_policy(window_cond=True)
        _perturb(cond)
        m, c, t, start, end = _random_inputs(B=2)
        end_other = start + torch.tensor([[8], [16]], dtype=torch.long)

        u1, b1 = cond(m, c, t, start, end)
        u2, b2 = cond(m, c, t, start, end_other)
        torch.testing.assert_close(b1, b2)
        self.assertFalse(torch.allclose(u1, u2))

    def test_window_cond_requires_block_end(self):
        cond = _make_policy(window_cond=True)
        m, c, t, start, _ = _random_inputs(B=2)
        with self.assertRaises(AssertionError):
            cond(m, c, t, start, None)

    def test_heads_match_forward(self):
        cond = _make_policy(window_cond=True)
        _perturb(cond)
        m, c, t, start, end = _random_inputs(B=2)
        u, b = cond(m, c, t, start, end)
        torch.testing.assert_close(cond.unmask_logits(m, c, t, start, end), u)
        torch.testing.assert_close(cond.block_logits(m, c, t, start), b)


# --------------------------------------------------------------------------------
# 3. Rollout record replays exactly through the policy
# --------------------------------------------------------------------------------


def _run_loop(policy: DiTBlockUnmaskPolicy, B: int, seed: int = 3):
    torch.manual_seed(seed)
    prompt_L = 3
    V = 5
    x = torch.full((B, prompt_L + L), MASK_ID, dtype=torch.long)
    x[:, :prompt_L] = 1
    steps_taken = torch.zeros(B, dtype=torch.int)
    g = torch.Generator().manual_seed(seed)

    def forward_logits():
        logits = torch.randn((B, L, V), generator=g)
        logits[..., MASK_ID] = -1e9  # never predict the mask token
        probs = torch.softmax(logits, dim=-1)
        return None, probs, probs.argmax(dim=-1)

    rec = _block_unmask_policy_loop(
        x,
        prompt_L,
        L,
        MASK_ID,
        L,  # max_steps
        steps_taken,
        policy,
        forward_logits,
        None,
        CANDIDATES,
        "bernoulli",
        "categorical",
        P,
        1.0,
    )
    return x, rec


class TestRecordReplay(unittest.TestCase):
    def setUp(self):
        self.policy = _make_policy(window_cond=True)
        _perturb(self.policy)
        with torch.no_grad():
            self.x, self.rec = _run_loop(self.policy, B=2)

    def test_makes_progress(self):
        # Training-mode bernoulli has no stall fallback, so a short loop need not
        # finish; it must decode something, in every row, and never touch the prompt.
        gen = self.x[:, 3:]
        self.assertTrue(bool((gen != MASK_ID).any(dim=-1).all()))
        self.assertTrue(bool((self.x[:, :3] == 1).all()))

    def test_policy_inputs_carry_the_decided_window(self):
        inputs = self.rec["policy_inputs"]
        self.assertEqual(len(inputs), 5)
        _, _, _, start, end = inputs
        B, T, _ = start.shape
        self.assertEqual(end.shape, (B, T, 1))
        self.assertEqual(end.dtype, torch.long)

        decided = self.rec["block_decisions"]
        chosen = self.rec["block_sizes_chosen"]
        self.assertTrue(bool(decided.any()))
        # On decision steps the recorded window is exactly the block just chosen ...
        self.assertTrue(
            torch.equal((end - start)[..., 0][decided], chosen[decided]),
        )
        # ... and the chosen sizes all come from the candidate set.
        for b in chosen[decided].tolist():
            self.assertIn(b, CANDIDATES)
        # The bernoulli slot is only ever sampled inside that window.
        positions = torch.arange(L)
        window = (positions >= start) & (positions < end)
        in_block_mask = self.rec["sampling_masks"][..., :L]
        self.assertFalse(bool((in_block_mask & ~window).any()))

    def test_replay_reproduces_rollout_logits_and_loglik(self):
        inputs = self.rec["policy_inputs"]
        with torch.no_grad():
            unmask_logits, block_logits = self.policy(*inputs)
        rec_unmask = self.rec["sampling_inputs"][..., :L]
        rec_block = self.rec["sampling_inputs"][..., L:]
        masks = self.rec["sampling_masks"]
        pos_mask = masks[..., :L]
        decided = masks[..., L]

        torch.testing.assert_close(unmask_logits[pos_mask], rec_unmask[pos_mask])
        torch.testing.assert_close(block_logits[decided], rec_block[decided])

        samples = self.rec["samples"]

        def joint(logits_u, logits_b):
            return bernoulli_batch_loglik(
                samples[..., :L], logits_u, mask_index=pos_mask, dtype=torch.float32
            ) + categorical_batch_loglik(
                samples[..., L], logits_b, action_mask=decided, dtype=torch.float32
            )

        old = joint(rec_unmask, rec_block)
        new = joint(unmask_logits, block_logits)
        self.assertTrue(torch.isfinite(old).all())
        torch.testing.assert_close(new, old)

    def test_off_decision_steps_contribute_no_block_term(self):
        masks = self.rec["sampling_masks"]
        decided = masks[..., L]
        rec_block = self.rec["sampling_inputs"][..., L:]
        # The loop records zeros in the block slot when no row decides; the loss must
        # never read them.
        ll = categorical_batch_loglik(
            self.rec["samples"][..., L], rec_block, action_mask=decided
        )
        self.assertTrue(bool((ll[~decided] == 0).all()))


# --------------------------------------------------------------------------------
# 4. The wrapper the trainer and eval actually hand to the loop
# --------------------------------------------------------------------------------


class TestThroughWrapper(unittest.TestCase):
    """train.train and eval.eval pass a PolicyHFWrapper, not the bare policy, so the
    loop's per-head calls must resolve on the wrapper and go through its dtype cast."""

    def test_loop_and_replay_through_wrapper(self):
        core = _make_policy(window_cond=True)
        _perturb(core)
        wrapper = PolicyHFWrapper(core, "dit_block_unmask").eval()
        with torch.no_grad():
            _, rec = _run_loop(wrapper, B=2)
            unmask_logits, block_logits = wrapper(*rec["policy_inputs"])
        masks = rec["sampling_masks"]
        pos_mask, decided = masks[..., :L], masks[..., L]
        torch.testing.assert_close(
            unmask_logits[pos_mask], rec["sampling_inputs"][..., :L][pos_mask]
        )
        torch.testing.assert_close(
            block_logits[decided], rec["sampling_inputs"][..., L:][decided]
        )

    def test_wrapper_casts_float_inputs_only(self):
        core = _make_policy(window_cond=True)
        wrapper = PolicyHFWrapper(core, "dit_block_unmask").to(torch.bfloat16).eval()
        m, c, t, start, end = _random_inputs(B=2)  # c, t are float32; indices long
        with torch.no_grad():
            b = wrapper.block_logits(m, c, t, start)
            u = wrapper.unmask_logits(m, c, t, start, end)
        self.assertEqual(b.dtype, torch.bfloat16)
        self.assertEqual(u.dtype, torch.bfloat16)
        self.assertEqual(u.shape, (2, L))
        self.assertEqual(b.shape, (2, len(CANDIDATES)))


if __name__ == "__main__":
    unittest.main()
