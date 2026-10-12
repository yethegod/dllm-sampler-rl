"""The --zero_window_cond eval ablation: zeroing a window-conditioned policy's window
embedding must turn it into the window_cond=False policy with the same weights.

What has to hold (CPU only):
  1. Before zeroing, the unmask logits depend on the window (the test is not vacuous).
  2. After zeroing they no longer depend on it, and the block logits are unchanged.
  3. They equal those of a window_cond=False DiTBlockUnmaskPolicy loaded with the
     same weights, which is the policy the ablation is meant to evaluate.

Run with:  python tests/test_zero_window_cond.py
"""

import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tests.test_block_unmask_window_cond import (  # noqa: E402
    _make_policy,
    _perturb,
    _random_inputs,
)


class ZeroWindowCond(unittest.TestCase):
    def test_zeroed_embedding_equals_unconditioned_policy(self):
        cond = _make_policy(window_cond=True)
        _perturb(cond)
        m, c, t, start, end = _random_inputs(2)
        wider = end + 4

        u_cond, b_cond = cond(m, c, t, start, end)
        u_wider, _ = cond(m, c, t, start, wider)
        self.assertFalse(torch.allclose(u_cond, u_wider), "window term is dead")

        with torch.no_grad():
            cond.window_embedding.weight.zero_()
        u_zero, b_zero = cond(m, c, t, start, end)
        u_zero_wider, _ = cond(m, c, t, start, wider)
        torch.testing.assert_close(u_zero, u_zero_wider)
        torch.testing.assert_close(b_zero, b_cond)
        self.assertFalse(torch.allclose(u_zero, u_cond))

        plain = _make_policy(window_cond=False)
        state = {
            k: v
            for k, v in cond.state_dict().items()
            if not k.startswith("window_embedding")
        }
        plain.load_state_dict(state)
        u_plain, b_plain = plain(m, c, t, start, None)
        torch.testing.assert_close(u_zero, u_plain)
        torch.testing.assert_close(b_zero, b_plain)


if __name__ == "__main__":
    unittest.main()
