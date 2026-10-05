"""Tests for the position-dependent Fast-dLLM threshold (remasking='fastdllm' with
thres_slope != 0), the training-free sampler distilled from the learned unmask head.

The learned head's STOP crossing rises along the answer (0.57 / 0.70 / 0.76 / 0.81 per
64-token quarter on GSM8K, nearly the same on MATH500), so the heuristic is Fast-dLLM
with tau(i) = thres + thres_slope * ((i + 0.5) / L - 0.5). What has to hold, on CPU:
  1. position_threshold has mean `thres` and rises by `thres_slope` across the answer;
     slope 0 is a constant.
  2. The rule compares each candidate with its own position's threshold, keeps the
     one-token fallback per row, and slope 0 reproduces plain Fast-dLLM token for token.
  3. The knob round-trips through the baseline name (-s<slope>, sign allowed) and is
     rejected for any remasking other than fastdllm.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.
Run with:  python tests/test_position_threshold.py
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation.generation import (  # noqa: E402
    _confidence_threshold_unmask_rowwise,
    generate_unified,
    position_threshold,
)
from eval.eval import parse_baseline_checkpoint  # noqa: E402

VOCAB = 40
MASK_ID = VOCAB - 1
L = 64


class StubDLLM:
    """Logits depend on (absolute position, forward index) only; never predicts mask."""

    def __init__(self, seed):
        self.seed = seed
        self.calls = 0
        self.dtype = torch.float32
        self.config = SimpleNamespace(hidden_size=8)

    def __call__(self, input_ids, attention_mask=None, output_hidden_states=False):
        B, S = input_ids.shape
        g = torch.Generator().manual_seed(self.seed * 100003 + self.calls)
        self.calls += 1
        logits = torch.randn(S, VOCAB, generator=g) * (torch.rand(S, 1, generator=g) * 10.0)
        logits[:, MASK_ID] = -1e4
        return SimpleNamespace(logits=logits.unsqueeze(0).expand(B, S, VOCAB).clone())


def _run(thres, slope, block=16, **kw):
    prompt = torch.randint(0, MASK_ID, (2, 5), generator=torch.Generator().manual_seed(1))
    return generate_unified(
        StubDLLM(3),
        prompt,
        remasking=kw.pop("remasking", "fastdllm"),
        thres=thres,
        thres_slope=slope,
        gen_length=L,
        block_length=block,
        mask_id=MASK_ID,
        record_unmask_order=True,
        **kw,
    )


class TestSchedule(unittest.TestCase):
    def test_mean_and_rise(self):
        t = position_threshold(0.7, 0.32, 256)
        self.assertEqual(tuple(t.shape), (1, 256))
        self.assertAlmostEqual(float(t.mean()), 0.7, places=5)
        self.assertAlmostEqual(float(t[0, -1] - t[0, 0]), 0.32 * 255 / 256, places=5)
        self.assertTrue(bool((t[0, 1:] > t[0, :-1]).all()))
        # Close to the learned GSM8K crossing at the quarter midpoints.
        for i, ref in zip((32, 96, 160, 224), (0.567, 0.697, 0.757, 0.811)):
            self.assertLess(abs(float(t[0, i]) - ref), 0.04)

    def test_slope_zero_is_constant_and_clamped(self):
        self.assertTrue(torch.allclose(position_threshold(0.9, 0.0, 8), torch.full((1, 8), 0.9)))
        t = position_threshold(0.9, 0.4, 8)
        self.assertLessEqual(float(t.max()), 1.0)
        self.assertGreaterEqual(float(position_threshold(0.1, -0.4, 8).min()), 0.0)


class TestRule(unittest.TestCase):
    def test_each_position_against_its_own_threshold(self):
        conf = torch.tensor([[0.60, 0.60, 0.60, 0.60], [0.20, 0.10, 0.30, 0.05]])
        probs = torch.stack([conf, 1 - conf], dim=-1)  # max over V is conf (all >= 0.5 here)
        probs[1] = torch.tensor([[0.2, 0.0], [0.1, 0.0], [0.3, 0.0], [0.05, 0.0]])
        thres = torch.tensor([[0.5, 0.55, 0.65, 0.7]])  # rising
        block = torch.ones(2, 4, dtype=torch.bool)
        out = _confidence_threshold_unmask_rowwise(block, probs, thres)
        self.assertEqual(out[0].tolist(), [True, True, False, False])
        # Row 1 clears nothing: one forced token, its most confident candidate.
        self.assertEqual(out[1].tolist(), [False, False, True, False])

    def test_slope_zero_matches_plain_fastdllm(self):
        a = _run(0.7, 0.0)
        b = _run(torch.tensor(0.7), 0.0)
        self.assertTrue(torch.equal(a.sequences, b.sequences))
        self.assertTrue(torch.equal(a.steps_taken, b.steps_taken))

    def test_rising_schedule_completes_and_changes_decoding(self):
        flat = _run(0.7, 0.0)
        rise = _run(0.7, 0.32)
        fall = _run(0.7, -0.32)
        for r in (rise, fall):
            self.assertFalse(bool((r.sequences[:, 5:] == MASK_ID).any()))
            self.assertTrue(bool((r.unmask_order >= 0).all()))
        self.assertFalse(torch.equal(flat.unmask_order, rise.unmask_order))

    def test_adaptive_block_accepts_the_schedule(self):
        prompt = torch.randint(0, MASK_ID, (1, 5), generator=torch.Generator().manual_seed(2))
        res = generate_unified(
            StubDLLM(4), prompt, remasking="fastdllm", thres=0.7, thres_slope=0.32,
            gen_length=L, block_length=16, mask_id=MASK_ID, adaptive_block=True,
            delimiter_ids=(7,), delimiter_threshold=0.05,
        )
        self.assertFalse(bool((res.sequences[:, 5:] == MASK_ID).any()))
        self.assertEqual(sum(res.block_sizes), L)

    def test_rejected_outside_fastdllm(self):
        with self.assertRaises(ValueError):
            _run(0.7, 0.32, remasking="low_confidence", steps=L)


class TestBaselineName(unittest.TestCase):
    def test_parse(self):
        p = parse_baseline_checkpoint("baseline-fastdllm-t0.7-s0.32")
        self.assertEqual((p["method"], p["thres"], p["thres_slope"]), ("fastdllm", 0.7, 0.32))
        p = parse_baseline_checkpoint("checkpoint-baseline-fastdllm-t0.7-s-0.32")
        self.assertEqual((p["thres"], p["thres_slope"]), (0.7, -0.32))
        p = parse_baseline_checkpoint("baseline-fastdllm-t0.7-s0.32-ada0.3")
        self.assertEqual((p["thres_slope"], p["delimiter_threshold"]), (0.32, 0.3))
        for name in ("baseline-fastdllm-t0.9", "baseline-fastdllm-t0.9-ada0.3", "baseline-cadllm"):
            self.assertNotIn("thres_slope", parse_baseline_checkpoint(name), name)


if __name__ == "__main__":
    unittest.main()
