"""Tests for the CadLLM port (common/generation/cadllm.py).

What is pinned here:

  1. the helpers reproduce CadLLM's own formulas on hand-picked inputs (block length,
     step budget, sawtooth threshold, commit rule);
  2. end to end against the ORIGINAL generate_cadllm from github.com/juchengshen/CadLLM
     (looked up at $CADLLM_REPO, default /u/zsun9/CadLLM; skipped if absent): with a
     stub dLLM whose logits depend only on (absolute position, forward index), the
     dual-KV-cache approximation has nothing to approximate, so both loops must commit
     the same tokens in the same order and spend the same NFE;
  3. through generate_unified: every position ends unmasked, NFE == steps_taken, and
     the recorded block sizes tile gen_length.

This repo has no pytest dependency, so the file is plain `unittest` and self-running.

Run with:  python tests/test_cadllm.py
"""

import importlib.util
import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

# Run directly (`python tests/...`) and sys.path[0] is tests/, not the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from common.generation import cadllm  # noqa: E402
from common.generation.generation import generate_unified  # noqa: E402

VOCAB = 1200  # >= CadLLM's max top-V of 1000
MASK_ID = VOCAB - 1
MAX_POS = 300  # prompt + gen_length in every test below


class StubDLLM:
    """Logits depend on (absolute position, forward index) only.

    Speaks both calling conventions: ours (full x, attention_mask) and CadLLM's dual
    cache (first call full x with use_cache, later calls the block only plus
    replace_position marking its absolute positions).
    """

    def __init__(self, seed):
        self.seed = seed
        self.calls = 0
        self.device = torch.device("cpu")

    def _logits_at(self, positions):
        g = torch.Generator().manual_seed(self.seed * 100003 + self.calls)
        # Always draw the full MAX_POS table, so a row's logits do not depend on how
        # many positions this particular call asked for.
        n = MAX_POS
        base = torch.randn(n, VOCAB, generator=g)
        # Per-row sharpness spans ~uniform to near one-hot, so confidences cover the
        # whole [0, 1] range and the threshold, block length and step budget all move.
        scale = torch.rand(n, 1, generator=g) * 12.0
        logits = base * scale
        # A two-token alternation on a subset of rows, so detect_repetition fires and
        # the adaptive top-V takes its repetition branch.
        rep = torch.rand(n, generator=g) < 0.3
        logits[rep, torch.arange(n)[rep] % 2] += 40.0
        logits[:, MASK_ID] = -1e4  # never predict the mask token
        return logits[positions].unsqueeze(0)

    def __call__(self, x, attention_mask=None, past_key_values=None, use_cache=False,
                 replace_position=None):
        if replace_position is not None:
            positions = replace_position[0].nonzero().squeeze(-1)
        else:
            positions = torch.arange(x.shape[1])
        out = SimpleNamespace(logits=self._logits_at(positions), past_key_values=None)
        self.calls += 1
        return out


class HelperTests(unittest.TestCase):
    def test_block_length(self):
        self.assertEqual(cadllm.next_block_length([], 256), 24)
        self.assertEqual(cadllm.next_block_length([], 10), 10)
        # 4 + int(60 * 0.5) = 34, mean of the last two only
        self.assertEqual(cadllm.next_block_length([0.0, 0.25, 0.75], 256), 34)
        self.assertEqual(cadllm.next_block_length([1.0, 1.0], 256), 64)
        self.assertEqual(cadllm.next_block_length([0.0, 0.0], 256), 4)

    def test_block_steps(self):
        self.assertEqual(cadllm.next_block_steps([], 40), 24)
        # conf 0.5 -> 24 + int(66 * 0.5) = 57, scaled by 48/24
        self.assertEqual(cadllm.next_block_steps([0.5, 0.5], 48), 114)
        self.assertEqual(cadllm.next_block_steps([1.0, 1.0], 4), 4)

    def test_task_settings(self):
        # eval_humaneval.sh: B0 48, S0 16, max_steps 32, block in [12, 96]
        he = cadllm.settings_for("humaneval")
        self.assertEqual(cadllm.next_block_length([], 256, he), 48)
        self.assertEqual(cadllm.next_block_length([1.0, 1.0], 256, he), 96)
        self.assertEqual(cadllm.next_block_length([0.0, 0.0], 256, he), 12)
        self.assertEqual(cadllm.next_block_steps([], 40, he), 16)
        # conf 0.5 -> 16 + int(16 * 0.5) = 24, scaled by 48/48
        self.assertEqual(cadllm.next_block_steps([0.5, 0.5], 48, he), 24)
        for task in ("gsm8k", "math", "mbpp", None):
            self.assertIs(cadllm.settings_for(task), cadllm.DEFAULT_SETTINGS)

    def test_threshold_sawtooth(self):
        self.assertAlmostEqual(cadllm.block_threshold(0, 24), 0.85)
        self.assertAlmostEqual(cadllm.block_threshold(12, 24), 0.625)
        self.assertAlmostEqual(cadllm.block_threshold(24, 24), 0.4)
        self.assertAlmostEqual(cadllm.block_threshold(50, 24), 0.4)

    def test_commit_rule(self):
        conf = torch.tensor([0.9, 0.2, 0.7, 0.95, 0.1], dtype=torch.float64)
        mask = torch.tensor([True, True, True, False, True])
        commit = cadllm.threshold_commit(conf, mask, 0.7)
        # position 3 is not masked, so it is never committed despite the top score
        self.assertEqual(commit.tolist(), [True, False, True, False, False])
        # nothing clears: the top masked position is still committed
        commit = cadllm.threshold_commit(conf, mask, 0.99)
        self.assertEqual(commit.tolist(), [True, False, False, False, False])


def _load_reference():
    repo = Path(os.environ.get("CADLLM_REPO", "/u/zsun9/CadLLM")) / "llada"
    path = repo / "cadllm_generate.py"
    if not path.exists():
        return None
    # The module imports its own modeling_llada at import time; only the pure-torch
    # decode function is needed, so load it with that import stubbed out.
    sys.modules.setdefault("model", SimpleNamespace())
    sys.modules.setdefault(
        "model.modeling_llada", SimpleNamespace(LLaDAModelLM=None)
    )
    spec = importlib.util.spec_from_file_location("cadllm_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.generate_cadllm


REFERENCE = _load_reference()


def _ours(prompt, gen_length, seed):
    stub = StubDLLM(seed)
    x = torch.full((1, prompt.shape[1] + gen_length), MASK_ID, dtype=torch.long)
    x[:, : prompt.shape[1]] = prompt
    steps = torch.zeros((1,), dtype=torch.int32)
    P = prompt.shape[1]
    blocks = cadllm.cadllm_loop(
        x, P, gen_length, MASK_ID, lambda: stub(x).logits[:, P:], steps
    )
    return x, int(steps.item()), blocks


@unittest.skipIf(REFERENCE is None, "CadLLM repo not found (set CADLLM_REPO)")
class MatchesReferenceTests(unittest.TestCase):
    def test_same_tokens_and_nfe(self):
        for seed in range(12):
            for gen_length in (64, 256):
                with self.subTest(seed=seed, gen_length=gen_length):
                    prompt = torch.randint(0, VOCAB - 1, (1, 7))
                    ref_x, ref_nfe, _ = REFERENCE(
                        StubDLLM(seed),
                        prompt,
                        initial_steps=24,
                        max_steps=90,
                        gen_length=gen_length,
                        initial_block_length=24,
                        temperature=0.0,
                        mask_id=MASK_ID,
                        max_block=64,
                        min_block=4,
                    )
                    x, nfe, blocks = _ours(prompt, gen_length, seed)
                    self.assertTrue(torch.equal(x, ref_x))
                    self.assertEqual(nfe, ref_nfe)
                    self.assertEqual(sum(blocks), gen_length)
                    self.assertFalse((x == MASK_ID).any())


class GenerateUnifiedTests(unittest.TestCase):
    def test_end_to_end(self):
        prompt = torch.randint(0, VOCAB - 1, (1, 9))
        res = generate_unified(
            StubDLLM(3),
            prompt,
            remasking="cadllm",
            gen_length=128,
            mask_id=MASK_ID,
            record_unmask_order=True,
        )
        gen = res.sequences[:, 9:]
        self.assertFalse((gen == MASK_ID).any())
        self.assertEqual(sum(res.block_sizes), 128)
        self.assertTrue(all(4 <= b <= 64 for b in res.block_sizes[:-1]))
        nfe = int(res.steps_taken.item())
        self.assertLessEqual(nfe, 128)
        self.assertEqual(int(res.unmask_order.max()) + 1, nfe)
        self.assertTrue((res.unmask_order >= 0).all())
        x, ref_nfe, blocks = _ours(prompt, 128, 3)
        self.assertTrue(torch.equal(res.sequences, x))
        self.assertEqual(nfe, ref_nfe)
        self.assertEqual(res.block_sizes, blocks)

    def test_rejects_batch_and_temperature(self):
        with self.assertRaises(ValueError):
            generate_unified(StubDLLM(0), torch.zeros((2, 4), dtype=torch.long),
                             remasking="cadllm", mask_id=MASK_ID)
        with self.assertRaises(ValueError):
            generate_unified(StubDLLM(0), torch.zeros((1, 4), dtype=torch.long),
                             remasking="cadllm", mask_id=MASK_ID, temperature=0.5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
