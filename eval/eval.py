#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import argparse
import json
import math
import os
import random
import re
import time
from pathlib import Path

import evaluate as hf_evaluate
import numpy as np
import torch
import torch.distributed as dist
from accelerate import Accelerator
from accelerate.utils import gather_object
from safetensors.torch import load_file
from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from tqdm import tqdm
from transformers import AutoModel
from transformers import AutoTokenizer
from trl import TrlParser

from common.config import Config
from common.generation.generation import generate_unified
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTHiddenProjPolicy
from common.models.policy import DiTBlockSizePolicy
from common.models.policy import DiTBlockSizeHiddenProjPolicy
from common.models.policy import DiTBlockUnmaskPolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from data.loaders.gsm8k import GSM8KDataset
from data.loaders.humaneval import HumanEvalDataset
from data.loaders.math500 import MATH500Dataset
from data.loaders.mbpp import MBPPDataset
from data.sanitize import sanitize_humaneval
from data.sanitize import sanitize_mbpp

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

DATASET_MAP = {
    "gsm8k": GSM8KDataset,
    "math": MATH500Dataset,
    "humaneval": HumanEvalDataset,
    "mbpp": MBPPDataset,
}


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}

FEW_SHOT_DEFAULTS = {
    "gsm8k": 0,  # NOTE: Fast-dLLM uses 5
    "math": 0,  # NOTE: Fast-dLLM uses 4
    "humaneval": 0,
    "mbpp": 3,
}


def init_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def parse_baseline_checkpoint(name):
    name = name.replace("checkpoint-", "")
    if not name.startswith("baseline-"):
        return None

    params = {"method": name.split("-")[1]}

    # Extract K<number> (tokens per step)
    if match := re.search(r"K(\d+)", name):
        params["diffusion_steps"] = int(match.group(1))

    # Extract t<number> (threshold)
    if match := re.search(r"t([\d.]+)", name):
        params["thres"] = float(match.group(1))

    # Extract ada<number> (adaptive block, AdaBlock delimiter threshold)
    if match := re.search(r"ada([\d.]+)", name):
        params["adaptive_block"] = True
        params["delimiter_threshold"] = float(match.group(1))

    return params


def evaluate(
    model,
    tokenizer,
    dataloader,
    dataset_name,
    accelerator=None,
    policy=None,
    gen_length=128,
    temperature=0.0,
    steps=64,
    block_length=32,
    remasking="low_confidence",
    thres=0.7,
    sampling_mode="bernoulli",
    dpls_stop_logit=0.0,
    temperature_policy=1.0,
    policy_full_context=True,
    confidences_top_p=1,
    mask_id=126336,
    model_type=None,
    adaptive_block=False,
    delimiter_ids=(198,),
    delimiter_threshold=0.3,
    block_size_candidates=(8, 16, 32, 64, 128),
    threshold_candidates=(0.5, 0.7, 0.9),
    block_schedule=None,
    block_sampling_mode="categorical",
):
    model.eval()
    total_processed = torch.tensor(0, device=model.device)
    wall_times = []
    all_generations = []
    device = model.device

    is_code_dataset = dataset_name in ["humaneval", "mbpp"]

    with torch.no_grad():
        for batch in tqdm(
            dataloader,
            disable=(not accelerator.is_main_process if accelerator else False),
        ):
            start_time = time.time()
            input_ids = batch["input_ids"].to(device)

            attn_masks = batch["attention_mask"].bool().to(device)
            prompts = batch["prompts"]

            if is_code_dataset:
                if dataset_name == "humaneval":
                    raw_prompts = batch["raw_prompts"]
                    task_ids = batch["task_ids"]
                    test_cases = batch["test_cases"]
                    entry_points = batch["entry_points"]
                elif dataset_name == "mbpp":
                    raw_prompts = batch["texts"]
                    task_ids = batch["task_ids"]
                    test_cases = batch["test_cases"]
                    entry_points = [None] * len(task_ids)
            else:
                gt_answers = batch["answers"]
                questions = batch["questions"]

            gen_kwargs = {
                "model": model,
                "prompt": input_ids,
                "remasking": remasking,
                "gen_length": gen_length,
                "block_length": block_length,
                "temperature": temperature,
                "mask_id": mask_id,
                "model_type": model_type,
                "attention_mask": attn_masks,
            }

            if adaptive_block:
                gen_kwargs.update(
                    {
                        "adaptive_block": True,
                        "delimiter_ids": tuple(delimiter_ids),
                        "delimiter_threshold": delimiter_threshold,
                    }
                )

            if remasking in ("policy", "block_policy", "block_unmask_policy"):
                if policy is None:
                    raise ValueError(
                        f"{remasking} remasking requires a policy to be provided"
                    )
                gen_kwargs.update(
                    {
                        "policy": policy,
                        "sampling_mode": sampling_mode,
                        "dpls_stop_logit": dpls_stop_logit,
                        "temperature_policy": temperature_policy,
                        "full_context": policy_full_context,
                        "confidences_top_p": confidences_top_p,
                    }
                )
            elif remasking == "fastdllm":
                gen_kwargs["thres"] = thres
            elif remasking != "block_schedule":
                gen_kwargs["steps"] = steps

            if remasking in ("block_policy", "block_schedule"):
                # block_length is ignored: the block size is picked per block, along
                # with the Fast-dLLM threshold.
                gen_kwargs.update(
                    {
                        "block_size_candidates": tuple(block_size_candidates),
                        "threshold_candidates": tuple(threshold_candidates),
                    }
                )
                if remasking == "block_schedule":
                    gen_kwargs["block_schedule"] = tuple(block_schedule)
            elif remasking == "block_unmask_policy":
                # Block size from the policy, unmasking from its per-position head:
                # neither block_length nor a threshold applies.
                gen_kwargs.update(
                    {
                        "block_size_candidates": tuple(block_size_candidates),
                        "block_sampling_mode": block_sampling_mode,
                    }
                )

            result = generate_unified(**gen_kwargs)
            out = result.sequences

            if remasking in (
                "policy",
                "block_policy",
                "block_schedule",
                "block_unmask_policy",
            ):
                steps_taken = result.steps_taken.tolist()
            elif remasking == "fastdllm":
                steps_taken = [result.steps_taken.item()]
            else:
                steps_taken = [result.steps_taken.item()] * len(input_ids)

            generated_texts = tokenizer.batch_decode(
                out[:, -gen_length:], skip_special_tokens=True
            )

            avg_block_size = None
            if result.block_sizes:
                avg_block_size = sum(result.block_sizes) / len(result.block_sizes)

            # Per-example decoding schedules. adaptive_block runs at batch size 1 and
            # reports a single shared list; block_policy chooses per row, so unpack
            # each row's real decisions (padded slots are dropped via sampling_masks).
            n_out = len(generated_texts)
            block_schedules = [result.block_sizes] * n_out
            thres_schedules = [None] * n_out
            avg_block_sizes = [avg_block_size] * n_out
            action_logits = [None] * n_out
            if result.block_sizes_chosen is not None:
                active = result.block_decisions
                n_b = len(block_size_candidates)
                # block_policy packs [block | thres] logits; block_unmask_policy packs
                # [unmask (L) | block] and has no thresholds.
                has_thres = result.thresholds_chosen is not None
                b_lo = 0 if has_thres else result.sampling_inputs.shape[-1] - n_b
                for j in range(n_out):
                    chosen = result.block_sizes_chosen[j][active[j]].tolist()
                    block_schedules[j] = chosen
                    if has_thres:
                        thres_schedules[j] = [
                            round(t, 4)
                            for t in result.thresholds_chosen[j][active[j]].tolist()
                        ]
                    avg_block_sizes[j] = (
                        sum(chosen) / len(chosen) if chosen else None
                    )
                    # The full action distribution at each decision, not just the
                    # action that got sampled. Sampling alone cannot separate "the
                    # policy conditions on this problem" from "one distribution drawn
                    # twice"; the logits can. Infeasible block sizes are -inf, which
                    # is not valid JSON, so they go out as null. The per-position
                    # unmask logits (L per step) are not dumped.
                    logits = result.sampling_inputs[j][active[j]].float().tolist()
                    action_logits[j] = [
                        {
                            "block": [
                                None if not math.isfinite(v) else round(v, 4)
                                for v in row[b_lo : b_lo + n_b]
                            ],
                            **(
                                {"thres": [round(v, 4) for v in row[n_b:]]}
                                if has_thres
                                else {}
                            ),
                        }
                        for row in logits
                    ]

            batch_wall_time = time.time() - start_time
            wall_time_per_sample = batch_wall_time / len(generated_texts)

            if is_code_dataset:
                sanitized_completions = []
                for j, gen_text in enumerate(generated_texts):
                    if dataset_name == "humaneval":
                        try:
                            full_completion = raw_prompts[j] + gen_text
                            sanitized = sanitize_humaneval(
                                full_completion, entry_points[j]
                            )
                            sanitized_completions.append(sanitized)
                        except Exception as e:
                            print(
                                f"Warning: Failed to sanitize HumanEval completion for {task_ids[j]}: {e}"
                            )
                            # for HumanEval, fall back to just doing prompt + generation
                            sanitized_completions.append(raw_prompts[j] + gen_text)
                    elif dataset_name == "mbpp":
                        try:
                            sanitized = sanitize_mbpp(gen_text)
                            sanitized_completions.append(sanitized)
                        except Exception as e:
                            print(
                                f"Warning: Failed to sanitize MBPP completion for {task_ids[j]}: {e}"
                            )
                            # for MBPP, fall back to just doing the generation
                            sanitized_completions.append(gen_text)

                example_result = [
                    {
                        "task_id": task_ids[j],
                        "prompt": raw_prompts[j],
                        "prompt_input": prompts[j],
                        "generation_raw": generated_texts[j],
                        "generation_sanitized": sanitized_completions[j],
                        "test_cases": test_cases[j],
                        "entry_point": entry_points[j],
                        "steps": steps_taken[j].item()
                        if hasattr(steps_taken[j], "item")
                        else steps_taken[j],
                        "wall_time": wall_time_per_sample,
                        "avg_block_size": avg_block_sizes[j],
                        "block_sizes": block_schedules[j],
                        "thresholds": thres_schedules[j],
                        "action_logits": action_logits[j],
                    }
                    for j in range(len(task_ids))
                ]

            else:
                example_result = [
                    {
                        "question": questions[j],
                        "prompt_input": prompts[j],
                        "generations": generated_texts[j],
                        "ground_truth": gt_answers[j].item()
                        if hasattr(gt_answers[j], "item")
                        else gt_answers[j],
                        "steps": steps_taken[j].item()
                        if hasattr(steps_taken[j], "item")
                        else steps_taken[j],
                        "wall_time": wall_time_per_sample,
                        "avg_block_size": avg_block_sizes[j],
                        "block_sizes": block_schedules[j],
                        "thresholds": thres_schedules[j],
                        "action_logits": action_logits[j],
                    }
                    for j in range(len(gt_answers))
                ]
            all_generations.extend(example_result)
            total_processed += len(generated_texts)
            wall_times.append(batch_wall_time)

            if accelerator and accelerator.is_main_process:
                idx = random.randint(0, len(prompts) - 1)
                if is_code_dataset:
                    if dataset_name == "humaneval":
                        print(f"Task ID: {task_ids[idx]}")
                        print("-" * 50)
                        print("Generation (sanitized):")
                        print(sanitized_completions[idx])
                        print("-" * 50)
                    elif dataset_name == "mbpp":
                        print(f"Task: {raw_prompts[idx]}")
                        print("-" * 50)
                        print("Generation (sanitized):")
                        print(sanitized_completions[idx])
                        print("-" * 50)
                else:
                    print(f"Question: {questions[idx]}")
                    print("-" * 50)
                    print("Generation:")
                    print(generated_texts[idx])
                    print("-" * 50)
                    print(f"Ground truth: {gt_answers[idx]}")

    avg_wall_time = sum(wall_times) / len(wall_times)
    metrics = {
        "wall_time": avg_wall_time,
        "generations": all_generations,
        "total_processed": total_processed.item(),
    }
    return metrics


def evaluate_code(generations, dataset_name):
    try:
        print(f"\n=== Running code evaluation for {dataset_name} ===")
        code_eval = hf_evaluate.load("code_eval")

        predictions = [[gen["generation_sanitized"]] for gen in generations]
        references = [gen["test_cases"] for gen in generations]

        print(f"Evaluating {len(predictions)} code samples...")
        pass_at_k, results = code_eval.compute(
            references=references, predictions=predictions, k=[1]
        )
        pass_at_1 = pass_at_k["pass@1"]

        print("Code evaluation results:")
        print(f"  pass@1: {pass_at_1:.4f}")

        for task_id, task_results in results.items():
            if len(task_results) > 0:
                _, result_dict = task_results[0]
                generations[task_id]["pass@1"] = 1.0 if result_dict["passed"] else 0.0
            else:
                generations[task_id]["pass@1"] = 0.0

        return {"pass@1": pass_at_1}

    except Exception as e:
        print(f"Error during code evaluation: {e}")
        import traceback

        traceback.print_exc()
        return None


def get_local_path_and_save_results(
    results: dict,
    args: argparse.Namespace,
    model_name: str,
) -> Path | None:
    file_path = None
    if not args.dont_save:
        filename_parts = [
            args.dataset,
            model_name,
            args.gen_length,
            args.diffusion_steps,
            args.block_length,
            args.remasking,
            0,  # for legacy reasons we include the rank of the process
            "generations",
        ]
        file_path = Path(args.output_dir) / (
            "_".join(map(str, filename_parts)) + ".json"
        )
        os.makedirs(args.output_dir, exist_ok=True)
        with open(file_path, "w") as f:
            json.dump(results, f, indent=2, sort_keys=False)
        print(f"Saved results locally to {file_path}")
    return file_path


class CustomDistributedSampler(DistributedSampler):
    """
    From torch docs:
    drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas

    We want drop_last = False, but don't want to have extra padding indices. Hence using a custom sampler.
    """

    def __init__(
        self,
        dataset,
        num_replicas=None,
        rank=None,
        shuffle=True,
        seed=0,
        drop_last=False,
    ) -> None:
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                f"Invalid rank {rank}, rank should be in the interval [0, {num_replicas - 1}]"
            )

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last

        if self.drop_last and len(self.dataset) % self.num_replicas != 0:
            self.num_samples = math.ceil(
                (len(self.dataset) - self.num_replicas) / self.num_replicas
            )
            self.total_size = self.num_samples * self.num_replicas
        else:
            self.total_size = len(self.dataset)
            self.num_samples = len(self.dataset) // self.num_replicas + int(
                rank < (self.total_size % self.num_replicas)
            )

        self.shuffle = shuffle
        self.seed = seed


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, required=True, help="Path to experiment config file"
    )
    parser.add_argument("--model_path", type=str, required=False, default=None)
    parser.add_argument(
        "--few_shot",
        type=int,
        default=-1,
        help="Number of few-shot examples (default: -1 -> dataset-specific defaults)",
    )
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["gsm8k", "math", "humaneval", "mbpp"],
        default="gsm8k",
    )
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--gen_length", type=int, default=None)
    parser.add_argument("--block_length", type=int, default=None)
    parser.add_argument("--diffusion_steps", type=int, default=0)
    parser.add_argument("--dont_save", action="store_true")
    parser.add_argument("--output_dir", type=str, default="results/")
    parser.add_argument("--remasking", type=str, default="policy")
    parser.add_argument("--policy_path", type=str, default=None)
    parser.add_argument("--thres", type=float, default=0.7)
    parser.add_argument("--n_test", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--temperature_policy", type=float, default=1.0)
    parser.add_argument(
        "--sampling_mode",
        type=str,
        default=None,
        help="Sampling mode override (optional, uses config value if not specified)",
    )
    parser.add_argument(
        "--block_sampling_mode",
        type=str,
        default=None,
        help="Block-size sampling for --remasking block_unmask_policy: 'categorical' "
        "or 'categorical-argmax' (optional, uses config value if not specified)",
    )
    parser.add_argument(
        "--adaptive_block",
        action="store_true",
        help="Use AdaBlock-style adaptive block sizes (requires batch_size 1)",
    )
    parser.add_argument(
        "--delimiter_threshold",
        type=float,
        default=0.3,
        help="AdaBlock delimiter confidence threshold",
    )
    parser.add_argument(
        "--delimiter_ids",
        type=str,
        default="198",
        help="Comma-separated delimiter token ids for AdaBlock (198=newline)",
    )
    parser.add_argument(
        "--block_schedule",
        type=str,
        default=None,
        help=(
            "Fixed per-decision actions for --remasking block_schedule, as "
            "'b:thres,b:thres,...' (e.g. '64:0.5,32:0.9'). The last entry repeats for "
            "any further blocks. Both values must be in the configured candidate sets."
        ),
    )
    args = parser.parse_args()
    args.delimiter_ids = tuple(int(t) for t in args.delimiter_ids.split(","))
    if args.block_schedule is not None:
        args.block_schedule = tuple(
            (int(b), float(t))
            for b, t in (e.split(":") for e in args.block_schedule.split(","))
        )
    if args.remasking == "block_schedule" and not args.block_schedule:
        parser.error("--remasking block_schedule requires --block_schedule")

    init_seed(args.seed)

    baseline_mode = False
    baseline_params = None
    if args.policy_path:
        checkpoint_name = Path(args.policy_path).parent.name
        baseline_params = parse_baseline_checkpoint(checkpoint_name)
        if baseline_params:
            baseline_mode = True
            print(f"Auto-detected baseline: {baseline_params}")

    # Load args from teh config (unless overriden)
    trl_parser = TrlParser((Config,))
    (grpo_config,) = trl_parser.parse_args_and_config(
        args=["--config", args.config], fail_with_unknown_args=False
    )
    args.grpo_config = grpo_config
    if args.sampling_mode is None:
        args.sampling_mode = grpo_config.sampling_mode
    if args.block_sampling_mode is None:
        args.block_sampling_mode = grpo_config.block_sampling_mode
    if args.block_length is None:
        args.block_length = grpo_config.block_length
    if args.gen_length is None:
        args.gen_length = grpo_config.max_completion_length
    # Override model_path from config if not explicitly provided
    if args.model_path is None:
        args.model_path = grpo_config.model_path
    args.dpls_stop_logit = grpo_config.dpls_stop_logit

    if args.remasking == "fastdllm":
        assert args.thres is not None, "thres must be provided for fastdllm"

    if args.adaptive_block:
        assert args.batch_size == 1, "adaptive_block requires batch_size 1"

    # NOTE: setting up the accelerator must be done after parsing config
    accelerator = Accelerator()

    # Check if we are running a baseline, if so get the args from the name
    args.baseline_mode = baseline_mode
    if baseline_mode:
        assert baseline_params is not None
        args.remasking = baseline_params["method"]
        if "thres" in baseline_params:
            args.thres = baseline_params["thres"]
        if "diffusion_steps" in baseline_params:
            args.diffusion_steps = baseline_params["diffusion_steps"]
        if baseline_params.get("adaptive_block"):
            args.adaptive_block = True
            args.delimiter_threshold = baseline_params["delimiter_threshold"]

        args.sampling_mode = None
        if args.remasking in {"random", "low_confidence"}:
            assert args.diffusion_steps > 0

    # Set few_shot to dataset-specific default if -1 is specified
    if args.few_shot == -1:
        args.few_shot = FEW_SHOT_DEFAULTS[args.dataset]
        if accelerator.is_main_process:
            print(
                f"Using dataset-specific few-shot setting for {args.dataset}: {args.few_shot}"
            )

    # Compute model name for output path
    model_name = "instruct" if "Instruct" in args.model_path else "base"

    if args.few_shot > 0:
        model_name = model_name + f"_fs{args.few_shot}"

    if len(args.suffix) > 0:
        model_name = model_name + f"_{args.suffix}"

    # Load the base model and tokenizer
    model = AutoModel.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    if "LLaDA" in args.model_path:
        mask_id = MASK_TOKENS_MAP["LLaDA"]
        _model_type = "LLaDA"
    elif "Dream" in args.model_path:
        mask_id = MASK_TOKENS_MAP["Dream"]
        _model_type = "Dream"
    else:
        raise ValueError(f"Model path {args.model_path} not supported")

    # Load the policy
    policy = None
    if (
        args.remasking in ("policy", "block_policy", "block_unmask_policy")
        and not args.baseline_mode
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        config = args.grpo_config
        if config.policy_type == "dit_hidden":
            assert _model_type == "LLaDA", (
                "dit_hidden policy is only supported with LLaDA models, not Dream"
            )
            policy_core = DiTHiddenStatePolicy(
                dllm=model,
                time_embed_dim=config.policy_time_embed_dim,
                num_blocks=config.policy_num_blocks,
                smart_init=config.policy_smart_init,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_hidden_proj":
            assert _model_type == "LLaDA", (
                "dit_hidden_proj policy is only supported with LLaDA models, not Dream"
            )
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTHiddenProjPolicy(
                dllm=model,
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_confidence":
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTConfidencePolicy(
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                confidences_top_p=config.confidences_top_p,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_block_size":
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTBlockSizePolicy(
                block_size_candidates=tuple(config.block_size_candidates),
                thresholds=tuple(config.threshold_candidates),
                block_size_prior_logits=(
                    tuple(config.block_size_prior_logits)
                    if config.block_size_prior_logits is not None
                    else None
                ),
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                confidences_top_p=config.confidences_top_p,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_block_size_hidden_proj":
            assert _model_type == "LLaDA", (
                "dit_block_size_hidden_proj policy is only supported with LLaDA "
                "models, not Dream"
            )
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTBlockSizeHiddenProjPolicy(
                dllm=model,
                block_size_candidates=tuple(config.block_size_candidates),
                thresholds=tuple(config.threshold_candidates),
                block_size_prior_logits=(
                    tuple(config.block_size_prior_logits)
                    if config.block_size_prior_logits is not None
                    else None
                ),
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        elif config.policy_type == "dit_block_unmask":
            hidden_dim = config.policy_hidden_dim or 128
            feedforward_dim = config.policy_feedforward_dim or (4 * hidden_dim)

            policy_core = DiTBlockUnmaskPolicy(
                block_size_candidates=tuple(config.block_size_candidates),
                block_size_prior_logits=(
                    tuple(config.block_size_prior_logits)
                    if config.block_size_prior_logits is not None
                    else None
                ),
                hidden_dim=hidden_dim,
                feedforward_dim=feedforward_dim,
                num_heads=config.policy_num_heads,
                dropout=config.policy_dropout,
                time_embed_dim=config.policy_time_embed_dim,
                smart_init=config.policy_smart_init,
                confidences_top_p=config.confidences_top_p,
                num_blocks=config.policy_num_blocks,
                time_period=config.policy_time_period,
            ).to(device)
        else:
            raise ValueError(
                f"Policy type {config.policy_type} not supported. "
                "Choose from ['dit_hidden', 'dit_hidden_proj', 'dit_confidence', "
                "'dit_block_size', 'dit_block_size_hidden_proj', 'dit_block_unmask']"
            )
        policy = PolicyHFWrapper(policy_core, config.policy_type)

        if args.policy_path is not None:
            if accelerator.is_main_process:
                print(f"Loading policy from {args.policy_path}")
            state = load_file(args.policy_path)
            policy.load_state_dict(state)

    # Create the dataset
    dataset_kwargs = {
        "tokenizer": tokenizer,
        "subsample": -1,
        "num_examples": args.few_shot,
    }
    if args.dataset in ["gsm8k", "math"]:
        dataset_kwargs["add_reasoning"] = True
    dataset = DATASET_MAP[args.dataset](**dataset_kwargs)

    # take only first args.n_test examples
    collate_fn = dataset.collate_fn
    if args.n_test is not None and len(dataset) > args.n_test:
        dataset = torch.utils.data.Subset(dataset, range(args.n_test))

    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        sampler=CustomDistributedSampler(dataset, shuffle=False),
        collate_fn=collate_fn,
    )

    # Use accelerator to prepare model and policy, but NOT the dataloader
    # We manage distribution manually with CustomDistributedSampler to avoid padding
    if policy is not None:
        model, policy = accelerator.prepare(model, policy)
    else:
        model = accelerator.prepare(model)

    # Run evaluation
    results = evaluate(
        model,
        tokenizer,
        dataloader,
        dataset_name=args.dataset,
        accelerator=accelerator,
        policy=policy,
        gen_length=args.gen_length,
        temperature=args.temperature,
        block_length=args.block_length,
        steps=args.diffusion_steps,
        remasking=args.remasking,
        thres=args.thres,
        sampling_mode=args.sampling_mode,
        dpls_stop_logit=args.dpls_stop_logit,
        temperature_policy=args.temperature_policy,
        mask_id=mask_id,
        model_type=_model_type,
        policy_full_context=args.grpo_config.policy_full_context
        if args.remasking in ("policy", "block_policy", "block_unmask_policy")
        else False,
        confidences_top_p=args.grpo_config.confidences_top_p
        if args.remasking in ("policy", "block_policy", "block_unmask_policy")
        else 1,
        adaptive_block=args.adaptive_block,
        delimiter_ids=args.delimiter_ids,
        delimiter_threshold=args.delimiter_threshold,
        block_size_candidates=tuple(args.grpo_config.block_size_candidates),
        threshold_candidates=tuple(args.grpo_config.threshold_candidates),
        block_schedule=args.block_schedule,
        block_sampling_mode=args.block_sampling_mode,
    )

    if accelerator.num_processes > 1:
        all_gpu_generations = gather_object(results["generations"])
        if accelerator.is_main_process:
            results["generations"] = all_gpu_generations

    if accelerator.is_main_process:
        if args.dataset in {"humaneval", "mbpp"}:
            results["code_eval_results"] = evaluate_code(
                results["generations"], args.dataset
            )
        results["metrics"] = {
            k: results.pop(k) for k in ("wall_time", "total_processed")
        }
        # Label adaptive-block runs distinctly so filenames and aggregation don't
        # collide with fixed-block results (block_length only serves as the
        # AdaBlock fallback length during generation, which is done by now).
        # B0 stays in the label: it is only the fallback, but it bounds how far
        # ahead a delimiter can move the boundary, so two runs differing only in
        # B0 are different runs and must not group together during aggregation.
        if args.adaptive_block:
            args.block_length = f"ada{args.delimiter_threshold}_B{args.block_length}"
        elif args.remasking == "block_policy":
            # The policy picks a block length (and a threshold) per block, so no
            # single value describes the run. Label it distinctly for the same
            # reason as adaptive_block: aggregation groups on block_length, and a
            # numeric label would silently merge these rows with fixed-block runs.
            args.block_length = "cat"
        elif args.remasking == "block_schedule":
            # Same reason, and it must not merge with the learned-policy rows either:
            # the whole point of this run is to be compared against them.
            args.block_length = "sched" + ",".join(
                f"{b}:{t}" for b, t in args.block_schedule
            )
        elif args.remasking == "block_unmask_policy":
            # Block size and unmasking both come from the policy; no fixed length.
            args.block_length = "blockunmask"
        results.update(
            {
                "model_path": args.model_path,
                "gen_length": args.gen_length,
                "diffusion_steps": args.diffusion_steps,
                "block_length": args.block_length,
                "remasking": args.remasking,
                "policy_path": args.policy_path,
                "thres": None
                if args.remasking
                in ("block_policy", "block_schedule", "block_unmask_policy")
                else args.thres,
                "block_sampling_mode": args.block_sampling_mode
                if args.remasking == "block_unmask_policy"
                else None,
                "n_test": args.n_test,
                "few_shot": args.few_shot,
                "adaptive_block": args.adaptive_block,
                "delimiter_threshold": args.delimiter_threshold
                if args.adaptive_block
                else None,
            }
        )
        # Verify and persist test-set coverage before saving so downstream
        # aggregation can detect incomplete evaluations.
        actual_samples_processed = len(results["generations"])
        expected_dataset_size = len(dataset) if hasattr(dataset, "__len__") else None
        if hasattr(dataset, "dataset"):  # Handle Subset wrapper
            expected_dataset_size = (
                len(dataset.dataset) if args.n_test is None else args.n_test
            )
        elif args.n_test is not None:
            expected_dataset_size = args.n_test

        coverage_complete = (
            actual_samples_processed == expected_dataset_size
            if expected_dataset_size is not None
            else None
        )
        results["test_set_verification"] = {
            "expected_dataset_size": expected_dataset_size,
            "actual_samples_processed": actual_samples_processed,
            "coverage_complete": coverage_complete,
        }
        get_local_path_and_save_results(results, args, model_name)

        print("\n=== Test Set Verification ===")
        print(f"Dataset: {args.dataset}")
        print(f"Samples processed: {actual_samples_processed}")
        print(f"Expected dataset size: {expected_dataset_size}")
        if expected_dataset_size:
            print(
                f"Coverage: {actual_samples_processed}/{expected_dataset_size} ({100 * actual_samples_processed / expected_dataset_size:.1f}%)"
            )
        print(f"Batch size: {args.batch_size}")
        print(f"Multi-GPU processes: {accelerator.num_processes}")
        print("=============================\n")

    accelerator.end_training()
    accelerator.free_memory()
