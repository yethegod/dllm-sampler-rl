#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
### Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import tempfile
from contextlib import nullcontext

import accelerate
import torch
import transformers
import trl
import wandb
from dotenv import load_dotenv
from transformers import AutoModel
from transformers import AutoTokenizer
from transformers import BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig
from trl import TrlParser

import train.reward_func as reward_func
from common.config import Config
from common.models.policy import DiTHiddenStatePolicy
from common.models.policy import DiTBlockSizePolicy
from common.models.policy import DiTConfidencePolicy
from common.models.policy import PolicyHFWrapper
from common.s3 import S3UploadCallback
from common.s3 import download_s3_checkpoint
from common.s3 import get_latest_s3_checkpoint
from data.data_utils import get_gsm8k_and_math_and_kodcode_questions
from data.data_utils import get_gsm8k_and_math_questions
from data.data_utils import get_gsm8k_questions
from data.data_utils import get_kodcode_questions
from data.data_utils import get_math_questions
from data.data_utils import set_random_seed
from train.trainer import Trainer

os.environ["HF_ALLOW_CODE_EVAL"] = "1"

torch.set_float32_matmul_precision("high")

print("=== Library Versions ===")
print(f"torch: {torch.__version__}")
print(f"transformers: {transformers.__version__}")
print(f"accelerate: {accelerate.__version__}")
print(f"trl: {trl.__version__}")
try:
    import flash_attn

    print(f"flash_attn: {flash_attn.__version__}")
except ImportError:
    print("flash_attn: not installed")
print("========================")


def get_reward_functions(config: Config):
    """Get reward functions based on config."""
    if config.reward_functions is not None:
        reward_functions = []
        for func_name in config.reward_functions:
            func = getattr(reward_func, func_name, None)
            if func is None:
                raise ValueError(
                    f"Unknown reward function: {func_name}. Function not found in reward_func module."
                )
            reward_functions.append(func)
        return reward_functions
    else:
        raise ValueError("Reward functions must be manually specified.")


load_dotenv()
token = os.getenv("HF_TOKEN")
if not token:
    raise ValueError(
        "Hugging Face token not found in environment variables. Please set HF_TOKEN."
    )


MASK_TOKENS_MAP = {"LLaDA": 126336, "Dream": 151666}


def main(grpo_config, model_config):
    set_random_seed(grpo_config.seed)

    # During training, remasking must be a learned strategy: "policy" learns which
    # positions to unmask at a fixed schedule, "block_policy" learns the schedule
    # itself (block size + threshold) and decodes within a block with Fast-dLLM.
    assert grpo_config.remasking in ("policy", "block_policy"), (
        f"Training only supports remasking in ('policy', 'block_policy'), "
        f"got '{grpo_config.remasking}'"
    )
    if grpo_config.remasking == "block_policy":
        assert grpo_config.policy_type == "dit_block_size", (
            f"remasking='block_policy' requires policy_type='dit_block_size', "
            f"got '{grpo_config.policy_type}'"
        )
        assert grpo_config.sampling_mode == "categorical", (
            f"remasking='block_policy' requires sampling_mode='categorical', "
            f"got '{grpo_config.sampling_mode}'"
        )
        assert grpo_config.policy_full_context, (
            "remasking='block_policy' requires policy_full_context=True: the policy "
            "scores boundaries across the whole generation region, not just one block"
        )
        assert not grpo_config.es_thresholds, (
            "Expert steering is not supported with remasking='block_policy': the ES "
            "expert is a fixed-threshold Fast-dLLM rollout, which produces no "
            "(block size, threshold) actions for the policy to imitate"
        )

    assert grpo_config.per_device_train_batch_size % grpo_config.num_generations == 0, (
        f"per_device_train_batch_size ({grpo_config.per_device_train_batch_size}) must be "
        f"divisible by num_generations ({grpo_config.num_generations}) to ensure complete groups per GPU."
    )

    # ES (Expert Steering) currently only supports 1 group per GPU (generates samples for first prompt only)
    if grpo_config.es_thresholds:
        assert grpo_config.num_generations == grpo_config.per_device_train_batch_size, (
            "ES requires exactly 1 group per GPU (num_generations == per_device_train_batch_size)"
        )
        assert grpo_config.block_length == 256
        assert grpo_config.policy_full_context

    if grpo_config.dataset in {"mbpp", "humaneval"}:
        raise ValueError(
            f"Training not supported for {grpo_config.dataset}. "
            "This dataset is evaluation-only."
        )
    elif grpo_config.dataset == "gsm8k":
        dataset = get_gsm8k_questions("train")
    elif grpo_config.dataset == "math":
        dataset = get_math_questions("train")
    elif grpo_config.dataset == "gsm8k_and_math":
        dataset = get_gsm8k_and_math_questions("train", seed=grpo_config.seed)
        assert (
            "mixed_correctness_mult_reward_func" in grpo_config.reward_functions
            or "mixed_correctness_add_reward_func" in grpo_config.reward_functions
        )
    elif grpo_config.dataset == "gsm8k_and_math_and_kodcode":
        dataset = get_gsm8k_and_math_and_kodcode_questions(
            "train", seed=grpo_config.seed
        )
        assert (
            "mixed_correctness_mult_reward_func" in grpo_config.reward_functions
            or "mixed_correctness_add_reward_func" in grpo_config.reward_functions
        )
    elif grpo_config.dataset == "kodcode":
        dataset = get_kodcode_questions()
    else:
        raise ValueError(f"Dataset {grpo_config.dataset} not supported")

    reward_functions = get_reward_functions(grpo_config)
    dataset = dataset.shuffle(seed=grpo_config.seed)
    train_set = dataset
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 4 bit quantization configuration (only if enabled in ModelConfig)
    # For the paper, we left this turned off.
    bnb_config = None
    if model_config.load_in_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    # Load model and tokenizer
    if "LLaDA" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["LLaDA"]
        grpo_config.model_type = "LLaDA"
    elif "Dream" in grpo_config.model_path:
        model = AutoModel.from_pretrained(
            grpo_config.model_path,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
        ).to(device)
        grpo_config.mask_id = MASK_TOKENS_MAP["Dream"]
        grpo_config.model_type = "Dream"
    else:
        raise ValueError(f"Model path {grpo_config.model_path} not supported")

    tokenizer = AutoTokenizer.from_pretrained(
        grpo_config.model_path, trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    model.config.use_cache = False

    # Create policy based on type
    if grpo_config.policy_type == "dit_hidden":
        assert grpo_config.model_type == "LLaDA", (
            "dit_hidden policy is only supported with LLaDA models, not Dream"
        )
        policy_core = DiTHiddenStatePolicy(
            dllm=model,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            num_blocks=grpo_config.policy_num_blocks,
            smart_init=grpo_config.policy_smart_init,
            time_period=grpo_config.policy_time_period,
        ).to(device)

    elif grpo_config.policy_type == "dit_confidence":
        hidden_dim = grpo_config.policy_hidden_dim or 128
        feedforward_dim = grpo_config.policy_feedforward_dim or (4 * hidden_dim)

        policy_core = DiTConfidencePolicy(
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=grpo_config.policy_num_heads,
            dropout=grpo_config.policy_dropout,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            smart_init=grpo_config.policy_smart_init,
            confidences_top_p=grpo_config.confidences_top_p,
            num_blocks=grpo_config.policy_num_blocks,
            time_period=grpo_config.policy_time_period,
        ).to(device)
    elif grpo_config.policy_type == "dit_block_size":
        hidden_dim = grpo_config.policy_hidden_dim or 128
        feedforward_dim = grpo_config.policy_feedforward_dim or (4 * hidden_dim)

        policy_core = DiTBlockSizePolicy(
            block_size_candidates=tuple(grpo_config.block_size_candidates),
            thresholds=tuple(grpo_config.threshold_candidates),
            block_size_prior_logits=(
                tuple(grpo_config.block_size_prior_logits)
                if grpo_config.block_size_prior_logits is not None
                else None
            ),
            hidden_dim=hidden_dim,
            feedforward_dim=feedforward_dim,
            num_heads=grpo_config.policy_num_heads,
            dropout=grpo_config.policy_dropout,
            time_embed_dim=grpo_config.policy_time_embed_dim,
            smart_init=grpo_config.policy_smart_init,
            confidences_top_p=grpo_config.confidences_top_p,
            num_blocks=grpo_config.policy_num_blocks,
            time_period=grpo_config.policy_time_period,
        ).to(device)
    else:
        raise ValueError(
            f"Policy type {grpo_config.policy_type} not supported. "
            "Choose from ['dit_hidden', 'dit_confidence', 'dit_block_size']"
        )

    policy = PolicyHFWrapper(policy_core, grpo_config.policy_type)

    # Log policy parameter count
    total_params = sum(p.numel() for p in policy_core.parameters())
    trainable_params = sum(
        p.numel() for p in policy_core.parameters() if p.requires_grad
    )

    print(f"Policy type: {grpo_config.policy_type}")
    print(f"Total policy parameters: {total_params:,}")
    print(f"Trainable policy parameters: {trainable_params:,}")

    if wandb.run is not None:
        wandb.log(
            {
                "policy/total_parameters": total_params,
                "policy/trainable_parameters": trainable_params,
                "policy/policy_type": grpo_config.policy_type,
            },
            step=0,
        )

    output_dir = grpo_config.output_dir
    s3_output = "s3" in output_dir
    if s3_output:
        # For remote paths we save checkpoints in a temp dir locally and then
        # use a callback to push them to aws
        context_manager = tempfile.TemporaryDirectory()
        callbacks = [S3UploadCallback(output_dir)]
    else:
        # Otherwise use a context that is basically a no-op, so
        # checkpoints will be written to the path indicated by output_dir
        context_manager = nullcontext(output_dir)
        callbacks = []
        # Do however need to make sure that the local path exists!
        os.makedirs(output_dir, exist_ok=True)

    with context_manager as local_output_dir:
        grpo_config.output_dir = local_output_dir

        # Check for existing checkpoint to resume from
        resume_from = None
        if s3_output:
            latest = get_latest_s3_checkpoint(output_dir)
            if latest:
                resume_from = download_s3_checkpoint(
                    output_dir, latest, local_output_dir
                )
                print(
                    f"=== Auto-resume: found {latest}, downloaded to {resume_from} ==="
                )
        else:
            resume_from = get_last_checkpoint(output_dir)
            if resume_from:
                print(f"=== Auto-resume: resuming from {resume_from} ===")

        trainer = Trainer(
            args=grpo_config,
            model=policy,
            dllm=model,
            reward_funcs=reward_functions,
            train_dataset=train_set,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
        trainer.train(resume_from_checkpoint=resume_from)

        if resume_from:
            print(f"=== Resumed training from step {trainer.state.global_step} ===")
        else:
            print(
                f"=== Started fresh training, now at step {trainer.state.global_step} ==="
            )


if __name__ == "__main__":
    parser = TrlParser((Config, ModelConfig))
    grpo_config, model_config = parser.parse_args_and_config(
        fail_with_unknown_args=False
    )
    main(grpo_config=grpo_config, model_config=model_config)
