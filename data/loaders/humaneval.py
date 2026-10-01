#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH


class HumanEvalDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.load_test_dataset()

        self.subsample = (
            np.random.choice(len(self.dataset), subsample, replace=False)
            if subsample != -1
            else np.arange(len(self.dataset))
        )
        print(f"evaluating {len(self.subsample)} examples")
        assert subsample <= len(self.dataset), (
            "Subsample size is greater than dataset size"
        )

    def __len__(self):
        return len(self.subsample)

    def load_test_dataset(self):
        local_path = DATASETS_PATH / "humaneval"
        if local_path.exists():
            self.dataset = load_from_disk(str(local_path))["test"]
        else:
            # The same 164 problems with the prompt / test / entry_point fields used here.
            self.dataset = load_dataset("openai/openai_humaneval")["test"]

    def create_prompt(self, prompt_text):
        # HumanEval is 0-shot (no few-shot examples)
        # Following lm-eval's humaneval_instruct.yaml format:
        # doc_to_text: "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{{ prompt }}\n```\n"
        # gen_prefix: "Here is the completed function:\n```python\n{{ prompt }}\n"

        doc_to_text = (
            f"Write a solution to the following problem and make sure that it passes the tests:\n"
            f"```python\n{prompt_text}\n```\n"
        )
        messages = [{"role": "user", "content": doc_to_text}]
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Append gen_prefix - the model continues from here
        gen_prefix = f"Here is the completed function:\n```python\n{prompt_text}\n"
        return user_input + gen_prefix

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        prompt = item["prompt"]  # Contains function signature + docstring
        task_id = item["task_id"]
        test_cases_raw = item["test"]
        entry_point = item["entry_point"]
        test_cases = f"{test_cases_raw}\ncheck({entry_point})\n"

        formatted_prompt = self.create_prompt(prompt)
        return formatted_prompt, prompt, task_id, test_cases, entry_point

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        raw_prompts = [item[1] for item in batch]
        task_ids = [item[2] for item in batch]
        test_cases = [item[3] for item in batch]
        entry_points = [item[4] for item in batch]

        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )

        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
            "raw_prompts": raw_prompts,
            "task_ids": task_ids,
            "test_cases": test_cases,
            "entry_points": entry_points,
        }
