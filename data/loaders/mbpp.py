#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH


class MBPPDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=3,  # MBPP uses 3-shot
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.load_test_dataset()
        self.create_few_shot_prompt()

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

    def _load_splits(self):
        local_path = DATASETS_PATH / "mbpp"
        if local_path.exists():
            return load_from_disk(str(local_path))
        # The "full" config: 500 test problems plus the 10-problem "prompt" split
        # (task_ids 1-10) that lm-eval draws its few-shot examples from.
        return load_dataset("google-research-datasets/mbpp", "full")

    def load_test_dataset(self):
        self.dataset = self._load_splits()["test"]

    def load_few_shot_examples(self):
        prompt_data = self._load_splits()["prompt"]
        n_examples = min(self.num_examples, len(prompt_data))
        return [prompt_data[i] for i in range(n_examples)]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples (matching lm-eval mbpp_instruct.yaml format)"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            text = example["text"]
            code = example["code"]
            test_list = example["test_list"]
            # Format matching lm-eval:
            # doc_to_text: "You are an expert Python programmer, and here is your task:\n{{text}}\nYour code should pass these tests:\n{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}"
            # gen_prefix: "\n```python\n"
            # doc_to_target (for fewshot): "{{code}}\n```"
            # Full few-shot example: doc_to_text + gen_prefix + doc_to_target
            tests_str = "\n".join(test_list[:3])  # Use first 3 tests
            formatted_examples.append(
                f"You are an expert Python programmer, and here is your task:\n{text}\n"
                f"Your code should pass these tests:\n{tests_str}\n```python\n{code}\n```"
            )
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def create_prompt(self, text, test_list):
        # MBPP is 3-shot, format matches lm-eval mbpp_instruct.yaml
        # doc_to_text: "You are an expert Python programmer, and here is your task:\n{{text}}\nYour code should pass these tests:\n{{test_list[0]}}\n{{test_list[1]}}\n{{test_list[2]}}"
        # gen_prefix: "\n```python\n"
        tests_str = "\n".join(test_list[:3])  # Use first 3 tests
        task_prompt = (
            f"You are an expert Python programmer, and here is your task:\n{text}\n"
            f"Your code should pass these tests:\n{tests_str}"
        )

        if self.num_examples > 0:
            full_prompt = f"{self.few_shot_prompt}\n\n{task_prompt}"
        else:
            full_prompt = task_prompt

        messages = [{"role": "user", "content": full_prompt}]
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )

        # Append gen_prefix - the model continues from here
        gen_prefix = "\n```python\n"
        return user_input + gen_prefix

    def __getitem__(self, idx):
        item = self.dataset[self.subsample[idx].item()]
        text = item["text"]  # Task description
        task_id = item["task_id"]
        test_list = item["test_list"]  # List of test cases (assertions)

        # Combine test cases into a single test string
        test_cases = "\n".join(test_list)

        formatted_prompt = self.create_prompt(text, test_list)
        return formatted_prompt, text, task_id, test_cases

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        task_ids = [item[2] for item in batch]
        test_cases = [item[3] for item in batch]

        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )

        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompts": prompts,
            "texts": texts,
            "task_ids": task_ids,
            "test_cases": test_cases,
        }
