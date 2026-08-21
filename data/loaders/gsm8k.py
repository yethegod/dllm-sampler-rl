#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
# Adapted from https://github.com/dllm-reasoning/d1 (Apache 2.0)
import os
import random
import re
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from datasets import load_from_disk

from common.parsing.parsers import Parser

DATASETS_PATH = Path(os.environ.get("RLDLLM_DATASETS_DIR", "/mnt/datasets"))

GSM_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"""


def format_demo_answer(answer: str) -> str:
    """Rewrite a raw GSM8K train answer into the format the system prompt asks for.

    The raw answers carry calculator annotations and end in "#### 72", but the
    grader (common/parsing/parse_and_get_acc.py:extract_gsm_answer) only reads
    \\boxed{} and <answer></answer>. Demonstrations in the raw format would teach
    the model to answer in a format that is scored as a miss, so the shots have
    to speak the same format as the system prompt and as the "<reasoning>" that
    create_prompt prefills.
    """
    rationale, _, final = answer.partition("####")
    rationale = re.sub(r"<<[^>]*>>", "", rationale).strip()
    final = final.strip().replace(",", "")
    return (
        f"<reasoning>\n{rationale}\n</reasoning>\n"
        f"<answer>\n\\boxed{{{final}}}\n</answer>"
    )


class GSM8KDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=GSM_SYSTEM_PROMPT,
        subsample=-1,
    ):
        self.tokenizer = tokenizer
        self.num_examples = num_examples
        self.add_reasoning = add_reasoning
        self.system_prompt = system_prompt
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

    def load_test_dataset(self):
        local_path = DATASETS_PATH / "gsm8k"
        if local_path.exists():
            self.dataset = load_from_disk(str(local_path))["test"]
        else:
            self.dataset = load_dataset("openai/gsm8k", "main")["test"]

    def create_prompt(self, input_text):
        # Format similar to your chat function
        if self.num_examples > 0:
            prompt = f"{self.few_shot_prompt}\n\nQuestion: {input_text}\nAnswer:\n"
        else:
            prompt = input_text
        messages = [{"role": "user", "content": self.system_prompt + "\n\n" + prompt}]
        user_input = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        if self.add_reasoning:
            return user_input + "<reasoning>"
        else:
            return user_input

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        local_path = DATASETS_PATH / "gsm8k"
        if local_path.exists():
            train_data = load_from_disk(str(local_path))["train"]
        else:
            train_data = load_dataset("openai/gsm8k", "main")["train"]
        examples = random.sample(range(len(train_data)), self.num_examples)
        return [
            {
                "question": train_data[i]["question"],
                "answer": format_demo_answer(train_data[i]["answer"]),
            }
            for i in examples
        ]

    def create_few_shot_prompt(self):
        """Create few-shot prompt from dataset examples"""
        few_shot_examples = self.load_few_shot_examples()

        formatted_examples = []
        for example in few_shot_examples:
            input_text = example["question"]
            answer = example["answer"]
            formatted_examples.append(f"Question: {input_text}\nAnswer:\n{answer}")
        self.few_shot_prompt = "\n\n".join(formatted_examples)

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["question"]
        answer = Parser.extract_answer_gsm8k(
            self.dataset[self.subsample[idx].item()]["answer"]
        )
        prompt = self.create_prompt(question)
        return prompt, question, answer

    def collate_fn(self, batch):
        prompts = [item[0] for item in batch]
        questions = [item[1] for item in batch]
        answers = [item[2] for item in batch]
        input_ids = self.tokenizer(
            prompts, padding_side="left", return_tensors="pt", padding="longest"
        )
        attention_mask = input_ids.attention_mask
        input_ids = input_ids.input_ids
        return {
            "input_ids": input_ids,
            "questions": questions,
            "answers": answers,
            "prompts": prompts,
            "attention_mask": attention_mask,
        }
