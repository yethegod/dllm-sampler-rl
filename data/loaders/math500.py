#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#
import random

from datasets import load_dataset
from datasets import load_from_disk

from data.loaders.gsm8k import DATASETS_PATH
from data.loaders.gsm8k import GSM8KDataset

MATH500_SYSTEM_PROMPT = """You are a math expert. You will be given a question to solve. Solve it step by step. Wrap the final answer in a \\boxed{}.
Respond in the following format:
<reasoning>
Your reasoning here
</reasoning>
<answer>
\\boxed{...}
</answer>"
"""


class MATH500Dataset(GSM8KDataset):
    def __init__(
        self,
        tokenizer,
        num_examples=0,
        add_reasoning=True,
        system_prompt=MATH500_SYSTEM_PROMPT,
        subsample=-1,
    ):
        super().__init__(
            tokenizer, num_examples, add_reasoning, system_prompt, subsample
        )

    def load_test_dataset(self):
        local_path = DATASETS_PATH / "math500"
        if local_path.exists():
            self.dataset = load_from_disk(str(local_path))["test"]
        else:
            # Same 500 problems, with the reference `answer` field __getitem__ reads.
            self.dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]

    def load_few_shot_examples(self):
        if self.num_examples <= 0:
            return []
        local_path = DATASETS_PATH / "hendrycks_math_algebra"
        if local_path.exists():
            train_data = load_from_disk(str(local_path))["train"]
        else:
            train_data = load_dataset("EleutherAI/hendrycks_math", "algebra")["train"]
        few_shot_examples = []
        samples = random.sample(range(len(train_data)), self.num_examples)
        for example in samples:
            few_shot_examples.append(
                {
                    "question": train_data[example]["problem"],
                    "answer": train_data[example]["solution"],
                }
            )
        return few_shot_examples

    def __getitem__(self, idx):
        question = self.dataset[self.subsample[idx].item()]["problem"]
        answer = self.dataset[self.subsample[idx].item()]["answer"]
        prompt = self.create_prompt(question)
        return prompt, question, answer
