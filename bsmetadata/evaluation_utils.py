# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This script provides functions for evaluating different kinds of metadata.
"""

import logging
from abc import ABC, abstractmethod
from typing import List

import numpy as np
import torch
from tqdm import tqdm


logger = logging.getLogger(__name__)


def calc_ppl(sentence, model, tokenizer):
    tokenize_input = tokenizer.encode(sentence)
    tensor_input = torch.tensor([tokenize_input])
    loss = model(tensor_input, labels=tensor_input)[0]
    return np.exp(loss.detach().numpy())


class MetadataEvaluation(ABC):
    def __init__(self, model) -> None:
        self.model = model
        self.tokenizer = model.tokenizer

    @abstractmethod
    def evaluate(self, correct_examples: List, incorrect_examples: List) -> List:
        pass


class WebsiteDescriptionEvaluation(MetadataEvaluation):
    def __init__(self, model) -> None:
        self.model = model
        self.tokenizer = model.tokenizer

    def evaluate(self, correct_examples: List, incorrect_examples: List) -> List:
        assert len(correct_examples) == len(incorrect_examples)
        results = []
        for correct_example, incorrect_example in tqdm(zip(correct_examples, incorrect_examples)):
            correct_ppl = calc_ppl(correct_example, self.model, self.tokenizer)
            incorrect_ppl = calc_ppl(incorrect_example, self.model, self.tokenizer)
            results.append(correct_ppl < incorrect_ppl)
        return results
