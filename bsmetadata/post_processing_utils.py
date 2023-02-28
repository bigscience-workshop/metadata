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
This script provides functions for adding different kinds of metadata to a pretraining corpus.
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Dict, List


logger = logging.getLogger(__name__)


class MetadataPostProcessor(ABC):
    """A metadata post processor can be used for post processing extracted metadata from a corpus."""

    def __init__(self, col_to_process: str) -> None:
        self.col_to_process = col_to_process
        super().__init__()

    @abstractmethod
    def post_process(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Post process a batch of examples for their extracted metadata."""
        pass


class WebsiteDescPostProcessor(MetadataPostProcessor):
    """website metadata post processor to remove noisy data"""

    def __init__(
        self,
        col_to_process="metadata",
    ) -> None:
        super().__init__(col_to_process=col_to_process)

    def post_process(self, examples: Dict[str, List]) -> Dict[str, List]:
        example_metadata_list = examples[self.col_to_process]
        # Iterate through the metadata associated with all examples in this batch.

        for example_metadata in example_metadata_list:
            if example_metadata and (
                self.is_noisy_data(example_metadata[0]["value"]) or self.is_outlier(example_metadata[0]["value"])
            ):
                example_metadata.clear()  # remove website description with empty list if metadata is invalid
        examples[self.col_to_process] = example_metadata_list
        return examples

    def is_noisy_data(self, data):
        corrupt_regex = [".* refer(|s) to.?:", "\[\[\w*:"]
        corrupt_regex_str = "|".join("({0})".format(x) for x in corrupt_regex)
        return re.match(corrupt_regex_str, data)

    def is_outlier(self, data):
        return len(data.split()) < 5 or len(data.split()) > 50  # caps tbd
