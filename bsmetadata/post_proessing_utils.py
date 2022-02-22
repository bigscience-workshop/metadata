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
from datasets import Value
from abc import ABC, abstractmethod
from typing import Any, Dict, List

logger = logging.getLogger(__name__)





class MetadataPostProcessor(ABC):
    """A metadata processor can be used for preprocessing text and adding or extracting metadata information."""

    def __init__(self, col_to_process: str) -> None:
        self.col_to_process = col_to_process
        super().__init__()

    @property
    @abstractmethod
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        """Returns a dictionary whose key corresponds to the name of a new column / a column modified by this processor
        and whose value corresponds to the minimal format of this column"""
        pass

    @abstractmethod
    def post_process(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples and process the corresponding extracted metadata."""
        pass


class WebsiteDescPostProcessor(MetadataPostProcessor):
    """Metadata preprocessor for adding website description based on URLs."""

    def __init__(
        self,
        col_to_process="metadata",
    ) -> None:
        
        super().__init__(col_to_process=col_to_process)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        features = {
            self.col_to_process: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def post_process(self, examples: Dict[str, List]) -> Dict[str, List]:

        example_metadata_list = examples[self.col_to_process]
        # Iterate through the metadata associated with all examples in this batch.
   
        for example_metadata in  example_metadata_list:
            if example_metadata and (self.is_noisy_data(example_metadata[0]["value"]) or self.is_outlier(example_metadata[0]["value"])):
                example_metadata = []
                
                

        examples[self.col_to_process] = example_metadata_list
        return examples

    
    def is_noisy_data(self, data):
        corrupt_regex = ['.* refer(|s) to.?:', '\[\[\w*:']
        corrupt_regex_str = '|'.join('({0})'.format(x) for x in corrupt_regex)
        return re.match(corrupt_regex_str, data)
    
    def is_outlier(self, data):
        return len(data.split()) < 5 or len(data.split()) > 50 #caps tbd


