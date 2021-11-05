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
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib.parse import unquote, urlsplit

from bsmetadata.vendor.dateutil.src.dateutil.parser import ParserError, parse


def get_path_from_url(url):
    """get the `path` part of `url`, with %xx escapes replaced by their single-character equivalent"""
    parts = urlsplit(url)
    return unquote(parts.path)


def parse_date(path):
    try:
        return parse(path, fuzzy=True, date_only=True)
    except ParserError:
        return None
    except OverflowError:
        # this happens sometimes, I don't know why, just ignore it
        return None


def remove_improbable_date(x):
    if x is not None and (x.year < 1983 or x.year > 2021):
        return None
    return x


class MetadataPreprocessor(ABC):
    """A metadata processor can be used for preprocessing text and adding or extracting metadata information."""

    @abstractmethod
    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples and add or extract corresponding metadata."""
        pass


class TimestampPreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        example_metadata_list = examples["metadata"]

        # Iterate through the metadata associated with all examples in this batch.
        for example_metadata in example_metadata_list:
            # Get the URL associated with this example.
            example_urls = [md["value"] for md in example_metadata if md["key"] == "url"]

            if not example_urls:
                continue

            # Try to extract a timestamp from the given URL and add it to the metadata.
            example_timestamp = self._extract_timestamp_from_url(example_urls[0])

            if example_timestamp:
                example_metadata.append({"key": "timestamp", "type": "global", "value": example_timestamp})

        return examples

    def _extract_timestamp_from_url(self, url: str) -> Optional[str]:
        path = get_path_from_url(url)
        date = parse_date(path)
        date = remove_improbable_date(date)
        date = str(date) if date is not None else None
        return date

class GenerationLengthPreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding generation length information based on text."""
    
    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        example_metadata_list = examples["metadata"]
        
        # Iterate through the metadata associated with all examples in this batch.
        for example_metadata in example_metadata_list:
            example_texts = [md["value"] for md in example_metadata if md["key"] == "text"] # check if text is right
            
            if not example_urls:
                continue
                
            # Try to extract a timestamp from the given URL and add it to the metadata.
            example_length = self._extract_length_from_text(example_texts[0])
            
            if example_timestamp:
                example_metadata.append({"key": "length", "type": "global", "value": example_length})
                
        return example_metadata # check if this is right
    
    def _extract_length_from_text(self, text: str) -> Optional[str]:
        return len(text) # global