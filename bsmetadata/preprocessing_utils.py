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
import re
import urllib
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib.parse import unquote, urlparse, urlsplit

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

    def __init__(self, mode):
        # The length can be calculated for the whole text or for each sentence of a text individually.
        # We can specify a global length of a TEXT or a local length for each SENTENCE of a text.
        # Therefore, we provide two different modes: text (global) or sentence (local).
        self.mode = mode  # {text, sentence}

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        """
        Iterate through all the examples retrieve the length meta information.
        """
        for example_text, example_metadata in zip(examples["text"], examples["metadata"]):
            if self.mode == "text":
                text_length = self._extract_length_from_text(example_text)
                example_length = {"key": "length", "type": "global", "value": text_length}
            elif self.mode == "sentence":
                example_length = self._extract_length_from_sentences(example_text)
            else:
                print("Please select a valid length type [text or sentence].")

            if not example_length:
                continue

            example_metadata.append(example_length)
        examples["metadata"] = (
            [m[0] for m in examples["metadata"]] if self.mode == "sentence" else examples["metadata"]
        )  # reformatting of nested lists

        return examples

    def _extract_length_from_text(self, text: str) -> Optional[str]:
        """
        Identify the length of a text.
        """

        return str(len(text))  # char-based length

    def _extract_length_from_sentences(self, text: str) -> Optional[str]:
        """
        Identify the length of each sentence in a text and add the length as local metadata.
        """

        meta_sentences = []

        # Find all points in a text and store their absolute position to determine the final position of a sentence.
        pos_sentences = [pos for pos, char in enumerate(text) if char == "."]

        # Calculate the length of each sentence in a text based on a simple sentence splitting using the dots as indicators.
        len_sentences = [self._extract_length_from_text(sent) for sent in text.split(".")]

        # Iterate through the sentences of a text, storing the absolute beginning and end of each sentence and the associated length of each sentence.
        for sent_pos, sent_len, i in zip(pos_sentences, len_sentences, range(len(len_sentences))):
            meta_sentence = {
                "key": "length",
                "type": "local",
                "char_start_idx": 0
                if i == 0
                else pos_sentences[i - 1],  # end position of the previous sentence in a text
                "char_end_idx": pos_sentences[i],  # end position of the current sentence in a text
                "value": len_sentences[i],  # sentence length
            }

            meta_sentences.append(meta_sentence)  # combine all metadata for all sentences of a text

        return meta_sentences


class DatasourcePreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding datasource information based on URLs."""

    def _check_numbers(self, sub_part: List[str]) -> List[str]:
        """Check for insignificant numbers (i.e. we delete all numbers at the end or beginning of a given URL part (w/o domain))"""

        # We delete all numbers at the beginning of a given URL sub-phrase
        if sub_part[0].isdigit():
            sub_part = sub_part[:1]

        # We delete all numbers at the end of a given URL sub-phrase
        if sub_part[-1].isdigit():
            sub_part = sub_part[:-1]

        return sub_part

    def _parse_words(self, sub_part):
        """Check for meaningful seperators (chars) to split a phrase into sub-elements."""

        # Separator for splitting a phrase into sub tokens
        tokens = re.split(r"-|_|\+|\.|&|=", sub_part)

        return tokens

    def _clean_url_parts(self, url_parts):
        """Clean up a URL to identify the inherent and meaningful data source information."""

        datasource_list = []
        # Split sub phrases by a defined set of separators
        url_parts = [self._parse_words(i) for i in url_parts]
        # Delete numbers that are not meaningful (e.g., id, timestamp, ect.)
        url_parts = [self._check_numbers(i) for i in url_parts]

        for s in url_parts:
            if len(s) == 1:
                datasource_list.append(str(s[0]))
            elif len(s) > 1:
                datasource_list.append(" ".join(s))

        return datasource_list

    def _extract_datasource_from_url(self, url: str) -> Optional[str]:
        """Given an input URL (str) this function returns a structured datasource text (str)."""

        parts = urlparse(url)
        # Split a raw URL with “/” as separator
        directories_parts = parts.path.strip("/").split("/")
        directories_parts = self._clean_url_parts(directories_parts)

        return parts.netloc + " > " + " > ".join(directories_parts)

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        for example_url, example_meta in zip(examples["url"], examples["metadata"]):
            example_datasource = self._extract_datasource_from_url(example_url)
            print(example_datasource)

            if not example_datasource:
                continue

            example_meta.append({"key": "datasource", "type": "global", "value": example_datasource})

        return examples
