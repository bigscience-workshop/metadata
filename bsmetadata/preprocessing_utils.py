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
import uuid
from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from urllib.parse import unquote, urlsplit

from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import load_flair_ner
from REL.utils import process_results

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


class EntityPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for adding entity information."""

    def __init__(self, base_url):
        self.base_url = base_url
        self.wiki_version = "wiki_2019"
        self.mention_detection = MentionDetection(self.base_url, self.wiki_version)
        self.tagger_ner = load_flair_ner("ner-fast")
        self.config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.model = EntityDisambiguation(self.base_url, self.wiki_version, self.config)

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        for example_text, example_metadata in zip(examples["text"], examples["metadata"]):
            res = self._extract_entity_from_text(example_text)
            result = self.postprocess_entity(res)
            if not result:
                continue
            for val in result:
                example_metadata.append(val)
        return examples

    def _extract_entity_from_text(self, text: str) -> Optional:
        input_text = self.preprocess_example(text)
        res = self.fetch_mention_predictions(input_text)
        res_list = []
        for key, value in res.items():
            res_list = [list(elem) for elem in value]
        return res_list

    def postprocess_entity(self, resu_list):
        entities = []
        for ent in range(len(resu_list)):
            entity = resu_list[ent][3]  # element at index = 3 in the result list corresponds to the predicted entity
            en = {
                "key": "entity",
                "type": "local",
                "char_start_idx": resu_list[ent][
                    0
                ],  # element at index = 0 in the result list corresponds to the char start index
                "char_end_idx": (
                    resu_list[ent][0] + resu_list[ent][1]
                ),  # element at index = 1 in the result list corresponds to length of the entity
                "value": entity,
            }
            entities.append(en)
        return entities

    def preprocess_example(self, text: str) -> Optional:
        id_ = uuid.uuid4().hex.upper()[0:6]
        text_ = text
        value = [text_, []]
        processed = {id_: value}
        return processed

    def fetch_mention_predictions(self, input_text: str) -> Optional:
        mentions_dataset, n_mentions = self.mention_detection.find_mentions(input_text, self.tagger_ner)
        predictions, timing = self.model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        return result
