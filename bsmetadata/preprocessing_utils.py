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

from datasets import Dataset

from src.rel.REL.entity_disambiguation import EntityDisambiguation
from src.rel.REL.mention_detection import MentionDetection
from src.rel.REL.ner import load_flair_ner
from src.rel.REL.utils import process_results


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

    def _extract_timestamp_from_url(self, url: str) -> Optional:
        # This would have to be implemented.
        return None


class EntityPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for adding entity information."""

    base_url = ".\preprocessing_scripts\entity_linking_files"
    wiki_version = "wiki_2019"

    def __init__(self):
        self.mention_detection = MentionDetection(self.base_url, self.wiki_version)
        self.tagger_ner = load_flair_ner("ner-fast")
        self.config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.model = EntityDisambiguation(self.base_url, self.wiki_version, self.config)

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        example_text_list = examples["text"]
        example_metadata_list = examples["metadata"]

        for i in range(len(example_text_list)):
            res = self._extract_entity_from_text(example_text_list[i])
            result = self.postprocess_entity(res)

            if not result:
                continue

            example_metadata_list[i].append(result)

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
