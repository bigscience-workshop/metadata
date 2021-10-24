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
    """An exemplary metadata preprocessor for adding entity information."""

    base_url = ".\preprocessing_scripts\entity_linking_files"
    wiki_version = "wiki_2019"

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        for example in examples:

            text = example["text"]
            res = self._extract_entity_from_text(text)
            result = self.postprocess_entity(res)

            if not result:
                continue

            if "metadata" in example:
                example["metadata"].append(result)
            else:
                example["metadata"] = result

        return examples

    def _extract_entity_from_text(self, text: str) -> Optional:
        input_text = self.preprocess_example(text)
        res = self.fetch_mention_predictions(self.base_url, self.wiki_version, input_text)
        res_list = []
        for key, value in res.items():
            res_list = [list(elem) for elem in value]
        return res_list

    def postprocess_entity(self, r_list):
        entities = []
        for ent in range(len(r_list)):
            entity = r_list[ent][3] if (r_list[ent][5] > r_list[ent][4]) else r_list[ent][2]
            en = {
                "key": "entity",
                "type": "local",
                "char_start_idx": r_list[ent][0],
                "char_end_idx": (r_list[ent][0] + r_list[ent][1]),
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

    def fetch_mention_predictions(self, base_url: str, wiki_version: str, input_text: str) -> Optional:
        mention_detection = MentionDetection(base_url, wiki_version)
        tagger_ner = load_flair_ner("ner-fast")
        mentions_dataset, n_mentions = mention_detection.find_mentions(input_text, tagger_ner)
        config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        model = EntityDisambiguation(base_url, wiki_version, config)
        predictions, timing = model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        return result
