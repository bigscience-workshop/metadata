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

from bs_dateutil.parser import ParserError, parse
from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import load_flair_ner
from REL.utils import process_results

from bsmetadata.preprocessing_tools import html_parser
from bsmetadata.preprocessing_tools.wikipedia_desc_utils import WikipediaDescUtils


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


def fetch_keyword_from_url(url: str) -> str:  # e.g http://www.californialandcan.org/Plumas -> californialandcan.org
    domain = urlsplit(url).netloc
    return domain.replace("www.", "")


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


class HtmlPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for extracting metadata from html text.

    Specifically, it separates the html text contained in the `name_html_column`` column into a text and a list of
    HTML metadata containing the tags, their attributes, their location in the text and their relative location to
    each other."""

    def __init__(self, name_html_column: str = "doc_html") -> None:
        self.name_html_column = name_html_column
        super().__init__()

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        tags_to_remove_with_content = [
            html_parser.objects.TagToRemoveWithContent(tag="script"),
            html_parser.objects.TagToRemoveWithContent(tag="style"),
            html_parser.objects.TagToRemoveWithContent(tag="header"),
            html_parser.objects.TagToRemoveWithContent(tag="iframe"),
            html_parser.objects.TagToRemoveWithContent(tag="footer"),  # copyright in footer
            html_parser.objects.TagToRemoveWithContent(tag="form"),
        ]

        new_texts = []
        for example_doc_html, example_metadata in zip(
            examples[self.name_html_column], examples["metadata"]
        ):  # if metadata already exists

            plain_text, metadata = html_parser.get_clean_text_and_metadata(
                example_doc_html,
                tags_to_remove_with_content=tags_to_remove_with_content,
                consecutive_tags_to_fold=["div"],
                convert_br_tag_to_breaking_line=True,
            )
            new_texts.append(plain_text)
            example_metadata.extend(
                [html_parser.objects.convert_html_metadata_dataclass_to_dict(node) for node in metadata]
            )

        examples["texts"] = new_texts
        return examples


class WebsiteDescPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for adding website description based on URLs."""

    def __init__(self, path_wiki_db: str = "../preprocessing_data/wiki_dump/wiki_en_dump_db") -> None:
        self.website_utils = WikipediaDescUtils(path_wiki_db)
        super().__init__()

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

        metadata_list = examples["metadata"]

        # Iterate through the metadata associated with all examples in this batch.
        for metadata in metadata_list:
            # Get the URL associated with this example.
            urls = [md["value"] for md in metadata if md["key"] == "url"]

            if not urls:
                continue

            # Try to extract a website description from the given URL and add it to the metadata.
            website_description = self._extract_website_desc_from_url(urls[0])

            if website_description:
                metadata.append({"key": "website_description", "type": "global", "value": website_description})
        return examples

    def _extract_website_desc_from_url(self, url: str) -> Optional:

        keyword = fetch_keyword_from_url(url)
        return self.website_utils.fetch_website_description_from_keyword(keyword)


class EntityPreprocessor(
    MetadataPreprocessor
):  # Note: To run this pre-processor, make sure that you have a column named "id" in the dataset.
    """Metadata preprocessor for adding entity information."""

    def __init__(self, base_url, path_wiki_db):
        self.wiki_db_path = path_wiki_db
        self.entity_utils = WikipediaDescUtils(path_wiki_db)
        self.base_url = base_url
        self.wiki_version = "wiki_2019"
        self.mention_detection = MentionDetection(self.base_url, self.wiki_version)
        self.tagger_ner = load_flair_ner("ner-fast")
        self.config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.model = EntityDisambiguation(self.base_url, self.wiki_version, self.config, reset_embeddings=True)
        super().__init__()

    def preprocess_example(self, examples: Dict[str, List]) -> Dict[str, List]:
        # preprocess all the examples in a particular batch in the required format
        processed = {}
        for example_id, example_text in zip(examples["id"], examples["text"]):
            id_ = example_id
            text_ = example_text
            value = [text_, []]
            d = {id_: value}
            processed.update(d)
        return processed

    def fetch_mention_predictions(self, examples: Dict[str, List]) -> Dict[str, List]:
        # fetch mention predictions for all the examples in a particular batch at once.
        input_text = self.preprocess_example(examples)
        mentions_dataset, n_mentions = self.mention_detection.find_mentions(input_text, self.tagger_ner)
        predictions, timing = self.model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        return result

    def _extract_desc_from_entity(self, keyword: str) -> Optional:
        # fetch description of an entity
        key = keyword
        key = key.lower()
        key = key.replace("_", " ")
        return self.entity_utils.fetch_entity_description_from_keyword(key)

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        # process all the examples in a particular batch and all the metadata extracted for entities for those examples

        res = self.fetch_mention_predictions(examples)

        for example_id, example_metadata in zip(examples["id"], examples["metadata"]):
            if example_id in res:  # fetch all the elements for which entity tags are present by mapping through "id"
                r = res[example_id]
                for i in range(len(r)):
                    entity = r[i][3]  # element at index = 3 in the result list corresponds to the predicted entity
                    ent_desc = self._extract_desc_from_entity(entity)
                    en = {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": r[i][
                            0
                        ],  # element at index = 0 in the result list corresponds to the char start index
                        "char_end_idx": (
                            r[i][0] + r[i][1]
                        ),  # element at index = 1 in the result list corresponds to length of the entity
                        "value": entity,
                        "ent_desc": ent_desc,
                    }
                    example_metadata.append(en)
            continue
        return examples
