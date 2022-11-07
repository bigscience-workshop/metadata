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

import copy
import logging
import re
from abc import ABC, abstractmethod
from datetime import datetime as DateTime
from typing import Any, Dict, List, Optional, Union
from urllib.parse import unquote, urlparse, urlsplit

from bs_dateutil.parser import ParserError, parse
from datasets import Value
from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import load_flair_ner
from REL.utils import process_results

from bsmetadata.paragraph_by_metadata_html import get_paragraphs
from bsmetadata.preprocessing_tools import html_parser
from bsmetadata.preprocessing_tools.wikipedia_desc_utils import WikipediaDescUtils


logger = logging.getLogger(__name__)


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


def convert_str_to_datetime(timestamp_str: str) -> DateTime:
    """A temporarily special treatment that converts a roughly 11-to-13-digit integer/float string to datetime."""
    try:
        # Unify int sec., long ms/ns, and float ms/ns FOR OUR CASES.
        return DateTime.fromtimestamp(float(f"{timestamp_str[:-3]}.{timestamp_str[-3:]}".replace("..", ".")))
    except ValueError:
        return parse(timestamp_str)


OneToOneFeature = Dict[str, Value]
OneToManyToListFeature = Dict[str, Dict[str, List[Value]]]
RawFeatures = Union[Value, List[Value]]
ComplexFeatures = Union[List[Union[OneToOneFeature, OneToManyToListFeature]], RawFeatures]


class MetadataTagger(ABC):
    """A metadata processor can be used for preprocessing text and adding or extracting metadata information."""

    def __init__(self, col_to_store_metadata: str) -> None:
        self.col_to_store_metadata = col_to_store_metadata
        super().__init__()

    @property
    @abstractmethod
    def new_columns_minimal_features(self) -> Dict[str, ComplexFeatures]:
        """Returns a dictionary whose key corresponds to the name of a new column / a column modified by this processor
        and whose value corresponds to the minimal format of this column"""
        pass

    @abstractmethod
    def tag(self, examples: Dict[str, List]) -> Dict[str, ComplexFeatures]:
        """Process a batch of examples and add or extract corresponding metadata."""
        pass


class TimestampPreprocessor(MetadataTagger):
    """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_metadata_url="metadata") -> None:
        self.col_metadata_url = col_metadata_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_metadata_url]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_metadata_url, example_metadata in zip(examples[self.col_metadata_url], example_metadata_list):
            # Get the URL associated with this example.
            example_urls = [md["value"] for md in example_metadata_url if md["key"] == "url"]

            if not example_urls:
                continue

            # Try to extract a timestamp from the given URL and add it to the metadata.
            example_timestamp = self._extract_timestamp_from_url(example_urls[0])

            if example_timestamp:
                example_metadata.append({"key": "timestamp", "type": "global", "value": example_timestamp})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples

    def _extract_timestamp_from_url(self, url: str) -> Optional[str]:
        path = get_path_from_url(url)
        date = parse_date(path)
        date = remove_improbable_date(date)
        date = str(date) if date is not None else None
        return date


class HtmlPreprocessor(MetadataTagger):
    """Metadata preprocessor for extracting metadata from html text.

    Specifically, it separates the html text contained in the `col_html`` column into a text and a list of
    HTML metadata containing the tags, their attributes, their location in the text and their relative location to
    each other."""

    def __init__(
        self,
        col_html: str = "doc_html",
        col_to_store_metadata="metadata",
        col_to_store_text="text",
        col_to_store_head="html_head",
        col_to_store_footer="html_footer",
        col_to_store_title="html_title",
    ) -> None:
        self.col_html = col_html
        self.col_to_store_text = col_to_store_text
        self.col_to_store_footer = col_to_store_footer
        self.col_to_store_head = col_to_store_head
        self.col_to_store_title = col_to_store_title
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, ComplexFeatures]:
        features = {
            self.col_to_store_metadata: [
                {
                    "char_end_idx": Value("int64"),
                    "char_start_idx": Value("int64"),
                    "html_attrs": {"attrs": [Value("string")], "values": [Value("string")]},
                    "key": Value("string"),
                    "relative_end_pos": Value("int64"),
                    "relative_start_pos": Value("int64"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ],
            self.col_to_store_text: Value("string"),
            self.col_to_store_footer: [Value("string")],
            self.col_to_store_head: [Value("string")],
            self.col_to_store_title: [Value("string")],
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, ComplexFeatures]:
        tags_to_remove_with_content = [
            html_parser.objects.TagToRemoveWithContent(tag="script"),
            html_parser.objects.TagToRemoveWithContent(tag="style"),
            html_parser.objects.TagToRemoveWithContent(tag="header"),
            html_parser.objects.TagToRemoveWithContent(tag="iframe"),
            html_parser.objects.TagToRemoveWithContent(tag="footer"),  # copyright in footer
            html_parser.objects.TagToRemoveWithContent(tag="form"),
            html_parser.objects.TagToRemoveWithContent(tag="body", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="div", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="p", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="section", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="table", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="ul", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="ol", content_max_char_length=64),
            html_parser.objects.TagToRemoveWithContent(tag="dl", content_max_char_length=64),
        ]
        head_tag = "head"
        footer_tag = "footer"
        title_tag = "title"

        new_texts = []
        new_head = []
        new_footer = []
        new_title = []
        new_metadata = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_html]))]
        )
        for example_doc_html, example_metadata in zip(
            examples[self.col_html], new_metadata
        ):  # if metadata already exists

            plain_text, metadata, additional_columns = html_parser.get_clean_text_and_metadata(
                example_doc_html,
                tags_to_remove_with_content=tags_to_remove_with_content,
                consecutive_tags_to_fold=["div"],
                convert_br_tag_to_breaking_line=True,
                tags_sub_tree_to_isolate=[head_tag, footer_tag, title_tag],
            )
            new_texts.append(plain_text)
            new_head.append(additional_columns.get(head_tag, []))
            new_footer.append(additional_columns.get(footer_tag, []))
            new_title.append(additional_columns.get(title_tag, []))
            example_metadata.extend(
                [html_parser.objects.convert_html_metadata_dataclass_to_dict(node) for node in metadata]
            )

        examples[self.col_to_store_text] = new_texts
        examples[self.col_to_store_metadata] = new_metadata
        examples[self.col_to_store_head] = new_head
        examples[self.col_to_store_footer] = new_footer
        examples[self.col_to_store_title] = new_title
        return examples


class WebsiteDescPreprocessor(MetadataTagger):
    """Metadata preprocessor for adding website description based on URLs."""

    def __init__(
        self,
        path_wiki_db: str = "../preprocessing_data/wiki_dump/wiki_en_dump_db",
        col_to_store_metadata="metadata",
        col_metadata_url="metadata",
    ) -> None:
        self.website_utils = WikipediaDescUtils(path_wiki_db)

        self.col_metadata_url = col_metadata_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:

        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_metadata_url]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_metadata_url, example_metadata in zip(examples[self.col_metadata_url], example_metadata_list):
            # Get the URL associated with this example.
            urls = [md["value"] for md in example_metadata_url if md["key"] == "url"]

            if not urls:
                continue

            # Try to extract a website description from the given URL and add it to the metadata.
            website_description = self._extract_website_desc_from_url(urls[0])

            if website_description:
                example_metadata.append({"key": "website_description", "type": "global", "value": website_description})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples

    def _extract_website_desc_from_url(self, url: str) -> Optional:

        keyword = fetch_keyword_from_url(url)
        return self.website_utils.fetch_website_description_from_keyword(keyword)


class WebsiteDescPostprocessor(MetadataTagger):
    """website metadata post-processor to remove noisy data"""

    def __init__(
        self,
        col_to_store_metadata="metadata",
    ) -> None:
        corrupt_patterns = [".* refer(|s) to.?:", "\[\[\w*:"]
        corrupt_pattern_str = "|".join("({0})".format(x) for x in corrupt_patterns)
        # TODO: Stricter regex patterns for speed

        self.corrupt_regex = re.compile(corrupt_pattern_str)
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List]:
        example_metadata_list = examples[self.col_to_store_metadata]
        # Iterate through the metadata associated with all examples in this batch.

        for example_metadata in example_metadata_list:
            if example_metadata and (
                self.is_noisy_data(example_metadata[0]["value"])
                or self.is_outlier(example_metadata[0]["value"])
                # TODO: Trim `example_metadata[0]["value"]` and share it for speed
                # TODO: Run `is_outlier()` first for speed
            ):
                example_metadata.clear()  # remove website description with empty list if metadata is invalid
        examples[self.col_to_store_metadata] = example_metadata_list
        return examples

    def is_noisy_data(self, data):
        return self.corrupt_regex.match(data)

    def is_outlier(self, data):
        return len(data.split()) < 5 or len(data.split()) > 50  # caps tbd
        # TODO: Run `data.split()` only once for speed


class EntityPreprocessor(
    MetadataTagger
):  # Note: To run this pre-processor, make sure that you have a column named "id" in the dataset.
    """Metadata preprocessor for adding entity information."""

    def __init__(
        self,
        base_url,
        path_or_url_flair_ner_model="ner-fast",
        col_to_store_metadata="metadata",
        col_text="text",
    ):
        self.base_url = base_url
        self.wiki_version = "wiki_2019"
        self.mention_detection = MentionDetection(self.base_url, self.wiki_version)
        self.tagger_ner = load_flair_ner(path_or_url_flair_ner_model)
        self.config = {
            "mode": "eval",
            "model_path": "ed-wiki-2019",
        }
        self.model = EntityDisambiguation(self.base_url, self.wiki_version, self.config, reset_embeddings=True)

        self.col_text = col_text
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "char_end_idx": Value("int64"),
                    "char_start_idx": Value("int64"),
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def preprocess_example(self, examples: Dict[str, List]) -> Dict[str, List]:
        # preprocess all the examples in a particular batch in the required format
        processed = {ex_id: [ex_text, []] for ex_id, ex_text in enumerate(examples[self.col_text])}
        return processed

    def fetch_mention_predictions(self, examples: Dict[str, List]) -> Dict[str, List]:
        # fetch mention predictions for all the examples in a particular batch at once.
        input_text = self.preprocess_example(examples)
        mentions_dataset, n_mentions = self.mention_detection.find_mentions(input_text, self.tagger_ner)
        predictions, timing = self.model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        return result

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        # process all the examples in a particular batch and all the metadata extracted for entities for those examples
        mentions_predicted = self.fetch_mention_predictions(examples)

        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_text]))]
        )

        for example_id, example_metadata in enumerate(example_metadata_list):
            if example_id not in mentions_predicted:
                continue

            # fetch all the elements for which entity tags are present by mapping through "id"
            mentions_predicted_for_id = mentions_predicted[example_id]
            for mention_predicted in mentions_predicted_for_id:
                # element at index = 3 in the result list corresponds to the predicted entity
                entity = mention_predicted[3]
                # element at index = 0 in the result list corresponds to the char start ind
                char_start_idx = mention_predicted[0]
                # element at index = 1 in the result list corresponds to length of the entity
                char_end_idx = mention_predicted[0] + mention_predicted[1]

                en = {
                    "key": "entity",
                    "type": "local",
                    "char_start_idx": char_start_idx,
                    "char_end_idx": char_end_idx,
                    "value": entity,
                }
                example_metadata.append(en)

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples


class GenerationLengthPreprocessor(MetadataTagger):
    """An exemplary metadata preprocessor for adding generation length information based on text."""

    def __init__(
        self,
        mode,
        col_to_store_metadata="metadata",
        col_text="text",
    ) -> None:
        # The length can be calculated for the whole text or for each sentence of a text individually.
        # We can specify a global length of a TEXT or a local length for each SENTENCE of a text.
        # Therefore, we provide two different modes: text (global) or sentence (local).
        self.mode = mode  # {text, sentence}

        self.col_text = col_text
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:

        if self.mode == "text":
            features = {
                self.col_to_store_metadata: [
                    {
                        "key": Value("string"),
                        "type": Value("string"),
                        "value": Value("string"),
                    }
                ]
            }
        elif self.mode == "sentence":
            features = {
                self.col_to_store_metadata: [
                    {
                        "char_end_idx": Value("int64"),
                        "char_start_idx": Value("int64"),
                        "key": Value("string"),
                        "type": Value("string"),
                        "value": Value("string"),
                    }
                ]
            }
        else:
            raise ValueError("Please select a valid length type [text or sentence].")

        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        """
        Iterate through all the examples retrieve the length meta information.
        """

        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_text]))]
        )

        for example_text, example_metadata in zip(examples[self.col_text], example_metadata_list):
            if self.mode == "text":
                text_length = self._extract_length_from_text(example_text)
                example_length = {"key": "length", "type": "global", "value": text_length}
                if not example_length:
                    continue
                example_metadata.append(example_length)
            elif self.mode == "sentence":
                example_length = self._extract_length_from_sentences(example_text)
                example_metadata.extend(example_length)
            else:
                print("Please select a valid length type [text or sentence].")

        examples[self.col_to_store_metadata] = example_metadata_list
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


class DatasourcePreprocessor(MetadataTagger):
    """An exemplary metadata preprocessor for adding datasource information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_url="url") -> None:
        self.col_url = col_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

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

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_url]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_url, example_metadata in zip(examples[self.col_url], example_metadata_list):
            example_datasource = self._extract_datasource_from_url(example_url)
            if not example_datasource:
                continue

            example_metadata.append({"key": "datasource", "type": "global", "value": example_datasource})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples


class UrlPreprocessor(MetadataTagger):
    """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_url="url") -> None:
        self.col_url = col_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_url]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_url, example_metadata in zip(examples[self.col_url], example_metadata_list):
            if example_url:
                example_metadata.append({"key": "url", "type": "global", "value": example_url})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples


class TitlePreprocessor(MetadataTagger):
    """An exemplary metadata preprocessor for adding titles information."""

    def __init__(self, col_to_store_metadata="metadata", col_title="html_title") -> None:
        self.col_title = col_title
        self.title_regex = re.compile(r"<title[^>]*>(.*)</title>")
        # TODO: A stricter regex pattern for speed

        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ]
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_title]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_title, example_metadata in zip(examples[self.col_title], example_metadata_list):

            # The number of titles retrieved on a page is not necessarily equal to 1. Here the choice is made to keep only the first title retrieved when there is one.
            if not example_title:
                continue
            title = example_title[0]
            # TODO: Trim string for speed

            title = self.title_regex.search(title)
            # If title is not None, we keep the first title retrieved.
            if title:
                title = title.group(1)
                example_metadata.append({"key": "title", "type": "global", "value": title})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples


class ParagraphPreprocessor(MetadataTagger):
    """ParagraphPreprocessor Extract paragraphs based on HTML and line-breaking markers."""

    def __init__(self, col_to_store_metadata: str = "metadata", col_metadata_html: str = "metadata_html") -> None:
        super().__init__(col_to_store_metadata=col_to_store_metadata)
        self._col_url = "url"
        self._col_mtdt_html = col_metadata_html
        self._col_text = "text"

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        """new_columns_minimal_features Paragraph's `Features`.

        Note:
            Added a new string field "marker".

        Returns:
            Dict[str, List[OneToOneFeature]]: Paragraph-specific `Features`.
        """
        features = {
            self.col_to_store_metadata: [
                {
                    "char_end_idx": Value("int64"),
                    "char_start_idx": Value("int64"),
                    "key": Value("string"),
                    "type": Value("string"),
                    "value": Value("string"),
                    "marker": Value("string"),
                }
            ],
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        exmpl_mtdt_list = examples.get(self.col_to_store_metadata, [[] for _ in range(len(examples[self._col_url]))])
        exmpl_mtdt_htmls = examples.get(self._col_mtdt_html)
        exmpl_txts = examples.get(self._col_text)

        if exmpl_txts and exmpl_mtdt_htmls:
            for exmpl_txt, exmpl_mtdt_html, exmpl_mtdt in zip(exmpl_txts, exmpl_mtdt_htmls, exmpl_mtdt_list):
                if exmpl_txt and exmpl_mtdt_html:
                    exmpl_mtdt += get_paragraphs(exmpl_mtdt_html, exmpl_txt)

        examples[self.col_to_store_metadata] = exmpl_mtdt_list
        return examples


class EntityParagraphPreprocessor(MetadataTagger):
    """A metadata preprocessor for updating entity information based on paragraphs."""

    def __init__(
        self, col_to_store_metadata="metadata", col_entity="metadata_entity", col_paragraph="metadata_paragraph"
    ) -> None:
        self.col_entity = col_entity
        self.col_paragraph = col_paragraph
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, List[OneToOneFeature]]:
        features = {
            self.col_to_store_metadata: [
                {
                    "char_end_idx": Value("int64"),
                    "char_start_idx": Value("int64"),
                    "key": Value("string"),
                    "relative_end_pos": Value("int64"),
                    "relative_start_pos": Value("int64"),
                    "type": Value("string"),
                    "value": Value("string"),
                }
            ],
        }
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, List[OneToOneFeature]]:
        example_metadata_list = (
            examples[self.col_to_store_metadata]
            if self.col_to_store_metadata in examples
            else [[] for _ in range(len(examples[self.col_entity]))]
        )

        # Iterate through the metadata associated with all examples in this batch.
        for example_entity, example_paragraph, example_metadata in zip(
            examples[self.col_entity], examples[self.col_paragraph], example_metadata_list
        ):

            if not example_entity or not example_paragraph:
                continue

            # Iterate through the entities associated with this example.
            for entity in example_entity:
                # If entity["key"] != "entity", we skip this entity. Added this check since tests were failing.
                if entity["key"] != "entity":
                    continue
                # Initialize the start and end index of an entity
                start_index = entity["char_start_idx"]
                end_index = entity["char_end_idx"]
                check = True

                # Search the start and end index of paragraph in between which the entity is present without any duplicate entity values.
                for paragraph in example_paragraph:
                    # If paragraph["key"] != "paragraph", we skip this paragraph. Added this check since tests were failing.
                    if paragraph["key"] != "paragraph":
                        check = False
                        continue
                    if start_index >= paragraph["char_start_idx"] and end_index <= paragraph["char_end_idx"]:
                        # Update the start and end index of an entity
                        start_index = paragraph["char_start_idx"]
                        end_index = paragraph["char_end_idx"]
                        check = True
                        break
                en = {
                    "key": "entity_paragraph",
                    "type": "local",
                    "char_start_idx": start_index,
                    "char_end_idx": end_index,
                    "value": entity["value"],
                }
                # Add the entity paragraph information to the example metadata if it is not already present.
                if en not in example_metadata and check:
                    example_metadata.append(en)

            # Add relative start and end position information to the example metadata.
            for index, entity in enumerate(example_metadata):
                if (
                    index > 0
                    and example_metadata[index - 1]["key"]
                    == "entity_paragraph"  # Added this check since tests were failing.
                    and example_metadata[index]["key"]
                    == "entity_paragraph"  # Added this check since tests were failing.
                    and example_metadata[index]["char_start_idx"] == example_metadata[index - 1]["char_start_idx"]
                ):
                    example_metadata[index].update(
                        {
                            "relative_start_pos": (example_metadata[index - 1]["relative_start_pos"] + 1),
                            "relative_end_pos": (example_metadata[index - 1]["relative_end_pos"] + 1),
                        }
                    )
                else:
                    example_metadata[index].update({"relative_start_pos": 0, "relative_end_pos": 0})

        examples[self.col_to_store_metadata] = example_metadata_list
        return examples


class ErrorWrapperPreprocessor:
    def __init__(
        self, metadata_preprocessor: MetadataTagger, output_keys: Dict[str, Any], verbose: bool = True
    ) -> None:
        self.metadata_preprocessor = metadata_preprocessor
        self.output_keys = output_keys
        self.verbose = verbose

        self.error_column_name = f"{type(metadata_preprocessor).__name__}_error"
        self.error_comment_column_name = f"{type(metadata_preprocessor).__name__}_error_comment"

    @property
    def new_columns_minimal_features(self) -> Dict[str, ComplexFeatures]:
        features = self.metadata_preprocessor.new_columns_minimal_features
        features.update(
            {
                # Types below are are different than ~`preprocess`'s.
                self.error_column_name: Value("int64"),
                self.error_comment_column_name: Value("string"),
            }
        )
        return features

    def tag(self, examples: Dict[str, List]) -> Dict[str, ComplexFeatures]:
        """Process a batch of examples and add or extract corresponding metadata."""
        num_errors = 0

        metadata_list_backup = {
            col_name: copy.deepcopy(examples[col_name])
            for col_name in self.metadata_preprocessor.new_columns_minimal_features.keys()
            if col_name in examples
        }
        try:
            processed_examples = self.metadata_preprocessor.tag(examples=examples)

            random_key = list(processed_examples)[0]
            num_examples = len(processed_examples[random_key])
            if self.error_column_name not in processed_examples:
                # `List[Value]`, cf. ~`new_columns_minimal_features`
                processed_examples[self.error_column_name] = [0 for _ in range(num_examples)]

            if self.error_comment_column_name not in processed_examples:
                # `List[Value]`, cf. ~`new_columns_minimal_features`
                processed_examples[self.error_comment_column_name] = ["" for _ in range(num_examples)]
        except:  # noqa
            # we try the example one by one to find the culprit(s) and strore the error
            processed_examples = {
                key: []
                for key in list(self.output_keys.keys()) + [self.error_column_name, self.error_comment_column_name]
            }

            for key, values in metadata_list_backup.items():
                examples[key] = copy.deepcopy(values)

            random_key = list(examples)[0]
            for idx in range(len(examples[random_key])):
                example = {key: [values[idx]] for key, values in examples.items()}
                try:
                    processed_example = self.metadata_preprocessor.tag(examples=example)

                    for key, value in processed_example.items():
                        processed_examples[key].append(value[0])

                    processed_examples[self.error_column_name].append(0)
                    processed_examples[self.error_comment_column_name].append("")
                except Exception as e:
                    for output_key in self.output_keys.keys():
                        if output_key in metadata_list_backup:
                            # We keep the initial value
                            processed_examples[output_key].append(metadata_list_backup[output_key][idx])
                        elif output_key in example:
                            # We keep the initial value
                            processed_examples[output_key].append(example[output_key][0])
                        else:
                            # We use the default value
                            processed_examples[output_key].append(self.output_keys[output_key])

                    processed_examples[self.error_column_name].append(1)
                    processed_examples[self.error_comment_column_name].append(str(e))
                    logger.info(f"An error occurred with the message: {str(e)}")
                    num_errors += 1
        if self.verbose and num_errors != 0:
            logger.warning(f"{num_errors} errors occurred during the preprocessing")
        return processed_examples
