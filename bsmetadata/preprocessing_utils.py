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
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote, urlparse, urlsplit

from bs_dateutil.parser import ParserError, parse
from datasets import Value
from REL.entity_disambiguation import EntityDisambiguation
from REL.mention_detection import MentionDetection
from REL.ner import load_flair_ner
from REL.utils import process_results

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


class MetadataPreprocessor(ABC):
    """A metadata processor can be used for preprocessing text and adding or extracting metadata information."""

    def __init__(self, col_to_store_metadata: str) -> None:
        self.col_to_store_metadata = col_to_store_metadata
        super().__init__()

    @property
    @abstractmethod
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        """Returns a dictionary whose key corresponds to the name of a new column / a column modified by this processor
        and whose value corresponds to the minimal format of this column"""
        pass

    @abstractmethod
    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
        """Process a batch of examples and add or extract corresponding metadata."""
        pass


class TimestampPreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_metadata_url="metadata") -> None:
        self.col_metadata_url = col_metadata_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class HtmlPreprocessor(MetadataPreprocessor):
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
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class WebsiteDescPreprocessor(MetadataPreprocessor):
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
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:

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


class EntityPreprocessor(
    MetadataPreprocessor
):  # Note: To run this pre-processor, make sure that you have a column named "id" in the dataset.
    """Metadata preprocessor for adding entity information."""

    def __init__(
        self,
        base_url,
        num_chars,
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
        self.num_chars = num_chars
        self.col_text = col_text
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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
        # preprocess all the examples in a particular batch in the required format and check if length of text is less than num_chars
        processed = {ex_id: [ex_text, []] for ex_id, ex_text in enumerate(examples[self.col_text]) if len(ex_text) <= self.num_chars}
        return processed

    def fetch_mention_predictions(self, examples: Dict[str, List]) -> Dict[str, List]:
        # fetch mention predictions for all the examples in a particular batch at once.
        input_text = self.preprocess_example(examples)
        mentions_dataset, n_mentions = self.mention_detection.find_mentions(input_text, self.tagger_ner)
        predictions, timing = self.model.predict(mentions_dataset)
        result = process_results(mentions_dataset, predictions, input_text)
        return result

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class GenerationLengthPreprocessor(MetadataPreprocessor):
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
    def new_columns_minimal_features(self) -> Dict[str, Any]:

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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class DatasourcePreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding datasource information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_url="url") -> None:
        self.col_url = col_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class UrlPreprocessor(MetadataPreprocessor):
    """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

    def __init__(self, col_to_store_metadata="metadata", col_url="url") -> None:
        self.col_url = col_url
        super().__init__(col_to_store_metadata=col_to_store_metadata)

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
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

    def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
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


class ErrorWrapperPreprocessor:
    def __init__(
        self, metadata_preprocessor: MetadataPreprocessor, output_keys: Dict[str, Any], verbose: bool = True
    ) -> None:
        self.metadata_preprocessor = metadata_preprocessor
        self.output_keys = output_keys
        self.verbose = verbose

        self.error_column_name = f"{type(metadata_preprocessor).__name__}_error"
        self.error_comment_column_name = f"{type(metadata_preprocessor).__name__}_error_comment"

    @property
    def new_columns_minimal_features(self) -> Dict[str, Any]:
        features = self.metadata_preprocessor.new_columns_minimal_features
        features.update(
            {
                self.error_column_name: Value("int64"),
                self.error_comment_column_name: Value("string"),
            }
        )
        return features

    def preprocess(self, examples: Dict[str, List]) -> Tuple[Dict[str, List], int]:
        """Process a batch of examples and add or extract corresponding metadata."""
        num_errors = 0

        metadata_list_backup = {
            col_name: copy.deepcopy(examples[col_name])
            for col_name in self.metadata_preprocessor.new_columns_minimal_features.keys()
            if col_name in examples
        }
        try:
            processed_examples = self.metadata_preprocessor.preprocess(examples=examples)

            random_key = list(processed_examples)[0]
            num_examples = len(processed_examples[random_key])
            if self.error_column_name not in processed_examples:
                processed_examples[self.error_column_name] = [0 for _ in range(num_examples)]

            if self.error_comment_column_name not in processed_examples:
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
                    processed_example = self.metadata_preprocessor.preprocess(examples=example)

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
