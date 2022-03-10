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
This script provides functions for processing different kinds of metadata.
"""
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote_plus

from dateutil.parser import parse


@dataclass
class MetadataConfig:
    metadata_list: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of metadata types to use. Metadata is added in order of appearance in this list."},
    )
    local_metadata_special_tokens: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "A dictionary whose keys correspond to a local metadata type and values to the associated  "
            "generation control token special. This dictionary will be used if "
            "`add_local_metadata_special_tokens_in_prefix` is `True`. If `add_local_metadata_special_tokens_in_prefix`"
            " is `True` and this argument is equal to `None` then the name of the local metadata will be used directly"
            " as special token.."
        },
    )
    metadata_sep: str = field(
        default=" | ",
        metadata={
            "help": "The character sequence that is used to separate two instances of global metadata and/or local "
            "metadata special tokens (if `add_local_metadata_special_tokens_in_prefix` is `True`)."
        },
    )
    metadata_key_value_sep: str = field(
        default=": ",
        metadata={"help": "The character sequence that is used by default to separate a metadata key and its value."},
    )
    random_sample_metadata: bool = field(
        default=False,
        metadata={"help": "Whether to random drop metadata, using bsmetadata.metadata_utils.random_sample_metadata."},
    )
    metadata_probability: float = field(
        default=1, metadata={"help": "The probability of adding metadata to an input example."}
    )
    treat_local_metadata_as_regular_text: bool = field(
        default=False,
        metadata={
            "help": "If True, local metadata token will be associated to a `0` int the metadata_mask list. If False, "
            "local metadata token will be associated to a `1` int the metadata_mask list"
        },
    )
    add_local_metadata_special_tokens_in_prefix: bool = field(
        default=False,
        metadata={
            "help": "If True, local metadata special tokens are added at the begining of the sample to indicate the "
            "type of metadata added in the sample. The special tokens used are equal to the string used in "
            "`metadata_list`"
        },
    )
    metadata_prefix_sep: str = field(
        default=" |||",
        metadata={
            "help": "The character sequence that is used to separate all global metadata and/or local metadata "
            "special tokens (if `add_local_metadata_special_tokens_in_prefix` is `True`) from the actual text."
        },
    )
    metadata_prefix_start_seq: str = field(
        default="",
        metadata={"help": "The character sequence to be concatenated at the beginning of the metadata prefix."},
    )
    max_seq_len: int = field(
        default=512, metadata={"help": "The maximum number of tokens to use for each training chunk."}
    )


class MetadataProcessor:
    """A metadata processor can be used to add both global and local metadata information to a given input text."""

    def __init__(self, cfg: MetadataConfig):
        """
        Args:
            cfg: The data configuration to use.
        """
        self.cfg = cfg

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        """Process a single instance of global metadata and compute the corresponding prefix.

        This prefix is added at the very beginning (that is, before the actual input text), along with all other global metadata.
        By default, global metadata is represented as a key-value pair and separated using `self.cfg.metadata_key_value_sep`, which
        defaults to ": ". For example, for a metadata instance with key "url" and value "wikipedia.com/Apple", the default global
        prefix will be "url: wikipedia.com/Apple".

        Args:
            metadata_attrs: All attributes of this metadata instance. Each global metadata instance is expected to have an attribute "key"
            of type string and a corresponding "value" of arbitrary type.

        Returns:
            A single string representing the prefix that should be added to the input for this metadata instance.
        """
        return "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, metadata_attrs["value"]])

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Process a single instance of local metadata and compute the corresponding prefix and suffix.

        Local metadata must have a character-level start and end index. The prefix returned by this function is added directly before the
        start index, the suffix is added directly at the end index. For example, for an input "a b c" and a metadata instance with start
        index 2 and end index 3, if this function returns the tuple `("<b>", "</b>")`, the input will be converted to "a <b>b</b> c".

        Args:
            metadata_attrs: All attributes of this metadata instance. Each local metadata instance is expected to have an attribute "key"
            of type string and a corresponding "value" of arbitrary type, as well as a "char_start_idx" and "char_end_idx" of type int.

        Returns:
            A tuple of two strings representing the prefix and suffix that should be added to the input for this metadata instance.
        """
        kv_pair = "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, metadata_attrs["value"]])
        return f"[{kv_pair}]", f"[/{kv_pair}]"


class TimestampProcessor(MetadataProcessor):
    """An example metadata processor for timestamps."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # We represent a timestamp using only the year and month.
        # Example: "Year: 2020 | Month: September".
        formatted_datetime = parse(metadata_attrs["value"])
        year_str = f"Year: {formatted_datetime.year}"
        month_str = f"Month: {formatted_datetime.strftime('%B')}"
        return self.cfg.metadata_sep.join((year_str, month_str))


class EntityProcessor(MetadataProcessor):
    """An example metadata processor for named entities."""

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent an entity by adding the entity name after the entity mention in double square brackets.
        # Example: "Biden [[Joe Biden]] studied at ..."
        return "", f" [[{metadata_attrs['value']}]]"


class HtmlProcessor(MetadataProcessor):
    """An example metadata processor for HTMl tags."""

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent a html tag `T` by enclosing the corresponding text span with "<T>" and "</T>".
        # Example: An <b>apple</b> is an edible fruit.
        attributes = " ".join(
            f"{attr}:{value}"
            for attr, value in zip(metadata_attrs["html_attrs"]["attrs"], metadata_attrs["html_attrs"]["values"])
        )
        if attributes:
            attributes = " " + attributes
        return f"<{metadata_attrs['value']}{attributes}>", f"</{metadata_attrs['value']}>"


class UrlProcessor(MetadataProcessor):
    """An example metadata processor for URLs."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # We represent a URL with unquoted format such that less confusion for a tokenizer.
        # Example: "foo.bar/Year 2021/" instead of "foo.bar/Year%202021/".
        return "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, unquote_plus(metadata_attrs["value"])])


class WebsiteDescriptionProcessor(MetadataProcessor):
    """An example metadata processor for website descriptions."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # Example: "website_description: BBC is a news organization".
        return "".join(["Website Description", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


class DatasourceProcessor(MetadataProcessor):
    """An example metadata processor for datasource types."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # We represent the DATASOURCE by using meaningful information of the URL.
        # URL: http://www.example.de/2015/forum/article/21-new-project
        # Example: example.de > forum > article > new project
        return "".join(["Datasource", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


class GenerationLengthProcessor(MetadataProcessor):
    """An example metadata processor for the text length."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # We represent the length of a text by the number of characters.
        # Example: Length: 123

        return "".join(["Text Length", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


class BasicStartLocalProcessor(MetadataProcessor):
    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # This is a basic processor that just creates a local start tag from the value stored in the metadata
        return metadata_attrs["value"], ""


PROCESSORS = {
    "timestamp": TimestampProcessor,
    "source": DatasourceProcessor,
    "length": GenerationLengthProcessor,
    "entity": EntityProcessor,
    "html": HtmlProcessor,
    "url": UrlProcessor,
    "website_description": WebsiteDescriptionProcessor,
    "basic_start_local": BasicStartLocalProcessor,
}
