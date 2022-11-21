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

from bsmetadata.preprocessing_tools import html_parser
from bsmetadata.preprocessing_utils import convert_str_to_datetime


@dataclass
class AllTagsRules:
    """Class containing rules common to all HTML tags to keep them or not"""

    attributes_to_keep: Optional[List[str]] = field(
        default=None,
        metadata={"help": "List of the html attributes that we wish to keep. If None we keep them all."},
    )
    txt_max_chr_len: float = field(
        default=-float("inf"),
        metadata={
            "help": "A tag will not be added if its opening tag is spaced from its ending tag by a number of "
            "characters between `txt_min_chr_len` and `txt_max_chr_len`."
        },
    )
    txt_min_chr_len: float = field(
        default=-float("inf"),
        metadata={
            "help": "A tag will not be added if its opening tag is spaced from its ending tag by a number of "
            "characters between `txt_min_chr_len` and `txt_max_chr_len`."
        },
    )
    tags_exceptions_to_txt_max_min_chr_len: List[str] = field(
        default_factory=(lambda: []),
        metadata={
            "help": "A list of tags that would be excluded from the filter out rule using the values of "
            "`txt_min_chr_len` and `txt_max_chr_len`."
        },
    )


@dataclass
class HTMLParserConfig:
    """A class to store all hyperparameters for adding or not adding html metadata"""

    all_tags_rules: AllTagsRules = AllTagsRules()
    tags_to_remove_alone_tag_name: List[str] = field(
        default_factory=(lambda: []),
        metadata={
            "help": "Lists of tags that are governed by a particular rule based on the spacing of their opening and "
            "closing tags to be kept or not in the final list. A tag will not be added if its opening tag is spaced "
            "from its ending tag by a number of characters between the value at the same index as it in the "
            "`tags_to_remove_alone_txt_min_chr_len` list and the corresponding value in the "
            "`tags_to_remove_alone_txt_max_chr_len` list."
        },
    )
    tags_to_remove_alone_txt_max_chr_len: List[float] = field(
        default_factory=(lambda: []),
        metadata={"help": "Corresponding text length lower bound."},
    )
    tags_to_remove_alone_txt_min_chr_len: List[float] = field(
        default_factory=(lambda: []),
        metadata={"help": "Corresponding text length upper bound."},
    )


@dataclass
class MetadataConfig:
    metadata_list: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of metadata types to use. Metadata is added in order of appearance in this list."},
    )
    metadata_column_list: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of column names containing metadata to use"},
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
    random_sample_metadata_calculate_size: int = field(
        default=-1,
        metadata={"help": "Number of examples to go through when calculating sampling proportion."},
    )
    metadata_probability: float = field(
        default=1, metadata={"help": "The probability of adding metadata to an input example."}
    )
    treat_local_metadata_as_regular_text: bool = field(
        default=False,
        metadata={
            "help": "If True, local metadata token will be associated to a `0` in the metadata_mask list. If False, "
            "local metadata token will be associated to a `1` in the metadata_mask list"
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
    entity_setting: str = field(
        default="beg",
        metadata={"help": "The settings in which you want to use entites. Valid choices: (beg, end)"},
    )
    local_metadata_special_token_start: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "A dictionary whose keys correspond to a local metadata type and value to the associated key will be prepended to the corresponding local metadata token."
        },
    )
    local_metadata_special_token_end: Optional[Dict[str, str]] = field(
        default=None,
        metadata={
            "help": "A dictionary whose keys correspond to a local metadata type and value to the associated key will be appended to the corresponding local metadata token."
        },
    )
    local_metadata_special_token_state: bool = field(
        default=False,
        metadata={"help": "If True, local metadata special tokens will be set as special tokens in the tokenizer"},
    )
    max_seq_len: int = field(
        default=512, metadata={"help": "The maximum number of tokens to use for each training chunk."}
    )
    html_parser_config: Optional[HTMLParserConfig] = HTMLParserConfig(
        AllTagsRules(
            attributes_to_keep=None,
            txt_max_chr_len=-float("inf"),
            txt_min_chr_len=-float("inf"),
            tags_exceptions_to_txt_max_min_chr_len=[],
        ),
        tags_to_remove_alone_tag_name=[],
        tags_to_remove_alone_txt_max_chr_len=[],
        tags_to_remove_alone_txt_min_chr_len=[],
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
        formatted_datetime = metadata_attrs["value"]
        if isinstance(formatted_datetime, str):
            formatted_datetime = convert_str_to_datetime(formatted_datetime)
        year_str = f"Year: {formatted_datetime.year}"
        month_str = f"Month: {formatted_datetime.strftime('%B')}"
        return self.cfg.metadata_sep.join((year_str, month_str))


class EntityProcessor(MetadataProcessor):
    """An example metadata processor for named entities."""

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent an entity by adding the entity name after the entity mention in double square brackets.
        # Example: "Biden [[Joe Biden]] studied at ..."
        return "", f" [[{metadata_attrs['value'].replace('_', ' ')}]]"


class EntityParagraphProcessor(MetadataProcessor):
    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent an entity_paragraph by adding the entity name in double square brackets at the beginning or end of the paragraph an entity resides.
        # Example: [[Joe Biden]] "Biden studied at ..."
        if self.cfg.entity_setting == "beg":
            return f" |{metadata_attrs['value'].replace('_', ' ')}|", ""
        elif self.cfg.entity_setting == "end":
            return "", f" |{metadata_attrs['value'].replace('_', ' ')}|"


class HtmlProcessor(MetadataProcessor):
    """An example metadata processor for HTMl tags."""

    def __init__(
        self,
        cfg: MetadataConfig,
    ):
        """
        Args:
            cfg: The data configuration to use.
        """
        super().__init__(cfg)
        attributes_to_keep = cfg.html_parser_config.all_tags_rules.attributes_to_keep
        txt_max_chr_len = cfg.html_parser_config.all_tags_rules.txt_max_chr_len
        txt_min_chr_len = cfg.html_parser_config.all_tags_rules.txt_min_chr_len
        tags_exceptions = cfg.html_parser_config.all_tags_rules.tags_exceptions_to_txt_max_min_chr_len
        tags_to_remove_alone = [
            html_parser.objects.TagToRemove(
                tag=tag, content_min_char_length=txt_min_chr_len, content_max_char_length=txt_max_chr_len
            )
            for (tag, txt_max_chr_len, txt_min_chr_len) in zip(
                cfg.html_parser_config.tags_to_remove_alone_tag_name,
                cfg.html_parser_config.tags_to_remove_alone_txt_max_chr_len,
                cfg.html_parser_config.tags_to_remove_alone_txt_min_chr_len,
            )
        ]

        self._tag_filter = html_parser.filters_and_cleaners.TagFilter(
            tags_to_remove_alone=tags_to_remove_alone,
            txt_min_chr_len_alone=txt_min_chr_len,
            txt_max_chr_len_alone=txt_max_chr_len,
            tags_exceptions_alone=tags_exceptions,
        )
        self._attributes_to_keep = attributes_to_keep

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent a html tag `T` by enclosing the corresponding text span with "<T>" and "</T>".
        # Example: An <b>apple</b> is an edible fruit.
        html_node = html_parser.objects.Metadata(
            char_start_idx=metadata_attrs["char_start_idx"],
            value=html_parser.objects.HtmlTag(
                tag=metadata_attrs["value"],
                attrs={
                    attr: attr_value
                    for attr, attr_value in zip(
                        metadata_attrs["html_attrs"]["attrs"], metadata_attrs["html_attrs"]["values"]
                    )
                },
            ),
            char_end_idx=metadata_attrs["char_end_idx"],
            key=metadata_attrs["key"],
            type=metadata_attrs["type"],
            relative_start_pos=0,  # unused
            relative_end_pos=0,  # unused
        )
        if self._tag_filter.drop_tag(html_node):
            return None

        attributes = " ".join(
            f"{attr}:{value}"
            for attr, value in zip(metadata_attrs["html_attrs"]["attrs"], metadata_attrs["html_attrs"]["values"])
            if (self._attributes_to_keep is None or attr in self._attributes_to_keep)
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


class TitleProcessor(MetadataProcessor):
    """An example metadata processor for titles."""

    def process_global(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
        # We represent a title by the title of the corresponding webpage content.
        # Example: "My Thoughts On It » Dad, I want to be an inventor".
        return "".join(["Title", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


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
    "datasource": DatasourceProcessor,
    "length": GenerationLengthProcessor,
    "entity": EntityProcessor,
    "entity_paragraph": EntityParagraphProcessor,
    "html": HtmlProcessor,
    "url": UrlProcessor,
    "website_description": WebsiteDescriptionProcessor,
    "title": TitleProcessor,
    "basic_start_local": BasicStartLocalProcessor,
}
