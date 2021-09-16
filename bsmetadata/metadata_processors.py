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
import datetime
from typing import Any, Dict, Optional, Tuple
from urllib.parse import unquote_plus

from bsmetadata.input_pipeline import DataConfig


class MetadataProcessor:
    """A metadata processor can be used to add both global and local metadata information to a given input text."""

    def __init__(self, cfg: DataConfig):
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
        formatted_datetime = datetime.datetime.strptime(metadata_attrs["value"], "%Y-%m-%dT%H:%M:%S.%fZ")
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
        return f"<{metadata_attrs['value']}>", f"</{metadata_attrs['value']}>"


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
        return "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


PROCESSORS = {
    "timestamp": TimestampProcessor,
    "entity": EntityProcessor,
    "html": HtmlProcessor,
    "url": UrlProcessor,
    "website_description":WebsiteDescriptionProcessor
}
