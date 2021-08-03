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
from typing import Dict, Any, Tuple, Optional

from input_pipeline import DataConfig


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

        Args:
            metadata_attrs: All attributes of this metadata instance, following our default format.

        Returns:
            A single string representing the prefix that should be added to the input for this metadata instance.
        """
        return "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, metadata_attrs["value"]])

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        """Process a single instance of local metadata and compute the corresponding prefix and suffix.

        Args:
            metadata_attrs: All attributes of this metadata instance, following our default format.

        Returns:
            A tuple of two strings representing the prefix and suffix that should be added to the input for this metadata instance.
        """
        kv_pair = "".join([metadata_attrs["key"], self.cfg.metadata_key_value_sep, metadata_attrs["value"]])
        return f"[{kv_pair}]", f"[/{kv_pair}]"


class EntityProcessor(MetadataProcessor):
    """An example metadata processor for named entities."""

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        return "", f" [[{metadata_attrs['value']}]]"


class HtmlProcessor(MetadataProcessor):
    """An example metadata processor for HTMl tags."""

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        return f"<{metadata_attrs['value']}>", f"</{metadata_attrs['value']}>"


PROCESSORS = {
    'entity': EntityProcessor,
    'html': HtmlProcessor
}
