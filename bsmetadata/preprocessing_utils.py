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
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Optional

import requests


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


class WebsiteDescPreprocessor(MetadataPreprocessor):
    """Metadata preprocessor for adding website description based on URLs."""

    website_description_cache = {}
    org_list = ["com", "co", "org", "go", "in"]

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
                metadata.append({"key": "timestamp", "type": "global", "value": website_description})

        return examples

    def _extract_website_desc_from_url(self, url: str) -> Optional:

        domain = url.str.split("/")[2]  # e.g http://www.californialandcan.org/Plumas -> www.californialandcan.org
        keywords = domain.str.split(".")

        keyword = (
            keywords[-2]
            if len(keywords[-2]) > 3
            else keywords[1]
            if (keywords[1] not in self.org_list)
            else keywords[0]
        )  # extracting the keyword from domain e.g.  www.californialandcan.org -> californialandcan

        if keyword not in self.website_description_cache:
            self.website_description_cache[keyword] = self.extract_wiki_desc(keyword)

        return self.website_description_cache[keyword]

    def extract_wiki_desc(self, keyword: str) -> Optional:

        keyword = keyword.replace(" ", "_")
        r = requests.get(
            "https://en.wikipedia.org/w/api.php?action=query&prop=extracts&titles="
            + keyword
            + "&exintro=&exsentences=2&explaintext=&redirects=&formatversion=2&format=json"
        )
        page = r.json()

        try:
            return page["query"]["pages"][0]["extract"]
        except:
            return None
