import re
import string
from collections import defaultdict
from typing import Optional

import nltk


try:
    from wikipedia2vec.dump_db import DumpDB

    wikipedia2vec_available = True
except ImportError:
    wikipedia2vec_available = False


class WikipediaDescUtils:
    def __init__(self, path_wiki_db) -> None:
        if not wikipedia2vec_available:
            raise ImportError(
                "Please install wikipedia2vec to use this feature. "
                "You can do so by running `pip install -e .'website_description'`."
            )
        self.cache = defaultdict(str)
        self.wiki_dump_db = DumpDB(path_wiki_db)
        self.redirects_map = {
            key.lower(): value for key, value in self.wiki_dump_db.redirects()
        }  # loading all redirect information: takes ~10s

    def fetch_wikipedia_title_from_keyword(self, keyword: str) -> str:
        title = self.redirects_map.get(
            keyword, keyword.split(".")[0].capitalize()
        )  # fallback to default for cases where domain is not recognized. We'll try to hit the db with the exact keyword directly (e.g. rightmove.com -> Rightmove) Capitalizing since wikipedia titles are so
        return title

    def fetch_wikipedia_description_for_title(self, title: str) -> Optional:
        try:
            paragraphs = self.wiki_dump_db.get_paragraphs(title)
        except KeyError:
            # If the title does not have a corresponding paragraph
            return None
        if len(paragraphs) == 0:
            # If there is no corresponding paragraphs
            return None

        text = paragraphs[0].text
        text = re.sub(r"\((?:[^)(]|\([^)(]*\))*\)", "", text)
        text = nltk.sent_tokenize(text)[0]  # Picking the first sentence
        return text

    def extract_wiki_desc(self, keyword: str) -> Optional:
        title = self.fetch_wikipedia_title_from_keyword(keyword)
        desc = self.fetch_wikipedia_description_for_title(title)
        return desc

    def fetch_website_description_from_keyword(self, keyword: str) -> Optional:
        if not self.cache[keyword]:
            self.cache[keyword] = self.extract_wiki_desc(keyword)

        return self.cache[keyword]

    def fetch_entity_description_from_keyword(self, keyword: str) -> str:
        title = string.capwords(keyword)
        text = self.fetch_wikipedia_description_for_title(title)

        if text is None and keyword in self.redirects_map:
            title = self.redirects_map[keyword]
            text = self.fetch_wikipedia_description_for_title(title)

        if text is None:
            text = ""

        return text
