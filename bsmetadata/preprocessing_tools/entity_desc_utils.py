import re
import string
from collections import defaultdict
from typing import Optional

import nltk
from wikipedia2vec.dump_db import DumpDB


class EntityDescUtils:
    def __init__(self, path_wiki_db) -> None:
        self.cache = defaultdict(str)
        self.wiki_dump_db = DumpDB(path_wiki_db)
        self.redirects_map = {
            key.lower(): value for key, value in self.wiki_dump_db.redirects()
        }  # loading all redirect information: takes ~10s
        nltk.download("punkt")

    def fetch_entity_description_from_keyword(self, keyword: str) -> str:
        try:
            key = string.capwords(keyword)
            text = self.wiki_dump_db.get_paragraphs(key)[0].text
            text = re.sub(r"\((?:[^)(]|\([^)(]*\))*\)", "", text)
            text = nltk.sent_tokenize(text)[0]
        except:
            try:
                text = self.wiki_dump_db.get_paragraphs(self.redirects_map[keyword])[0].text
                text = nltk.tokenize.sent_tokenize(text)[0]
            except:
                text = ""
        return text
