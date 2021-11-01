from collections import defaultdict
from typing import Optional

from wikipedia2vec.dump_db import DumpDB


class WebsiteDescUtils:
    def __init__(self, path_wiki_db) -> None:
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
            text = self.wiki_dump_db.get_paragraphs(title)[0].text
            text = ". ".join(
                text.split(". ")[:2]
            )  # Picking the first two sentences from the text (Splitting on '. ' might not give the desired sentence for some corner cases but mostly works)
            if not text.endswith("."):
                text += "."
        except Exception:
            return None
        return text

    def extract_wiki_desc(self, keyword: str) -> Optional:

        title = self.fetch_wikipedia_title_from_keyword(keyword)
        desc = self.fetch_wikipedia_description_for_title(title)
        return desc

    def fetch_website_description_from_keyword(self, keyword: str) -> Optional:
        if not self.cache[keyword]:
            self.cache[keyword] = self.extract_wiki_desc(keyword)

        return self.cache[keyword]
