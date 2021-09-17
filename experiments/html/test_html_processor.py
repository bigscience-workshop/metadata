import unittest

from html_processor import AllTagsRules, HTMLParserConfig, HtmlProcessor, TagToRemove
from start_training import DataConfigWithHTML
from transformers import GPT2TokenizerFast

from bsmetadata.metadata_processors import PROCESSORS
from bsmetadata.metadata_utils import add_local_metadata_to_text


class MetadataUtilsTester(unittest.TestCase):
    def setUp(self) -> None:
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        self.examples = [
            {
                "id": "0004",
                "text": "useless text The Walking Dead (season 8)\n",
                "metadata": [
                    {
                        "char_start_idx": 0,
                        "value": {"tag": "a", "attrs": {"attr": [], "value": []}},
                        "char_end_idx": 12,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {
                            "tag": "div",
                            "attrs": {"attr": ["id", "class"], "value": ["mw-page-base", "noprint"]},
                        },
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {
                            "tag": "div",
                            "attrs": {"attr": ["id", "class"], "value": ["mw-head-base", "noprint"]},
                        },
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {"tag": "a", "attrs": {"attr": ["id"], "value": ["top"]}},
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {
                            "tag": "div",
                            "attrs": {
                                "attr": ["id", "class"],
                                "value": ["siteNotice centralNotice", "mw-body-content"],
                            },
                        },
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {"tag": "i", "attrs": {"attr": [], "value": []}},
                        "char_end_idx": 29,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": {
                            "tag": "h1",
                            "attrs": {
                                "attr": ["id", "class", "lang"],
                                "value": ["firstHeading", "firstHeading", "en"],
                            },
                        },
                        "char_end_idx": 40,
                        "key": "html",
                        "type": "local",
                    },
                ],
            },
            {
                "id": "0004",
                "text": ("this is a title that we keep\n" "blablabla\n" "tidi tidi2 this one keep his tag\n"),
                "metadata": [
                    {
                        "char_start_idx": 0,
                        "value": {"tag": "h1", "attrs": {"attr": ["id"], "value": ["title"]}},
                        "char_end_idx": 28,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 50,
                        "value": {"tag": "span", "attrs": {"attr": ["id"], "value": ["3"]}},
                        "char_end_idx": 71,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 29,
                        "value": {
                            "tag": "div",
                            "attrs": {
                                "attr": ["class", "id", "href"],
                                "value": ["div-level-1 div-level-2", "1", "http"],
                            },
                        },
                        "char_end_idx": 72,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 0,
                        "value": {"tag": "body", "attrs": {"attr": [], "value": []}},
                        "char_end_idx": 72,
                        "key": "html",
                        "type": "local",
                    },
                ],
            },
        ]

    def test_add_html_tags(self):
        cfg = DataConfigWithHTML(
            html_parser_config=HTMLParserConfig(
                all_tags_rules=AllTagsRules(attributes_to_keep=["class", "id", "href"])
            )
        )
        cfg.metadata_list = ["html"]
        PROCESSORS["html"] = HtmlProcessor

        text1, mask1 = add_local_metadata_to_text(self.examples[0], cfg)
        target_text = '<a>useless text</a> <div id="siteNotice centralNotice" class="mw-body-content"><a id="top"><div id="mw-head-base" class="noprint"><div id="mw-page-base" class="noprint"></div></div></a></div><h1 id="firstHeading" class="firstHeading"><i>The Walking Dead</i> (season 8)</h1>\n'

        self.assertEqual(text1, target_text)

    def test_add_html_tags_remove_tag(self):
        tags_to_remove_alone = [TagToRemove("span", txt_max_chr_len=5), TagToRemove("body")]

        cfg = DataConfigWithHTML(
            html_parser_config=HTMLParserConfig(
                all_tags_rules=AllTagsRules(attributes_to_keep=["class", "id", "href"]),
                tags_to_remove_alone_tag_name=[tag_to_remove.tag for tag_to_remove in tags_to_remove_alone],
                tags_to_remove_alone_txt_max_chr_len=[
                    tag_to_remove.txt_max_chr_len for tag_to_remove in tags_to_remove_alone
                ],
                tags_to_remove_alone_txt_min_chr_len=[
                    tag_to_remove.txt_min_chr_len for tag_to_remove in tags_to_remove_alone
                ],
            )
        )
        cfg.metadata_list = ["html"]
        PROCESSORS["html"] = HtmlProcessor

        text1, mask1 = add_local_metadata_to_text(self.examples[1], cfg)
        target_text = (
            '<h1 id="title">this is a title that we keep</h1>\n'
            '<div class="div-level-1 div-level-2" id="1" href="http">blablabla\ntidi tidi2 <span id="3">this one keep his tag</span>\n</div>'
        )

        print(repr(text1))

        self.assertEqual(text1, target_text)
