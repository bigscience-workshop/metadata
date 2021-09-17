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
                "text": "useless text\nThe Walking Dead (season 8)\n",
                "metadata": [
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 1,
                        "char_end_idx": 12,
                        "relative_end_pos": 0,
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 3,
                        "char_end_idx": 13,
                        "relative_end_pos": 4,
                        "value": "div",
                        "html_attrs": {"attrs": ["id", "class"], "values": ["mw-page-base", "noprint"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 2,
                        "char_end_idx": 13,
                        "relative_end_pos": 5,
                        "value": "div",
                        "html_attrs": {"attrs": ["id", "class"], "values": ["mw-head-base", "noprint"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 1,
                        "char_end_idx": 13,
                        "relative_end_pos": 6,
                        "value": "a",
                        "html_attrs": {"attrs": ["id"], "values": ["top"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 0,
                        "char_end_idx": 13,
                        "relative_end_pos": 7,
                        "value": "div",
                        "html_attrs": {
                            "attrs": ["id", "class"],
                            "values": ["siteNotice centralNotice", "mw-body-content"],
                        },
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 9,
                        "char_end_idx": 29,
                        "relative_end_pos": 0,
                        "value": "i",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 13,
                        "relative_start_pos": 8,
                        "char_end_idx": 40,
                        "relative_end_pos": 0,
                        "value": "h1",
                        "html_attrs": {"attrs": ["id", "class"], "values": ["firstHeading", "firstHeading"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 0,
                        "char_end_idx": 41,
                        "relative_end_pos": 0,
                        "value": "body",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                ],
            },
            {
                "id": "0004",
                "text": ("this is a title that we keep\nblablabla tidi tidi2 this one keep his tag\n"),
                "metadata": [
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 1,
                        "char_end_idx": 28,
                        "relative_end_pos": 0,
                        "value": "h1",
                        "html_attrs": {"attrs": ["id"], "values": ["title"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 50,
                        "relative_start_pos": 0,
                        "char_end_idx": 71,
                        "relative_end_pos": 0,
                        "value": "span",
                        "html_attrs": {"attrs": ["id"], "values": ["3"]},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 29,
                        "relative_start_pos": 0,
                        "char_end_idx": 72,
                        "relative_end_pos": 0,
                        "value": "div",
                        "html_attrs": {
                            "attrs": ["class", "id", "href"],
                            "values": ["div-level-1 div-level-2", "1", "http"],
                        },
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 0,
                        "char_end_idx": 72,
                        "relative_end_pos": 1,
                        "value": "body",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                ],
            },
            {
                "id": "0005",
                "text": "event_id\nyear\nmonth\nidghtu 1998 may\n",
                "metadata": [
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 3,
                        "char_end_idx": 8,
                        "relative_end_pos": 0,
                        "value": "th",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 9,
                        "relative_start_pos": 0,
                        "char_end_idx": 13,
                        "relative_end_pos": 0,
                        "value": "th",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 14,
                        "relative_start_pos": 0,
                        "char_end_idx": 19,
                        "relative_end_pos": 0,
                        "value": "th",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 2,
                        "char_end_idx": 20,
                        "relative_end_pos": 0,
                        "value": "tr",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 20,
                        "relative_start_pos": 2,
                        "char_end_idx": 26,
                        "relative_end_pos": 0,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 27,
                        "relative_start_pos": 0,
                        "char_end_idx": 31,
                        "relative_end_pos": 0,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 32,
                        "relative_start_pos": 0,
                        "char_end_idx": 35,
                        "relative_end_pos": 0,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 20,
                        "relative_start_pos": 1,
                        "char_end_idx": 36,
                        "relative_end_pos": 0,
                        "value": "tr",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 36,
                        "relative_start_pos": 2,
                        "char_end_idx": 36,
                        "relative_end_pos": 3,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 36,
                        "relative_start_pos": 4,
                        "char_end_idx": 36,
                        "relative_end_pos": 5,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 36,
                        "relative_start_pos": 6,
                        "char_end_idx": 36,
                        "relative_end_pos": 7,
                        "value": "td",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 36,
                        "relative_start_pos": 1,
                        "char_end_idx": 36,
                        "relative_end_pos": 8,
                        "value": "tr",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 1,
                        "char_end_idx": 36,
                        "relative_end_pos": 9,
                        "value": "table",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                    {
                        "key": "html",
                        "type": "local",
                        "char_start_idx": 0,
                        "relative_start_pos": 0,
                        "char_end_idx": 36,
                        "relative_end_pos": 10,
                        "value": "body",
                        "html_attrs": {"attrs": [], "values": []},
                    },
                ],
            },
        ]

    def test_add_html_tags(self):
        tags_to_remove_alone = [TagToRemove("body")]
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

        text1, mask1 = add_local_metadata_to_text(self.examples[0], cfg)
        target_text = "<a>useless text</a>\n<div id:siteNotice centralNotice class:mw-body-content><a id:top><div id:mw-head-base class:noprint><div id:mw-page-base class:noprint></div></div></a></div><h1 id:firstHeading class:firstHeading><i>The Walking Dead</i> (season 8)</h1>\n"

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
            "<h1 id:title>this is a title that we keep</h1>\n"
            "<div class:div-level-1 div-level-2 id:1 href:http>blablabla tidi tidi2 <span id:3>this one keep his tag</span>\n</div>"
        )

        self.assertEqual(text1, target_text)

    def test_add_html_table(self):
        tags_to_remove_alone = [TagToRemove("body")]
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

        text1, mask1 = add_local_metadata_to_text(self.examples[2], cfg)
        target_text = (
            "<table>"
            "<tr>"
            "<th>event_id</th>\n<th>year</th>\n<th>month</th>\n"
            "</tr>"
            "<tr>"
            "<td>idghtu</td> <td>1998</td> <td>may</td>\n"
            "</tr>"
            "<tr>"
            "<td></td><td></td><td></td>"
            "</tr>"
            "</table>"
        )

        self.assertEqual(text1, target_text)
