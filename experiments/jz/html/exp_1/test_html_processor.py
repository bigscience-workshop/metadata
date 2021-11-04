import unittest

from html_processor import AllTagsRules, HTMLParserConfig, HtmlProcessor, TagToRemove
from start_training import MetadataConfigWithHTML
from transformers import GPT2TokenizerFast

from bsmetadata.metadata_processors import PROCESSORS
from bsmetadata.metadata_utils import add_local_metadata_to_text, add_metadata_and_chunk_examples
from bsmetadata.metadata_processors import MetadataConfig


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
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_end_idx": 12,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "div",
                        "html_attrs": {"attrs": ["id", "class"], "values": ["mw-page-base", "noprint"]},
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "div",
                        "html_attrs": {"attrs": ["id", "class"], "values": ["mw-head-base", "noprint"]},
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "a",
                        "html_attrs": {"attrs": ["id"], "values": ["top"]},
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "div",
                        "html_attrs": {
                            "attrs": ["id", "class"],
                            "values": ["siteNotice centralNotice", "mw-body-content"],
                        },
                        "char_end_idx": 13,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "i",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_end_idx": 29,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 13,
                        "value": "h1",
                        "html_attrs": {
                            "attrs": ["id", "class", "lang"],
                            "values": ["firstHeading", "firstHeading", "en"],
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
                        "value": "h1",
                        "html_attrs": {"attrs": ["id"], "values": ["title"]},
                        "char_end_idx": 28,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 50,
                        "value": "span",
                        "html_attrs": {"attrs": ["id"], "values": ["3"]},
                        "char_end_idx": 71,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 29,
                        "value": "div",
                        "html_attrs": {
                            "attrs": ["class", "id", "href"],
                            "values": ["div-level-1 div-level-2", "1", "http"],
                        },
                        "char_end_idx": 72,
                        "key": "html",
                        "type": "local",
                    },
                    {
                        "char_start_idx": 0,
                        "value": "body",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_end_idx": 72,
                        "key": "html",
                        "type": "local",
                    },
                ],
            },
        ]

    def test_add_html_tags(self):
        cfg = MetadataConfigWithHTML(
            html_parser_config=HTMLParserConfig(
                all_tags_rules=AllTagsRules(attributes_to_keep=["class", "id", "href"])
            )
        )
        cfg.metadata_list = ["html"]
        PROCESSORS["html"] = HtmlProcessor

        text1, mask1 = add_local_metadata_to_text(self.examples[0], cfg)
        self.maxDiff = None
        target_text = "<a>useless text</a> <div id:siteNotice centralNotice class:mw-body-content><a id:top><div id:mw-head-base class:noprint><div id:mw-page-base class:noprint></div></div></a></div><h1 id:firstHeading class:firstHeading lang:en><i>The Walking Dead</i> (season 8)</h1>\n"

        self.assertEqual(text1, target_text)

    def test_add_html_tags_remove_tag(self):
        tags_to_remove_alone = [TagToRemove("span", txt_max_chr_len=5), TagToRemove("body")]

        cfg = MetadataConfigWithHTML(
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
            "<div class:div-level-1 div-level-2 id:1 href:http>blablabla\ntidi tidi2 <span id:3>this one keep his tag</span>\n</div>"
        )

        print(repr(text1))

        self.assertEqual(text1, target_text)

    def test_tmp(self):
        tags_to_remove_alone = [
            TagToRemove("body"),
            # TagToRemove("div", txt_max_chr_len=0),
            # TagToRemove("a", txt_max_chr_len=0),
        ]

        cfg = MetadataConfigWithHTML(
            html_parser_config=HTMLParserConfig(
                all_tags_rules=AllTagsRules(
                    attributes_to_keep=["class", "id"],
                    txt_max_chr_len=float("inf"),
                    txt_min_chr_len=float("inf"),
                    tags_exceptions_to_txt_max_min_chr_len=None,
                ),
                tags_to_remove_alone_tag_name=[tag_to_remove.tag for tag_to_remove in tags_to_remove_alone],
                tags_to_remove_alone_txt_max_chr_len=[
                    tag_to_remove.txt_max_chr_len for tag_to_remove in tags_to_remove_alone
                ],
                tags_to_remove_alone_txt_min_chr_len=[
                    tag_to_remove.txt_min_chr_len for tag_to_remove in tags_to_remove_alone
                ],
            ),
            add_local_metadata_special_tokens_in_prefix=True,
            local_metadata_special_tokens={"html": "HtmlOn"},
            treat_local_metadata_as_regular_text=True,
        )
        cfg.metadata_list = ["html"]
        PROCESSORS["html"] = HtmlProcessor

        import json
        import os

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sample.json")) as json_file:
            data = json.load(json_file)

        text1, mask1 = add_local_metadata_to_text(data, cfg)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "text_with_local.txt"), "w") as f:
            f.write(text1)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "mask_text_with_local.json"), "w") as f:
            f.write(json.dumps(mask1))

        print(text1)

        output = add_metadata_and_chunk_examples({key:[value] for key, value in data.items()}, tokenizer=self.tokenizer, cfg=cfg)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "ext_with_local_output.json"), "w") as f:
            output["tokens"] = [ self.tokenizer.convert_ids_to_tokens(input_ids) for input_ids in output["input_ids"]]
            f.write(json.dumps(output))

        # import pprint

        # pprint.pprint(mask1)
