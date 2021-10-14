import functools
import unittest

from datasets import Dataset
from transformers import GPT2TokenizerFast

from bsmetadata.metadata_processors import (
    PROCESSORS,
    EntityProcessor,
    HtmlProcessor,
    MetadataProcessor,
    TimestampProcessor,
    UrlProcessor,
)
from bsmetadata.metadata_utils import (
    MetadataConfig,
    add_local_metadata_to_text,
    add_metadata_and_chunk_examples,
    chunks,
    create_global_metadata_prefix,
)


class MetadataUtilsTester(unittest.TestCase):
    def setUp(self) -> None:
        self.maxDiff = None
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        self.examples = [
            {
                "id": "0001",
                "text": "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is Yafai has got to go him.",
                "metadata": [
                    {"key": "url", "type": "global", "value": "https://www.bbc.com/sport/live/olympics/50974152"},
                    {"key": "timestamp", "type": "global", "value": "2018-12-10T13:45:00.000Z"},
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 132,
                        "char_end_idx": 137,
                        "value": "Galal Yafai",
                    },
                ],
            },
            {
                "id": "0002",
                "text": "An apple is an edible fruit produced by an apple tree (Malus domestica).",
                "metadata": [
                    {"key": "url", "type": "global", "value": "https://en.wikipedia.org/wiki/Apple"},
                    {
                        "key": "html",
                        "relative_start_pos": 0,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "value": "Malus domestica",
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 1,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": ["class"], "values": ["level1"]},
                        "char_start_idx": 43,
                        "char_end_idx": 53,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 2,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "i",
                        "html_attrs": {"attrs": ["class"], "values": ["level2"]},
                        "char_start_idx": 43,
                        "char_end_idx": 48,
                    },
                ],
            },
            {
                "id": "0003",
                "text": "Wubba Lubba Dub Dub!",
                "metadata": [
                    {"key": "url", "type": "global", "value": "callto:RickAndMorty/Year%202021/"},
                ],
            },
            {
                "id": "0004",
                "text": "Amazon.com: Customer Reviews: Contracts and the Legal Environment for Engineers and Architects\nCustomer Reviews63",
                "metadata": [
                    {
                        "key": "website_description",
                        "type": "global",
                        "value": "Amazon.com, Inc. ( AM-ə-zon) is an American multinational conglomerate which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence.",
                    },
                ],
            },
            {
                "id": "0002",
                "text": "An apple is an edible fruit produced by an apple tree (Malus domestica).",
                "metadata": [
                    {"key": "url", "type": "global", "value": "https://en.wikipedia.org/wiki/Apple"},
                    {
                        "key": "html",
                        "relative_start_pos": 0,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "value": "Malus domestica",
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 0,
                        "relative_end_pos": 1,
                        "type": "local",
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 43,
                        "char_end_idx": 43,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 6,
                        "relative_end_pos": 7,
                        "type": "local",
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 43,
                        "char_end_idx": 43,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 4,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": ["class"], "values": ["level2"]},
                        "char_start_idx": 43,
                        "char_end_idx": 53,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 3,
                        "relative_end_pos": 1,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": ["class"], "values": ["level1"]},
                        "char_start_idx": 43,
                        "char_end_idx": 53,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 5,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "i",
                        "html_attrs": {"attrs": ["class"], "values": ["level3"]},
                        "char_start_idx": 43,
                        "char_end_idx": 48,
                    },
                ],
            },
            {
                "id": "0002",
                "text": "An apple is an edible fruit produced by an apple tree (Malus domestica).",
                "metadata": [
                    {"key": "url", "type": "global", "value": "https://en.wikipedia.org/wiki/Apple"},
                    {
                        "key": "html",
                        "relative_start_pos": 0,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 0,
                        "relative_end_pos": 1,
                        "type": "local",
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 43,
                        "char_end_idx": 43,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 6,
                        "relative_end_pos": 7,
                        "type": "local",
                        "value": "a",
                        "html_attrs": {"attrs": [], "values": []},
                        "char_start_idx": 43,
                        "char_end_idx": 43,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 4,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": ["class"], "values": ["level2"]},
                        "char_start_idx": 43,
                        "char_end_idx": 53,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 3,
                        "relative_end_pos": 1,
                        "type": "local",
                        "value": "b",
                        "html_attrs": {"attrs": ["class"], "values": ["level1"]},
                        "char_start_idx": 43,
                        "char_end_idx": 53,
                    },
                    {
                        "key": "html",
                        "relative_start_pos": 5,
                        "relative_end_pos": 0,
                        "type": "local",
                        "value": "i",
                        "html_attrs": {"attrs": ["class"], "values": ["level3"]},
                        "char_start_idx": 43,
                        "char_end_idx": 48,
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "value": "Malus domestica",
                        "char_start_idx": 3,
                        "char_end_idx": 8,
                    },
                ],
            },
        ]

    def test_chunks(self):
        list1 = ["a", "b", "c", "d", "e", "f", "g"]
        list2 = [0, 1, 2, 3, 4, 5, 6]
        self.assertEqual(list([x for x, *_ in chunks(1, list1)]), [["a"], ["b"], ["c"], ["d"], ["e"], ["f"], ["g"]])
        self.assertEqual(list([x for x, *_ in chunks(len(list1), list1)]), [list1])
        self.assertEqual(list([x for x, *_ in chunks(3, list1)]), [["a", "b", "c"], ["d", "e", "f"], ["g"]])
        self.assertEqual(list([x for x, *_ in chunks(3, list1)]), [["a", "b", "c"], ["d", "e", "f"], ["g"]])
        self.assertEqual(list([x for x, _ in chunks(3, list1, list2)]), [["a", "b", "c"], ["d", "e", "f"], ["g"]])
        self.assertEqual(list([x for _, x in chunks(3, list1, list2)]), [[0, 1, 2], [3, 4, 5], [6]])

    def test_create_global_metadata_prefix(self):
        cfg = MetadataConfig()
        cfg.metadata_key_value_sep = ": "
        cfg.metadata_sep = " | "
        cfg.metadata_prefix_sep = " |||"
        cfg.metadata_list = ["url", "timestamp", "website_description"]
        PROCESSORS["timestamp"] = MetadataProcessor

        global_metadata_prefix_1, global_metadata_special_tokens_prefix_1 = create_global_metadata_prefix(
            self.examples[0], cfg
        )
        global_metadata_prefix_2, global_metadata_special_tokens_prefix_2 = create_global_metadata_prefix(
            self.examples[1], cfg
        )
        global_metadata_prefix_3, global_metadata_special_tokens_prefix_3 = create_global_metadata_prefix(
            self.examples[2], cfg
        )
        global_metadata_prefix_4, global_metadata_special_tokens_prefix_4 = create_global_metadata_prefix(
            self.examples[3], cfg
        )

        self.assertEqual(
            global_metadata_prefix_1,
            "url: https://www.bbc.com/sport/live/olympics/50974152 | timestamp: 2018-12-10T13:45:00.000Z",
        )
        self.assertEqual(
            global_metadata_special_tokens_prefix_1,
            "url timestamp",
        )

        self.assertEqual(global_metadata_prefix_2, "url: https://en.wikipedia.org/wiki/Apple")
        self.assertEqual(
            global_metadata_special_tokens_prefix_2,
            "url",
        )

        self.assertEqual(global_metadata_prefix_3, "url: callto:RickAndMorty/Year 2021/")
        self.assertEqual(
            global_metadata_special_tokens_prefix_3,
            "url",
        )

        self.assertEqual(
            global_metadata_prefix_4,
            "Website Description: Amazon.com, Inc. ( AM-ə-zon) is an American multinational conglomerate which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence.",
        )
        self.assertEqual(
            global_metadata_special_tokens_prefix_4,
            "website_description",
        )

    def test_add_local_metadata_to_text(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["html", "entity"]
        PROCESSORS["html"] = MetadataProcessor
        PROCESSORS["entity"] = MetadataProcessor
        text1, mask1, local_metadata_special_tokens_prefix_1 = add_local_metadata_to_text(self.examples[0], cfg)
        text2, mask2, local_metadata_special_tokens_prefix_2 = add_local_metadata_to_text(self.examples[1], cfg)

        self.assertEqual(
            text1,
            "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is [entity: Galal Yafai]Yafai[/entity: Galal Yafai] has got to go him.",
        )
        self.assertEqual(
            "".join(str(int(x)) for x in mask1),
            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111111111111111111110000011111111111111111111110000000000000000000",
        )
        self.assertEqual(
            local_metadata_special_tokens_prefix_1,
            "entity",
        )

        self.assertEqual(
            text2,
            "An [html: b][entity: Malus domestica]apple[/entity: Malus domestica][/html: b] is an edible fruit produced by an [html: b][html: i]apple[/html: i] tree[/html: b] (Malus domestica).",
        )
        self.assertEqual(
            "".join(str(int(x)) for x in mask2),
            "000111111111111111111111111111111111100000111111111111111111111111111111111111000000000000000000000000000000000001111111111111111110000011111111110000011111111110000000000000000000",
        )
        self.assertEqual(
            local_metadata_special_tokens_prefix_2,
            "html entity",
        )

    def test_add_no_metadata_and_chunk_examples(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 64
        cfg.metadata_probability = 0

        ds_dict = {
            key: [self.examples[0][key], self.examples[1][key], self.examples[2][key], self.examples[3][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        for example in mapped_ds:
            self.assertTrue(all(not x for x in example["metadata_mask"]))

    def test_add_metadata_and_chunk_examples(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity", "website_description"]
        cfg.max_seq_len = 64
        cfg.metadata_probability = 1

        PROCESSORS["timestamp"] = MetadataProcessor
        ds_dict = {
            key: [self.examples[0][key], self.examples[1][key], self.examples[3][key]]
            for key in self.examples[0].keys()
        }

        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]),
            [
                "url",
                ":",
                "Ġhttps",
                "://",
                "www",
                ".",
                "bb",
                "c",
                ".",
                "com",
                "/",
                "s",
                "port",
                "/",
                "live",
                "/",
                "oly",
                "mp",
                "ics",
                "/",
                "509",
                "74",
                "152",
                "Ġ|",
                "Ġtimestamp",
                ":",
                "Ġ2018",
                "-",
                "12",
                "-",
                "10",
                "T",
                "13",
                ":",
                "45",
                ":",
                "00",
                ".",
                "000",
                "Z",
                "Ġ||",
                "|",
                "ĠIt",
                "Ġwas",
                "Ġa",
                "Ġbrilliant",
                "Ġfirst",
                "Ġround",
                ".",
                "ĠYou",
                "Ġhave",
                "Ġto",
                "Ġbreak",
                "Ġdown",
                "Ġthe",
                "ĠCuban",
                "'s",
                "Ġrhythm",
                "Ġyou",
                "Ġcan",
                "'t",
                "Ġlet",
                "Ġthem",
                "Ġget",
            ],
        )
        self.assertEqual(
            mapped_ds[0]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        )
        self.assertEqual(
            mapped_ds[0]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )

        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]),
            [
                "url",
                ":",
                "Ġhttps",
                "://",
                "www",
                ".",
                "bb",
                "c",
                ".",
                "com",
                "/",
                "s",
                "port",
                "/",
                "live",
                "/",
                "oly",
                "mp",
                "ics",
                "/",
                "509",
                "74",
                "152",
                "Ġ|",
                "Ġtimestamp",
                ":",
                "Ġ2018",
                "-",
                "12",
                "-",
                "10",
                "T",
                "13",
                ":",
                "45",
                ":",
                "00",
                ".",
                "000",
                "Z",
                "Ġ||",
                "|",
                "Ġinto",
                "Ġrhythm",
                ".",
                "ĠThe",
                "Ġrisk",
                "Ġwith",
                "Ġthat",
                "Ġis",
                "Ġ[",
                "entity",
                ":",
                "ĠGal",
                "al",
                "ĠY",
                "af",
                "ai",
                "]",
                "Y",
                "af",
                "ai",
                "[/",
                "entity",
            ],
        )
        self.assertEqual(
            mapped_ds[1]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        )
        self.assertEqual(
            mapped_ds[1]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
            ],
        )

        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]),
            [
                "url",
                ":",
                "Ġhttps",
                "://",
                "www",
                ".",
                "bb",
                "c",
                ".",
                "com",
                "/",
                "s",
                "port",
                "/",
                "live",
                "/",
                "oly",
                "mp",
                "ics",
                "/",
                "509",
                "74",
                "152",
                "Ġ|",
                "Ġtimestamp",
                ":",
                "Ġ2018",
                "-",
                "12",
                "-",
                "10",
                "T",
                "13",
                ":",
                "45",
                ":",
                "00",
                ".",
                "000",
                "Z",
                "Ġ||",
                "|",
                ":",
                "ĠGal",
                "al",
                "ĠY",
                "af",
                "ai",
                "]",
                "Ġhas",
                "Ġgot",
                "Ġto",
                "Ġgo",
                "Ġhim",
                ".",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
            ],
        )
        self.assertEqual(
            mapped_ds[2]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )
        self.assertEqual(
            mapped_ds[2]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )

        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[3]["input_ids"]),
            [
                "url",
                ":",
                "Ġhttps",
                "://",
                "en",
                ".",
                "wikipedia",
                ".",
                "org",
                "/",
                "wiki",
                "/",
                "Apple",
                "Ġ||",
                "|",
                "ĠAn",
                "Ġ[",
                "html",
                ":",
                "Ġb",
                "][",
                "entity",
                ":",
                "ĠMal",
                "us",
                "Ġdomest",
                "ica",
                "]",
                "apple",
                "[/",
                "entity",
                ":",
                "ĠMal",
                "us",
                "Ġdomest",
                "ica",
                "][/",
                "html",
                ":",
                "Ġb",
                "]",
                "Ġis",
                "Ġan",
                "Ġedible",
                "Ġfruit",
                "Ġproduced",
                "Ġby",
                "Ġan",
                "Ġ[",
                "html",
                ":",
                "Ġb",
                "][",
                "html",
                ":",
                "Ġi",
                "]",
                "apple",
                "[/",
                "html",
                ":",
                "Ġi",
                "]",
                "Ġtree",
            ],
        )
        self.assertEqual(
            mapped_ds[3]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
            ],
        )
        self.assertEqual(
            mapped_ds[3]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                1,
                1,
                1,
                0,
            ],
        )

        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[4]["input_ids"]),
            [
                "url",
                ":",
                "Ġhttps",
                "://",
                "en",
                ".",
                "wikipedia",
                ".",
                "org",
                "/",
                "wiki",
                "/",
                "Apple",
                "Ġ||",
                "|",
                "[/",
                "html",
                ":",
                "Ġb",
                "]",
                "Ġ(",
                "Mal",
                "us",
                "Ġdomest",
                "ica",
                ").",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
            ],
        )
        self.assertEqual(
            mapped_ds[4]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )
        self.assertEqual(
            mapped_ds[4]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )
        self.assertEqual(
            self.tokenizer.convert_ids_to_tokens(mapped_ds[5]["input_ids"]),
            [
                "Website",
                "ĠDescription",
                ":",
                "ĠAmazon",
                ".",
                "com",
                ",",
                "ĠInc",
                ".",
                "Ġ(",
                "ĠAM",
                "-",
                "É",
                "Ļ",
                "-",
                "zon",
                ")",
                "Ġis",
                "Ġan",
                "ĠAmerican",
                "Ġmultinational",
                "Ġconglomerate",
                "Ġwhich",
                "Ġfocuses",
                "Ġon",
                "Ġe",
                "-",
                "commerce",
                ",",
                "Ġcloud",
                "Ġcomputing",
                ",",
                "Ġdigital",
                "Ġstreaming",
                ",",
                "Ġand",
                "Ġartificial",
                "Ġintelligence",
                ".",
                "Ġ||",
                "|",
                "ĠAmazon",
                ".",
                "com",
                ":",
                "ĠCustomer",
                "ĠReviews",
                ":",
                "ĠContracts",
                "Ġand",
                "Ġthe",
                "ĠLegal",
                "ĠEnvironment",
                "Ġfor",
                "ĠEngineers",
                "Ġand",
                "ĠArchitects",
                "Ċ",
                "Customer",
                "ĠReviews",
                "63",
                "<|endoftext|>",
                "<|endoftext|>",
                "<|endoftext|>",
            ],
        )

        self.assertEqual(
            mapped_ds[5]["attention_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
            ],
        )

        self.assertEqual(
            mapped_ds[5]["metadata_mask"],
            [
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ],
        )

    def test_add_metadata_and_chunk_examples_with_true_processor(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 80
        cfg.metadata_probability = 1

        PROCESSORS["url"] = UrlProcessor
        PROCESSORS["timestamp"] = TimestampProcessor
        PROCESSORS["html"] = HtmlProcessor
        PROCESSORS["entity"] = EntityProcessor

        ds_dict = {
            key: [self.examples[1][key], self.examples[4][key], self.examples[4][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            self.tokenizer.decode(mapped_ds[0]["input_ids"]),
            "url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            "url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            "url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

    def test_add_metadata_and_chunk_examples_with_true_processor_and_metadata_special_tokens(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 85
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_special_token_for_generation_start_seq = " "
        cfg.metadata_global_start_seq = " "

        PROCESSORS["url"] = UrlProcessor
        PROCESSORS["timestamp"] = TimestampProcessor
        PROCESSORS["html"] = HtmlProcessor
        PROCESSORS["entity"] = EntityProcessor

        ds_dict = {
            key: [self.examples[1][key], self.examples[4][key], self.examples[4][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            self.tokenizer.decode(mapped_ds[0]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        # fmt: on

    def test_add_metadata_and_chunk_examples_with_true_processor_and_metadata_special_tokens_without_global(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["html", "entity"]
        cfg.max_seq_len = 68
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_special_token_for_generation_start_seq = " "
        cfg.metadata_global_start_seq = " "

        PROCESSORS["url"] = UrlProcessor
        PROCESSORS["timestamp"] = TimestampProcessor
        PROCESSORS["html"] = HtmlProcessor
        PROCESSORS["entity"] = EntityProcessor

        ds_dict = {
            key: [self.examples[1][key], self.examples[4][key], self.examples[4][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            self.tokenizer.decode(mapped_ds[0]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'],)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0])
        # fmt: on

    def test_add_metadata_and_chunk_examples_treat_local_metadata_as_regular_text_without_global(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["html", "entity"]
        cfg.max_seq_len = 68
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_special_token_for_generation_start_seq = " "
        cfg.metadata_global_start_seq = " "
        cfg.treat_local_metadata_as_regular_text = True

        PROCESSORS["url"] = UrlProcessor
        PROCESSORS["timestamp"] = TimestampProcessor
        PROCESSORS["html"] = HtmlProcessor
        PROCESSORS["entity"] = EntityProcessor

        ds_dict = {
            key: [self.examples[1][key], self.examples[4][key], self.examples[4][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            self.tokenizer.decode(mapped_ds[0]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " html entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'],)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġhtml', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # fmt: on

    def test_add_metadata_and_chunk_examples_treat_local_metadata_as_regular_text_with_global(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 85
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_special_token_for_generation_start_seq = " "
        cfg.metadata_global_start_seq = " "
        cfg.treat_local_metadata_as_regular_text = True

        PROCESSORS["url"] = UrlProcessor
        PROCESSORS["timestamp"] = TimestampProcessor
        PROCESSORS["html"] = HtmlProcessor
        PROCESSORS["entity"] = EntityProcessor

        ds_dict = {
            key: [self.examples[1][key], self.examples[4][key], self.examples[4][key]]
            for key in self.examples[0].keys()
        }
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            self.tokenizer.decode(mapped_ds[0]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url html entity ||| url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', 'Ġhtml', 'Ġentity', 'Ġ||', '|', 'Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # fmt: on


if __name__ == "__main__":
    unittest.main()
