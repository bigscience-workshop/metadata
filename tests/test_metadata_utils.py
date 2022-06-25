import functools
import unittest

from datasets import Dataset
from transformers import GPT2TokenizerFast

from bsmetadata.metadata_processors import (
    PROCESSORS,
    AllTagsRules,
    EntityParagraphProcessor,
    EntityProcessor,
    HTMLParserConfig,
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
    create_metadata_prefix,
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
                        "html_attrs": {
                            "attrs": ["class", "id", "href"],
                            "values": ["level1", "4", "https://test.org"],
                        },
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
            {
                "id": "0007",
                "text": "Your article struck a cord, as a child I would imagine being an inventor. . As an adult I still love it when an insight to a new product or process reveals itself .",
                "metadata": [
                    {"key": "title", "type": "global", "value": "My Thoughts On It » Dad, I want to be an inventor"},
                ],
            },
            {
                "id": "6",  # To be used for entity_setting = "beg" and "end"
                "text": "Hints and tips for media appearances, speaking and social media. This week; wall-to-wall politicians; Great Britain: Louis Vuitton condoms; Billy Connolly,; Lisa Dutton; Something in Common; What was I saying?: We’re all publishers; An interview with Lembit Opik; Music from The Good Suns.",
                "metadata": [
                    {
                        "key": "entity_paragraph",
                        "type": "local",
                        "char_start_idx": 0,
                        "char_end_idx": 289,
                        "value": "United_Kingdom",
                        "relative_start_pos": 0,
                        "relative_end_pos": 0,
                    },
                    {
                        "key": "entity_paragraph",
                        "type": "local",
                        "char_start_idx": 0,
                        "char_end_idx": 289,
                        "value": "Louis_Vuitton",
                        "relative_start_pos": 1,
                        "relative_end_pos": 1,
                    },
                    {
                        "key": "entity_paragraph",
                        "type": "local",
                        "char_start_idx": 0,
                        "char_end_idx": 289,
                        "value": "Billy_Connolly",
                        "relative_start_pos": 2,
                        "relative_end_pos": 2,
                    },
                    {
                        "key": "entity_paragraph",
                        "type": "local",
                        "char_start_idx": 0,
                        "char_end_idx": 289,
                        "value": "Something_in_Common",
                        "relative_start_pos": 3,
                        "relative_end_pos": 3,
                    },
                    {
                        "key": "entity_paragraph",
                        "type": "local",
                        "char_start_idx": 0,
                        "char_end_idx": 289,
                        "value": "Lembit_Öpik",
                        "relative_start_pos": 4,
                        "relative_end_pos": 4,
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 103,
                        "char_end_idx": 115,
                        "value": "United_Kingdom",
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 118,
                        "char_end_idx": 130,
                        "value": "Louis_Vuitton",
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 141,
                        "char_end_idx": 154,
                        "value": "Billy_Connolly",
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 171,
                        "char_end_idx": 189,
                        "value": "Something_in_Common",
                    },
                    {
                        "key": "entity",
                        "type": "local",
                        "char_start_idx": 252,
                        "char_end_idx": 262,
                        "value": "Lembit_Öpik",
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

    def test_create_metadata_prefix_with_only_global_metadata(self):
        cfg = MetadataConfig()
        cfg.metadata_key_value_sep = ": "
        cfg.metadata_sep = " | "
        cfg.metadata_prefix_sep = " |||"
        cfg.metadata_list = ["url", "timestamp", "website_description", "title"]
        PROCESSORS["timestamp"] = MetadataProcessor

        global_metadata_prefix_1 = create_metadata_prefix(self.examples[0], cfg)
        global_metadata_prefix_2 = create_metadata_prefix(self.examples[1], cfg)
        global_metadata_prefix_3 = create_metadata_prefix(self.examples[2], cfg)
        global_metadata_prefix_4 = create_metadata_prefix(self.examples[3], cfg)
        global_metadata_prefix_5 = create_metadata_prefix(self.examples[6], cfg)

        self.assertEqual(
            global_metadata_prefix_1,
            "url: https://www.bbc.com/sport/live/olympics/50974152 | timestamp: 2018-12-10T13:45:00.000Z |||",
        )

        self.assertEqual(global_metadata_prefix_2, "url: https://en.wikipedia.org/wiki/Apple |||")

        self.assertEqual(global_metadata_prefix_3, "url: callto:RickAndMorty/Year 2021/ |||")

        self.assertEqual(
            global_metadata_prefix_4,
            "Website Description: Amazon.com, Inc. ( AM-ə-zon) is an American multinational conglomerate which focuses on e-commerce, cloud computing, digital streaming, and artificial intelligence. |||",
        )
        self.assertEqual(global_metadata_prefix_5, "title: My Thoughts On It » Dad, I want to be an inventor |||")

    def test_entity_settings(self):
        from transformers import AddedToken

        cfg = MetadataConfig()
        PROCESSORS["entity"] = EntityProcessor
        PROCESSORS["entity_paragraph"] = EntityParagraphProcessor
        cfg.metadata_list = ["entity", "entity_paragraph"]
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "
        cfg.local_metadata_special_tokens = {
            "entity": "EntityOn",
            "entity_paragraph": "EntityParagraphOn",
        }
        cfg.treat_local_metadata_as_regular_text = True
        cfg.max_seq_len = 142
        cfg.local_metadata_special_token_start = {"entity_paragraph": " <ENTITY_CHAIN>"}
        cfg.local_metadata_special_token_end = {"entity_paragraph": " </ENTITY_CHAIN>"}
        cfg.entity_setting = "end"
        text3, mask3 = add_local_metadata_to_text(self.examples[7], cfg)

        cfg.local_metadata_special_token_start = {"entity_paragraph": "<ENTITY_CHAIN>"}
        cfg.local_metadata_special_token_end = {"entity_paragraph": " </ENTITY_CHAIN> "}
        cfg.entity_setting = "beg"
        text4, mask4 = add_local_metadata_to_text(self.examples[7], cfg)

        text5, mask5 = add_local_metadata_to_text(self.examples[0], cfg)

        self.assertEqual(
            text3,
            "Hints and tips for media appearances, speaking and social media. This week; wall-to-wall politicians; Great Britain [[United Kingdom]]: Louis Vuitton [[Louis Vuitton]] condoms; Billy Connolly [[Billy Connolly]],; Lisa Dutton; Something in Common [[Something in Common]]; What was I saying?: We’re all publishers; An interview with Lembit Opik [[Lembit Öpik]]; Music from The Good Suns. <ENTITY_CHAIN> |United Kingdom| |Louis Vuitton| |Billy Connolly| |Something in Common| |Lembit Öpik| </ENTITY_CHAIN>",
        )

        self.assertEqual(
            text4,
            "<ENTITY_CHAIN> |United Kingdom| |Louis Vuitton| |Billy Connolly| |Something in Common| |Lembit Öpik| </ENTITY_CHAIN> Hints and tips for media appearances, speaking and social media. This week; wall-to-wall politicians; Great Britain [[United Kingdom]]: Louis Vuitton [[Louis Vuitton]] condoms; Billy Connolly [[Billy Connolly]],; Lisa Dutton; Something in Common [[Something in Common]]; What was I saying?: We’re all publishers; An interview with Lembit Opik [[Lembit Öpik]]; Music from The Good Suns.",
        )
        self.assertEqual(
            text5,
            "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is Yafai [[Galal Yafai]] has got to go him.",
        )

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        tokenizer.add_tokens(
            [AddedToken(special_token, lstrip=True) for special_token in cfg.local_metadata_special_tokens.values()]
        )

        ds_dict = {key: [self.examples[7][key]] for key in self.examples[0].keys()}
        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            tokenizer.decode(mapped_ds[0]["input_ids"]),
            "EntityOn |EntityParagraphOn ||| <ENTITY_CHAIN> |United Kingdom| |Louis Vuitton| |Billy Connolly| |Something in Common| |Lembit Öpik| </ENTITY_CHAIN> Hints and tips for media appearances, speaking and social media. This week; wall-to-wall politicians; Great Britain [[United Kingdom]]: Louis Vuitton [[Louis Vuitton]] condoms; Billy Connolly [[Billy Connolly]],; Lisa Dutton; Something in Common [[Something in Common]]; What was I saying?: We’re all publishers; An interview with Lembit Opik [[Lembit Öpik]]; Music from The Good Suns",
        )

    def test_add_local_metadata_to_text(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["html", "entity"]
        PROCESSORS["html"] = MetadataProcessor
        PROCESSORS["entity"] = MetadataProcessor
        text1, mask1 = add_local_metadata_to_text(self.examples[0], cfg)
        text2, mask2 = add_local_metadata_to_text(self.examples[1], cfg)

        self.assertEqual(
            text1,
            "It was a brilliant first round. You have to break down the Cuban's rhythm you can't let them get into rhythm. The risk with that is [entity: Galal Yafai]Yafai[/entity: Galal Yafai] has got to go him.",
        )
        self.assertEqual(
            "".join(str(int(x)) for x in mask1),
            "0000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001111111111111111111110000011111111111111111111110000000000000000000",
        )

        self.assertEqual(
            text2,
            "An [html: b][entity: Malus domestica]apple[/entity: Malus domestica][/html: b] is an edible fruit produced by an [html: b][html: i]apple[/html: i] tree[/html: b] (Malus domestica).",
        )
        self.assertEqual(
            "".join(str(int(x)) for x in mask2),
            "000111111111111111111111111111111111100000111111111111111111111111111111111111000000000000000000000000000000000001111111111111111110000011111111110000011111111110000000000000000000",
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
        cfg.metadata_list = ["url", "timestamp", "html", "entity", "website_description", "title"]
        cfg.max_seq_len = 64
        cfg.metadata_probability = 1

        PROCESSORS["timestamp"] = MetadataProcessor
        # :func:`~test_add_local_metadata_to_text` as of writing can cause inconsistent outcome if not reset.
        PROCESSORS["entity"] = MetadataProcessor
        PROCESSORS["html"] = MetadataProcessor

        ds_dict = {
            key: [self.examples[0][key], self.examples[1][key], self.examples[3][key], self.examples[6][key]]
            for key in self.examples[0].keys()
        }

        ds = Dataset.from_dict(ds_dict)

        mapped_ds = ds.map(
            functools.partial(add_metadata_and_chunk_examples, tokenizer=self.tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.assertEqual(len(mapped_ds), 6)

        # fmt: off
        pad_tkn = self.tokenizer.eos_token

        """
        example[0]
        ==========
        """
        expected_g_mtdt0 = [
            "url", ":", "Ġhttps", "://", "www", ".", "bb", "c",
            ".", "com", "/", "s", "port", "/", "live", "/",
            "oly", "mp", "ics", "/", "509", "74", "152", "Ġ|",
            "Ġtimestamp", ":", "Ġ2018", "-", "12", "-", "Ġ||", "|",
        ]  # 32 tokens

        expected_tkns00 = expected_g_mtdt0 + [
            "ĠIt", "Ġwas", "Ġa", "Ġbrilliant", "Ġfirst", "Ġround", ".", "ĠYou",
            "Ġhave", "Ġto", "Ġbreak", "Ġdown", "Ġthe", "ĠCuban", "'s", "Ġrhythm",
            "Ġyou", "Ġcan", "'t", "Ġlet", "Ġthem", "Ġget", "Ġinto", "Ġrhythm",
            ".", "ĠThe", "Ġrisk", "Ġwith", "Ġthat", "Ġis", "Ġ[", "entity",
        ]
        actual_tkns00 = self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"])
        self.assertEqual(actual_tkns00, expected_tkns00, actual_tkns00)
        self.assertEqual(mapped_ds[0]["attention_mask"], [1] * cfg.max_seq_len)
        self.assertEqual(mapped_ds[0]["metadata_mask"], [1] * 32 + [0] * (cfg.max_seq_len - 32 - 2) + [1] * 2)

        expected_tkns01 = expected_g_mtdt0 + [
            ":", "ĠGal", "al", "ĠY", "af", "ai", "]", "Y",
            "af", "ai", "[/", "entity", ":", "ĠGal", "al", "ĠY",
            "af", "ai", "]", "Ġhas", "Ġgot", "Ġto", "Ġgo", "Ġhim",
            ".",
        ]
        pad_len01 = cfg.max_seq_len - len(expected_tkns01)
        expected_tkns01 += [pad_tkn] * pad_len01
        actual_tkns01 = self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"])
        self.assertEqual(actual_tkns01, expected_tkns01, actual_tkns01)
        self.assertEqual(mapped_ds[1]["attention_mask"], [1] * (cfg.max_seq_len - pad_len01) + [0] * pad_len01)
        self.assertEqual(
            mapped_ds[1]["metadata_mask"],
            [1] * len(expected_g_mtdt0)
            + [1] * 7 + [0]
            + [0] * 2 + [1] * 6
            + [1] * 3 + [0] * 5
            + [0]
            + [0] * pad_len01
        )

        """
        example[1]
        ==========
        """
        expected_g_mtdt1 = [
            "url", ":", "Ġhttps", "://", "en", ".", "wikipedia", ".",
            "org", "/", "wiki", "/", "Apple", "Ġ||", "|",
        ]

        expected_tkns10 = expected_g_mtdt1 + [
            "ĠAn", "Ġ[", "html", ":", "Ġb", "][", "entity", ":",
            "ĠMal", "us", "Ġdomest", "ica", "]", "apple", "[/", "entity",
            ":", "ĠMal", "us", "Ġdomest", "ica", "][/", "html", ":",
            "Ġb", "]", "Ġis", "Ġan", "Ġedible", "Ġfruit", "Ġproduced", "Ġby",
            "Ġan", "Ġ[", "html", ":", "Ġb", "][", "html", ":",
            "Ġi", "]", "apple", "[/", "html", ":", "Ġi", "]",
            "Ġtree",
        ]
        actual_tkns10 = self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"])
        self.assertEqual(actual_tkns10, expected_tkns10, actual_tkns10)
        self.assertEqual(mapped_ds[2]["attention_mask"], [1] * cfg.max_seq_len)
        self.assertEqual(
            mapped_ds[2]["metadata_mask"],
            [1] * len(expected_g_mtdt1)
            + [0] + [1] * 7
            + [1] * 5 + [0] + [1] * 2
            + [1] * 8
            + [1] * 2 + [0] * 6
            + [0] + [1] * 7
            + [1] * 2 + [0] + [1] * 5
            + [0]
        )

        expected_tkns11 = expected_g_mtdt1 + [
            "[/", "html", ":", "Ġb", "]", "Ġ(", "Mal", "us",
            "Ġdomest", "ica", ").",
        ]
        pad_len11 = cfg.max_seq_len - len(expected_tkns11)
        expected_tkns11 += [pad_tkn] * pad_len11
        actual_tkns11 = self.tokenizer.convert_ids_to_tokens(mapped_ds[3]["input_ids"])
        self.assertEqual(actual_tkns11, expected_tkns11, actual_tkns11)
        self.assertEqual(mapped_ds[3]["attention_mask"], [1] * (cfg.max_seq_len - pad_len11) + [0] * pad_len11)
        self.assertEqual(
            mapped_ds[3]["metadata_mask"],
            [1] * len(expected_g_mtdt1)
            + [1] * 5 + [0] * 3
            + [0] * 3
            + [0] * pad_len11
        )

        """
        example[3]
        ==========
        """
        expected_g_mtdt3 = [
            "Website", "ĠDescription", ":", "ĠAmazon", ".", "com", ",", "ĠInc",
            ".", "Ġ(", "ĠAM", "-", "É", "Ļ", "-", "zon",
            ")", "Ġis", "Ġan", "ĠAmerican", "Ġmultinational", "Ġconglomerate", "Ġwhich", "Ġfocuses",
            "Ġon", "Ġe", "-", "commerce", ",", "Ġcloud", "Ġ||", "|",
        ]

        expected_tkns30 = expected_g_mtdt3 + [
            "ĠAmazon", ".", "com", ":", "ĠCustomer", "ĠReviews", ":", "ĠContracts",
            "Ġand", "Ġthe", "ĠLegal", "ĠEnvironment", "Ġfor", "ĠEngineers", "Ġand", "ĠArchitects",
            "Ċ", "Customer", "ĠReviews", "63",
        ]
        pad_len30 = cfg.max_seq_len - len(expected_tkns30)
        expected_tkns30 += [pad_tkn] * pad_len30
        actual_tkns30 = self.tokenizer.convert_ids_to_tokens(mapped_ds[4]["input_ids"])
        self.assertEqual(actual_tkns30, expected_tkns30, actual_tkns30)
        self.assertEqual(mapped_ds[4]["attention_mask"], [1] * (cfg.max_seq_len - pad_len30) + [0] * pad_len30)
        self.assertEqual(mapped_ds[4]["metadata_mask"], [1] * (cfg.max_seq_len // 2) + [0] * (cfg.max_seq_len // 2))

        """
        example[6]
        ==========
        """
        expected_g_mtdt6 = [
            "title", ":", "ĠMy", "ĠThoughts", "ĠOn", "ĠIt", "ĠÂ»", "ĠDad",
            ",", "ĠI", "Ġwant", "Ġto", "Ġbe", "Ġan", "Ġinventor", "Ġ||",
            "|",
        ]  # 17 tokens
        expected_tkns60 = expected_g_mtdt6 + [
            "ĠYour", "Ġarticle", "Ġstruck", "Ġa", "Ġcord", ",", "Ġas", "Ġa",
            "Ġchild", "ĠI", "Ġwould", "Ġimagine", "Ġbeing", "Ġan", "Ġinventor", ".",
            "Ġ.", "ĠAs", "Ġan", "Ġadult", "ĠI", "Ġstill", "Ġlove", "Ġit",
            "Ġwhen", "Ġan", "Ġinsight", "Ġto", "Ġa", "Ġnew", "Ġproduct", "Ġor",
            "Ġprocess", "Ġreveals", "Ġitself", "Ġ.",
        ]
        pad_len60 = cfg.max_seq_len - len(expected_tkns60)
        expected_tkns60 += [pad_tkn] * pad_len60
        actual_tkns60 = self.tokenizer.convert_ids_to_tokens(mapped_ds[5]["input_ids"])
        self.assertEqual(actual_tkns60, expected_tkns60, actual_tkns60)
        self.assertEqual(mapped_ds[5]["attention_mask"], [1] * (cfg.max_seq_len - pad_len60) + [0] * pad_len60)
        self.assertEqual(mapped_ds[5]["metadata_mask"], [1] * 17 + [0] * (cfg.max_seq_len - 17))
        # fmt: on

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
            "url: https://en.wikipedia.org/wiki/Apple ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
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
        cfg.max_seq_len = 84
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "

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
            f" url: https://en.wikipedia.org/wiki/Apple | html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).{'<|endoftext|>'*7}",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple | html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple | html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', 'Ġid', ':', '4', 'Ġhref', ':', 'https', '://', 'test', '.', 'org', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        # fmt: on

    def test_add_metadata_and_chunk_examples_with_true_processor_and_metadata_special_tokens_without_global(self):
        cfg = MetadataConfig()
        cfg.metadata_list = ["html", "entity"]
        cfg.max_seq_len = 69
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "

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
            " html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[1]["input_ids"]),
            " html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )
        self.assertEqual(
            self.tokenizer.decode(mapped_ds[2]["input_ids"]),
            " html | entity ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).",
        )

        # fmt: off
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', 'Ġid', ':', '4', 'Ġhref', ':', 'https', '://', 'test', '.', 'org', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)
        self.assertEqual(self.tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġhtml', 'Ġ|', 'Ġentity', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').'],)
        # fmt: on

    def test_add_metadata_and_chunk_examples_with_true_processor_and_metadata_special_tokens_specifying_special_token(
        self,
    ):
        from transformers import AddedToken

        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 84
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "
        cfg.local_metadata_special_tokens = {
            "html": "HtmlOn",
            "entity": "EntityOn",
        }

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        tokenizer.add_tokens(
            [AddedToken(special_token, lstrip=True) for special_token in cfg.local_metadata_special_tokens.values()]
        )

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
            functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            tokenizer.decode(mapped_ds[0]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

        # fmt: off
        self.assertEqual(mapped_ds[0]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"),)
        self.assertEqual(mapped_ds[1]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)
        self.assertEqual(mapped_ds[2]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)
        # fmt: on

        # fmt: off
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', 'Ġid', ':', '4', 'Ġhref', ':', 'https', '://', 'test', '.', 'org', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        # fmt: on

    def test_add_metadata_and_chunk_examples_with_true_processor_and_metadata_special_tokens_specifying_special_token_and_special_html_config(
        self,
    ):
        from transformers import AddedToken

        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 84
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "
        cfg.local_metadata_special_tokens = {
            "html": "HtmlOn",
            "entity": "EntityOn",
        }
        cfg.html_parser_config = HTMLParserConfig(
            AllTagsRules(
                attributes_to_keep=["class"],
                txt_max_chr_len=-float("inf"),
                txt_min_chr_len=-float("inf"),
                tags_exceptions_to_txt_max_min_chr_len=None,
            ),
            tags_to_remove_alone_tag_name=[],
            tags_to_remove_alone_txt_max_chr_len=[],
            tags_to_remove_alone_txt_min_chr_len=[],
        )

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        tokenizer.add_tokens(
            [AddedToken(special_token, lstrip=True) for special_token in cfg.local_metadata_special_tokens.values()]
        )

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
            functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            tokenizer.decode(mapped_ds[0]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )

        # fmt: off
        self.assertEqual(mapped_ds[0]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"),)
        self.assertEqual(mapped_ds[1]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)
        self.assertEqual(mapped_ds[2]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)
        # fmt: on

        # fmt: off
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
        # fmt: on

    def test_add_metadata_and_chunk_examples_treat_local_metadata_as_regular_text(self):
        from transformers import AddedToken

        cfg = MetadataConfig()
        cfg.metadata_list = ["url", "timestamp", "html", "entity"]
        cfg.max_seq_len = 84
        cfg.metadata_probability = 1
        cfg.add_local_metadata_special_tokens_in_prefix = True
        cfg.metadata_prefix_start_seq = " "
        cfg.local_metadata_special_tokens = {
            "html": "HtmlOn",
            "entity": "EntityOn",
        }
        cfg.treat_local_metadata_as_regular_text = True

        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2-xl")
        tokenizer.add_tokens(
            [AddedToken(special_token, lstrip=True) for special_token in cfg.local_metadata_special_tokens.values()]
        )

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
            functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg),
            batched=True,
            remove_columns=ds.column_names,
            load_from_cache_file=False,
        )

        self.maxDiff = None

        self.assertEqual(
            tokenizer.decode(mapped_ds[0]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[1]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        self.assertEqual(
            tokenizer.decode(mapped_ds[2]["input_ids"]),
            " url: https://en.wikipedia.org/wiki/Apple |HtmlOn |EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>",
        )
        # fmt: off
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[0]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'b', 'Ġclass', ':', 'level', '1', 'Ġid', ':', '4', 'Ġhref', ':', 'https', '://', 'test', '.', 'org', '><', 'i', 'Ġclass', ':', 'level', '2', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[1]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])
        self.assertEqual(tokenizer.convert_ids_to_tokens(mapped_ds[2]["input_ids"]), ['Ġurl', ':', 'Ġhttps', '://', 'en', '.', 'wikipedia', '.', 'org', '/', 'wiki', '/', 'Apple', 'Ġ|', 'HtmlOn', 'Ġ|', 'EntityOn', 'Ġ||', '|', 'ĠAn', 'Ġ<', 'b', '>', 'apple', 'Ġ[[', 'Mal', 'us', 'Ġdomest', 'ica', ']]', '</', 'b', '>', 'Ġis', 'Ġan', 'Ġedible', 'Ġfruit', 'Ġproduced', 'Ġby', 'Ġan', 'Ġ<', 'a', '></', 'a', '><', 'b', 'Ġclass', ':', 'level', '1', '><', 'b', 'Ġclass', ':', 'level', '2', '><', 'i', 'Ġclass', ':', 'level', '3', '><', 'a', '></', 'a', '>', 'apple', '</', 'i', '>', 'Ġtree', '</', 'b', '></', 'b', '>', 'Ġ(', 'Mal', 'us', 'Ġdomest', 'ica', ').', '<|endoftext|>'])

        self.assertEqual(mapped_ds[0]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <b class:level1 id:4 href:https://test.org><i class:level2>apple</i> tree</b> (Malus domestica).<|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|><|endoftext|>"),)
        self.assertEqual(mapped_ds[1]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)
        self.assertEqual(mapped_ds[2]["input_ids"], tokenizer.encode(" url: https://en.wikipedia.org/wiki/Apple | HtmlOn | EntityOn ||| An <b>apple [[Malus domestica]]</b> is an edible fruit produced by an <a></a><b class:level1><b class:level2><i class:level3><a></a>apple</i> tree</b></b> (Malus domestica).<|endoftext|>"),)

        self.assertEqual(mapped_ds[0]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[1]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        self.assertEqual(mapped_ds[2]["metadata_mask"], [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        # fmt: on


if __name__ == "__main__":
    unittest.main()
