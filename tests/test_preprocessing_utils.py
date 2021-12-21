import unittest
from unittest import mock

from datasets import Dataset, Features, Value
from mocks.mock_dump_db import MockDumpDB

from bsmetadata.preprocessing_tools.wikipedia_desc_utils import WikipediaDescUtils
from bsmetadata.preprocessing_utils import (
    EntityPreprocessor,
    HtmlPreprocessor,
    TimestampPreprocessor,
    UrlPreprocessor,
    WebsiteDescPreprocessor,
)


def mock_sent_tokenize(text):
    return [text]


class WebsiteDescPreprocessorTester(unittest.TestCase):
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    def setUp(self, mock_db) -> None:
        mock_db.return_value = MockDumpDB("some/path")
        self.website_processor = WebsiteDescPreprocessor()
        self.example_ids = [0, 1, 2]
        self.example_text = ["test text 1", "test text 2", "test text 3"]
        self.example_metadata = [
            [{"key": "url", "type": "global", "value": "https://www.xyz.com"}],
            [
                {"key": "url", "type": "global", "value": "http://sometitle.com"},
                {"key": "url", "type": "global", "value": "http://notfound.com"},
            ],
            [{"key": "url", "type": "global", "value": "https://www.test.com"}],
        ]

        self.example_dict = {"id": self.example_ids, "metadata": self.example_metadata, "text": self.example_text}

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    def test_website_metadata_processor(self):
        ds = Dataset.from_dict(self.example_dict)
        ds = ds.map(lambda ex: self.website_processor.preprocess(ex), batched=True)
        target_metadata = [
            [
                {"key": "url", "type": "global", "value": "https://www.xyz.com"},
                {"key": "website_description", "type": "global", "value": "XYZ is a U.S. based company."},
            ],
            [
                {"key": "url", "type": "global", "value": "http://sometitle.com"},
                {"key": "url", "type": "global", "value": "http://notfound.com"},
                {"key": "website_description", "type": "global", "value": "SomeTitle is a U.S. based company."},
            ],
            [
                {"key": "url", "type": "global", "value": "https://www.test.com"},
                {"key": "website_description", "type": "global", "value": "Test is a U.S. based company."},
            ],
        ]
        self.assertEqual(ds[:]["metadata"], target_metadata)


class HtmlPreprocessorTester(unittest.TestCase):
    def setUp(self) -> None:
        self.html_processor = HtmlPreprocessor()

    def test_toy_dataset(self):
        # Define toy data
        my_dict = {
            "doc_html": [
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n    </body>\n    </html>\n",
                "<html><body><p>this is a simple paragraph</p></body></html>",
                "<html><body><p id=1>paragraph 1</p><p id=2>paragraph 2</p></body></html>",
                '<html><body><div class="div-level-1">blablabla<div class="div-level-2">tidi tidi</div></div></body></html>',
            ],
            "metadata": [[], [], [], []],
        }

        # Define target values
        target_texts = [
            "This is a title\n",
            "this is a simple paragraph\n",
            "paragraph 1\nparagraph 2\n",
            "blablabla\ntidi tidi\n",
        ]

        target_metadata = [
            [
                {
                    "char_end_idx": 15,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "h1",
                },
                {
                    "char_end_idx": 16,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 26,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 27,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 11,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["id"], "values": ["1"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 23,
                    "char_start_idx": 12,
                    "html_attrs": {"attrs": ["id"], "values": ["2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 24,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 20,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["class"], "values": ["div-level-1 div-level-2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "div",
                },
                {
                    "char_end_idx": 20,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 1,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
        ]

        # Apply function
        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: self.html_processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["text"], target_texts)
        self.assertEqual(ds[:]["metadata"], target_metadata)


def mock_fetch_mention_predictions(self, examples):
    hardcoded_dict = {"Paris": "Paris", "Obama": "Barack_Obama", "Merkel": "Angela_Merkel", "Bieber": "Justin_Bieber"}

    result = {}
    for idx, example_text in enumerate(examples["text"]):
        result[idx] = []
        for keyword, entity_value in hardcoded_dict.items():
            index = example_text.find(keyword)
            print(index)
            if index == -1:
                continue
            result[idx].append(
                [
                    index,
                    len(keyword),
                    None,
                    entity_value,
                ]
            )
    return result


def mock_EntityPreprocessor__init__(
    self,
    base_url,
    path_wiki_db,
    path_or_url_flair_ner_model="ner-fast",
    col_to_store_metadata="metadata",
    col_text="text",
):
    self.wiki_db_path = path_wiki_db
    self.entity_utils = WikipediaDescUtils(path_wiki_db)
    self.base_url = base_url
    self.wiki_version = "wiki_2019"
    self.config = {
        "mode": "eval",
        "model_path": "ed-wiki-2019",
    }
    self.col_to_store_metadata = col_to_store_metadata
    self.col_text = col_text


class PipelinePreprocessorTester(unittest.TestCase):
    def setUp(self) -> None:
        self.create_data()

    def create_data(self):
        # Define toy data
        self.init_dict = {
            "doc_html": [
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n    </body>\n    </html>\n",
                "<html><body><p>this is a simple paragraph with Obama and Merkel mentioned </p></body></html>",
                "<html><body><p id=1>paragraph 1</p><p id=2>paragraph 2 is in Paris</p></body></html>",
                '<html><body><div class="div-level-1">blablabla<div class="div-level-2">tidi tidi</div></div></body></html>',
            ],
            "url": [
                "https://www.nytimes.com/1998/03/08/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html",
                "https://www.xyz.com",
                "https://www.test.com",
                "http://notfound.com",
            ],
        }

        # Define target values
        self.target_texts = [
            "This is a title\n",
            "this is a simple paragraph with Obama and Merkel mentioned\n",
            "paragraph 1\nparagraph 2 is in Paris\n",
            "blablabla\ntidi tidi\n",
        ]

        self.target_metadata_url = [
            [
                {
                    "key": "url",
                    "type": "global",
                    "value": "https://www.nytimes.com/1998/03/08/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html",
                },
            ],
            [{"key": "url", "type": "global", "value": "https://www.xyz.com"}],
            [{"key": "url", "type": "global", "value": "https://www.test.com"}],
            [{"key": "url", "type": "global", "value": "http://notfound.com"}],
        ]

        self.target_metadata_timestamp = [
            [
                {"key": "timestamp", "type": "global", "value": "1998-03-08 00:00:00"},
            ],
            [],
            [],
            [],
        ]

        self.target_metadata_website_desc = [
            [],
            [
                {"key": "website_description", "type": "global", "value": "XYZ is a U.S. based company."},
            ],
            [{"key": "website_description", "type": "global", "value": "Test is a U.S. based company."}],
            [],
        ]

        self.target_metadata_html = [
            [
                {
                    "char_end_idx": 15,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "h1",
                },
                {
                    "char_end_idx": 16,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 59,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 59,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 1,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 11,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["id"], "values": ["1"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 35,
                    "char_start_idx": 12,
                    "html_attrs": {"attrs": ["id"], "values": ["2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 36,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [
                {
                    "char_end_idx": 20,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["class"], "values": ["div-level-1 div-level-2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "div",
                },
                {
                    "char_end_idx": 20,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 1,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
        ]

    target_metadata_entities = [
        [],
        [
            {
                "char_end_idx": 37,
                "char_start_idx": 32,
                "ent_desc": "",
                "key": "entity",
                "type": "local",
                "value": "Barack_Obama",
                "ent_desc": "Barack Hussein Obama II is an American politician.",
            },
            {
                "char_end_idx": 48,
                "char_start_idx": 42,
                "ent_desc": "",
                "key": "entity",
                "type": "local",
                "value": "Angela_Merkel",
            },
        ],
        [
            {
                "char_end_idx": 35,
                "char_start_idx": 30,
                "ent_desc": "",
                "key": "entity",
                "type": "local",
                "value": "Paris",
            }
        ],
        [],
    ]

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    @mock.patch.object(EntityPreprocessor, "fetch_mention_predictions", new=mock_fetch_mention_predictions)
    @mock.patch.object(EntityPreprocessor, "__init__", new=mock_EntityPreprocessor__init__)
    def test_extraction_in_different_columns(self, mock_db):
        mock_db.return_value = MockDumpDB("some/path")
        # Define preprocessors
        col_to_store_text = "text"
        col_to_store_metadata_html = "metadata_html"
        col_to_store_metadata_url = "metadata_url"
        col_to_store_metadata_timestamp = "metadata_timestamp"
        col_to_store_metadata_website_desc = "metadata_website_desc"
        col_to_store_metadata_entities = "metadata_entity"

        html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html, col_to_store_text=col_to_store_text
        )
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url="url")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc, col_metadata_url=col_to_store_metadata_url
        )
        entity_processor = EntityPreprocessor(
            base_url="", path_wiki_db="", col_to_store_metadata=col_to_store_metadata_entities
        )

        # Apply function
        ds = Dataset.from_dict(self.init_dict)
        ds = ds.map(lambda ex: html_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: url_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: timestamp_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: website_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: entity_processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:][col_to_store_text], self.target_texts)
        self.assertEqual(ds[:][col_to_store_metadata_html], self.target_metadata_html)
        self.assertEqual(ds[:][col_to_store_metadata_url], self.target_metadata_url)
        self.assertEqual(ds[:][col_to_store_metadata_timestamp], self.target_metadata_timestamp)
        self.assertEqual(ds[:][col_to_store_metadata_website_desc], self.target_metadata_website_desc)
        self.assertEqual(ds[:][col_to_store_metadata_entities], self.target_metadata_entities)

        ds.set_format("pandas")
        print(ds[:])

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    @mock.patch.object(EntityPreprocessor, "fetch_mention_predictions", new=mock_fetch_mention_predictions)
    @mock.patch.object(EntityPreprocessor, "__init__", new=mock_EntityPreprocessor__init__)
    def test_extraction_in_same_column(self, mock_db):
        mock_db.return_value = MockDumpDB("some/path")
        # Define preprocessors
        col_to_store_text = "text"
        col_to_store_metadata_html = "metadata"
        col_to_store_metadata_url = "metadata"
        col_to_store_metadata_timestamp = "metadata"
        col_to_store_metadata_website_desc = "metadata"
        col_to_store_metadata_entities = "metadata"

        html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html, col_to_store_text=col_to_store_text
        )
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url="url")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc, col_metadata_url=col_to_store_metadata_url
        )
        entity_processor = EntityPreprocessor(
            base_url="", path_wiki_db="", col_to_store_metadata=col_to_store_metadata_entities
        )

        features = Features(
            {
                "doc_html": Value("string"),
                "metadata": [
                    {
                        "char_end_idx": Value("int64"),
                        "char_start_idx": Value("int64"),
                        "html_attrs": {"attrs": [Value("string")], "values": [Value("string")]},
                        "key": Value("string"),
                        "relative_end_pos": Value("int64"),
                        "relative_start_pos": Value("int64"),
                        "type": Value("string"),
                        "value": Value("string"),
                        "ent_desc": Value("string"),
                    }
                ],
                "text": Value("string"),
                "url": Value("string"),
            }
        )

        # Apply function
        ds = Dataset.from_dict(self.init_dict)
        ds = ds.map(lambda ex: html_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: url_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: timestamp_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: website_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: entity_processor.preprocess(ex), batched=True, batch_size=3, features=features)

        self.assertEqual(ds[:][col_to_store_text], self.target_texts)

        for id, metadata_example in enumerate(self.target_metadata_html):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_html])

        for id, metadata_example in enumerate(self.target_metadata_url):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_url])

        for id, metadata_example in enumerate(self.target_metadata_timestamp):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_timestamp])

        for id, metadata_example in enumerate(self.target_metadata_website_desc):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_website_desc])

        for id, metadata_example in enumerate(self.target_metadata_entities):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "relative_start_pos": None,
                        "relative_end_pos": None,
                        "html_attrs": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_entities])

        ds.set_format("pandas")
        print(ds[:])


if __name__ == "__main__":
    unittest.main()
