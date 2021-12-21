import unittest
from unittest import mock

from datasets import Dataset
from mocks.mock_dump_db import MockDumpDB

from bsmetadata.preprocessing_utils import (
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


class PipelinePreprocessorTester(unittest.TestCase):
    def setUp(self) -> None:
        self.create_data()

    def create_data(self):
        # Define toy data
        self.init_dict = {
            "doc_html": [
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n    </body>\n    </html>\n",
                "<html><body><p>this is a simple paragraph</p></body></html>",
                "<html><body><p id=1>paragraph 1</p><p id=2>paragraph 2</p></body></html>",
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
            "this is a simple paragraph\n",
            "paragraph 1\nparagraph 2\n",
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

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    def test_toy_dataset(self, mock_db):
        mock_db.return_value = MockDumpDB("some/path")
        # Define preprocessors
        col_to_store_text = "text"
        col_to_store_metadata_html = "metadata_html"
        col_to_store_metadata_url = "metadata_url"
        col_to_store_metadata_timestamp = "metadata_timestamp"
        col_to_store_metadata_website_desc = "metadata_website_desc"

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

        # Apply function
        ds = Dataset.from_dict(self.init_dict)
        ds = ds.map(lambda ex: html_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: url_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: timestamp_processor.preprocess(ex), batched=True, batch_size=3)
        ds = ds.map(lambda ex: website_processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:][col_to_store_text], self.target_texts)
        self.assertEqual(ds[:][col_to_store_metadata_html], self.target_metadata_html)
        self.assertEqual(ds[:][col_to_store_metadata_url], self.target_metadata_url)
        self.assertEqual(ds[:][col_to_store_metadata_timestamp], self.target_metadata_timestamp)
        self.assertEqual(ds[:][col_to_store_metadata_website_desc], self.target_metadata_website_desc)

        ds.set_format("pandas")
        print(ds[:])


if __name__ == "__main__":
    unittest.main()
