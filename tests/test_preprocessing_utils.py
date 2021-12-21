import unittest
from typing import Dict, List
from unittest import mock

from datasets import Dataset
from mocks.mock_dump_db import MockDumpDB

from bsmetadata.preprocessing_utils import (
    ErrorWrapperPreprocessor,
    HtmlPreprocessor,
    MetadataPreprocessor,
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
            [{"key": "prev_metadata", "type": "global", "value": "1"}],
            [
                {"key": "prev_metadata", "type": "global", "value": "2"},
                {"key": "prev_metadata", "type": "global", "value": "3"},
            ],
            [{"key": "prev_metadata", "type": "global", "value": "4"}],
        ]
        self.url = [
            "https://www.xyz.com",
            "http://sometitle.com",
            "https://www.test.com",
        ]

        self.example_dict = {
            "id": self.example_ids,
            "metadata": self.example_metadata,
            "text": self.example_text,
            "url": self.url,
        }

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    def test_website_metadata_processor(self):
        ds = Dataset.from_dict(self.example_dict)
        ds = ds.map(lambda ex: self.website_processor.preprocess(ex), batched=True)
        target_metadata = [
            [
                {"key": "prev_metadata", "type": "global", "value": "1"},
                {"key": "website_description", "type": "global", "value": "XYZ is a U.S. based company."},
            ],
            [
                {"key": "prev_metadata", "type": "global", "value": "2"},
                {"key": "prev_metadata", "type": "global", "value": "3"},
                {"key": "website_description", "type": "global", "value": "SomeTitle is a U.S. based company."},
            ],
            [
                {"key": "prev_metadata", "type": "global", "value": "4"},
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
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n    this is a simple long enough text with at least 64 characters </body>\n    </html>\n",
                "<html><body><p>this is a simple long enough paragraph with at least 64 characters</p></body></html>",
                "<html><body><p id=1>this is a simple long enough paragraph with at least 64 characters</p><p id=2>this is a second simple long enough paragraph with at least 64 characters</p></body></html>",
                '<html><body><div class="div-level-1">blablabla<div class="div-level-2">tidi tidi</div></div></body></html>',
            ],
            "metadata": [[], [], [], []],
        }

        # Define target values
        target_texts = [
            "This is a title\nthis is a simple long enough text with at least 64 characters\n",
            "this is a simple long enough paragraph with at least 64 characters\n",
            "this is a simple long enough paragraph with at least 64 characters\nthis is a second simple long enough paragraph with at least 64 characters\n",
            "",
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
                    "char_end_idx": 78,
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
                    "char_end_idx": 66,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 67,
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
                    "char_end_idx": 66,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["id"], "values": ["1"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 140,
                    "char_start_idx": 67,
                    "html_attrs": {"attrs": ["id"], "values": ["2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 141,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "body",
                },
            ],
            [],
        ]

        # Apply function
        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: self.html_processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["text"], target_texts)
        self.assertEqual(ds[:]["metadata"], target_metadata)


class ErrorWrapperPreprocessorTester(unittest.TestCase):
    def test_error_wrapper(self):
        class ToyMetadataPreprocessor(MetadataPreprocessor):
            """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

            def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
                example_metadata_list = examples["metadata"]
                example_url_list = examples["urls"]
                new_column = []
                new_metadata_lst = []

                # Iterate through the metadata associated with all examples in this batch.
                for example_metadata, example_url in zip(example_metadata_list, example_url_list):
                    new_metadata_lst.append([{"key": "toy_new", "type": "global", "value": 2}])
                    if example_url == 1:
                        raise ValueError("this is an error")
                    new_column.append(True)

                examples["new_col"] = new_column
                examples["new_metadata"] = new_metadata_lst
                return examples

        toy_metadata_preprocessor = ToyMetadataPreprocessor()
        error_wrapper_preprocessor = ErrorWrapperPreprocessor(
            metadata_preprocessor=toy_metadata_preprocessor,
            output_keys={"metadata": [], "new_metadata": [], "urls": 10, "new_col": False},
        )

        examples = {
            "metadata": [[], [], [{"key": "toy_before", "type": "global", "value": 1}]],
            "urls": [0, 0, 0],
        }

        examples_preprocessed = error_wrapper_preprocessor.preprocess(examples)
        print(examples_preprocessed)
        print("\n***\n")
        assert examples_preprocessed["ToyMetadataPreprocessor_error"] == [0, 0, 0]
        assert examples_preprocessed["ToyMetadataPreprocessor_error_comment"] == ["", "", ""]
        assert examples_preprocessed["new_metadata"] == [
            [{"key": "toy_new", "type": "global", "value": 2}],
            [{"key": "toy_new", "type": "global", "value": 2}],
            [{"key": "toy_new", "type": "global", "value": 2}],
        ]

        # Bad apple in the batch
        examples = {
            "metadata": [[], [], [{"key": "toy_before", "type": "global", "value": 1}]],
            "urls": [0, 0, 1],
        }

        examples_preprocessed = error_wrapper_preprocessor.preprocess(examples)

        print(examples_preprocessed)

        assert examples_preprocessed["ToyMetadataPreprocessor_error"] == [0, 0, 1]
        assert examples_preprocessed["ToyMetadataPreprocessor_error_comment"] == ["", "", "this is an error"]
        assert examples_preprocessed["new_metadata"] == [
            [{"key": "toy_new", "type": "global", "value": 2}],
            [{"key": "toy_new", "type": "global", "value": 2}],
            [],
        ]


if __name__ == "__main__":
    unittest.main()
