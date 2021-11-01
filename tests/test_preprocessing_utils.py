import unittest
from unittest import mock

from datasets import Dataset
from mocks.mock_dump_db import MockDumpDB

from bsmetadata.preprocessing_utils import WebsiteDescPreprocessor


def mock_sent_tokenize(text):
    return [text]


class WebsiteDescPreprocessorTester(unittest.TestCase):
    @mock.patch("mocks.website_desc_utils.DumpDB")
    def setUp(self, mock_db) -> None:
        mock_db.return_value = MockDumpDB("some/path")
        self.website_processor = WebsiteDescPreprocessor("some/path")
        self.example_ids = [0, 1, 2]
        self.example_text = ["test text 1", "test text 2", "test text 3"]
        self.example_metadata = [
            [{"key": "url", "type": "global", "value": "https://www.xyz.com"}],
            [{"key": "url", "type": "global", "value": "http://sometitle.com"}],
            [{"key": "url", "type": "global", "value": "http://www.sometitle.com"}],
            [{"key": "url", "type": "global", "value": "https://www.test.com"}],
        ]

        self.example_dict = {"id": self.example_ids, "metadata": self.example_metadata, "text": self.example_text}

    @mock.patch("mocks.website_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    def test_website_metadata_processor(self):
        ds = Dataset.from_dict(self.example_dict)
        ds = ds.map(lambda ex: self.website_processor.preprocess(ex), batched=True)

        target_metadata = ["XYZ is a U.S. based company. Another test line."]
        self.assertEqual(ds[:]["metadata"], target_metadata)


if __name__ == "__main__":
    unittest.main()
