import unittest
from unittest import mock

from mocks.dump_db import DumpDB

from bsmetadata.preprocessing_utils import WebsiteDescPreprocessor


class WebsiteDescPreprocessorTester(unittest.TestCase):
    # def setUp(self) -> None:
    #     self.website_processor = WebsiteDescPreprocessor("some/path")

    @mock.patch("bsmetadata.preprocessing_utils.DumpDB")
    def test_website_preprocessing(self, mock_db):
        mock_db.return_value = DumpDB
        print(mock_db)


if __name__ == "__main__":
    unittest.main()
