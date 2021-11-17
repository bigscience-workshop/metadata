import unittest
from unittest import mock

from datasets import Dataset

#import sys
#sys.path.insert(0, "/Users/christopher/git/metadata/")

from bsmetadata.preprocessing_utils import DatasourcePreprocessor

class TestDatasourcePreprocessor(unittest.TestCase):
    def test_extract_datasource(self):
        my_dict = {
            "id": [0, 1, 2],
            "url": [
                "https://seattlejob.us/finance-internships-miami.html",
                "https://hawkeyesports.com/news/2005/03/22/on-time-and-on-budget/",
                "http://www.plumberscrib.com/steam-generators.html?cat=142&manufacturer=390&p=2",
            ],
            "metadata": [[], [], []],
        }

        target_id = [0, 1, 2]
        target_url = [
            "https://seattlejob.us/finance-internships-miami.html",
            "https://hawkeyesports.com/news/2005/03/22/on-time-and-on-budget/",
            "http://www.plumberscrib.com/steam-generators.html?cat=142&manufacturer=390&p=2",
        ]

        target_metadata = [
            [{"key": "datasource", "type": "global", "value": "seattlejob.us > finance internships miami html"}],
            [{"key": "datasource", "type": "global", "value": "hawkeyesports.com > news > on time and on budget"}],
            [{"key": "datasource", "type": "global", "value": "www.plumberscrib.com > steam generators html"}],
        ]

        processor = DatasourcePreprocessor()

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["id"], target_id)
        self.assertEqual(ds[:]["url"], target_url)
        self.assertEqual(ds[:]["metadata"], target_metadata)

if __name__ == "__main__":
    unittest.main()
