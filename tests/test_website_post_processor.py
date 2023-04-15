import unittest

from datasets import Dataset

from bsmetadata.post_processing_utils import WebsiteDescPostProcessor


class TestWebsitePostProcessor(unittest.TestCase):
    def test_extract_datasource(self):
        website_metadata = [
            [{"key": "website_description", "type": "global", "value": "Remove this example."}],
            [
                {
                    "key": "website_description",
                    "type": "global",
                    "value": "This is a valid website and should be retained",
                }
            ],
            [{"key": "website_description", "type": "global", "value": "Website refers to:"}],
        ]
        my_dict = {
            "id": [0, 1, 2],
            "url": [
                "https://seattlejob.us/finance-internships-miami.html",
                "https://hawkeyesports.com/news/2005/03/22/on-time-and-on-budget/",
                "http://www.plumberscrib.com/steam-generators.html?cat=142&manufacturer=390&p=2",
            ],
            "metadata": website_metadata,
        }

        target_metadata = [
            [],
            [
                {
                    "key": "website_description",
                    "type": "global",
                    "value": "This is a valid website and should be retained",
                }
            ],
            [],
        ]

        processor = WebsiteDescPostProcessor()

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.post_process(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["metadata"], target_metadata)


if __name__ == "__main__":
    unittest.main()
