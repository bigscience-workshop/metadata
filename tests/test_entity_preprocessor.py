import unittest
from unittest import mock

from datasets import Dataset

from bsmetadata.preprocessing_utils import EntityPreprocessor


class TestEntityPreprocessor(unittest.TestCase):
    def test_extract_entities(self):

        my_dict = {
            "id": [0, 1, 2],
            "text": [
                "Paris is the beautiful place to visit",
                "Obama was the guest of honor at the conference.",
                "Bieber performed at the concert last night",
            ],
            "metadata": [[], [], []],
        }  # toy dataset

        target_id = [0, 1, 2]

        target_text = [
            "Paris is the beautiful place to visit",
            "Obama was the guest of honor at the conference.",
            "Bieber performed at the concert last night",
        ]

        target_metadata = [
            [{"key": "entity", "type": "local", "char_start_idx": 0, "char_end_idx": 5, "value": "Paris"}],
            [{"key": "entity", "type": "local", "char_start_idx": 0, "char_end_idx": 5, "value": "Barack_Obama"}],
            [{"key": "entity", "type": "local", "char_start_idx": 0, "char_end_idx": 6, "value": "Justin_Bieber"}],
        ]
        processor = EntityPreprocessor()

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["id"], target_id)
        self.assertEqual(ds[:]["text"], target_text)
        self.assertEqual(ds[:]["metadata"], target_metadata)


if __name__ == "__main__":
    unittest.main()
