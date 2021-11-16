import unittest
from unittest import mock

from datasets import Dataset

from bsmetadata.preprocessing_utils import GenerationLengthPreprocessor

class TestGenerationLengthPreprocessor(unittest.TestCase):
    def test_extract_length(self):
        my_dict = {
            "id": [0, 1, 2],
            "text": [
                "Paris is the beautiful place to visit",
                "Obama was the guest of honor at the conference.",
                "Bieber performed at the concert last night",
                ],
            "metadata": [[], [], []],
        }
        
        target_id = [0, 1, 2]
        target_text = [
            "Paris is the beautiful place to visit",
            "Obama was the guest of honor at the conference.",
            "Bieber performed at the concert last night",
        ]
        
        target_metadata = [
            [{"key": "length", "type": "global", "value": "37"}],
            [{"key": "length", "type": "global", "value": "47"}],
            [{"key": "length", "type": "global", "value": "42"}],
        ]
        
        processor = GenerationLengthPreprocessor()
        
        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.preprocess(ex), batched=True, batch_size=3)
        
        self.assertEqual(ds[:]["id"], target_id)
        self.assertEqual(ds[:]["text"], target_text)
        self.assertEqual(ds[:]["metadata"], target_metadata)

if __name__ == "__main__":
    unittest.main()
