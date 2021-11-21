import unittest
from unittest import mock

from datasets import Dataset

from bsmetadata.preprocessing_utils import GenerationLengthPreprocessor

class TestGenerationLengthPreprocessor(unittest.TestCase):

    def test_text_length(self):
        my_dict = {
            "id": [0, 1, 2],
            "text": [
                "Paris is the beautiful place to visit. Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
                "Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
                "Bieber performed at the concert last night. Paris is the beautiful place to visit.",
            ],
            "metadata": [[], [], []],
        }

        target_id = [0, 1, 2]
        target_text = [
            "Paris is the beautiful place to visit. Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
            "Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
            "Bieber performed at the concert last night. Paris is the beautiful place to visit.",
        ]

        target_metadata = [
            [{"key": "length", "type": "global", "value": "130"}],
            [{"key": "length", "type": "global", "value": "91"}],
            [{"key": "length", "type": "global", "value": "82"}],
        ]

        processor = GenerationLengthPreprocessor("text")

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["id"], target_id)
        self.assertEqual(ds[:]["text"], target_text)
        self.assertEqual(ds[:]["metadata"], target_metadata)

    def test_sentences_length(self):
        my_dict = {
            "id": [0, 1, 2],
            "text": [
                "Paris is the beautiful place to visit. Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
                "Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
                "Bieber performed at the concert last night. Paris is the beautiful place to visit.",
            ],
            "metadata": [[], [], []],
        }

        target_id = [0, 1, 2]
        target_text = [
            "Paris is the beautiful place to visit. Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
            "Obama was the guest of honor at the conference. Bieber performed at the concert last night.",
            "Bieber performed at the concert last night. Paris is the beautiful place to visit.",
        ]

        target_metadata = [
            [
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 0, 
                    "char_end_idx": 37, 
                    "value": "37"
                }, 
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 37, 
                    "char_end_idx": 85, 
                    "value": "47"
                }, 
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 85, 
                    "char_end_idx": 129, 
                    "value": "43"
                }
            ],
            [
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 0, 
                    "char_end_idx": 46, 
                    "value": "46"
                }, 
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 46, 
                    "char_end_idx": 90, 
                    "value": "43"
                }
            ],
            [
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 0, 
                    "char_end_idx": 42, 
                    "value": '42'
                }, 
                {
                    "key": "length", 
                    "type": "local", 
                    "char_start_idx": 42, 
                    "char_end_idx": 81, 
                    "value": "38"
                }
            ],
        ]

        processor = GenerationLengthPreprocessor("sentence")

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.preprocess(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["id"], target_id)
        self.assertEqual(ds[:]["text"], target_text)
        self.assertEqual(ds[:]["metadata"], target_metadata)

if __name__ == "__main__":
    unittest.main()
