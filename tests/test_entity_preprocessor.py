import unittest

from datasets import Dataset

from bsmetadata.preprocessing_utils import EntityPreprocessor


class TestEntityPreprocessor(unittest.TestCase):
    def test_extract_entities(self):

        my_dict = {
            "text": [
                "Paris is the beautiful place to visit",
                "This Friday, Obama and Merkel will be meeting to discuss on issues related to climate change",
                "Bieber performed at the concert last night",
                "He was playing a game",
            ],
            "metadata": [[], [], [], []],
        }  # toy dataset

        target_text = [
            "Paris is the beautiful place to visit",
            "This Friday, Obama and Merkel will be meeting to discuss on issues related to climate change",
            "Bieber performed at the concert last night",
            "He was playing a game",
        ]

        target_metadata = [
            [{"char_end_idx": 5, "char_start_idx": 0, "key": "entity", "type": "local", "value": "Paris"}],
            [
                {"char_end_idx": 18, "char_start_idx": 13, "key": "entity", "type": "local", "value": "Barack_Obama"},
                {"char_end_idx": 29, "char_start_idx": 23, "key": "entity", "type": "local", "value": "Angela_Merkel"},
            ],
            [{"char_end_idx": 6, "char_start_idx": 0, "key": "entity", "type": "local", "value": "Justin_Bieber"}],
            [],
        ]
        processor = EntityPreprocessor(
            base_url="Enter the path to the folder having the files downloaded after running the bsmetadata\preprocessing_scripts\download_entity_processing_files.sh script",
            path_or_url_flair_ner_model="Enter the path where you runned `wget https://nlp.informatik.hu-berlin.de/resources/models/ner-fast/en-ner-fast-conll03-v0.4.pt`",
        )

        ds = Dataset.from_dict(my_dict)
        ds = ds.map(lambda ex: processor.tag(ex), batched=True, batch_size=3)

        self.assertEqual(ds[:]["text"], target_text)
        self.assertEqual(ds[:]["metadata"], target_metadata)


if __name__ == "__main__":
    unittest.main()
