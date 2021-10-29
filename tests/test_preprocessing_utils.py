import pprint
import unittest
from datasets import Dataset

from bsmetadata.preprocessing_utils import HtmlPreprocessor


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

        self.assertEqual(ds[:]["texts"], target_texts)
        self.assertEqual(ds[:]["metadata"], target_metadata)
