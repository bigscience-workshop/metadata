import itertools
import unittest
from typing import Dict, List
from unittest import mock

from datasets import Dataset, Features, Value
from mocks.mock_dump_db import MockDumpDB

from bsmetadata.preprocessing_tools.wikipedia_desc_utils import WikipediaDescUtils
from bsmetadata.preprocessing_utils import (
    DatasourcePreprocessor,
    EntityPreprocessor,
    ErrorWrapperPreprocessor,
    GenerationLengthPreprocessor,
    HtmlPreprocessor,
    MetadataPreprocessor,
    TimestampPreprocessor,
    UrlPreprocessor,
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
            [{"key": "url", "type": "global", "value": "https://www.xyz.com"}],
            [
                {"key": "url", "type": "global", "value": "http://sometitle.com"},
                {"key": "url", "type": "global", "value": "http://notfound.com"},
            ],
            [{"key": "url", "type": "global", "value": "https://www.test.com"}],
        ]

        self.example_dict = {"id": self.example_ids, "metadata": self.example_metadata, "text": self.example_text}

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    def test_website_metadata_processor(self):
        ds = Dataset.from_dict(self.example_dict)
        ds = ds.map(lambda ex: self.website_processor.preprocess(ex), batched=True)
        target_metadata = [
            [
                {"key": "url", "type": "global", "value": "https://www.xyz.com"},
                {"key": "website_description", "type": "global", "value": "XYZ is a U.S. based company."},
            ],
            [
                {"key": "url", "type": "global", "value": "http://sometitle.com"},
                {"key": "url", "type": "global", "value": "http://notfound.com"},
                {"key": "website_description", "type": "global", "value": "SomeTitle is a U.S. based company."},
            ],
            [
                {"key": "url", "type": "global", "value": "https://www.test.com"},
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
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n   with some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada </body>\n    </html>\n",
                "<html><body><p>this is a simple paragraph with Obama and Merkel mentioned. tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada</p></body></html>",
                "<html><body><p id=1>paragraph 1 tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada.</p><p id=2>paragraph 2 is in Paris tidi tadada tidi tadada tidi tadada tidi tadada.</p></body></html>",
                '<html><body><div class="div-level-1">blablabla blablabla blablabla blablabla blablabla blablabla<div class="div-level-2">tidi tidi tidi tidi</div></div></body></html>',
            ],
            "metadata": [[], [], [], []],
        }

        # Define target values
        target_texts = [
            "This is a title\nwith some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada\n",
            "this is a simple paragraph with Obama and Merkel mentioned. tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada\n",
            "paragraph 1 tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada.\nparagraph 2 is in Paris tidi tadada tidi tadada tidi tadada tidi tadada.\n",
            "blablabla blablabla blablabla blablabla blablabla blablabla\ntidi tidi tidi tidi\n",
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
                    "char_end_idx": 101,
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
                    "char_end_idx": 119,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 120,
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
                    "char_end_idx": 72,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["id"], "values": ["1"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 145,
                    "char_start_idx": 73,
                    "html_attrs": {"attrs": ["id"], "values": ["2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 146,
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
                    "char_end_idx": 80,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["class"], "values": ["div-level-1 div-level-2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "div",
                },
                {
                    "char_end_idx": 80,
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

        self.assertEqual(ds[:]["text"], target_texts)
        self.assertEqual(ds[:]["metadata"], target_metadata)


def mock_fetch_mention_predictions(self, examples):
    hardcoded_dict = {"Paris": "Paris", "Obama": "Barack_Obama", "Merkel": "Angela_Merkel", "Bieber": "Justin_Bieber"}

    result = {}
    for idx, example_text in enumerate(examples["text"]):
        result[idx] = []
        for keyword, entity_value in hardcoded_dict.items():
            index = example_text.find(keyword)
            if index == -1:
                continue
            result[idx].append(
                [
                    index,
                    len(keyword),
                    None,
                    entity_value,
                ]
            )
    return result


def mock_EntityPreprocessor__init__(
    self,
    base_url,
    path_wiki_db,
    path_or_url_flair_ner_model="ner-fast",
    col_to_store_metadata="metadata",
    col_text="text",
):
    self.wiki_db_path = path_wiki_db
    self.entity_utils = WikipediaDescUtils(path_wiki_db)
    self.base_url = base_url
    self.wiki_version = "wiki_2019"
    self.config = {
        "mode": "eval",
        "model_path": "ed-wiki-2019",
    }
    self.col_to_store_metadata = col_to_store_metadata
    self.col_text = col_text


def concat_columns(columns_names_to_concat: List[str], new_colum_name: str):
    def concat_columns_(examples: Dict[str, List]):
        new_column = []
        extracted_examples = [examples[key] for key in columns_names_to_concat if key in examples]
        for values_one_item in zip(*extracted_examples):
            new_column.append(list(itertools.chain.from_iterable(values_one_item)))
        examples[new_colum_name] = new_column
        return examples

    return concat_columns_


class PipelinePreprocessorTester(unittest.TestCase):
    def setUp(self) -> None:
        self.create_data()

    def create_data(self):
        # Define toy data
        self.init_dict = {
            "doc_html": [
                "\n    <html>\n    <head>\n    </head>\n    <body>\n    <h1>This is a title</h1>\n   with some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada </body>\n    </html>\n",
                "<html><body><p>this is a simple paragraph with Obama and Merkel mentioned. tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada</p></body></html>",
                "<html><body><p id=1>paragraph 1 tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada.</p><p id=2>paragraph 2 is in Paris tidi tadada tidi tadada tidi tadada tidi tadada.</p></body></html>",
                '<html><body><div class="div-level-1">blablabla blablabla blablabla blablabla blablabla blablabla<div class="div-level-2">tidi tidi tidi tidi</div></div></body></html>',
            ],
            "url": [
                "https://www.nytimes.com/1998/03/08/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html",
                "https://www.xyz.com",
                "https://www.test.com",
                "http://notfound.com",
            ],
        }

        # Define target values
        self.target_texts = [
            "This is a title\nwith some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada\n",
            "this is a simple paragraph with Obama and Merkel mentioned. tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada\n",
            "paragraph 1 tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada.\nparagraph 2 is in Paris tidi tadada tidi tadada tidi tadada tidi tadada.\n",
            "blablabla blablabla blablabla blablabla blablabla blablabla\ntidi tidi tidi tidi\n",
        ]

        self.target_metadata_url = [
            [
                {
                    "key": "url",
                    "type": "global",
                    "value": "https://www.nytimes.com/1998/03/08/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html",
                },
            ],
            [{"key": "url", "type": "global", "value": "https://www.xyz.com"}],
            [{"key": "url", "type": "global", "value": "https://www.test.com"}],
            [{"key": "url", "type": "global", "value": "http://notfound.com"}],
        ]

        self.target_metadata_timestamp = [
            [
                {"key": "timestamp", "type": "global", "value": "1998-03-08 00:00:00"},
            ],
            [],
            [],
            [],
        ]

        self.target_metadata_website_desc = [
            [],
            [
                {"key": "website_description", "type": "global", "value": "XYZ is a U.S. based company."},
            ],
            [{"key": "website_description", "type": "global", "value": "Test is a U.S. based company."}],
            [],
        ]

        self.target_metadata_html = [
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
                    "char_end_idx": 101,
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
                    "char_end_idx": 119,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": [], "values": []},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 120,
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
                    "char_end_idx": 72,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["id"], "values": ["1"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 145,
                    "char_start_idx": 73,
                    "html_attrs": {"attrs": ["id"], "values": ["2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 0,
                    "type": "local",
                    "value": "p",
                },
                {
                    "char_end_idx": 146,
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
                    "char_end_idx": 80,
                    "char_start_idx": 0,
                    "html_attrs": {"attrs": ["class"], "values": ["div-level-1 div-level-2"]},
                    "key": "html",
                    "relative_end_pos": 0,
                    "relative_start_pos": 1,
                    "type": "local",
                    "value": "div",
                },
                {
                    "char_end_idx": 80,
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

        self.target_metadata_entities = [
            [],
            [
                {
                    "char_end_idx": 37,
                    "char_start_idx": 32,
                    "key": "entity",
                    "type": "local",
                    "value": "Barack_Obama",
                    "ent_desc": "Barack Hussein Obama II is an American politician.",
                },
                {
                    "char_end_idx": 48,
                    "char_start_idx": 42,
                    "key": "entity",
                    "type": "local",
                    "value": "Angela_Merkel",
                    "ent_desc": "",
                },
            ],
            [
                {
                    "char_end_idx": 96,
                    "char_start_idx": 91,
                    "key": "entity",
                    "type": "local",
                    "value": "Paris",
                    "ent_desc": "",
                }
            ],
            [],
        ]

        self.target_metadata_generation_length_sentence = [
            [],
            [{"char_end_idx": 58, "char_start_idx": 0, "key": "length", "type": "local", "value": "58"}],
            [
                {"char_end_idx": 71, "char_start_idx": 0, "key": "length", "type": "local", "value": "71"},
                {"char_end_idx": 144, "char_start_idx": 71, "key": "length", "type": "local", "value": "72"},
            ],
            [],
        ]

        self.target_metadata_generation_length_text = [
            [{"key": "length", "type": "global", "value": "101"}],
            [{"key": "length", "type": "global", "value": "120"}],
            [{"key": "length", "type": "global", "value": "146"}],
            [{"key": "length", "type": "global", "value": "80"}],
        ]

        self.target_metadata_datasource = [
            [
                {
                    "key": "datasource",
                    "type": "global",
                    "value": "www.nytimes.com > sports > on pro basketball one last hurrah for the bulls reinsdorf isn t quite saying html",
                }
            ],
            [{"key": "datasource", "type": "global", "value": "www.xyz.com > "}],
            [{"key": "datasource", "type": "global", "value": "www.test.com > "}],
            [{"key": "datasource", "type": "global", "value": "notfound.com > "}],
        ]

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    @mock.patch.object(EntityPreprocessor, "fetch_mention_predictions", new=mock_fetch_mention_predictions)
    @mock.patch.object(EntityPreprocessor, "__init__", new=mock_EntityPreprocessor__init__)
    def test_extraction_in_different_columns(self, mock_db):
        mock_db.return_value = MockDumpDB("some/path")
        # Define preprocessors
        col_to_store_text = "text"
        col_to_store_metadata_html = "metadata_html"
        col_to_store_metadata_url = "metadata_url"
        col_to_store_metadata_timestamp = "metadata_timestamp"
        col_to_store_metadata_website_desc = "metadata_website_desc"
        col_to_store_metadata_entities = "metadata_entity"
        col_to_store_metadata_generation_length_text = "metadata_generation_length_text"
        col_to_store_metadata_generation_length_sentence = "metadata_generation_length_sentence"
        col_to_store_metadata_datasource = "metadata_generation_datasource"

        html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html, col_to_store_text=col_to_store_text
        )
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url="url")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc, col_metadata_url=col_to_store_metadata_url
        )
        entity_processor = EntityPreprocessor(
            base_url="", path_wiki_db="", col_to_store_metadata=col_to_store_metadata_entities
        )
        generation_length_preprocessor_text = GenerationLengthPreprocessor(
            mode="text", col_to_store_metadata=col_to_store_metadata_generation_length_text
        )
        generation_length_preprocessor_sentence = GenerationLengthPreprocessor(
            mode="sentence", col_to_store_metadata=col_to_store_metadata_generation_length_sentence
        )
        datasource_preprocessor = DatasourcePreprocessor(
            col_to_store_metadata=col_to_store_metadata_datasource, col_url="url"
        )

        # Apply function
        ds = Dataset.from_dict(self.init_dict)
        features_dict = dict(ds.features)

        def apply_processor(ds: Dataset, processor: MetadataPreprocessor):
            for col_name, feature_type in processor.new_columns_minimal_features.items():
                assert col_name not in features_dict
                features_dict[col_name] = feature_type
            return ds.map(
                processor.preprocess,
                batched=True,
                batch_size=2,
                num_proc=2,
                features=Features(features_dict),
            )

        ds = apply_processor(ds=ds, processor=html_processor)
        ds = apply_processor(ds=ds, processor=url_processor)
        ds = apply_processor(ds=ds, processor=timestamp_processor)
        ds = apply_processor(ds=ds, processor=website_processor)
        ds = apply_processor(ds=ds, processor=entity_processor)
        ds = apply_processor(ds=ds, processor=generation_length_preprocessor_text)
        ds = apply_processor(ds=ds, processor=generation_length_preprocessor_sentence)
        ds = apply_processor(ds=ds, processor=datasource_preprocessor)

        print("\n", col_to_store_text, ds[:][col_to_store_text])
        print("\n", col_to_store_metadata_html, ds[:][col_to_store_metadata_html])
        print("\n", col_to_store_metadata_url, ds[:][col_to_store_metadata_url])
        print("\n", col_to_store_metadata_timestamp, ds[:][col_to_store_metadata_timestamp])
        print("\n", col_to_store_metadata_website_desc, ds[:][col_to_store_metadata_website_desc])
        print("\n", col_to_store_metadata_entities, ds[:][col_to_store_metadata_entities])
        print("\n", col_to_store_metadata_generation_length_text, ds[:][col_to_store_metadata_generation_length_text])
        print(
            "\n",
            col_to_store_metadata_generation_length_sentence,
            ds[:][col_to_store_metadata_generation_length_sentence],
        )
        print("\n", col_to_store_metadata_datasource, ds[:][col_to_store_metadata_datasource])

        self.assertEqual(ds[:][col_to_store_text], self.target_texts)
        self.assertEqual(ds[:][col_to_store_metadata_html], self.target_metadata_html)
        self.assertEqual(ds[:][col_to_store_metadata_url], self.target_metadata_url)
        self.assertEqual(ds[:][col_to_store_metadata_timestamp], self.target_metadata_timestamp)
        self.assertEqual(ds[:][col_to_store_metadata_website_desc], self.target_metadata_website_desc)
        self.assertEqual(ds[:][col_to_store_metadata_entities], self.target_metadata_entities)
        self.assertEqual(
            ds[:][col_to_store_metadata_generation_length_text], self.target_metadata_generation_length_text
        )
        self.assertEqual(
            ds[:][col_to_store_metadata_generation_length_sentence], self.target_metadata_generation_length_sentence
        )
        self.assertEqual(ds[:][col_to_store_metadata_datasource], self.target_metadata_datasource)

        col_to_store_all_metadata = "metadata"
        columns_names_to_concat = [
            col_to_store_metadata_html,
            col_to_store_metadata_url,
            col_to_store_metadata_timestamp,
            col_to_store_metadata_website_desc,
            col_to_store_metadata_entities,
            col_to_store_metadata_generation_length_text,
            col_to_store_metadata_generation_length_sentence,
            col_to_store_metadata_datasource,
        ]
        concat_columns_fn = concat_columns(
            columns_names_to_concat=columns_names_to_concat,
            new_colum_name=col_to_store_all_metadata,
        )
        features = Features(
            {
                "doc_html": Value("string"),
                "metadata": [
                    {
                        "char_end_idx": Value("int64"),
                        "char_start_idx": Value("int64"),
                        "html_attrs": {"attrs": [Value("string")], "values": [Value("string")]},
                        "key": Value("string"),
                        "relative_end_pos": Value("int64"),
                        "relative_start_pos": Value("int64"),
                        "type": Value("string"),
                        "value": Value("string"),
                        "ent_desc": Value("string"),
                    }
                ],
                "text": Value("string"),
                "url": Value("string"),
            }
        )

        ds = ds.map(
            concat_columns_fn,
            batched=True,
            batch_size=2,
            num_proc=2,
            remove_columns=columns_names_to_concat,
            features=features,
        )

        for metadata_type in [
            self.target_metadata_html,
            self.target_metadata_url,
            self.target_metadata_timestamp,
            self.target_metadata_website_desc,
            self.target_metadata_entities,
            self.target_metadata_generation_length_text,
            self.target_metadata_generation_length_sentence,
            self.target_metadata_datasource,
        ]:
            for id, metadata_example in enumerate(metadata_type):
                for metadata in metadata_example:
                    for potential_missing_key in [
                        "char_end_idx",
                        "char_start_idx",
                        "relative_end_pos",
                        "relative_start_pos",
                        "html_attrs",
                        "ent_desc",
                    ]:
                        if potential_missing_key in metadata:
                            continue
                        metadata[potential_missing_key] = None
                    self.assertIn(metadata, ds[id][col_to_store_all_metadata])

    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.DumpDB")
    @mock.patch("bsmetadata.preprocessing_tools.wikipedia_desc_utils.nltk.sent_tokenize", new=mock_sent_tokenize)
    @mock.patch.object(EntityPreprocessor, "fetch_mention_predictions", new=mock_fetch_mention_predictions)
    @mock.patch.object(EntityPreprocessor, "__init__", new=mock_EntityPreprocessor__init__)
    def test_extraction_in_same_column(self, mock_db):
        mock_db.return_value = MockDumpDB("some/path")
        # Define preprocessors
        col_to_store_text = "text"
        col_to_store_metadata_html = "metadata"
        col_to_store_metadata_url = "metadata"
        col_to_store_metadata_timestamp = "metadata"
        col_to_store_metadata_website_desc = "metadata"
        col_to_store_metadata_entities = "metadata"
        col_to_store_metadata_generation_length_text = "metadata"
        col_to_store_metadata_generation_length_sentence = "metadata"
        col_to_store_metadata_datasource = "metadata"

        html_processor = HtmlPreprocessor(
            col_to_store_metadata=col_to_store_metadata_html, col_to_store_text=col_to_store_text
        )
        url_processor = UrlPreprocessor(col_to_store_metadata=col_to_store_metadata_url, col_url="url")
        timestamp_processor = TimestampPreprocessor(
            col_to_store_metadata=col_to_store_metadata_timestamp, col_metadata_url=col_to_store_metadata_url
        )
        website_processor = WebsiteDescPreprocessor(
            col_to_store_metadata=col_to_store_metadata_website_desc, col_metadata_url=col_to_store_metadata_url
        )
        entity_processor = EntityPreprocessor(
            base_url="", path_wiki_db="", col_to_store_metadata=col_to_store_metadata_entities
        )
        generation_length_preprocessor_text = GenerationLengthPreprocessor(
            mode="text", col_to_store_metadata=col_to_store_metadata_generation_length_text
        )
        generation_length_preprocessor_sentence = GenerationLengthPreprocessor(
            mode="sentence", col_to_store_metadata=col_to_store_metadata_generation_length_sentence
        )
        datasource_preprocessor = DatasourcePreprocessor(
            col_to_store_metadata=col_to_store_metadata_datasource, col_url="url"
        )

        features = Features(
            {
                "doc_html": Value("string"),
                "metadata": [
                    {
                        "char_end_idx": Value("int64"),
                        "char_start_idx": Value("int64"),
                        "html_attrs": {"attrs": [Value("string")], "values": [Value("string")]},
                        "key": Value("string"),
                        "relative_end_pos": Value("int64"),
                        "relative_start_pos": Value("int64"),
                        "type": Value("string"),
                        "value": Value("string"),
                        "ent_desc": Value("string"),
                    }
                ],
                "text": Value("string"),
                "url": Value("string"),
            }
        )

        # Apply function
        ds = Dataset.from_dict(self.init_dict)
        ds = ds.map(lambda ex: html_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: url_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: timestamp_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: website_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(lambda ex: entity_processor.preprocess(ex), batched=True, batch_size=3, features=features)
        ds = ds.map(
            lambda ex: generation_length_preprocessor_text.preprocess(ex),
            batched=True,
            batch_size=3,
            features=features,
        )
        ds = ds.map(
            lambda ex: generation_length_preprocessor_sentence.preprocess(ex),
            batched=True,
            batch_size=3,
            features=features,
        )
        ds = ds.map(lambda ex: datasource_preprocessor.preprocess(ex), batched=True, batch_size=3, features=features)

        self.assertEqual(ds[:][col_to_store_text], self.target_texts)

        for id, metadata_example in enumerate(self.target_metadata_html):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_html])

        for id, metadata_example in enumerate(self.target_metadata_url):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_url])

        for id, metadata_example in enumerate(self.target_metadata_timestamp):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_timestamp])

        for id, metadata_example in enumerate(self.target_metadata_website_desc):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_website_desc])

        for id, metadata_example in enumerate(self.target_metadata_entities):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "relative_start_pos": None,
                        "relative_end_pos": None,
                        "html_attrs": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_entities])

        for id, metadata_example in enumerate(self.target_metadata_generation_length_sentence):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_generation_length_sentence])

        for id, metadata_example in enumerate(self.target_metadata_generation_length_text):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_generation_length_text])

        for id, metadata_example in enumerate(self.target_metadata_datasource):
            for metadata in metadata_example:
                metadata.update(
                    {
                        "char_end_idx": None,
                        "char_start_idx": None,
                        "relative_end_pos": None,
                        "relative_start_pos": None,
                        "html_attrs": None,
                        "ent_desc": None,
                    }
                )
                self.assertIn(metadata, ds[id][col_to_store_metadata_datasource])


# class ErrorWrapperPreprocessorTester(unittest.TestCase):
#     def test_error_wrapper(self):
#         class ToyMetadataPreprocessor(MetadataPreprocessor):
#             """An exemplary metadata preprocessor for adding timestamp information based on URLs."""

#             def preprocess(self, examples: Dict[str, List]) -> Dict[str, List]:
#                 example_metadata_list = examples["metadata"]
#                 example_url_list = examples["urls"]
#                 new_column = []

#                 print(example_metadata_list, example_url_list)

#                 # Iterate through the metadata associated with all examples in this batch.
#                 for example_metadata, example_url in zip(example_metadata_list, example_url_list):
#                     example_metadata.append({"key": "toy", "type": "global", "value": 2})
#                     if example_url == 1:
#                         raise ValueError("this is an error")
#                     new_column.append(True)

#                 examples["new_col"] = new_column
#                 return examples

#         toy_metadata_preprocessor = ToyMetadataPreprocessor()
#         error_wrapper_preprocessor = ErrorWrapperPreprocessor(
#             metadata_preprocessor=toy_metadata_preprocessor, output_keys={"metadata": [], "urls": 10, "new_col": False}
#         )

#         examples = {
#             "metadata": [[], [], [{"key": "toy_before", "type": "global", "value": 1}]],
#             "urls": [0, 0, 0],
#         }

#         examples_preprocessed, num_errors = error_wrapper_preprocessor.preprocess(examples)
#         assert examples_preprocessed["ToyMetadataPreprocessor_error"] == [0, 0, 0]
#         assert examples_preprocessed["ToyMetadataPreprocessor_error_comment"] == ["", "", ""]
#         assert examples_preprocessed["metadata"] == [
#             [{"key": "toy", "type": "global", "value": 2}],
#             [{"key": "toy", "type": "global", "value": 2}],
#             [{"key": "toy_before", "type": "global", "value": 1}, {"key": "toy", "type": "global", "value": 2}],
#         ]
#         assert num_errors == 0

#         # Bad apple in the batch
#         examples = {
#             "metadata": [[], [], [{"key": "toy_before", "type": "global", "value": 1}]],
#             "urls": [0, 0, 1],
#         }

#         examples_preprocessed, num_errors = error_wrapper_preprocessor.preprocess(examples)

#         assert examples_preprocessed["ToyMetadataPreprocessor_error"] == [0, 0, 1]
#         assert examples_preprocessed["ToyMetadataPreprocessor_error_comment"] == ["", "", "this is an error"]
#         assert examples_preprocessed["metadata"] == [
#             [{"key": "toy", "type": "global", "value": 2}],
#             [{"key": "toy", "type": "global", "value": 2}],
#             [{"key": "toy_before", "type": "global", "value": 1}],
#         ]
#         assert num_errors == 1


if __name__ == "__main__":
    unittest.main()
