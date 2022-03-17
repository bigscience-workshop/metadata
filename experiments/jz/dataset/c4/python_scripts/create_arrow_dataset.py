import argparse
from pathlib import Path
import logging

import datasets
from datasets.utils.logging import set_verbosity_info
from datasets import load_dataset, Features

set_verbosity_info()
logger = logging.getLogger(__name__)

def get_args():
    parser = argparse.ArgumentParser(description='Load a dataset.')
    parser.add_argument('--dataset-path', type=Path)
    parser.add_argument('--save-path', type=Path)
    args = parser.parse_args()
    return args

def convert_types(features):
    if isinstance(features, dict) and "_type" in features:
        return getattr(datasets, features["_type"])(features["dtype"])
    elif isinstance(features, dict):
        return {key: convert_types(value) for key, value in features.items()}
    elif isinstance(features, list):
        return [convert_types(value) for value in features]

def get_features():
    null = None
    features = {"c4_shard": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
        },
        "c4_timestamp": {
        "dtype": "timestamp[s]",
        "id": null,
        "_type": "Value"
        },
        "html": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
        },
        "url": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
        },
        "metadata_html": [
        {
            "char_end_idx": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "char_start_idx": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "html_attrs": {
            "attrs": [
                {
                "dtype": "string",
                "id": null,
                "_type": "Value"
                }
            ],
            "values": [
                {
                "dtype": "string",
                "id": null,
                "_type": "Value"
                }
            ]
            },
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "relative_end_pos": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "relative_start_pos": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "text": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
        },
        "html_footer": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
        ],
        "html_head": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
        ],
        "html_title": [
        {
            "dtype": "string",
            "id": null,
            "_type": "Value"
        }
        ],
        "HtmlPreprocessor_error": {
        "dtype": "int64",
        "id": null,
        "_type": "Value"
        },
        "HtmlPreprocessor_error_comment": {
        "dtype": "string",
        "id": null,
        "_type": "Value"
        },
        "metadata_url": [
        {
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "metadata_timestamp": [
        {
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "metadata_generation_length_text": [
        {
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "metadata_generation_length_sentence": [
        {
            "char_end_idx": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "char_start_idx": {
            "dtype": "int64",
            "id": null,
            "_type": "Value"
            },
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "metadata_generation_datasource": [
        {
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ],
        "metadata_website_desc": [
        {
            "key": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "type": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            },
            "value": {
            "dtype": "string",
            "id": null,
            "_type": "Value"
            }
        }
        ]
    }
    return Features(convert_types(features))

if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    logger.info(f" ===== Loading {args.dataset_path} =====")
    ds = load_dataset(
        str(args.dataset_path.parent), 
        data_files=[f"*{args.dataset_path}"],
        features = get_features(),
        split="train"
    )
    
    logger.info(f"ds info: {ds}")
    
    logger.info(f" ===== Saving Final dataset =====")
    logger.info(f"Saving to final dataset at {args.save_path}.")
    tmp_save_path = Path(args.save_path.parent, f"tmp-{args.save_path.name}")
    ds.save_to_disk(tmp_save_path)
    tmp_save_path.rename(args.save_path)
    logger.info(f" ===== Final dataset saved successfully =====")