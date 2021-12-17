import logging
import os
import sys
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional

import hydra
import wandb
from datasets import config, load_dataset
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bsmetadata.preprocessing_utils import (
    EntityPreprocessor,
    TimestampPreprocessor,
    WebsiteDescPreprocessor,
    HtmlPreprocessor,
    ErrorWrapperPreprocessor,
)
from bsmetadata.train import show_help


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    file_name: str = field(metadata={"help": "The input file name(a jsonl file, eventually compressed)."})
    out_file_name: str = field(metadata={"help": "The output file name(a jsonl file)."})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    path_wiki_db: str = field(
        metadata={"help": "The path to the wikipedia database file necessary for the website descriptions"}
    )
    entity_path_data_dir: str = field(
        metadata={
            "help": "The path to the directory containing the directories `ed-wiki-2019`, `generic` and `wiki_2019` "
        }
    )
    path_or_url_flair_ner_model: Optional[str] = field(
        default=None, metadata={"help": "TThe path or name of the flair ner model to use to preprocess entities"}
    )
    metadata_to_include: Optional[list] = field(
        default_factory=lambda: ["website_description", "entity", "timestamp"],
        metadata={"help": "The list of metadata to extract"},
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether the local cache containing datasets should be overwritten."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3?"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    map_batch_size: Optional[int] = field(
        default=1,
        metadata={
            "help": "This is the size of the batch size that will be used for the mapping operation when generating"
            " the dataset. If you are using `with_metadata` the recommended batch size is 1.."
        },
    )
    project_name: str = field(default="metadata_lm_exploration", metadata={"help": "The project name."})


class Logger:
    def __init__(self, *args, **kwargs):
        self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        wandb.log(dic)

    def close(self):
        wandb.finish()


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


def add_url_as_metadata(examples: Dict[str, List], column_name_url: str = "url") -> Dict[str, List]:

    example_url_list = examples[column_name_url]
    example_metadata = []

    for example_url in example_url_list:
        example_metadata.append([{"key": "url", "type": "global", "value": example_url}])

    examples["metadata"] = example_metadata
    examples["id"] = example_url_list  # to change
    return examples

@profile
@hydra.main(config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:
    data_files = {"file": args.file_name}

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config_dict = OmegaConf.to_container(args)
    metrics_logger = Logger(project=args.project_name, config=config_dict)

    logger.info(config.HF_DATASETS_CACHE)
    logger.info(
        "Downloading and loading a dataset from the hub"
        f"{args.dataset_name}, {args.dataset_config_name}, data_files={data_files}, cache_dir={args.cache_dir},"
    )
    # Downloading and loading a dataset from the hub.

    metrics_logger.log({"load_dataset": 0})
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        data_files=data_files,
        cache_dir=args.cache_dir,
        keep_in_memory=False,
        download_mode="force_redownload",
        split="file[0:1000]"
    )
    metrics_logger.log({"load_dataset": 1})
    logger.info(f"Dataset loaded :{raw_datasets}")

    if "url" in args.metadata_to_include:
        metrics_logger.log({"add_url_as_metadata": 0})
        raw_datasets = raw_datasets.map(partial(add_url_as_metadata, column_name_url="url"), batched=True)
        metrics_logger.log({"add_url_as_metadata": 1})

    if "html" in args.metadata_to_include:
        html_preprocessor = HtmlPreprocessor(name_html_column="html")
        output_keys = {
            "metadata": [],
            "metadata_html": [],
            "html": "",
            "text": "",
            "timestamp": 0,
            "id": "",
            "url": "",
        }
        # for col_name in raw_datasets["file"].features.keys():
        #     if col_name not in output_keys:
        #         output_keys[col_name] = None

        error_wrapper_preprocessor = ErrorWrapperPreprocessor(
            metadata_preprocessor=html_preprocessor, output_keys=output_keys
        )

        logger.info("Start HTML preprocessing")
        metrics_logger.log({"html_preprocessor": 0})
        raw_datasets = raw_datasets.map(
            error_wrapper_preprocessor.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running html_preprocessor on dataset",
            batch_size=args.map_batch_size,
        )
        metrics_logger.log({"html_preprocessor": 1})

    if "timestamp" in args.metadata_to_include:
        timestamp_preprocessor = TimestampPreprocessor()
        logger.info("Start timestamp preprocessing")
        metrics_logger.log({"timestamp_preprocessor": 0})
        raw_datasets = raw_datasets.map(
            timestamp_preprocessor.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running timestamp_preprocessor on dataset",
            batch_size=args.map_batch_size,
        )
        metrics_logger.log({"timestamp_preprocessor": 1})

    if "website_description" in args.metadata_to_include:
        logger.info("Start website description preprocessing")
        website_desc_preprocessor = WebsiteDescPreprocessor(path_wiki_db=args.path_wiki_db)
        metrics_logger.log({"website_desc_preprocessor": 0})
        raw_datasets = raw_datasets.map(
            website_desc_preprocessor.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running website_desc_preprocessor on dataset",
            batch_size=args.map_batch_size,
        )
        metrics_logger.log({"website_desc_preprocessor": 1})

    if "entity" in args.metadata_to_include:
        logger.info("Start entity preprocessing")
        entity_preprocessing = EntityPreprocessor(
            base_url=args.entity_path_data_dir,
            path_wiki_db=args.path_wiki_db,
            path_or_url_flair_ner_model=args.path_or_url_flair_ner_model,
        )
        metrics_logger.log({"entity_preprocessing": 0})
        raw_datasets = raw_datasets.map(
            entity_preprocessing.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running entity_preprocessing on dataset",
            batch_size=args.map_batch_size,
        )
        metrics_logger.log({"entity_preprocessing": 1})

    saving_path = os.path.join(args.out_dir, args.out_file_name)
    logger.info(f"Save resulting dataset at {saving_path}")
    raw_datasets.to_json(saving_path)
    metrics_logger.close()


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
