import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import hydra
from datasets import config, load_dataset
from hydra.core.config_store import ConfigStore

from bsmetadata.preprocessing_utils import EntityPreprocessor, TimestampPreprocessor, WebsiteDescPreprocessor
from bsmetadata.train import show_help


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    file_name: str = field(metadata={"help": "The input file name(a jsonl file, eventually compressed)."})
    out_file_name: str = field(metadata={"help": "The output file name(a jsonl file)."})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    website_desc_path_wiki_db: str = field(
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


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


@hydra.main(config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:
    data_files = {"file": args.file_name}

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    logger.info(config.HF_DATASETS_CACHE)
    logger.info(
        "Downloading and loading a dataset from the hub"
        f"{args.dataset_name}, {args.dataset_config_name}, data_files={data_files}, cache_dir={args.cache_dir},"
    )
    # Downloading and loading a dataset from the hub.
    raw_datasets = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        data_files=data_files,
        cache_dir=args.cache_dir,
        keep_in_memory=False,
        download_mode="force_redownload",
    )
    raw_datasets = raw_datasets.map(lambda batch: {"metadata": [[] for i in range(len(batch["text"]))]}, batched=True)

    if "timestamp" in args.metadata_to_include:
        timestamp_preprocessor = TimestampPreprocessor()
        logger.info("Start timestamp preprocessing")
        raw_datasets = raw_datasets.map(
            timestamp_preprocessor.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running timestamp_preprocessor on dataset",
            batch_size=args.map_batch_size,
        )

    if "website_description" in args.metadata_to_include:
        logger.info("Start website description preprocessing")
        website_desc_preprocessor = WebsiteDescPreprocessor(path_wiki_db=args.website_desc_path_wiki_db)
        raw_datasets = raw_datasets.map(
            website_desc_preprocessor.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running website_desc_preprocessor on dataset",
            batch_size=args.map_batch_size,
        )

    if "entity" in args.metadata_to_include:
        logger.info("Start entity preprocessing")
        entity_preprocessing = EntityPreprocessor(base_url=args.entity_path_data_dir, path_or_url_flair_ner_model=args.path_or_url_flair_ner_model)
        raw_datasets = raw_datasets.map(
            entity_preprocessing.preprocess,
            batched=True,
            num_proc=args.preprocessing_num_workers,
            load_from_cache_file=not args.overwrite_cache,
            desc="Running entity_preprocessing on dataset",
            batch_size=args.map_batch_size,
        )

    saving_path = os.path.join(args.out_dir, args.out_file_name)
    logger.info(f"Save resulting dataset at {saving_path}")
    raw_datasets["file"].to_json(saving_path)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
