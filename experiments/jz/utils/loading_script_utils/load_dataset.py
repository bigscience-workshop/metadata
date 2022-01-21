import logging
import sys

import hydra
from datasets import config, load_dataset
from hydra.core.config_store import ConfigStore

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.train import show_help


logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(name="data_config", node=DataConfig)


@hydra.main(config_name="data_config")
def main(args: DataConfig) -> None:
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if not data_files:
        data_files = None

    logger.info(config.HF_DATASETS_CACHE)
    if args.dataset_name is not None:
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

        if "validation" not in raw_datasets.keys():
            logger.info("validation not in raw_datasets.keys()")
            raw_datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        extension = args.train_file.split(".")[-1] if not args.extension else args.extension
        if extension == "txt":
            raise ValueError(
                "You have entered a text file for the train data, but this type of file cannot contain metadata "
                "columns. Wouldn't you rather have a file in json/jsonl or pandas format?"
            )
        if extension == "jsonl":
            extension = "json"
        raw_datasets = load_dataset(
            extension, data_files=data_files, cache_dir=args.cache_dir, download_mode="force_redownload"
        )

        if "validation" not in raw_datasets.keys():
            raw_datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            raw_datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )

    train_dataset = raw_datasets["train"]
    val_dataset = raw_datasets["validation"]

    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")

    logger.info("  Train sample:")
    logger.info(f"  Train sample n°{0} text:\n{train_dataset[0]['text']}")
    logger.info(f"  Train sample n°{0} metadata:\n{train_dataset[0]['metadata']}")


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
