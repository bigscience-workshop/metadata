import json
import logging
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

import datasets
import hydra
import wandb
from datasets import config, load_dataset, load_from_disk
from hydra.core.config_store import ConfigStore
from omegaconf import OmegaConf

from bsmetadata.train import show_help


logger = logging.getLogger(__name__)


@dataclass
class PreprocessingConfig:
    task_id: int = field(metadata={"help": "The id of the task"})
    out_dir: str = field(metadata={"help": "where to save the resulting dataset."})
    num_files_to_process: int = field(metadata={"help": "the number of files to process"})
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
    project_name: str = field(default="metadata_lm_exploration", metadata={"help": "The project name."})
    save_batch_size: int = field(
        default=datasets.config.DEFAULT_MAX_BATCH_SIZE,
        metadata={"help": " Size of the batch to load in memory and write at once."},
    )
    use_load_from_disk: bool = field(
        default=False,
        metadata={
            "help": "If false, the program will load the dataset with `load_dataset` and if false, it will load it "
            "with `load_from_disk`."
        },
    )


class Logger:
    def __init__(self, *args, **kwargs):
        self.run = wandb.init(*args, **kwargs)

    def log(self, dic):
        wandb.log(dic)

    def close(self):
        wandb.finish()


cs = ConfigStore.instance()
cs.store(name="preprocessing_config", node=PreprocessingConfig)


@hydra.main(config_name="preprocessing_config")
def main(args: PreprocessingConfig) -> None:  # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )

    config_dict = OmegaConf.to_container(args)
    metrics_logger = Logger(project=args.project_name, config=config_dict)

    poss_files = sorted(os.listdir(args.dataset_name))

    if args.use_load_from_disk:
        poss_files = [file_name for file_name in poss_files if file_name.startswith("c4-en-html")]
    else:
        poss_files = [
            file_name
            for file_name in poss_files
            if (file_name.endswith("jsonl.gz") or file_name.endswith("jsonl")) and file_name.startswith("c4-en-html")
        ]

    def process_file(file_name: str):
        if args.use_load_from_disk:
            dataset_name = os.path.join(args.dataset_name, file_name)
            logger.info(f"Loading the dataset {dataset_name} with `load_from_disk`")
            ds = load_from_disk(dataset_name)
        else:
            data_files = {"file": file_name}
            logger.info(
                "Loading a dataset with `load_dataset`"
                f"{args.dataset_name}, {args.dataset_config_name}, data_files={data_files}, cache_dir={args.cache_dir},"
            )
            ds = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                data_files=data_files,
                cache_dir=args.cache_dir,
                keep_in_memory=False,
                download_mode="force_redownload",
            )["file"]

        features_dict = dict(ds.features)
        logger.info(f"the initial features of the dataset are: {features_dict}")

        if file_name.endswith(".jsonl.gz"):
            out_file_name = file_name
        elif file_name.endswith(".jsonl"):
            out_file_name = f"{file_name}.gz"
        else:
            out_file_name = f"{file_name}.jsonl.gz"
        out_file_name_tmp = f"tmp-{out_file_name}"

        saving_path = os.path.join(args.out_dir, out_file_name)
        saving_path_tmp = os.path.join(args.out_dir, out_file_name_tmp)

        logger.info(f"Save resulting dataset {ds} at {saving_path_tmp}")
        metrics_logger.log({"save_result": 0})
        ds.to_json(
            saving_path_tmp,
            batch_size=args.save_batch_size,
            num_proc=args.preprocessing_num_workers,
            compression="gzip",
        )

        metrics_logger.log({"save_result": 1})
        logger.info(f"Moving the saved dataset to {saving_path}")
        subprocess.run(["mv", saving_path_tmp, saving_path])
        logger.info(f"Processing of {file_name} ended successfully")

        # Get json serializable dataset info
        dataset_info = asdict(ds._info)
        dataset_info_filename = Path(args.out_dir, config.DATASET_INFO_FILENAME).as_posix()
        if not os.path.isfile(dataset_info_filename):
            logger.info(f"Creating {dataset_info_filename}")
            with open(dataset_info_filename, "w", encoding="utf-8") as dataset_info_file:
                # Sort only the first level of keys, or we might shuffle fields of nested features if we use sort_keys=True
                sorted_keys_dataset_info = {key: dataset_info[key] for key in sorted(dataset_info)}
                json.dump(sorted_keys_dataset_info, dataset_info_file, indent=2)

    for file_name in poss_files[
        args.task_id * args.num_files_to_process : args.task_id * args.num_files_to_process + args.num_files_to_process
    ]:
        logger.info(f"Start to process {file_name}")
        process_file(file_name=file_name)


if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
