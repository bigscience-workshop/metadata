import logging
import os
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from datasets import concatenate_datasets, load_from_disk
from datasets.utils.logging import set_verbosity_info


set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dir-dataset-path", type=Path, required=True, help="Dataset path.")
    parser.add_argument("--dataset-name", type=str, required=True, help="Dataset name.")
    parser.add_argument("--save-path", type=Path, required=True, help="Where to save the dataset.")
    parser.add_argument("--number_shards", type=int, help="Number of shards.")
    args = parser.parse_args()
    return args


def save_dataset(
    ds_shard,
    shard_id,
    save_split_path,
    num_shards,
):
    logger.info(f"Saving: {shard_id} / {num_shards}")
    save_path = Path(f"{str(save_split_path)}--shard-id-{shard_id}--{num_shards}")
    if save_path.exists():
        logger.info("Shard was already saved")
        return


def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(f"** The job is runned with the following arguments: **\n{args}\n **** ")

    ds_shard_paths = []
    for file_name in os.listdir(args.dir_dataset_path):
        if file_name.startswith(args.dataset_name):
            ds_shard_paths.append(args.dir_dataset_path / file_name)
    ds_shard_paths = sorted(ds_shard_paths)

    if args.number_shards is not None:
        assert len(ds_shard_paths) == args.number_shards

    ds_shards_list = []
    col_to_store_text = "text"
    for id_shard, ds_shard_path in enumerate(ds_shard_paths):
        sub_ds = load_from_disk(str(ds_shard_path))
        logger.info(
            f"   Example of 1st example 100 first characters of shard n°{id_shard}:\n    {repr(sub_ds[0][col_to_store_text][:100])}"
        )
        ds_shards_list.append(sub_ds)

    ds_full = concatenate_datasets(ds_shards_list)

    logger.info(f"The reconcatenated dataset is: {ds_full}")

    folder_name = args.dataset_name
    save_path: Path = args.save_path / folder_name

    tmp_save_path = Path(save_path.parent, f"tmp-{save_path.name}")
    ds_full.save_to_disk(str(tmp_save_path.absolute()))
    tmp_save_path.rename(save_path)
    logger.info(" ===== Final dataset saved successfully =====")


if __name__ == "__main__":
    main()

import os
from pathlib import Path


dir_dataset_path = Path("/gpfsscratch/rech/six/uue59kq/new_dataset/process-v2/c4-en-sharded-with-entity")
dataset_name = "c4-en-html_cc-main-2019-18_pq00-205"

ds_shard_paths = []
for file_name in os.listdir(dir_dataset_path):
    if file_name.startswith(dataset_name):
        ds_shard_paths.append(dir_dataset_path / file_name)
ds_shard_paths = sorted(ds_shard_paths)
