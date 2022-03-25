import os
import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from datasets import load_from_disk, concatenate_datasets
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
    for ds_shard_path in ds_shard_paths:
        ds_shards_list.append(load_from_disk(str(ds_shard_path)))

    ds_full = concatenate_datasets(ds_shards_list)

    logger.info(f"The reconcatenated dataset is: {ds_full}")

    folder_name = args.dataset_name
    save_path: Path = args.save_path / folder_name
    save_path.mkdir(parents=True, exist_ok=True)

    ds_full.save_to_disk(
        f"{str(save_path.absolute())}.tmp",
    )
    subprocess.run(["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())])


if __name__ == "__main__":
    main()

from pathlib import Path
import os

dir_dataset_path = Path("/gpfsscratch/rech/six/uue59kq/new_dataset/process-v2/c4-en-sharded-with-entity")
dataset_name = "c4-en-html_cc-main-2019-18_pq00-205"

ds_shard_paths = []
for file_name in os.listdir(dir_dataset_path):
    if file_name.startswith(dataset_name):
        ds_shard_paths.append(dir_dataset_path / file_name)
ds_shard_paths = sorted(ds_shard_paths)