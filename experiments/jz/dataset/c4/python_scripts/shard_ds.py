import logging
import subprocess
from argparse import ArgumentParser
from pathlib import Path
from typing import List

from datasets import Dataset, load_from_disk
from datasets.utils.logging import set_verbosity_info

set_verbosity_info()
logger = logging.getLogger(__name__)


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--dataset-path", type=str, required=True, help="Dataset path.")
    parser.add_argument("--number_shards", type=int, required=True, help="Number of shards.")
    parser.add_argument(
        "--save-path", type=str, required=True, help="Where to save the dataset."
    )
    parser.add_argument("--index-slice", type=int)
    parser.add_argument("--total-number-slice", type=int)
    args = parser.parse_args()

    args.dataset_path = Path(args.dataset_path)
    args.save_path = Path(args.save_path)

    if args.index_slice is None:
        assert args.total_number_slice is None
    else:
        assert isinstance(args.index_slice, int)
        assert isinstance(args.total_number_slice, int)
    return args

def shard_dataset(ds: Dataset, number_shards: int) -> List[Dataset]:
    if number_shards <= 1:
        return [ds]

    results = []
    logger.info(f"Shard dataset in {number_shards} shards")
    for shard_id in range(number_shards):
        logger.info(f"Shard {shard_id}/{number_shards}")
        shard = ds.shard(num_shards=number_shards, index=shard_id)
        results.append(shard)
    return results

def save_dataset(
    ds_shard,
    shard_id,
    save_split_path,
    num_shards,
):
    logger.info(f"Saving: {shard_id} / {num_shards}")
    save_path = save_split_path / f"shard-id-{shard_id}--{num_shards}"
    if save_path.exists():
        logger.info("Shard was already saved")
        return
    ds_shard.save_to_disk(
        f"{str(save_path.absolute())}.tmp",
    )
    subprocess.run(
        ["mv", f"{str(save_path.absolute())}.tmp", str(save_path.absolute())]
    )

def main():
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    args = get_args()
    logger.info(
        f"** The job is runned with the following arguments: **\n{args}\n **** "
    )

    ds = load_from_disk(str(args.dataset_path.absolute()))

    shards_per_split = shard_dataset(ds, args.number_shards) 

    folder_name = str(args.dataset_path.name)
    save_split_path: Path = args.save_path / folder_name
    save_split_path.mkdir(parents=True, exist_ok=True)
    num_shards = len(shards_per_split)
    for i, ds_shard in enumerate(shards_per_split):
        if args.index_slice is not None:
            if args.index_slice != i % args.total_number_slice:
                continue
        logger.info(f"Shard has {len(ds_shard)} rows")
        save_dataset(
            ds_shard,
            i,
            save_split_path,
            num_shards,
        )


if __name__ == "__main__":
    main()