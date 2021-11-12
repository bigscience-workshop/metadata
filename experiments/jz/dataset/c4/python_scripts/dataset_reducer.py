import argparse
import gzip
import json
import logging
import os
import pprint
import random
from collections import defaultdict

from tqdm.auto import tqdm


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


data_path = "toy_random_sampler/data"
new_dataset_path = "toy_random_sampler/new_dataset"

total_number_examples = 11


def main(data_path, new_dataset_path, total_number_examples):
    samples_index = []

    for file_name in tqdm(os.listdir(data_path), desc="Counts the number of samples per file"):
        file_path = os.path.join(data_path, file_name)
        logger.info(f"Opening {file_path}")
        num = sum(1 for line in gzip.open(file_path))
        samples_index.extend([(file_name, idx) for idx in range(num)])

    if not os.path.isdir(new_dataset_path):
        os.makedirs(new_dataset_path)

    sampled_indexes = random.sample(samples_index, total_number_examples)
    sampled_indexes_dict = defaultdict(list)
    for sample_index in sampled_indexes:
        sampled_indexes_dict[sample_index[0]].append(sample_index[1])

    first_example = True
    for file_name in tqdm(os.listdir(data_path), desc="Write new examples"):
        file_path_original = os.path.join(data_path, file_name)
        logger.info(f"Opening {file_path_original}")
        with gzip.open(file_path_original, "r") as fi_org:
            with gzip.open(os.path.join(new_dataset_path, file_name), "w") as fi_new:
                for idx, line in enumerate(fi_org):
                    if idx in sampled_indexes_dict[file_name]:
                        fi_new.write(line)

                        if first_example:
                            json_example = json.loads(line)
                            logger.info(f"Example n°{idx}: {pprint.pformat(json_example, indent=4)}")
                            first_example = False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data-path", required=True)
    parser.add_argument("--new-dataset-path", required=True)
    parser.add_argument("--total-number-examples", type=int, required=True)

    args = parser.parse_args()
    main(
        data_path=args.data_path,
        new_dataset_path=args.new_dataset_path,
        total_number_examples=args.total_number_examples,
    )
