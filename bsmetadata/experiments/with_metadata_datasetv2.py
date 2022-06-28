import functools
import logging
from collections import Counter
from copy import deepcopy
from itertools import chain

from datasets import DatasetDict, Features, concatenate_datasets, load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import default_data_collator

from bsmetadata.experiments.datasetv2 import get_files, load_dataset_by_files
from bsmetadata.experiments.without_metadata import preprocess_no_metadata
from bsmetadata.metadata_utils import add_metadata_and_chunk_examples, random_sample_metadata_v2


logger = logging.getLogger(__name__)


# not to be confused with datasets.load_dataset
def my_load_dataset(args):
    """Do the loading
    Args:
        args: a DataConfig
    Returns:
        a dataset
    """
    assert args.train_file is not None, "This experiment requires a train file (can be a wildcard pattern)"
    assert args.validation_file is not None, "This experiment requires a validation file (can be a wildcard pattern)"
    train_files = list(get_files(args.train_file))
    validation_files = list(get_files(args.validation_file))

    tmp = []
    for file in train_files:
        if file in validation_files:
            logger.info(f"{file} is in both train and validation files, removing from train files")
        else:
            tmp.append(file)
    train_files = tmp

    # log number of train & val files before downloading
    logger.info(f"{len(train_files)} train files, starting with {train_files[0]}")
    logger.info(f"{len(validation_files)} validation files, starting with {validation_files[0]}")
    train_dataset = load_dataset_by_files(train_files)
    validation_dataset = load_dataset_by_files(validation_files)
    # datasets = DatasetDict(train=train_dataset, validation=validation_dataset)
    # return datasets
    return train_dataset, validation_dataset


def copy_dataset_for_each_metadata_type(dataset):
    d = dataset
    new_datasets = DatasetDict()
    for col in dataset.column_names:
        if col.startswith("metadata_"):
            logger.info(f"Copying dataset for {col}")
            columns_to_remove = [x for x in d.column_names if x.startswith("metadata_") and x != col]
            new_dataset = d.remove_columns(columns_to_remove)
            for c in columns_to_remove:
                new_column = [[] for _ in range(len(new_dataset))]
                new_dataset = new_dataset.add_column(c, new_column)
            new_datasets[f"validation_{col}"] = new_dataset
            logger.info(f"Copied dataset columns: {new_dataset.column_names}")
    return new_datasets


def preprocess_datasets(datasets, tokenizer, args, is_train=True):
    """
    Args:
        dataset: a huggingface DatasetDict
        tokenizer: a huggingface/transformers tokenizer
        args: a DataConfig
    Returns:
        a dataset
    """
    datasets = datasets.filter(
        lambda x: x["text"],
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="filter out data with empty text",
    )
    key = "train" if is_train else "validation"
    column_names = datasets[key].column_names

    logger.info("Removing metadata not used")
    for key in args.metadata_config.metadata_list:
        assert f"metadata_{key}" in column_names, f"{key} is not in the dataset, column names are {column_names}"
    keep_metadata_columns = [f"metadata_{key}" for key in args.metadata_config.metadata_list]
    remove_columns = [key for key in column_names if key.startswith("metadata_") and key not in keep_metadata_columns]
    logger.info(f"Removing columns {remove_columns}")
    datasets = datasets.remove_columns(remove_columns)
    column_names = datasets[key].column_names

    if is_train:
        if args.metadata_config.random_sample_metadata:
            logger.info("getting stats for dataset")

            def get_metadata_types(example):
                results = []
                for metadata_type in args.metadata_config.metadata_list:
                    if example[f"metadata_{metadata_type}"]:
                        results.append(metadata_type)
                return results

            # get statistics of the dataset for sampling metadata
            metadata_type_counter = Counter(
                chain.from_iterable(
                    get_metadata_types(x)
                    for x in tqdm(datasets["train"], desc="iterate over training set to count metadata types")
                )
            )
            metadata_type_weight_sum = sum(metadata_type_counter.values())
            metadata_type_sample_weights = {k: metadata_type_weight_sum / v for k, v in metadata_type_counter.items()}
            logger.info(f"Metadata type sample weights: {metadata_type_sample_weights}")

            logger.info("Randomly sampling metadata")
            datasets = datasets.map(
                functools.partial(
                    random_sample_metadata_v2, metadata_type_sample_weights=metadata_type_sample_weights
                ),
                batched=True,
                num_proc=args.preprocessing_num_workers,
                load_from_cache_file=not args.overwrite_cache,
                desc="Randomly dropping metadata",
                batch_size=args.map_batch_size,
            )
        else:
            logger.info("Not randomly sampling metadata")
    else:
        validation = datasets["validation"]
        datasets = copy_dataset_for_each_metadata_type(validation)
        datasets["validation"] = validation
        datasets["validation_no_metadata"] = preprocess_no_metadata(validation, tokenizer, args)

    logger.info("Start to add metadata and chunk examples")

    # First we pre-process our text and metadata
    # note: chunking data in batches reqires remove_columns, see https://github.com/huggingface/datasets/issues/1817#issuecomment-774066254
    # but each subset has different columns, so call .map() separately on each subset
    # deepcopy
    cfg = deepcopy(args.metadata_config)
    if not is_train:
        cfg.metadata_probability = 1.0
    datasets = datasets.map(
        functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Pre-process the text and metadata to create new samples",
        remove_columns=column_names,
        batch_size=args.map_batch_size,
    )

    logger.info("Add metadata and chunk examples finished")

    def create_labels_column(examples):
        # remove columns
        examples = {key: value for key, value in examples.items() if key not in column_names}
        examples["labels"] = examples["input_ids"].copy()
        return examples

    # labels column will be generated from input_ids on the fly
    datasets = datasets.with_transform(create_labels_column)
    return datasets


def build_dataset(tokenizer, args):
    """
    Args:
        tokenizer: a huggingface/transformers tokenizer
        args: a DataConfig
    Returns:
        a dataset
    """
    train_dataset, validation_dataset = my_load_dataset(args)

    train_datasets = preprocess_datasets(DatasetDict(train=train_dataset), tokenizer, args, is_train=True)
    validation_datasets = preprocess_datasets(
        DatasetDict(validation=validation_dataset), tokenizer, args, is_train=False
    )
    return train_datasets, validation_datasets


def get_dataloaders(tokenizer, args):
    """
    Args:
        tokenizer: a huggingface/transformers tokenizer
        args: a DataConfig
    Returns:
        a training dataloader and one or more validation dataloaders
        validation dataloaders should be in a dictionary
        each dataloader should yield  {str: torch.Tensor(cpu) }
        dictionary keys may have 'metadata_mask'
        other fields will be passed to model
        note: metadata_mask should be padded
    Example:
       train_dataloader, val_dataloaders = get_dataloaders(...)
       for batch in train_dataloader:
           metadata_mask = batch.get('metadata_mask', None)
           outputs = model(**batch)
           metrics = loss_fn(batch, outputs, metadata_mask)
    """
    train_datasets, validation_datasets = build_dataset(tokenizer, args)

    train_dataset = train_datasets["train"]
    val_dataset = validation_datasets["validation"]

    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")
    logger.info(f"{train_datasets=}")
    logger.info(f"{validation_datasets=}")

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
        drop_last=True,
    )

    val_dataloaders = {
        key: DataLoader(
            (
                val_dataset
                if args.validation_size_max is None
                else val_dataset.select(range(min(args.validation_size_max, len(val_dataset))))
            ),
            collate_fn=default_data_collator,
            batch_size=args.per_device_eval_batch_size,
            drop_last=True,
        )
        for key, val_dataset in validation_datasets.items()
        if key.startswith("validation")
    }
    logger.info(f"validation dataloaders: {val_dataloaders.keys()}")
    return train_dataloader, val_dataloaders
