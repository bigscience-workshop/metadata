import datetime
import functools
import logging

from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from metadata.metadata_utils import add_metadata_and_chunk_examples


logger = logging.getLogger(__name__)


def format_time(ordinal):
    date = datetime.date.fromordinal(725121)
    return date.strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def convert_to_metadata_format(row):
    return {
        "id": row["id"],
        "text": row["abstract"],
        "metadata": [{"key": "timestamp", "type": "global", "value": format_time(row["time"])}],
    }


def get_dataset(args):
    data_files = {}
    data_files["train"] = args.train_file
    extension = "json"
    raw_datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)

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
    raw_datasets = raw_datasets.map(
        convert_to_metadata_format,
        num_proc=args.preprocessing_num_workers,
    )
    return raw_datasets


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
    # Mostly copy/paste from https://github.com/huggingface/transformers/blob/master/examples/pytorch/language-modeling/run_clm_no_trainer.py
    # Get the datasets: you can either provide your own CSV/JSON/TXT training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files, this script will use the column called 'text' or the first column if no column called
    # 'text' is found. You can easily tweak this behavior (see below).
    #
    # In distributed training, the load_dataset function guarantees that only one local process can concurrently
    # download the dataset.
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.
    raw_datasets = get_dataset(args)

    # Preprocessing the datasets.
    column_names = raw_datasets["train"].column_names

    # First we pre-process our text and metadata
    lm_datasets = raw_datasets.map(
        functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=args),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Pre-process the text and metadata to create new samples",
        remove_columns=column_names,
    )

    def create_labels_column(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    # Then we add the column containing the labels
    lm_datasets = lm_datasets.map(
        create_labels_column,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Create labels column",
    )

    train_dataset = lm_datasets["train"]
    val_dataset = lm_datasets["validation"]

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    val_dataloader1 = DataLoader(
        val_dataset,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, {"val1": val_dataloader1}
