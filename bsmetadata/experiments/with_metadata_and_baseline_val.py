import copy
import functools
import logging

from datasets import config, load_dataset
from torch.utils.data import DataLoader
from transformers import default_data_collator

from bsmetadata.metadata_utils import add_metadata_and_chunk_examples


logger = logging.getLogger(__name__)


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
    data_files = {}
    if args.train_file is not None:
        data_files["train"] = args.train_file
    if args.validation_file is not None:
        data_files["validation"] = args.validation_file

    if not data_files:
        data_files = None

    logger.info(f"Start to load dataset, the result will be cached at {config.HF_DATASETS_CACHE}")
    if args.dataset_name is not None:
        logger.info(
            "Downloading with arguments: "
            f"dataset_name={args.dataset_name}, "
            f"dataset_config_name={args.dataset_config_name}, "
            f"data_files={data_files}, "
            f"cache_dir={args.cache_dir},"
        )
        # Downloading and loading a dataset from the hub.
        datasets = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            data_files=data_files,
            cache_dir=args.cache_dir,
            keep_in_memory=False,
        )

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            datasets["train"] = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    else:
        logger.info("Loading dataset from extension script")
        extension = args.train_file.split(".")[-1] if not args.extension else args.extension
        if extension == "txt":
            raise ValueError(
                "You have entered a text file for the train data, but this type of file cannot contain metadata "
                "columns. Wouldn't you rather have a file in json/jsonl or pandas format?"
            )
        if extension == "jsonl":
            extension = "json"
        datasets = load_dataset(extension, data_files=data_files, cache_dir=args.cache_dir)

        if "validation" not in datasets.keys():
            datasets["validation"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[:{args.validation_split_percentage}%]",
                cache_dir=args.cache_dir,
            )
            datasets["train"] = load_dataset(
                extension,
                data_files=data_files,
                split=f"train[{args.validation_split_percentage}%:]",
                cache_dir=args.cache_dir,
            )
    logger.info("Dataset loaded")
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    column_names = datasets["train"].column_names

    logger.info("Start to add metadata and chunk examples")

    # Sets the attributes of the args object that have no influence on the calculation of the next map. This is useful
    # for using the cache efficiently.
    tmp_data_args = copy.deepcopy(args)
    tmp_data_args.preprocessing_num_workers = 80
    tmp_data_args.overwrite_cache = False
    tmp_data_args.per_device_eval_batch_size = 2
    tmp_data_args.per_device_train_batch_size = 2
    tmp_data_args.map_batch_size = 1

    # First we pre-process our text and metadata
    datasets_with_metadata = datasets.map(
        functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=tmp_data_args),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Pre-process the text and metadata to create new samples",
        remove_columns=column_names,
        batch_size=args.map_batch_size,
    )
    logger.info("Add metadata and chunk examples finished")

    def create_labels_column(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    logger.info("Create labels column")
    # Then we add the column containing the labels
    datasets_with_metadata = datasets_with_metadata.map(
        create_labels_column,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Create labels column",
        batch_size=args.map_batch_size,
    )
    logger.info("Creating labels column finished")

    train_dataset = datasets_with_metadata["train"]
    val_dataset1 = datasets_with_metadata["validation"]

    # We create another validation dataset without metadata
    logger.info("Start to add metadata and chunk examples")
    tmp_data_args.metadata_probability = 0
    val_dataset_without_metadata = datasets["validation"].map(
        functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=tmp_data_args),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Pre-process the text and metadata to create new samples",
        remove_columns=column_names,
        batch_size=args.map_batch_size,
    )
    logger.info("Add metadata and chunk examples finished")

    def create_labels_column(examples):
        examples["labels"] = examples["input_ids"].copy()
        return examples

    logger.info("Create labels column")
    # Then we add the column containing the labels
    val_dataset_without_metadata = val_dataset_without_metadata.map(
        create_labels_column,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc="Create labels column",
        batch_size=args.map_batch_size,
    )
    logger.info("Creating labels column finished")
    val_dataset2 = val_dataset_without_metadata

    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples dataloader 1 = {len(val_dataset1)}")
    logger.info(f"  Num validation examples dataloader 2 = {len(val_dataset2)}")

    logger.info(f"  Train examples = {train_dataset[0]}")
    logger.info(f"  Validation examples dataloader 1 = {val_dataset1[0]}")
    logger.info(f"  Validation examples dataloader 2 = {val_dataset2[0]}")

    logger.info(f'  Train examples = {tokenizer.convert_ids_to_tokens(train_dataset[0]["input_ids"])}')
    logger.info(
        f'  Validation examples dataloader 1 = {tokenizer.convert_ids_to_tokens(val_dataset1[0]["input_ids"])}'
    )
    logger.info(
        f'  Validation examples dataloader 2 = {tokenizer.convert_ids_to_tokens(val_dataset2[0]["input_ids"])}'
    )

    # DataLoaders creation:
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    val_dataloader1 = DataLoader(
        val_dataset1,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    val_dataloader2 = DataLoader(
        val_dataset2,
        collate_fn=default_data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, {"val1": val_dataloader1, "val2": val_dataloader2}
