import logging

from accelerate import DistributedType
from datasets import config, load_dataset
from torch.utils.data import DataLoader
from transformers import DataCollatorWithPadding, default_data_collator


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
            extension = "text"
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
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Preprocessing the datasets.
    # First we tokenize all the texts.
    column_names = datasets["train"].column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name], truncation=True, max_length=args.max_seq_len)

    logger.info("Tokenize dataset")
    tokenized_datasets = datasets.map(
        tokenize_function,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        load_from_cache_file=not args.overwrite_cache,
        desc="Running tokenizer on dataset",
        batch_size=args.map_batch_size,
    )
    logger.info("Tokenize dataset finished")

    if args.block_size is None:
        block_size = tokenizer.model_max_length
        if block_size > 1024:
            logger.warning(
                f"The tokenizer picked seems to have a very large `model_max_length` ({tokenizer.model_max_length}). "
                "Picking 1024 instead. You can change that default value by passing --block_size xxx."
            )
        block_size = 1024
    else:
        if args.block_size > tokenizer.model_max_length:
            logger.warning(
                f"The block_size passed ({args.block_size}) is larger than the maximum length for the model"
                f"({tokenizer.model_max_length}). Using block_size={tokenizer.model_max_length}."
            )
        block_size = min(args.block_size, tokenizer.model_max_length)

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(examples):
        # Concatenate all texts.
        concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, we could add padding if the model supported it instead of this drop, you can
        # customize this part to your needs.
        if total_length >= block_size:
            total_length = (total_length // block_size) * block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

    # Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws away a remainder
    # for each of those groups of 1,000 texts. You can adjust that batch_size here but a higher value might be slower
    # to preprocess.
    #
    # To speed up this part, we use multiprocessing. See the documentation of the map method for more information:
    # https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map

    logger.info("Group texts")
    datasets = tokenized_datasets.map(
        group_texts,
        batched=True,
        num_proc=args.preprocessing_num_workers,
        load_from_cache_file=not args.overwrite_cache,
        desc=f"Grouping texts in chunks of {block_size}",
        batch_size=args.map_batch_size,
    )
    logger.info("Group texts finished")

    train_dataset = datasets["train"]
    val_dataset = datasets["validation"]

    logger.info(f"  Num train examples = {len(train_dataset)}")
    logger.info(f"  Num validation examples = {len(val_dataset)}")

    # DataLoaders creation:
    data_collator = default_data_collator
    if args.distributed_type == DistributedType.TPU:
        data_collator = DataCollatorWithPadding(tokenizer, padding="max_length", max_length=args.max_seq_len)
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=data_collator,
        batch_size=args.per_device_train_batch_size,
    )
    val_dataloader1 = DataLoader(
        val_dataset,
        collate_fn=data_collator,
        batch_size=args.per_device_eval_batch_size,
    )
    return train_dataloader, {"val1": val_dataloader1}
