from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    experiment: str = "sample"
    per_device_eval_batch_size: int = 2
    per_device_train_batch_size: int = 2
    metadata_list: List[str] = field(default_factory=list)  # A sorted list of all kinds of metadata to be used
    metadata_sep: str = (
        " | "  # The separator to be used between different kinds of global metadata (e.g., timestamp and URL)
    )
    metadata_key_value_sep: str = ": "  # The default separator used between a metadata key and its associated value
    metadata_probability: float = 1  # The probability of adding metadata to an input example
    global_metadata_sep: str = " |||"  # The separator to be used between global metadata and the actual input text
    max_seq_len: int = 512  # The maximum sequence length to be used for training
    dataset_name: Optional[str] = None  # The name of the dataset to use (via the datasets library)
    dataset_config_name: Optional[
        str
    ] = None  # The configuration name of the dataset to use (via the datasets library)
    train_file: Optional[str] = None  # The input training data file (a jsonl file).
    validation_file: Optional[
        str
    ] = None  # An optional input evaluation data file to evaluate the perplexity on (a jsonl file)
    overwrite_cache: Optional[bool] = False
    cache_dir: Optional[str] = None  # Where do you want to store the pretrained models downloaded from s3
    preprocessing_num_workers: Optional[int] = None  # The number of processes to use for the preprocessing
    validation_split_percentage: Optional[
        int
    ] = 5  # The percentage of the train set used as validation set in case there's no validation split
    block_size: Optional[int] = None  # "Optional input sequence length after tokenization. "


def get_dataloaders(tokenizer, cfg: DataConfig):
    """
    Args:
        tokenizer: a huggingface/transformers tokenizer
        cfg: a DataConfig
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
    if cfg.experiment == "sample":
        from bsmetadata.experiments.sample import get_dataloaders as fn

        return fn(tokenizer, cfg)
    if cfg.experiment == "without_metadata":
        from bsmetadata.experiments.without_metadata import get_dataloaders as fn

        return fn(tokenizer, cfg)
    if cfg.experiment == "with_metadata":
        from bsmetadata.experiments.with_metadata import get_dataloaders as fn

        return fn(tokenizer, cfg)
    else:
        raise ValueError("You have not entered a valid experience name")
