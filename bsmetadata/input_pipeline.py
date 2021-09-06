from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DataConfig:
    experiment: str = field(default="sample", metadata={"help": "The name of the experiment."})
    per_device_eval_batch_size: int = field(
        default=2, metadata={"help": "The per-device batch size to use for evaluation."}
    )
    per_device_train_batch_size: int = field(
        default=2, metadata={"help": "The per-device batch size to use for training."}
    )
    metadata_list: List[str] = field(
        default_factory=list,
        metadata={"help": "The list of metadata types to use. Metadata is added in order of appearance in this list."},
    )
    metadata_sep: str = field(
        default=" | ",
        metadata={"help": "The character sequence that is used to separate two instances of global metadata."},
    )
    metadata_key_value_sep: str = field(
        default=": ",
        metadata={"help": "The character sequence that is used by default to separate a metadata key and its value."},
    )
    metadata_probability: float = field(
        default=1, metadata={"help": "The probability of adding metadata to an input example."}
    )
    global_metadata_sep: str = field(
        default=" |||",
        metadata={"help": "The character sequence that is used to separate all global metadata from the actual text."},
    )
    max_seq_len: int = field(
        default=512, metadata={"help": "The maximum number of tokens to use for each training chunk."}
    )
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)"}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)"}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonl file)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a jsonl file)."},
    )
    overwrite_cache: Optional[bool] = field(
        default=False, metadata={"help": "Whether the local cache containing datasets should be overwritten."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3?"}
    )
    extension: Optional[str] = field(default=None, metadata={"help": "the file extension of the dataset"})
    preprocessing_num_workers: Optional[int] = field(
        default=None, metadata={"help": "The number of processes to use for the preprocessing."}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={
            "help": "The percentage of the train set used as validation set in case there's no validation split."
        },
    )
    block_size: Optional[int] = field(
        default=None, metadata={"help": "Optional input sequence length after tokenization."}
    )


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
