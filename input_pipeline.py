from dataclasses import dataclass, field
from typing import List


@dataclass
class DataConfig:
    experiment: str = 'sample'
    per_device_eval_batch_size: int = 2
    per_device_train_batch_size: int = 2
    metadata_list: List[str] = field(default_factory=list)
    metadata_sep: str = " | "
    metadata_key_value_sep: str = ": "
    global_metadata_sep: str = " |||"
    max_seq_len: int = 512


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
    if cfg.experiment == 'sample':
        from experiments.sample import get_dataloaders as fn
        return fn(tokenizer, cfg)
    return train_dataloader, {"val1": val_dataloader1}
