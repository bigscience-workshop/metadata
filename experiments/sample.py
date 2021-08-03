from dataclasses import dataclass
from transformers import PreTrainedTokenizerBase
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset

@dataclass
class DataCollatorForCLM:
    tokenizer: PreTrainedTokenizerBase
    pad_to_multiple_of: int = 16

    def __call__(self, batch):
        batch = self.tokenizer(
            [x["text"] for x in batch],
            truncation=True,
            padding="max_length",
            max_length=512, # TODO: make this configurable
            return_tensors="pt",
            pad_to_multiple_of=self.pad_to_multiple_of,
        )
        labels = batch["input_ids"].clone()
        # force an error in no pad_token
        # if self.tokenizer.pad_token_id is not None:
        labels[labels == self.tokenizer.pad_token_id] = -100
        batch["labels"] = labels
        return batch

def get_dataloaders(tokenizer, cfg: "DataConfig"):
    datasets = load_dataset('wikitext', 'wikitext-2-raw-v1')
    data_collator = DataCollatorForCLM(tokenizer)
    train_dataloader = DataLoader(
	datasets['train'],
	shuffle=True,
	collate_fn=data_collator,
	batch_size=cfg.per_device_train_batch_size,
	num_workers=1,
    )
    eval_dataloader = DataLoader(
	datasets['validation'],
	collate_fn=data_collator,
	batch_size=cfg.per_device_eval_batch_size,
	num_workers=1,
    )
    return train_dataloader, {'val': eval_dataloader}

