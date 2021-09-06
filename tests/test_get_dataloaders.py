import logging
import os
from pathlib import Path

import pytest
import torch
from transformers import AutoTokenizer

from bsmetadata.experiments.with_metadata import get_dataloaders as get_dataloaders_with_metadata
from bsmetadata.experiments.without_metadata import get_dataloaders as get_dataloaders_without_metadata
from bsmetadata.input_pipeline import DataConfig


logging.basicConfig(level=logging.DEBUG)

logger = logging.getLogger()


@pytest.mark.parametrize(
    "metadata_list",
    [
        ["html"],
        ["entity"],
    ],
)
def test_get_dataloaders_with_metadata(metadata_list):
    path = Path(__file__)
    path_test_folder = path.parent.absolute()

    data_config = DataConfig(
        train_file=os.path.join(path_test_folder, "data", "train_toy_wikitext_with_metadata.jsonl"),
        validation_file=os.path.join(path_test_folder, "data", "val_toy_wikitext_with_metadata.jsonl"),
        metadata_list=metadata_list,
        max_seq_len=100,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataloader, eval_dataloaders = get_dataloaders_with_metadata(tokenizer=tokenizer, args=data_config)

    batch = next(iter(train_dataloader))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "metadata_mask" in batch.keys()

    batch_val = next(iter(list(eval_dataloaders.values())[0]))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch_val.keys()
    assert "input_ids" in batch_val.keys()
    assert "metadata_mask" in batch_val.keys()


@pytest.mark.parametrize(
    "metadata_list",
    [
        ["url"],
        ["timestamp"],
        ["url", "timestamp"],
        ["url", "entity"],
        ["timestamp", "entity"],
        ["url", "timestamp", "entity"],
    ],
)
def test_get_dataloaders_with_metadata_start_with_metadata_mask(metadata_list):
    path = Path(__file__)
    path_test_folder = path.parent.absolute()

    data_config = DataConfig(
        train_file=os.path.join(path_test_folder, "data", "train_toy_wikitext_with_metadata.jsonl"),
        validation_file=os.path.join(path_test_folder, "data", "val_toy_wikitext_with_metadata.jsonl"),
        metadata_list=metadata_list,
        max_seq_len=100,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataloader, eval_dataloaders = get_dataloaders_with_metadata(tokenizer=tokenizer, args=data_config)

    batch = next(iter(train_dataloader))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch.keys()
    assert "input_ids" in batch.keys()
    assert "metadata_mask" in batch.keys()

    # Check if the samples start with a metadata token
    assert torch.all(batch["metadata_mask"][:, 0])
    assert ~torch.all(batch["metadata_mask"][:, -1])

    batch_val = next(iter(list(eval_dataloaders.values())[0]))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch_val.keys()
    assert "input_ids" in batch_val.keys()
    assert "metadata_mask" in batch_val.keys()

    # Check if the samples start with a metadata token
    assert torch.all(batch_val["metadata_mask"][:, 0])
    assert ~torch.all(batch_val["metadata_mask"][:, -1])


def test_get_dataloaders_without_metadata():
    path = Path(__file__)
    path_test_folder = path.parent.absolute()

    data_config = DataConfig(
        train_file=os.path.join(path_test_folder, "data", "train_toy_raw_wikitext.jsonl"),
        validation_file=os.path.join(path_test_folder, "data", "val_toy_raw_wikitext.jsonl"),
        block_size=50,
        max_seq_len=100,
    )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    train_dataloader, eval_dataloaders = get_dataloaders_without_metadata(tokenizer=tokenizer, args=data_config)

    batch = next(iter(train_dataloader))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch.keys()
    assert "input_ids" in batch.keys()

    batch_val = next(iter(list(eval_dataloaders.values())[0]))

    # Check if the right keys are in the train batch
    assert "attention_mask" in batch_val.keys()
    assert "input_ids" in batch_val.keys()
