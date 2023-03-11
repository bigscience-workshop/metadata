# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
This script provides utility functions for linearizing, encoding and chunking a given input text with metadata information.
"""
import logging
import random
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
from transformers import PreTrainedTokenizerFast

from bsmetadata.metadata_processors import PROCESSORS, MetadataConfig, MetadataProcessor


logger = logging.getLogger(__name__)


@dataclass
class MetadataIdxStorage:
    start_idx_tag_with_content: dict = field(default_factory=(lambda: defaultdict(list)))
    end_idx_tag_with_content: dict = field(default_factory=(lambda: defaultdict(list)))
    start_idx_tag_without_content: dict = field(default_factory=(lambda: defaultdict(list)))
    end_idx_tag_without_content: dict = field(default_factory=(lambda: defaultdict(list)))


@dataclass
class BasicMetadata:
    char_start_idx: int
    key: str
    type: str
    value: str
    char_end_idx: Optional[int] = None
    relative_start_pos: Optional[int] = None
    relative_end_pos: Optional[int] = None


def add_metadata_and_chunk_examples(
    examples: Dict[str, List], tokenizer: PreTrainedTokenizerFast, cfg: MetadataConfig
) -> Dict[str, List]:
    """Adds metadata to the provided input examples, encodes them and groups them in chunks of size `cfg.max_seq_len`.

    Args:
        examples: The examples to process, with required keys "text" and "metadata".
        tokenizer: The pretrained tokenizer to use.
        cfg: The config to use for adding metadata and chunking.

    Returns:
        A new (potentially larger) collection of examples with keys "input_ids", "attention_mask" and "metadata_mask", where:
            - the input ids are a list of token ids corresponding to the input text with metadata;
            - the attention mask is 0 for padding tokens and 1 everywhere else;
            - the metadata mask is 1 for tokens corresponding to metadata and 0 for all other tokens.
    """
    num_examples = len(next(iter(examples.values())))
    linearized_examples = defaultdict(list)

    for example_idx in range(num_examples):
        example = {k: v[example_idx] for k, v in examples.items()}

        # Determine whether metadata should be used.
        add_metadata = random.random() < cfg.metadata_probability

        if add_metadata:
            # Get the global metadata prefix that is prepended to each training example.
            metadata_prefix = create_metadata_prefix(example, cfg)
            metadata_prefix_encoded = (
                tokenizer.encode_plus(cfg.metadata_prefix_start_seq + metadata_prefix).input_ids
                if metadata_prefix
                else []
            )
            # NOTE: this is added when testing very short input length
            # limit how much length is used in prefix
            # to suppress error when prefix is longer than max len
            prefix_len = len(metadata_prefix_encoded)
            max_prefix_len = cfg.max_seq_len // 2
            if prefix_len > max_prefix_len:
                # `-2`'s are for preseving "|||"; it can be wrong if a tokenizer sometimes outputs different tokens.
                metadata_prefix_encoded = metadata_prefix_encoded[: max_prefix_len - 2] + metadata_prefix_encoded[-2:]

        else:
            metadata_prefix_encoded = []

        if add_metadata:
            # Get the actual text with local metadata inserted.
            text_with_local_metadata, char_level_metadata_mask = add_local_metadata_to_text(example, cfg)
        else:
            text_with_local_metadata = example["text"]
            char_level_metadata_mask = [False] * len(text_with_local_metadata)

        if metadata_prefix_encoded:
            text_with_local_metadata = " " + text_with_local_metadata
            char_level_metadata_mask = [False] + char_level_metadata_mask

        text_with_local_metadata_encoded = tokenizer.encode_plus(text_with_local_metadata)

        def is_metadata(idx: int) -> bool:
            char_span = text_with_local_metadata_encoded.token_to_chars(idx)
            char_range = range(char_span.start, char_span.end)
            return any(char_level_metadata_mask[c] for c in char_range)

        if cfg.treat_local_metadata_as_regular_text:
            token_level_metadata_mask = [0] * len(text_with_local_metadata_encoded.input_ids)
        else:
            token_level_metadata_mask = [
                is_metadata(idx) for idx, _ in enumerate(text_with_local_metadata_encoded.input_ids)
            ]

        # Create chunks of `max_seq_len` tokens.
        prefix_len = len(metadata_prefix_encoded)
        max_text_len = cfg.max_seq_len - prefix_len
        if cfg.apply_cm3_loss_to_sequences:
            max_text_len -= 2

        for text_chunk_encoded, chunk_metadata_mask in chunks(
            max_text_len, text_with_local_metadata_encoded.input_ids, token_level_metadata_mask
        ):
            if cfg.apply_cm3_loss_to_sequences:
                span_ids = sorted([random.randint(0, len(text_chunk_encoded)) for x in range(2)])
                span_start, span_end = span_ids[0], span_ids[1]
                if span_end - span_start > 0:
                    text_chunk_encoded = (
                        text_chunk_encoded[:span_start]
                        + [tokenizer.mask_token_id]
                        + text_chunk_encoded[span_end:]
                        + [tokenizer.mask_token_id]
                        + text_chunk_encoded[span_start:span_end]
                    )
                    chunk_metadata_mask = (
                        chunk_metadata_mask[:span_start]
                        + [1]
                        + chunk_metadata_mask[span_end:]
                        + [1]
                        + chunk_metadata_mask[span_start:span_end]
                    )

            total_len = prefix_len + len(text_chunk_encoded)
            padding_len = max_text_len - len(text_chunk_encoded)

            input_ids = metadata_prefix_encoded + text_chunk_encoded + [tokenizer.eos_token_id] * padding_len
            attention_mask = [1] * total_len + [0] * padding_len
            metadata_mask = [1] * prefix_len + [int(x) for x in chunk_metadata_mask] + [0] * padding_len

            linearized_examples["input_ids"].append(input_ids)
            linearized_examples["attention_mask"].append(attention_mask)
            linearized_examples["metadata_mask"].append(metadata_mask)

    return linearized_examples


def convert_v2_dataset_to_v1_format(example):
    metadata_list = []
    key_prefix = "metadata_"
    for key, value in example.items():
        if key.startswith(key_prefix) and value is not None:
            key = key[len(key_prefix) :]
            for metadata in value:
                metadata = deepcopy(metadata)
                if "key" not in metadata:
                    metadata["key"] = key
                metadata_list.append(metadata)
    example["metadata"] = metadata_list
    return example


def convert_v2_dataset_to_v1_format_v1_compatible(example):
    if "metadata" in example:
        return example
    return convert_v2_dataset_to_v1_format(example=example)


def get_metadata_types(metadata_list):
    return list(set(m["key"] for m in metadata_list))


def random_sample_metadata(
    examples: Dict[str, List],
    metadata_type_sample_weights: Dict[str, float],
) -> Dict[str, List]:
    """Randomly drop some of the metadata from the provided examples.
    Uniformly decide the number of metadata types to keep. And sample the metadata types to keep.

    Args:
        examples: The examples to process, with required "metadata".

    Returns:
        A new collection of examples, with some metadata dropped.
    """
    new_metadata = []
    for example_metadata_list in examples["metadata"]:
        if not example_metadata_list:
            new_metadata.append([])
            continue

        metadata_types = get_metadata_types(example_metadata_list)
        num_metadata_to_keep = random.randint(1, len(metadata_types))
        vec = np.arange(len(metadata_types))
        weights = np.array([metadata_type_sample_weights[m] for m in metadata_types])
        weights = weights / weights.sum()
        metadata_types_ids = np.random.choice(vec, num_metadata_to_keep, replace=False, p=weights)
        metadata_types = [metadata_types[i] for i in metadata_types_ids]
        new_metadata.append([m for m in example_metadata_list if m["key"] in metadata_types])
    examples["metadata"] = new_metadata
    return examples


def random_sample_metadata_v2(
    examples: Dict[str, List],
    metadata_type_sample_weights: Dict[str, float],
) -> Dict[str, List]:
    """Randomly drop some of the metadata from the provided examples.
    Uniformly decide the number of metadata types to keep. And sample the metadata types to keep.

    Args:
        examples: The examples to process, with required "metadata".
        metadata_type_sample_weights: Dict[str, float], metadata_{key} is the column name

    Returns:
        A new collection of examples, with some metadata dropped.
    """
    only_metadata_types = [key for key in metadata_type_sample_weights.keys() if f"metadata_{key}" in examples]
    for i in range(len(examples["text"])):
        example = {k: v[i] for k, v in examples.items()}
        metadata_types = [key for key in only_metadata_types if example[f"metadata_{key}"]]
        if len(metadata_types) == 0:
            continue
        num_metadata_to_keep = random.randint(1, len(metadata_types))
        weights = np.array([metadata_type_sample_weights[m] for m in metadata_types])
        weights = weights / weights.sum()
        ids = np.arange(len(metadata_types))
        metadata_types_ids = np.random.choice(ids, num_metadata_to_keep, replace=False, p=weights)
        metadata_types = set([metadata_types[i] for i in metadata_types_ids])
        for key in only_metadata_types:
            if key not in metadata_types:
                examples[f"metadata_{key}"][i] = []
    return examples


def create_metadata_prefix(example: Dict[str, Any], cfg: MetadataConfig) -> str:
    """Creates a prefix containing all global metadata information (including URLs, timestamps, etc)
    and/or local metadata special tokens

    Args:
        example: The example to create a metadata prefix for.
        cfg: The data config to use.

    Returns:
        A string containing the metadata prefix.
    """
    example = convert_v2_dataset_to_v1_format_v1_compatible(example=example)
    processed_metadata = {}
    for metadata in example["metadata"]:
        key, type_ = metadata["key"], metadata["type"]
        if key not in cfg.metadata_list:
            logger.warning(f"metadata key not in metadata_list, skipping. {key}, {cfg.metadata_list}")
            continue

        if type_ == "global":
            processor = PROCESSORS.get(key, MetadataProcessor)(cfg)
            processed_metadata[key] = processor.process_global(metadata)
        elif (
            cfg.add_local_metadata_special_tokens_in_prefix
            and cfg.local_metadata_special_tokens
            and key in cfg.local_metadata_special_tokens
        ):
            processed_metadata[key] = cfg.local_metadata_special_tokens[key]
        elif cfg.add_local_metadata_special_tokens_in_prefix:
            processed_metadata[key] = key

    sorted_metadata = [processed_metadata.get(md, None) for md in cfg.metadata_list]
    sorted_metadata = [md for md in sorted_metadata if md is not None]
    return cfg.metadata_sep.join(sorted_metadata) + cfg.metadata_prefix_sep if sorted_metadata else ""


def _collate_metadata(metadata_list: List[dict], cfg: MetadataConfig):
    """Transforms a list of local metadata that may be more than one at a same `char_start_idx` or `char_end_idx` index
    into a list of metadata with only one metadata per `char_start_idx`.
    The new metadata at `char_start_idx` corresponds to the concatenation of the local metadata that appear at the
    same idx (for its start value and/or its end value). The order is determined by the values of `relative_start_pos`
    and `relative_end_pos`.

    Note that this function requires that for each metadata has `char_start_idx`, `char_end_idx`,
    `relative_start_pos` and `relative_end_pos`

    Args:
        metadata_list: list of metadata dict to collate
        cfg: The data config to use.

    Returns:
        A list of metadata with only one metadata per `char_start_idx`
    """
    processor = PROCESSORS.get(metadata_list[0]["key"], MetadataProcessor)(cfg)

    new_metadata_list = []

    metadata_dict_idx = DefaultDict(dict)
    for metadata_node in metadata_list:
        processed_metadata = processor.process_local(metadata_node)
        if processed_metadata is None:
            continue

        metadata_node = BasicMetadata(
            char_start_idx=metadata_node["char_start_idx"],
            key=metadata_node["key"],
            type=metadata_node["type"],
            value=metadata_node["value"],
            char_end_idx=metadata_node["char_end_idx"],
            relative_start_pos=metadata_node["relative_start_pos"],
            relative_end_pos=metadata_node["relative_end_pos"],
        )
        start_text, end_text = processed_metadata

        assert metadata_node.relative_start_pos not in metadata_dict_idx[metadata_node.char_start_idx]
        assert metadata_node.relative_end_pos not in metadata_dict_idx[metadata_node.char_end_idx]

        if start_text:
            metadata_dict_idx[metadata_node.char_start_idx][metadata_node.relative_start_pos] = start_text
        if end_text:
            metadata_dict_idx[metadata_node.char_end_idx][metadata_node.relative_end_pos] = end_text

    for absolute_idx, value in metadata_dict_idx.items():
        pos_sorted = sorted(list(value.keys()))
        local_metadata = ""
        for pos in pos_sorted:
            local_metadata += metadata_dict_idx[absolute_idx][pos]

        # We add here a local special token if needed around the metadata list of a type if needed
        if (
            cfg.local_metadata_special_token_start
            and local_metadata
            and metadata_list[0]["key"] in cfg.local_metadata_special_token_start
        ):
            local_special_token_start = cfg.local_metadata_special_token_start[metadata_list[0]["key"]]
            local_metadata = f"{local_special_token_start}{local_metadata}"
        if (
            cfg.local_metadata_special_token_end
            and local_metadata
            and metadata_list[0]["key"] in cfg.local_metadata_special_token_end
        ):
            local_special_token_end = cfg.local_metadata_special_token_end[metadata_list[0]["key"]]
            local_metadata = f"{local_metadata}{local_special_token_end}"

        new_metadata_list.append(
            asdict(
                BasicMetadata(
                    char_start_idx=absolute_idx,
                    key=f"basic_start_local_{metadata_list[0]['key']}",
                    type="local",
                    value=local_metadata,
                    char_end_idx=absolute_idx,
                    relative_start_pos=None,
                    relative_end_pos=None,
                )
            )
        )
    return new_metadata_list


def add_local_metadata_to_text(example: Dict[str, Any], cfg: MetadataConfig) -> Tuple[str, List[bool]]:
    """Adds local metadata (such as HTML tags and entity names) to the given input text.
    Args:
        example: The example for which local metadata should be added.
        cfg: The data config to use.
    Returns:
        A tuple of two elements, where:
            - the first element is the text with metadata;
            - the second element is a boolean mask where `mask[i]` is set iff `text[i]` is some kind of metadata.
    """
    metadata_idx_storage = MetadataIdxStorage()

    # Filter and sort all metadata so that they are processed in the requested order.

    filtered_metadata = defaultdict(list)
    example = convert_v2_dataset_to_v1_format_v1_compatible(example=example)
    for md in example["metadata"]:
        if md["type"] == "local" and md["key"] in cfg.metadata_list:
            filtered_metadata[md["key"]].append(md)

    for metadata_type, metadata_list in filtered_metadata.items():
        if metadata_list and metadata_list[0].get("relative_start_pos") is not None:
            assert all(md.get("relative_start_pos") is not None for md in metadata_list), (
                "We have a type of tag that has its `relative_start_pos` field partially defined and "
                "we don't know how to handle this case."
            )

            filtered_metadata[metadata_type] = _collate_metadata(metadata_list, cfg)

    filtered_metadata = sum(filtered_metadata.values(), [])

    # A list is created to define the order between the metadata types
    metadata_list_priority = [
        metadata_key if pos == 0 else f"basic_start_local_{metadata_key}"
        for metadata_key in cfg.metadata_list
        for pos in (0, 1)
    ]
    sorted_metadata = sorted(
        filtered_metadata, key=lambda md: (metadata_list_priority.index(md["key"]), md["char_end_idx"])
    )

    for md in filtered_metadata:
        if "basic_start_local" in md["key"]:
            md["key"] = "basic_start_local"

    # Compute the text sequences to add at the start and end of each metadata entry.
    for metadata in sorted_metadata:
        processor = PROCESSORS.get(metadata["key"], MetadataProcessor)(cfg)
        processed_metadata = processor.process_local(metadata)
        if processed_metadata is None:
            continue
        start_text, end_text = processed_metadata

        char_start_idx = metadata.get("char_start_idx", -1)
        char_end_idx = metadata.get("char_end_idx", -1)

        if char_start_idx == char_end_idx:
            metadata_idx_storage.start_idx_tag_without_content[char_start_idx].insert(0, start_text)
            metadata_idx_storage.end_idx_tag_without_content[char_end_idx].append(end_text)
        else:
            metadata_idx_storage.start_idx_tag_with_content[char_start_idx].insert(0, start_text)
            metadata_idx_storage.end_idx_tag_with_content[char_end_idx].append(end_text)

    # Build the final text with local metadata and the corresponding mask.
    text_with_local_metadata = []
    metadata_mask = []

    def _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask):
        for metadata_text in metadata_text_list:
            text_with_local_metadata.append(metadata_text)
            metadata_mask += [True] * len(metadata_text)

    idx = 0
    for idx, char in enumerate(example["text"]):
        if idx in metadata_idx_storage.end_idx_tag_with_content:
            metadata_text_list = metadata_idx_storage.end_idx_tag_with_content[idx]
            _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

        if idx in metadata_idx_storage.start_idx_tag_without_content:
            metadata_text_list = metadata_idx_storage.start_idx_tag_without_content[idx]
            _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

        if idx in metadata_idx_storage.end_idx_tag_without_content:
            metadata_text_list = metadata_idx_storage.end_idx_tag_without_content[idx]
            _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

        if idx in metadata_idx_storage.start_idx_tag_with_content:
            metadata_text_list = metadata_idx_storage.start_idx_tag_with_content[idx]
            _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

        text_with_local_metadata.append(char)
        metadata_mask += [False]

    idx += 1
    if idx in metadata_idx_storage.end_idx_tag_with_content:
        metadata_text_list = metadata_idx_storage.end_idx_tag_with_content[idx]
        _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

    if idx in metadata_idx_storage.start_idx_tag_without_content:
        metadata_text_list = metadata_idx_storage.start_idx_tag_without_content[idx]
        _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

    if idx in metadata_idx_storage.end_idx_tag_without_content:
        metadata_text_list = metadata_idx_storage.end_idx_tag_without_content[idx]
        _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

    if idx in metadata_idx_storage.start_idx_tag_with_content:
        metadata_text_list = metadata_idx_storage.start_idx_tag_with_content[idx]
        _add_metadata_to_text(metadata_text_list, text_with_local_metadata, metadata_mask)

    return "".join(text_with_local_metadata), metadata_mask


def chunks(n: int, *lists):
    """Yield successive n-sized chunks from the provided lists."""
    assert n > 0, f"Chunk size must be positive, got {n}"
    assert lists, "At least one list must be given."
    assert len(set(len(lst) for lst in lists)) == 1, "All lists must be of the same size."
    for i in range(0, len(lists[0]), n):
        yield (lst[i : i + n] for lst in lists)
