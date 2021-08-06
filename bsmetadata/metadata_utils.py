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
import random
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from transformers import PreTrainedTokenizerFast

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.metadata_processors import PROCESSORS, MetadataProcessor


def add_metadata_and_chunk_examples(
    examples: Dict[str, List], tokenizer: PreTrainedTokenizerFast, cfg: DataConfig
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
            global_metadata_prefix = create_global_metadata_prefix(example, cfg)
            global_metadata_prefix_encoded = tokenizer.encode_plus(global_metadata_prefix).input_ids
        else:
            global_metadata_prefix_encoded = []

        if add_metadata:
            # Get the actual text with local metadata inserted.
            text_with_local_metadata, char_level_metadata_mask = add_local_metadata_to_text(example, cfg)

        else:
            text_with_local_metadata = example["text"]
            char_level_metadata_mask = [False] * len(text_with_local_metadata)

        text_with_local_metadata = " " + text_with_local_metadata
        char_level_metadata_mask = [False] + char_level_metadata_mask
        text_with_local_metadata_encoded = tokenizer.encode_plus(text_with_local_metadata)

        def is_metadata(idx: int) -> bool:
            char_span = text_with_local_metadata_encoded.token_to_chars(idx)
            char_range = range(char_span.start, char_span.end)
            return any(char_level_metadata_mask[c] for c in char_range)

        token_level_metadata_mask = [
            is_metadata(idx) for idx, _ in enumerate(text_with_local_metadata_encoded.input_ids)
        ]

        # Create chunks of `max_seq_len` tokens.
        global_metadata_len = len(global_metadata_prefix_encoded)
        max_text_len = cfg.max_seq_len - global_metadata_len

        for text_chunk_encoded, chunk_metadata_mask in chunks(
            max_text_len, text_with_local_metadata_encoded.input_ids, token_level_metadata_mask
        ):
            total_len = len(global_metadata_prefix_encoded) + len(text_chunk_encoded)
            padding_len = max_text_len - len(text_chunk_encoded)

            input_ids = global_metadata_prefix_encoded + text_chunk_encoded + [tokenizer.eos_token_id] * padding_len
            attention_mask = [1] * total_len + [0] * padding_len
            metadata_mask = [1] * global_metadata_len + [int(x) for x in chunk_metadata_mask] + [0] * padding_len

            linearized_examples["input_ids"].append(input_ids)
            linearized_examples["attention_mask"].append(attention_mask)
            linearized_examples["metadata_mask"].append(metadata_mask)

    return linearized_examples


def create_global_metadata_prefix(example: Dict[str, Any], cfg: DataConfig) -> str:
    """Creates a prefix containing all global metadata information (including URLs, timestamps, etc).

    Args:
        example: The example to create a global metadata prefix for.
        cfg: The data config to use.

    Returns:
        A string containing the global metadata prefix.
    """
    processed_metadata = {}
    for metadata in example["metadata"]:
        key, type_ = metadata["key"], metadata["type"]
        if type_ != "global" or key not in cfg.metadata_list:
            continue

        processor = PROCESSORS.get(key, MetadataProcessor)(cfg)
        processed_metadata[key] = processor.process_global(metadata)

    sorted_metadata = [processed_metadata.get(md, None) for md in cfg.metadata_list]
    sorted_metadata = [md for md in sorted_metadata if md is not None]
    return cfg.metadata_sep.join(sorted_metadata) + cfg.global_metadata_sep


def add_local_metadata_to_text(example: Dict[str, Any], cfg: DataConfig) -> Tuple[str, List[bool]]:
    """Adds local metadata (such as HTML tags and entity names) to the given input text.

    Args:
        example: The example for which local metadata should be added.
        cfg: The data config to use.

    Returns:
        A tuple of two elements, where:
            - the first element is the text with metadata;
            - the second element is a boolean mask where `mask[i]` is set iff `text[i]` is some kind of metadata.
    """
    metadata_start_texts, metadata_end_texts = defaultdict(list), defaultdict(list)

    # Filter and sort all metadata so that they are processed in the requested order.
    filtered_metadata = [md for md in example["metadata"] if md["type"] == "local" and md["key"] in cfg.metadata_list]
    sorted_metadata = sorted(
        filtered_metadata, key=lambda md: (cfg.metadata_list.index(md["key"]), md["char_end_idx"])
    )

    # Compute the text sequences to add at the start and end of each metadata entry.
    for metadata in sorted_metadata:
        processor = PROCESSORS.get(metadata["key"], MetadataProcessor)(cfg)
        processed_metadata = processor.process_local(metadata)
        if processed_metadata is None:
            continue
        start_text, end_text = processed_metadata
        char_start_idx = metadata.get("char_start_idx", -1)
        char_end_idx = metadata.get("char_end_idx", -1)

        metadata_start_texts[char_start_idx].insert(0, start_text)
        metadata_end_texts[char_end_idx].append(end_text)

    # Build the final text with local metadata and the corresponding mask.
    text_with_local_metadata = []
    metadata_mask = []

    for idx, char in enumerate(example["text"]):

        if idx in metadata_start_texts:
            for start_text in metadata_start_texts[idx]:
                text_with_local_metadata.append(start_text)
                metadata_mask += [True] * len(start_text)

        text_with_local_metadata.append(char)
        metadata_mask += [False]

        if idx + 1 in metadata_end_texts:
            for end_text in metadata_end_texts[idx + 1]:
                text_with_local_metadata.append(end_text)
                metadata_mask += [True] * len(end_text)

    return "".join(text_with_local_metadata), metadata_mask


def chunks(n: int, *lists):
    """Yield successive n-sized chunks from the provided lists."""
    assert n > 0, f"Chunk size must be positive, got {n}"
    assert lists, "At least one list must be given."
    assert len(set(len(lst) for lst in lists)) == 1, "All lists must be of the same size."
    for i in range(0, len(lists[0]), n):
        yield (lst[i : i + n] for lst in lists)
