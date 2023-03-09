# %%writefile bsmetadata/evaluation.py
import argparse
import functools
import itertools
import json
from typing import Any, Dict, Optional

import rich
import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from rich.text import Text
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from bsmetadata import metadata_utils
from bsmetadata.metadata_processors import (
    PROCESSORS,
    DatasourceProcessor,
    GenerationLengthProcessor,
    MetadataConfig,
    MetadataProcessor,
)
from bsmetadata.metadata_utils import add_metadata_and_chunk_examples, convert_v2_dataset_to_v1_format_v1_compatible


def format_by_one_mask(input_ids, mask, tokenizer):
    i = 0
    data = []
    for key, igroup in itertools.groupby(mask):
        size = len(list(igroup))
        text = tokenizer.decode(input_ids[i : i + size])
        i += size
        data.append((text, "green" if key else None))
    return Text.assemble(*data)


@torch.no_grad()
def ppl_fn(
    batch: Dict[str, torch.Tensor],
    outputs: CausalLMOutputWithCrossAttentions,
    metadata_mask: torch.Tensor = None,
    save_data: bool = False,
    idx: int = None,
) -> torch.Tensor:
    """Calculates the perplexity for a given batch.

    Args:
        batch: A dict with keys "input_ids" and "attention_mask".
        outputs: The model outputs for the batch.
        metadata_mask: 1 for tokens corresponding to metadata and 0 for all other tokens.
        save_data: Whether to tokens & losses.
        idx: The index of the batch.

    Returns:
        The perplexity of the given batch.
    """
    b = outputs.logits.size(0)
    lm_logits = outputs.logits
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if metadata_mask is not None:
        metadata_mask = metadata_mask.bool()
        nonmetadata_cumsum = torch.cumsum(~metadata_mask, dim=-1)
        first_nonmetadata = nonmetadata_cumsum == 1
        # rich.print(f"{~(metadata_mask.bool())=}")
        # rich.print("(attention_mask.bool())")
        # rich.print(attention_mask.bool())
        loss_mask = torch.logical_and(attention_mask.bool(), ~(metadata_mask.bool()))
        loss_mask = torch.logical_and(loss_mask, ~first_nonmetadata)
        # rich.print(f"{loss_mask=}")
    else:
        loss_mask = attention_mask.bool()
    shift_mask = loss_mask[..., 1:].contiguous()

    """

    max len: 10
    (label, by convention, is unshifted)
    label: a b c d e f g x x x
    input: a b c d e f g x x x
    mask : 1 1 1 1 1 1 1 0 0 0

    shift label : b c d e f g x x x
    shift logit : a b c d e f g x x
    shift a mask: 1 1 1 1 1 1 0 0 0


    calculated part
    input: a b c d e f
    label: b c d e f g

    metdata example:
    label : M M a b c d e f g x
    input : M M a b c d e f g x
    a mask: 1 1 1 1 1 1 1 1 1 0
    m mask: 1 1 0 0 0 0 0 0 0 0
    a & !m: 0 0 1 1 1 1 1 1 1 0

    shift label : M a b c d e f g x
    shift logit : M M a b c d e f g
    shift a mask: 1 1 1 1 1 1 1 1 0
    shift (a&!m): 0 1 1 1 1 1 1 1 0
    diff (bug)  :   x

    # fix: mask out the loss if ((the source token is metadata) or (the target token is padding))
    #

    shift m mask:
    ideal mask :
    """

    # if metadata_mask is not None:
    #    shift_metadata_mask = metadata_mask[..., 1:].contiguous().bool()
    #    shift_mask = torch.logical_and(shift_mask, ~shift_metadata_mask)
    # rich.print(f"shift_mask{shift_mask}")
    # rich.print(f"{shift_mask.sum()=}")

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(b, -1)

    if save_data:
        # Save the non-masked tokens & their loss
        suffix = "_meta" if metadata_mask is not None else ""
        torch.save(
            batch["input_ids"],
            f"{idx}_input_ids{suffix}.pt",
        )
        torch.save(
            loss.cpu().squeeze(),
            f"{idx}_loss{suffix}.pt",
        )

    # loss = loss.cpu().squeeze().numpy().tolist()
    # shift_mask = shift_mask.cpu().squeeze().numpy().tolist()
    # return loss, shift_mask, shift_labels.cpu().squeeze().numpy().tolist()
    # return loss, shift_mask

    # Normalize to avoid an overflow when there are many tokens
    normed_loss_weights = shift_mask / shift_mask.sum()
    loss = (loss * normed_loss_weights).sum()

    # Per-example ppl
    ppl = torch.exp((loss * shift_mask).sum(-1) / shift_mask.sum(-1))

    return ppl


@torch.no_grad()
def get_ppl(
    batch: Dict[str, torch.Tensor],
    save_data: bool = False,
    idx: int = None,
) -> torch.Tensor:
    """Prepares the arguments for perplexity calculation and passes them to the perplexity function.

    Args:
        batch: A dict with keys "input_ids", "attention_mask" and "metadata_mask", where:
            - the input ids are a list of token ids corresponding to the input text with metadata;
            - the attention mask is 0 for padding tokens and 1 everywhere else;
            - the metadata mask is 1 for tokens corresponding to metadata and 0 for all other tokens.
        save_data: Whether to save tokens & losses
        idx: The index of the batch for saving
    Returns:
        The perplexity of the given batch.
    """
    labels = batch.pop("labels")
    metadata_mask = batch.pop("metadata_mask", None)
    outputs = model(**batch)
    batch["labels"] = labels
    ppl = ppl_fn(batch, outputs, metadata_mask, save_data=save_data, idx=idx)
    return ppl


def datasource_process_global_for_prompt(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
    # We represent the DATASOURCE by using meaningful information of the URL.
    # URL: http://www.example.de/2015/forum/article/21-new-project
    # Example: example.de > forum > article > new project
    return "".join(["Data source", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


def generation_length_process_global_for_prompt(self, metadata_attrs: Dict[str, Any]) -> Optional[str]:
    # We represent the length of a text by the number of characters.
    # Example: Length: 123
    return "".join(["Number of characters", self.cfg.metadata_key_value_sep, metadata_attrs["value"]])


def create_metadata_prompt(example: Dict[str, Any], cfg: MetadataConfig) -> str:
    """Creates a prompt containing all global metadata information (including URLs, timestamps, etc)
    and/or local metadata special tokens
    Args:
        example: The example to create a metadata prefix for.
        cfg: The data config to use.
    Returns:
        A string containing the metadata prefix.
    """
    LIST_LIKE_METADATA_PROMPT_FIELDS = {
        "entity": "Entities",
        "entity_paragraph": "Entity Paragraphs",
    }
    example = convert_v2_dataset_to_v1_format_v1_compatible(example=example)
    processed_metadata = {}
    for metadata in example["metadata"]:
        key, type_ = metadata["key"], metadata["type"]
        if key not in cfg.metadata_list:
            # rich.print(f"metadata key not in metadata_list, skipping. {key}, {cfg.metadata_list}")
            continue

        if type_ == "global":
            processor = PROCESSORS.get(key, MetadataProcessor)(cfg)
            processed_metadata[key] = processor.process_global(metadata)
        elif key in LIST_LIKE_METADATA_PROMPT_FIELDS:
            if key not in processed_metadata:
                processed_metadata[key] = set()  # Same entities may occurr at different positions
            processed_metadata[key].add(metadata["value"])
        elif (
            cfg.add_local_metadata_special_tokens_in_prefix
            and cfg.local_metadata_special_tokens
            and key in cfg.local_metadata_special_tokens
        ):
            processed_metadata[key] = cfg.local_metadata_special_tokens[key]
        else:
            processed_metadata[key] = key.title()

    for list_like_metadata in LIST_LIKE_METADATA_PROMPT_FIELDS:
        if list_like_metadata in processed_metadata:
            processed_metadata[list_like_metadata] = (
                LIST_LIKE_METADATA_PROMPT_FIELDS[list_like_metadata]
                + cfg.metadata_key_value_sep
                + ", ".join(v.replace("_", " ") for v in processed_metadata[list_like_metadata])
            )

    sorted_metadata = [processed_metadata.get(md, None) for md in cfg.metadata_list]
    sorted_metadata = [md for md in sorted_metadata if md is not None]
    return cfg.metadata_sep.join(sorted_metadata) + cfg.metadata_prefix_sep if sorted_metadata else ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="bs-modeling-metadata/checkpoints_v0.4",
        help="Repository ID for the model to compute perplexity for",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="checkpoint-10000step",
        help="subfolder in the respository with the specific checkpoint to evaluate perplexity for",
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation.txt", help="Path to the file the perplexity is written to"
    )
    parser.add_argument("--no_cuda", action="store_true", help="If set to true, all computations are performed on CPU")
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set to true, the script runs in test mode and only takes 10 examples per dataset",
    )
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="If set to true, save tokens & losses",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="If set to true, the script runs in test mode and only takes 10 examples per dataset",
    )
    parser.add_argument(
        "--metadata_to_test",
        type=str,
        default="html,entity,entity_paragraph,website_desc,generation_datasource,timestamp,title,generation_length_sentence,generation_length_text,url,paragraph",
        help="metadata types to test",
    )
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="If set to true, will load gpt2-xl",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="If set to true, the script evaluates metadata in prompt style",
    )

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # Load config
    if args.local:
        import os

        config_file_path = os.path.join(args.repo_id, "actual_config.yaml")
    else:
        config_file_path = hf_hub_download(repo_id=args.repo_id, filename="actual_config.yaml", use_auth_token=True)
    repo_args = OmegaConf.load(config_file_path)
    data_config = repo_args.data_config

    # make sure loss (ppl) masking is on for local metadata
    data_config.metadata_config.treat_local_metadata_as_regular_text = False

    # Load model
    print("Loading model...")
    if args.untrained:
        model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.repo_id, subfolder=args.subfolder, use_auth_token=True)
    model.eval().cuda() if not args.no_cuda else model.eval()

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(repo_args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Config preprocess function
    cfg = data_config.metadata_config
    cfg.metadata_probability = 1.0
    cfg.entity_setting = "beg"
    cfg.metadata_list.append("entity")
    cfg.metadata_list.append("paragraph")

    if args.prompt:
        cfg.metadata_sep = "; "  # Instead of " | "
        cfg.metadata_prefix_sep = ""  # Instead of " |||"; there's already an implicit " "
        DatasourceProcessor.process_global = datasource_process_global_for_prompt
        GenerationLengthProcessor.process_global = generation_length_process_global_for_prompt
        metadata_utils.create_metadata_prefix = create_metadata_prompt

    preprocess_fn = functools.partial(add_metadata_and_chunk_examples, tokenizer=tokenizer, cfg=cfg)

    # Validation datasets
    dataset_paths = [
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_html",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_entity",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_entity_paragraph",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_website_desc",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_datasource",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_timestamp",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_title",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_length_sentence",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_length_text",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_url",
        "bs-modeling-metadata/c4-en-html-with-validation_metadata_paragraph",
    ]
    dataset_paths = [path for path in dataset_paths if path.split("_metadata_")[1] in args.metadata_to_test.split(",")]

    for path in dataset_paths:
        n_examples = 0
        total_normal_len = 0
        total_normal_ppl = 0
        total_metadata_len = 0
        total_metadata_ppl = 0
        exit_flag = False

        # Load validation dataset from hugging face
        metadata_type = path.split("_metadata_")[1]
        print(f"Loading {metadata_type} data...")
        split = "validation" if not args.test else "validation[:10]"
        validation_dataset = load_dataset(path, use_auth_token=True, split=split)

        for idx, example in tqdm(enumerate(validation_dataset), desc=f"Calculating perplexity for {metadata_type}..."):
            # Preprocess examples
            examples = {k: [v] for k, v in example.items()}
            try:
                processed_examples = preprocess_fn(examples)
            except Exception as e:
                # Write error to output file and continue with next dataset
                print(e)
                with open(args.output_file, "a", encoding="utf8") as f:
                    f.write(f"=== RESULT [{metadata_type}] ===\n")
                    f.write(f"{e}\n\n")
                exit_flag = True
                break

            # Get token sequence length
            normal_example = tokenizer(examples["text"][0])
            normal_example_len = len(normal_example["input_ids"])
            metadata_example = {k: v[0] for k, v in processed_examples.items()}
            # rich.print(f"{metadata_example['attention_mask']=}")
            # rich.print(f"{normal_example['attention_mask']=}")
            # import sys
            # sys.exit()
            # print(metadata_example)
            if "input_ids" not in metadata_example:
                print("Skipping")
                continue
            metadata_example_len = len(metadata_example["input_ids"])
            min_seq_len = min(normal_example_len, metadata_example_len)
            max_seq_len = max(normal_example_len, metadata_example_len)

            # For fair comparison, only choose
            # 1) processed_examples with exactly one example
            # 2) examples fitting the model sequence length
            if len(processed_examples["input_ids"]) == 1 and min_seq_len > 0 and max_seq_len <= 1024:
                # Keep track of considered examples and total length
                if n_examples % 10 == 0:
                    print("n_examples completed.")
                n_examples += 1
                total_normal_len += normal_example_len
                total_metadata_len += metadata_example_len

                # Prepare batches
                normal_example["labels"] = normal_example["input_ids"]
                normal_batch = default_data_collator([normal_example])
                metadata_example["labels"] = metadata_example["input_ids"]
                metadata_batch = default_data_collator([metadata_example])
                if not args.no_cuda:
                    normal_batch = {k: v.cuda() for k, v in normal_batch.items()}
                    metadata_batch = {k: v.cuda() for k, v in metadata_batch.items()}
                if n_examples == 1:
                    ex = format_by_one_mask(normal_batch["input_ids"][0], normal_batch["attention_mask"][0], tokenizer)
                    # rich.print(f"Normal example:")
                    # rich.print(ex)

                    ex = format_by_one_mask(
                        metadata_batch["input_ids"][0], metadata_batch["metadata_mask"][0], tokenizer
                    )
                    # rich.print(f"Metadata example:")
                    # rich.print(ex)
                    # rich.print(tokenizer.decode(metadata_batch["input_ids"][0]))

                # Calculate ppl
                normal_ppl = get_ppl(normal_batch, save_data=args.save_data, idx=idx)  # [0]
                # print("PPL")
                # print(normal_ppl)
                total_normal_ppl += float(normal_ppl) * normal_example_len
                metadata_ppl = get_ppl(metadata_batch, save_data=args.save_data, idx=idx)  # [0]
                # print(metadata_ppl)
                total_metadata_ppl += float(metadata_ppl) * metadata_example_len
                if False:  # n_examples == 1:
                    loss, mask, shift_labels = normal_ppl
                    # print("normal ppl")
                    printed = 0
                    for i, (l, m, sl) in enumerate(zip(loss, mask, shift_labels)):
                        if m:
                            if printed < 10:
                                # rich.print(f"Loss {json.dumps(tokenizer.decode(sl))}: {l}")
                                printed += 1

                    unmasked_labels = [label for label, m in zip(shift_labels, mask) if m]
                    # print(f"first 10 unmasked labels: {[tokenizer.decode(x) for x in unmasked_labels[:10]]}")
                    # print(f"first 10 unmasked labels: {tokenizer.decode(unmasked_labels[:10])}")
                    # ex = format_by_one_mask(normal_batch["input_ids"][0], mask, tokenizer)
                    # rich.print(ex)

                    loss, mask, shift_labels = metadata_ppl
                    printed = 0
                    # print("metadata ppl")
                    for i, (l, m, sl) in enumerate(zip(loss, mask, shift_labels)):
                        if m:
                            if printed < 10:
                                # rich.print(f"Loss {json.dumps(tokenizer.decode(sl))}: {l}")
                                printed += 1

                    unmasked_labels = [label for label, m in zip(shift_labels, mask) if m]
                    # print(f"first 10 unmasked labels: {tokenizer.decode(unmasked_labels[:10])}")
                    # ex = format_by_one_mask(metadata_batch["input_ids"][0], mask, tokenizer)
                    # rich.print(ex)

                    # ex = format_by_one_mask(normal_batch["input_ids"][0], normal_batch["attention_mask"][0], tokenizer)
                    # rich.print(ex)
                    # rich.print(f"Normal example: (ppl={normal_ppl[0]})")

                    # ex = format_by_one_mask(
                    #    metadata_batch["input_ids"][0], metadata_batch["metadata_mask"][0], tokenizer
                    # )
                    # rich.print(ex)
                    # rich.print(f"Metadata example: (ppl={metadata_ppl[0]})")
                    # rich.print(f"Normal example: (mask={normal_ppl[1]})")
                    # rich.print(f"Metadata example: (mask={metadata_ppl[1]})")
                    # import sys

                    # sys.exit()

                if n_examples > 100:
                    break

        if exit_flag:
            continue

        # Get number of skipped examples
        skipped_examples = len(validation_dataset) - n_examples
        print(f"Skipped {skipped_examples} of {len(validation_dataset)} examples")

        # Get average ppl weighted by token sequence length
        if n_examples > 0:
            final_normal_ppl = total_normal_ppl / total_normal_len
            final_metadata_ppl = total_metadata_ppl / total_metadata_len
        else:
            final_metadata_ppl = final_normal_ppl = 0

        # Write results to output file
        with open(args.output_file, "a", encoding="utf8") as f:
            f.write(f"=== RESULT [{metadata_type}] ===\n")
            f.write("Perplexity (metadata): {:>6,.3f}\n".format(final_metadata_ppl))
            f.write("Perplexity (normal):   {:>6,.3f}\n\n".format(final_normal_ppl))
