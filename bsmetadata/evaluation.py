# %%writefile bsmetadata/evaluation.py
import argparse
import functools
import gc
import itertools
import os
from typing import Any, Dict, List, Optional

import rich
import torch
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from huggingface_hub import dataset_info, hf_hub_download
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


# Raw validation datasets ordered by approximate size after deduplicating and chunking,
# so the dataset augmentation and intersection may be slightly more efficient.
VLD_DS_IDS = [
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_timestamp",  # 3341
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_website_desc",  # 3585
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_title",  # 35578
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_entity",
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_entity_paragraph",  # 36856
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_paragraph",
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_html",  # 37849
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_datasource",  # 37862
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_length_text",  # 37862
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_generation_length_sentence",
    "bs-modeling-metadata/c4-en-html-with-validation_metadata_url",
]


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
def mean_loss_fn(
    batch: Dict[str, torch.Tensor],
    outputs: CausalLMOutputWithCrossAttentions,
    metadata_mask: torch.Tensor = None,
    save_data: bool = False,
    idx: int = None,
    additional_special_token_ids: List[int] = None,
) -> torch.Tensor:
    """Calculates the perplexity for a given batch.

    Args:
        batch: A dict with keys "input_ids" and "attention_mask".
        outputs: The model outputs for the batch.
        metadata_mask: 1 for tokens corresponding to metadata and 0 for all other tokens.
        save_data: Whether to tokens & losses.
        idx: The index of the batch.

    Returns:
        The normalized loss of the given batch.
    """
    b = outputs.logits.size(0)
    lm_logits = outputs.logits

    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    if metadata_mask is not None:
        # Only patch special tokens when metadata is on
        for special_token_id in additional_special_token_ids:
            shift_logits[:, :, special_token_id] = float("-inf")

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
    # -log(softmax(-inf)) approaches to inf,
    # So temporarily convert it to the biggest float,
    # And then the normalization below with shift_mask will make it 0
    loss = torch.nan_to_num(loss)

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

    if save_data:
        # Save the non-masked tokens & their loss
        suffix = "_meta" if metadata_mask is not None else ""
        torch.save(
            {
                "loss": loss,
                "shift_mask": shift_mask,
                "input_ids": batch["input_ids"],
                "attention_mask": attention_mask,
                "metadata_mask": metadata_mask,
            },
            f"{idx}_data{suffix}.pt",
        )

    # Normalize to avoid an overflow when there are many tokens
    normed_loss_weights = shift_mask / shift_mask.sum()
    loss = (loss * normed_loss_weights).sum()

    # Per-example ppl
    # ppl = torch.exp((loss * shift_mask).sum(-1) / shift_mask.sum(-1))

    return loss, shift_mask.sum()


@torch.no_grad()
def get_mean_loss(
    batch: Dict[str, torch.Tensor],
    save_data: bool = False,
    idx: int = None,
    additional_special_token_ids: List[int] = None,
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
        The normalized loss of the given batch.
    """
    labels = batch.pop("labels")
    metadata_mask = batch.pop("metadata_mask", None)
    outputs = model(**batch)
    batch["labels"] = labels
    nll = mean_loss_fn(
        batch,
        outputs,
        metadata_mask,
        save_data=save_data,
        idx=idx,
        additional_special_token_ids=additional_special_token_ids,
    )
    return nll


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


def aug_examples_by_max_seq_len(
    examples,
    fn_add_metadata_and_chunk_examples,
    tokenizer,
    cfg,
    include_chunked_examples,
):
    texts = []
    metadatae = []
    normal_examples = []
    metadata_examples = []

    texts_append = texts.append
    metadatae_append = metadatae.append
    normal_examples_append = normal_examples.append
    metadata_examples_append = metadata_examples.append

    if include_chunked_examples:
        alt_cfg = cfg.copy()
        alt_cfg.without_metadata_same_context = True

    check_metadata_mask = False
    if "metadata" in examples:
        for mtdt_record in examples["metadata"][0]:
            if "html" == mtdt_record["key"]:
                check_metadata_mask = True
                break
    elif "metadata_html" in examples:
        check_metadata_mask = True

    n_examples = len(next(iter(examples.values())))
    for example_idx in range(n_examples):
        # fn_add_metadata_and_chunk_examples implicitly does convert_v2_dataset_to_v1_format_v1_compatible without retaining it,
        # so explicitly doing it here to make its call inside fn_add_metadata_and_chunk_examples a no-op.
        example = convert_v2_dataset_to_v1_format_v1_compatible({k: v[example_idx] for k, v in examples.items()})
        boxed_example = {k: [v] for k, v in example.items()}
        annotated_chunks = fn_add_metadata_and_chunk_examples(boxed_example, tokenizer, cfg)
        if include_chunked_examples:
            is_new_text = True
            plain_chunks = None
            for chunk_idx, metadata_mask in enumerate(annotated_chunks["metadata_mask"]):
                if chunk_idx > 0:  # The look is designed to add more than one chunk, but the condition turns it off.
                    break
                if not check_metadata_mask or not all(metadata_mask):
                    texts_append(example["text"])
                    if is_new_text:
                        plain_chunks = fn_add_metadata_and_chunk_examples(boxed_example, tokenizer, alt_cfg)
                        metadatae_append(example["metadata"])
                        is_new_text = False
                    else:
                        metadatae_append([])
                    normal_examples_append(
                        {
                            "input_ids": plain_chunks["input_ids"][chunk_idx],
                            "attention_mask": plain_chunks["attention_mask"][chunk_idx],
                        }
                    )
                    metadata_examples_append(
                        {
                            "input_ids": annotated_chunks["input_ids"][chunk_idx],
                            "attention_mask": annotated_chunks["attention_mask"][chunk_idx],
                            "metadata_mask": annotated_chunks["metadata_mask"][chunk_idx],
                        }
                    )
        else:
            # For fair comparison, only choose
            # 1) annotated_chunks with exactly one chunk, which implies
            # 2) the example fitting the model sequence length.
            annotated_1st_chunk = {k: v[0] for k, v in annotated_chunks.items()}
            if len(annotated_chunks["input_ids"]) == 1 and (
                not check_metadata_mask or not all(annotated_1st_chunk["metadata_mask"])
            ):
                texts_append(example["text"])
                metadatae_append(example["metadata"])
                plain_example = tokenizer(example["text"])
                normal_examples_append(
                    {
                        "input_ids": plain_example["input_ids"],
                        "attention_mask": plain_example["attention_mask"],
                    }
                )
                metadata_examples_append(
                    {
                        "input_ids": annotated_1st_chunk["input_ids"],
                        "attention_mask": annotated_1st_chunk["attention_mask"],
                        "metadata_mask": annotated_1st_chunk["metadata_mask"],
                    }
                )
    gc.collect(0)
    return {
        "text": texts,
        "metadata": metadatae,
        "normal_example": normal_examples,
        "metadata_example": metadata_examples,
    }


def aug_raw_vld_ds(raw_vld_ds, augmented_vld_ds_id, augment_examples_fn):
    print(f"Building {augmented_vld_ds_id}...")
    augmented_vld_ds = raw_vld_ds.map(
        augment_examples_fn,
        remove_columns=raw_vld_ds.column_names,
        batched=True,
        batch_size=64,
        num_proc=len(os.sched_getaffinity(0)),
    )
    print(f"#rows: {len(raw_vld_ds)} -> {len(augmented_vld_ds)}")
    return augmented_vld_ds


def dedupe_raw_vld_ds(raw_vld_ds):
    print("Deduplicating...")
    orig_len = len(raw_vld_ds)
    raw_vld_df = raw_vld_ds.to_pandas()
    del raw_vld_ds
    raw_vld_df.drop_duplicates(subset="text", inplace=True)
    raw_vld_ds = Dataset.from_pandas(raw_vld_df, split="validation", preserve_index=False)
    del raw_vld_df
    print(f"#rows: {orig_len} -> {len(raw_vld_ds)}")
    gc.collect()
    return raw_vld_ds


def shorten_metadata_name(orig_name):
    words = orig_name.split("_")
    if len(words) == 1:
        return words[0][0] + words[0][2]
    else:
        return "".join(word[0] for word in words)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo_id",
        type=str,
        default="bs-modeling-metadata/checkpoints_all_04_23",
        help="Repository ID for the model to compute perplexity for",
    )
    parser.add_argument(
        "--subfolder",
        type=str,
        default="checkpoint-2500step",
        help="subfolder in the respository with the specific checkpoint to evaluate perplexity for",
    )
    parser.add_argument(
        "--config_file_path",
        type=str,
        help="The path actual_config.yaml if available, otherwise repo_id/actual_config.yaml or git clone's v2.yaml",
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation.txt", help="Path to the file the perplexity is written to"
    )
    parser.add_argument("--no_cuda", action="store_true", help="If set to true, all computations are performed on CPU")
    parser.add_argument(
        "--save_data",
        action="store_true",
        help="If set to true, save tokens & losses",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="If set to true, the script runs in test mode and only takes 10 examples per dataset",
    )
    parser.add_argument(
        "--max_n_examples",
        type=int,
        default=1500,
        help="how many examples per metadata type to evaluate",
    )
    parser.add_argument(
        "--metadata_to_test",
        type=str,
        default="html,entity_paragraph,website_desc,generation_datasource,timestamp,title,generation_length_text",
        help="metadata types to test",
    )
    parser.add_argument(
        "--test_aforesaid_metadata_together",
        action="store_true",
        help="If true, it will test all of designated (by --metadata_to_test) types per example at once",
    )
    parser.add_argument(
        "--include_chunked_examples",
        action="store_true",
        help="If true, an example longer than max_seq_len will be included with the same text chunks",
    )
    parser.add_argument("--dedupe", action="store_true", help="If true, drop duplicated text")
    parser.add_argument(
        "--untrained",
        action="store_true",
        help="If set to true, will load --untrained_model_name (default to gpt2-xl)",
    )
    parser.add_argument(
        "--untrained_model_name",
        type=str,
        default="gpt2-xl",
        help="If --untrained, will load this model and its tokenizer; mostly for --prompt",
    )
    parser.add_argument(
        "--prompt",
        action="store_true",
        help="If set to true, the script evaluates metadata in prompt style",
    )

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # Load config
    if args.config_file_path:
        config_file_path = args.config_file_path
    else:
        try:
            config_file_path = hf_hub_download(
                repo_id=args.repo_id, filename="actual_config.yaml", use_auth_token=True
            )
        except Exception:
            config_file_path = "bsmetadata/hydra_configs/v2.yaml"
    repo_args = OmegaConf.load(config_file_path)
    data_config = repo_args.data_config

    # make sure loss (ppl) masking is on for local metadata
    data_config.metadata_config.treat_local_metadata_as_regular_text = False

    # Load tokenizer
    if args.untrained:
        tokenizer = AutoTokenizer.from_pretrained(args.untrained_model_name)
        tokenizer.pad_token = tokenizer.eos_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            args.repo_id,  # "bs-modeling-metadata/checkpoints_all_04_23" by default but it can change
            subfolder="tokenizer",
            use_auth_token=True,
        )

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

    #     cfg.max_seq_len = 2048
    augment_examples_fn = functools.partial(
        aug_examples_by_max_seq_len,
        fn_add_metadata_and_chunk_examples=add_metadata_and_chunk_examples,
        tokenizer=tokenizer,
        cfg=cfg,
        include_chunked_examples=args.include_chunked_examples,
    )

    # This preserves the order of VLD_DS_IDS such that larger datasets go first
    mtdt2test_name_id_pairs = [
        (ds_id.split("_metadata_")[1], ds_id)
        for ds_id in VLD_DS_IDS
        if ds_id.split("_metadata_")[1] in args.metadata_to_test.split(",")
    ]
    tknzr_id = args.untrained_model_name if args.untrained else args.repo_id
    tknzr_alias = f"-tknzr{cfg.max_seq_len//1024}k_" + tknzr_id.split("/")[-1]
    deduped = "-deduped" if args.dedupe else ""
    selection = "-plus_chunked" if args.include_chunked_examples else ""
    prefix = "bs-modeling-metadata/c4-en-htm-vld-"
    vld_ds_name_ids_dict = {
        mtdt_name: (ds_id, f"{prefix}{shorten_metadata_name(mtdt_name)}{tknzr_alias}{deduped}{selection}")
        for (mtdt_name, ds_id) in mtdt2test_name_id_pairs
    }
    augmented_merged_vld_ds_id = (
        prefix
        + "_".join([shorten_metadata_name(mtdt_name) for (mtdt_name, _) in mtdt2test_name_id_pairs])
        + f"{tknzr_alias}{deduped}{selection}"
    )

    if args.test_aforesaid_metadata_together:
        try:
            print(f"Checking existence of {augmented_merged_vld_ds_id}...")
            dataset_info(augmented_merged_vld_ds_id)
            is_merged = True
        except Exception:
            is_merged = False

    augmented_vld_dfs = []
    for mtdt_name, (raw_vld_ds_id, augmented_vld_ds_id) in vld_ds_name_ids_dict.items():
        try:
            print(f"Checking existence of {augmented_vld_ds_id}...")
            dataset_info(augmented_vld_ds_id)
        except Exception:
            raw_vld_ds = load_dataset(raw_vld_ds_id, split="validation", use_auth_token=True)
            cols_to_remove = [
                col for col in raw_vld_ds.column_names if col != "text" and not col.startswith("metadata_")
            ]
            raw_vld_ds = raw_vld_ds.remove_columns(cols_to_remove)
            print(f"Processing {raw_vld_ds}\n -> {augmented_vld_ds_id}")
            if args.dedupe:
                raw_vld_ds = dedupe_raw_vld_ds(raw_vld_ds)
            augmented_vld_ds = aug_raw_vld_ds(raw_vld_ds, augmented_vld_ds_id, augment_examples_fn)
            print(f"Pushing {augmented_vld_ds}\n -> {augmented_vld_ds_id}...")
            augmented_vld_ds.push_to_hub(augmented_vld_ds_id, split="validation", private=True)
            del raw_vld_ds
            gc.collect()
        if args.test_aforesaid_metadata_together and not is_merged and len(mtdt2test_name_id_pairs) > 1:
            print(f"Loading {augmented_vld_ds_id},,,")
            augmented_vld_ds = load_dataset(augmented_vld_ds_id, split="validation", use_auth_token=True)
            print(augmented_vld_ds)

            print(f"Converting {augmented_vld_ds_id} to Pandas for merging...")
            augmented_vld_dfs.append(
                augmented_vld_ds.remove_columns(
                    [
                        "normal_example",
                        "metadata_example",
                    ]
                )
                .to_pandas()
                .drop_duplicates(subset="text")  # The intersection is always deduped, otherwise it becomes artifical.
            )
            del augmented_vld_ds
        gc.collect()

    if args.test_aforesaid_metadata_together:
        if not is_merged:
            print(f"Merging {augmented_merged_vld_ds_id}...")
            left_df = augmented_vld_dfs.pop()
            while augmented_vld_dfs:
                right_df = augmented_vld_dfs.pop()
                left_df = left_df.merge(right_df, on="text")
                try:
                    metadata_x_s = left_df["metadata_x"].apply(lambda l: l.tolist())
                except Exception:
                    metadata_x_s = left_df["metadata_x"]
                metadata_y_s = left_df["metadata_y"].apply(lambda l: l.tolist())

                left_df["metadata"] = metadata_x_s + metadata_y_s
                del metadata_x_s, metadata_y_s
                left_df = left_df.drop(columns=["metadata_x", "metadata_y"])
                gc.collect()
            print(f"#rows: {len(left_df)}")
            raw_vld_ds = Dataset.from_pandas(left_df, split="validation", preserve_index=False)
            del left_df
            gc.collect()
            augmented_merged_vld_ds = aug_raw_vld_ds(raw_vld_ds, augmented_merged_vld_ds_id, augment_examples_fn)
            del raw_vld_ds

            print(f"Pushing {augmented_merged_vld_ds}\n -> {augmented_vld_ds_id}...")
            augmented_merged_vld_ds.push_to_hub(augmented_merged_vld_ds_id, split="validation", private=True)
            del augmented_merged_vld_ds
            gc.collect()
        merged_mtdt_names = " ⋂ ".join(vld_ds_name_ids_dict.keys())
        vld_ds_name_ids_dict[merged_mtdt_names] = ("", augmented_merged_vld_ds_id)

    # Load model
    if args.untrained:
        print(f"Loading model {args.untrained_model_name}...")
        model = AutoModelForCausalLM.from_pretrained(args.untrained_model_name)
    else:
        print(f"Loading model {args.repo_id}/{args.subfolder}...")
        model = AutoModelForCausalLM.from_pretrained(args.repo_id, subfolder=args.subfolder, use_auth_token=True)
    model.eval().cuda() if not args.no_cuda else model.eval()

    torch.set_printoptions(threshold=cfg.max_seq_len)  # For debugging

    for mtdt_name, (_, ds_id) in vld_ds_name_ids_dict.items():
        #         if mtdt_name != merged_mtdt_names:
        #             continue

        total_normal_len = []
        total_normal_nll = []
        total_metadata_len = []
        total_metadata_nll = []

        # Load validation dataset from hugging face
        print(f"Loading {mtdt_name}\n@ {ds_id}...")
        n_examples = args.max_n_examples if not args.test else 10
        vld_ds = load_dataset(ds_id, use_auth_token=True, split="validation")
        vls_ds_len = len(vld_ds)
        print(f"{vls_ds_len} loaded for {mtdt_name}\n@ {ds_id}")
        n_examples = min(vls_ds_len, n_examples)
        #         vld_ds = vld_ds.select(range(n_examples))
        #         print(f"{n_examples} selected")

        print(f"Computing PPL with {mtdt_name}...")
        data = []
        example_cnt = 0
        for idx, example in tqdm(enumerate(vld_ds), total=n_examples):
            # Prepare batches
            normal_example = example["normal_example"]
            normal_example["labels"] = normal_example["input_ids"]
            normal_batch = default_data_collator([normal_example])

            metadata_example = example["metadata_example"]
            metadata_example["labels"] = metadata_example["input_ids"]
            metadata_batch = default_data_collator([metadata_example])

            if not args.no_cuda:
                normal_batch = {k: v.cuda() for k, v in normal_batch.items()}
                metadata_batch = {k: v.cuda() for k, v in metadata_batch.items()}

            # Calculate nll (natural-log loss)
            normal_nll, normal_example_len = get_mean_loss(
                normal_batch,
                save_data=args.save_data,
                idx=idx,
            )
            metadata_nll, metadata_example_len = get_mean_loss(
                metadata_batch,
                save_data=args.save_data,
                idx=idx,
                additional_special_token_ids=tokenizer.additional_special_tokens_ids,
            )

            # Debug
            if torch.isnan(normal_nll):
                print(f"NaN: normal_example[{idx}]")
                print("Highlighted by attention_mask:")
                rich.print(
                    format_by_one_mask(
                        normal_example["input_ids"],
                        normal_example["attention_mask"],
                        tokenizer,
                    )
                )
            elif torch.isnan(metadata_nll):
                print(f"NaN: metadata_examples[{idx}] for {mtdt_name}")
                print("Highlighted by metadata_mask:")
                rich.print(
                    format_by_one_mask(
                        metadata_example["input_ids"],
                        metadata_example["metadata_mask"],
                        tokenizer,
                    )
                )
                print("Highlighted by attention_mask:")
                rich.print(
                    format_by_one_mask(
                        metadata_example["input_ids"],
                        metadata_example["attention_mask"],
                        tokenizer,
                    )
                )
            else:
                example_cnt += 1
                total_normal_nll.append(normal_nll)
                total_normal_len.append(normal_example_len)
                total_metadata_nll.append(metadata_nll)
                total_metadata_len.append(metadata_example_len)
                data.append({"idx": idx, "normal_nll": normal_nll, "metadata_nll": metadata_nll})

            if example_cnt == n_examples:
                break

        torch.save(
            {
                "total_normal_nll": total_normal_nll,
                "total_metadata_nll": total_metadata_nll,
                "total_normal_len": total_normal_len,
                "total_metadata_len": total_metadata_len,
            },
            "eva.data2",
        )

        # Get average ppl weighted by token sequence length
        def ppl(examples_mean_loss, examples_len):
            examples_mean_loss = torch.tensor(examples_mean_loss)
            examples_len = torch.tensor(examples_len)
            weight = examples_len / examples_len.sum()
            return torch.exp((examples_mean_loss * weight).sum()).item()

        final_normal_ppl = ppl(total_normal_nll, total_normal_len)
        final_metadata_ppl = ppl(total_metadata_nll, total_metadata_len)

        # Write results to output file
        with open(args.output_file, "a", encoding="utf8") as f:
            f.write(f"=== RESULT [{mtdt_name}{tknzr_alias}{deduped}{selection}]({example_cnt}) ===\n")
            f.write("Perplexity (metadata): {:>6,.3f}\n".format(final_metadata_ppl))
            f.write("Perplexity (normal):   {:>6,.3f}\n\n".format(final_normal_ppl))
        torch.save(data, "eva.data")
