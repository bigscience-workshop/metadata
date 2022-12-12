import argparse
import functools
from typing import Dict

import torch
import torch.nn.functional as F
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from omegaconf import OmegaConf
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator
from transformers.modeling_outputs import CausalLMOutputWithCrossAttentions

from bsmetadata.metadata_utils import add_metadata_and_chunk_examples


@torch.no_grad()
def ppl_fn(
    batch: Dict[str, torch.Tensor], outputs: CausalLMOutputWithCrossAttentions, metadata_mask: torch.Tensor = None
) -> torch.Tensor:
    """Calculates the perplexity for a given batch.

    Args:
        batch: A dict with keys "input_ids" and "attention_mask".
        outputs: The model outputs for the batch.
        metadata_mask: 1 for tokens corresponding to metadata and 0 for all other tokens.

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
        loss_mask = torch.logical_and(attention_mask, ~metadata_mask)
    else:
        loss_mask = attention_mask

    shift_mask = loss_mask[..., 1:].contiguous()

    # Flatten the tokens
    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        reduction="none",
    ).view(b, -1)

    # Normalize to avoid an overflow when there are many tokens
    normed_loss_weights = shift_mask / shift_mask.sum()
    loss = (loss * normed_loss_weights).sum()

    # Per-example ppl
    ppl = torch.exp((loss * shift_mask).sum(-1) / shift_mask.sum(-1))

    return ppl


@torch.no_grad()
def get_ppl(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Prepares the arguments for perplexity calculation and passes them to the perplexity function.

    Args:
        batch: A dict with keys "input_ids", "attention_mask" and "metadata_mask", where:
            - the input ids are a list of token ids corresponding to the input text with metadata;
            - the attention mask is 0 for padding tokens and 1 everywhere else;
            - the metadata mask is 1 for tokens corresponding to metadata and 0 for all other tokens.

    Returns:
        The perplexity of the given batch.
    """
    labels = batch.pop("labels")
    metadata_mask = batch.pop("metadata_mask", None)
    outputs = model(**batch)
    batch["labels"] = labels
    ppl = ppl_fn(batch, outputs, metadata_mask)
    return ppl


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

    args = parser.parse_args()
    print(f"Parameters: {args}")

    # Load config
    config_file_path = hf_hub_download(repo_id=args.repo_id, filename="actual_config.yaml", use_auth_token=True)
    repo_args = OmegaConf.load(config_file_path)
    data_config = repo_args.data_config

    # Load model
    print("Loading model...")
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

        for example in tqdm(validation_dataset, desc=f"Calculating perplexity for {metadata_type}..."):
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
            metadata_example_len = len(metadata_example["input_ids"])
            min_seq_len = min(normal_example_len, metadata_example_len)
            max_seq_len = max(normal_example_len, metadata_example_len)

            # For fair comparison, only choose
            # 1) processed_examples with exactly one example
            # 2) examples fitting the model sequence length
            if len(processed_examples["input_ids"]) == 1 and min_seq_len > 0 and max_seq_len <= 1024:
                # Keep track of considered examples and total length
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

                # Calculate ppl
                normal_ppl = get_ppl(normal_batch)
                total_normal_ppl += float(normal_ppl) * normal_example_len
                metadata_ppl = get_ppl(metadata_batch)
                total_metadata_ppl += float(metadata_ppl) * metadata_example_len

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
