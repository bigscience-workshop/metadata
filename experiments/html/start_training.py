import sys
from functools import partial

from datasets import load_dataset
from html_processor import HtmlProcessor, TagToRemove
from transformers import AutoTokenizer

from bsmetadata.experiments.with_metadata import get_dataloaders
from bsmetadata.input_pipeline import DataConfig
from bsmetadata.metadata_processors import PROCESSORS
from bsmetadata.metadata_utils import (
    add_local_metadata_to_text,
    add_metadata_and_chunk_examples,
    chunks,
    create_global_metadata_prefix,
)
from bsmetadata.train import main, show_help


tags_to_remove_alone = [
    TagToRemove("body"),
    TagToRemove("div", content_max_char_length=0),
    TagToRemove("a", content_max_char_length=0),
]
tags_table = ["table" "tr", "th", "td", "caption", "colgroup", "thead", "tfoot", "tbody"]
tags_list = [
    "li",
    "ol",
    "ul",
]
PROCESSORS["html"] = partial(
    HtmlProcessor,
    tags_to_remove_alone=tags_to_remove_alone,
    attributes_to_keep=["class", "id"],
    content_max_char_length=128,
    tags_exceptions=[
        *tags_table,
        *tags_list,
        "span",
    ],
)

# args = DataConfig(
#     train_file="/home/lucile/mini-html-parser/data/v1.0/pre-process-body-v3/nq-train-00.jsonl.gz",
#     extension="json",
#     metadata_list=["html"],
#     preprocessing_num_workers=8
# )
# tokenizer = AutoTokenizer.from_pretrained("gpt2")

# # dataloaders = get_dataloaders(tokenizer, args)

# # dataloaders

# # train_dataloader = dataloaders[0]

# # sample = next(iter(train_dataloader))
# # print(tokenizer.convert_ids_to_tokens(sample["input_ids"][0]))
# # dataset = load_dataset(args.extension, data_files=[args.train_file])

# # # dataset["train"][0]

# # examples = dataset["train"][:2]

# # output = add_metadata_and_chunk_examples(examples=examples, tokenizer=tokenizer, cfg=args)


# # print("******")
# # print(tokenizer.decode(output["input_ids"][0]))


# dataloaders = get_dataloaders(tokenizer, args)
# print(list(next(iter(dataloaders[0])).keys()))
# print(tokenizer.decode(next(iter(dataloaders[0]))["input_ids"][0]))

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
