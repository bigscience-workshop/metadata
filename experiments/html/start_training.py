import sys
from dataclasses import dataclass, field
from typing import Optional

from html_processor import AllTagsRules, HTMLParserConfig, HtmlProcessor, TagToRemove
from hydra.core.config_store import ConfigStore

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.metadata_processors import PROCESSORS
from bsmetadata.train import main, show_help


tags_to_remove_alone = [
    TagToRemove("body"),
    TagToRemove("div", txt_max_chr_len=0),
    TagToRemove("a", txt_max_chr_len=0),
]
tags_table = ["table" "tr", "th", "td", "caption", "colgroup", "thead", "tfoot", "tbody"]
tags_list = [
    "li",
    "ol",
    "ul",
]
attributes_to_keep = ["class", "id"]
txt_max_chr_len = 128
txt_min_chr_len = -float("inf")
tags_exceptions = [
    *tags_table,
    *tags_list,
    "span",
]

PROCESSORS["html"] = HtmlProcessor


@dataclass
class DataConfigWithHTML(DataConfig):
    html_parser_config: HTMLParserConfig = HTMLParserConfig(
        AllTagsRules(
            attributes_to_keep=attributes_to_keep,
            txt_max_chr_len=txt_max_chr_len,
            txt_min_chr_len=txt_min_chr_len,
            tags_exceptions_to_txt_max_min_chr_len=tags_exceptions,
        ),
        tags_to_remove_alone_tag_name=[tag_to_remove.tag for tag_to_remove in tags_to_remove_alone],
        tags_to_remove_alone_txt_max_chr_len=[tag_to_remove.txt_max_chr_len for tag_to_remove in tags_to_remove_alone],
        tags_to_remove_alone_txt_min_chr_len=[tag_to_remove.txt_min_chr_len for tag_to_remove in tags_to_remove_alone],
    )


@dataclass
class CFG:
    data_config: DataConfigWithHTML = DataConfigWithHTML()
    weight_decay: float = field(default=0.0, metadata={"help": "The weight decay to use for training."})
    learning_rate: float = field(default=5e-5, metadata={"help": "The initial learning rate."})
    gradient_accumulation_steps: int = field(
        default=1,
        metadata={"help": "The number of gradient accumulation steps to perform before updating model parameters."},
    )
    num_train_epochs: int = field(default=1, metadata={"help": "The number of epochs to train the model for."})
    max_train_steps: Optional[int] = field(
        default=None, metadata={"help": "The maximum number of training steps (overrides num_train_epochs)."}
    )
    lr_scheduler_type: str = field(default="linear", metadata={"help": "The type of learning rate schedule to use."})
    num_warmup_steps: int = field(
        default=1000, metadata={"help": "The number of warmup steps during which the learning rate is increased."}
    )
    seed: int = field(default=42, metadata={"help": "The seed used for RNG initialization."})
    out_dir: str = field(
        default="output_dir", metadata={"help": "The output directory in which the trained model is saved."}
    )
    num_eval: int = field(default=3, metadata={"help": "The number of evaluations to perform during training."})
    model_name: str = field(default="gpt2", metadata={"help": "The name of the pretrained model to use."})
    project_name: str = field(default="metadata_lm", metadata={"help": "The project name."})
    do_train: bool = field(default=True, metadata={"help": "Whether to run training."})
    do_eval: bool = field(default=True, metadata={"help": "Whether to run eval on the dev set."})


cs = ConfigStore.instance()
cs.store(name="config", node=CFG)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
