import sys
from dataclasses import dataclass

from html_processor import AllTagsRules, HTMLParserConfig, HtmlProcessor, TagToRemove
from hydra.core.config_store import ConfigStore

from bsmetadata.input_pipeline import DataConfig, MetadataConfig
from bsmetadata.metadata_processors import PROCESSORS
from bsmetadata.train import CFG, main, show_help


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
class MetadataConfigWithHTML(MetadataConfig):
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
class CFGAugmented(CFG):
    data_config: DataConfig = DataConfig(metadata_config=MetadataConfigWithHTML())


cs = ConfigStore.instance()
cs.store(name="config", node=CFGAugmented)

if __name__ == "__main__":
    if "--help" in sys.argv or "-h" in sys.argv:
        show_help()
        sys.exit()
    main()
