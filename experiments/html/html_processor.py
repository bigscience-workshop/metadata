from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.metadata_processors import MetadataProcessor


@dataclass
class TagToRemove:
    tag: str
    txt_min_chr_len: int = 0
    txt_max_chr_len: int = float("inf")


@dataclass
class HtmlTag:
    tag: str
    attrs: dict


@dataclass
class Metadata:
    char_start_idx: int
    value: HtmlTag
    char_end_idx: Optional[int] = None
    key: str = "html"
    type: str = "local"


@dataclass
class AllTagsRules:
    attributes_to_keep: List[str] = field(default_factory=(lambda: []), metadata={"help": "TODO."})
    txt_max_chr_len: float = field(default=-float("inf"), metadata={"help": "TODO."})
    txt_min_chr_len: float = field(default=-float("inf"), metadata={"help": "TODO."})
    tags_exceptions_to_txt_max_min_chr_len: List[str] = field(default_factory=(lambda: []), metadata={"help": "TODO."})


@dataclass
class HTMLParserConfig:
    all_tags_rules: AllTagsRules = AllTagsRules()
    tags_to_remove_alone_tag_name: List[str] = field(
        default_factory=(lambda: []),
        metadata={"help": "TODO."},
    )
    tags_to_remove_alone_txt_max_chr_len: List[float] = field(
        default_factory=(lambda: []),
        metadata={"help": "TODO."},
    )
    tags_to_remove_alone_txt_min_chr_len: List[float] = field(
        default_factory=(lambda: []),
        metadata={"help": "TODO."},
    )


class TagFilter:
    def __init__(
        self,
        txt_max_chr_len: Optional[float] = -float("inf"),
        txt_min_chr_len: Optional[float] = -float("inf"),
        tags_exceptions: Optional[List[str]] = None,
        tags_to_remove_alone: Optional[List[TagToRemove]] = None,
    ):
        self.tags_to_remove_alone = (
            {tag_to_remove.tag: tag_to_remove for tag_to_remove in tags_to_remove_alone}
            if isinstance(tags_to_remove_alone, list)
            else {}
        )
        self.txt_max_chr_len = txt_max_chr_len
        self.txt_min_chr_len = txt_min_chr_len
        self.tags_exceptions = tags_exceptions if tags_exceptions else []

    def drop_tag(self, metadata_node):
        tag = str(metadata_node.value.tag)

        drop_tag = False
        content_char_length = (
            metadata_node.char_end_idx - metadata_node.char_start_idx if metadata_node.char_end_idx is not None else 0
        )
        if tag in self.tags_to_remove_alone:
            tag_to_remove_characteristics = self.tags_to_remove_alone[tag]
            if (
                content_char_length <= tag_to_remove_characteristics.txt_max_chr_len
                and content_char_length >= tag_to_remove_characteristics.txt_min_chr_len
            ):
                drop_tag = True

        if tag not in self.tags_exceptions:
            if content_char_length <= self.txt_max_chr_len and content_char_length >= self.txt_min_chr_len:
                drop_tag = True

        # raise TypeError(f"tag need to be a string not a {type(tag)}")
        return drop_tag


class HtmlProcessor(MetadataProcessor):
    """An example metadata processor for HTMl tags."""

    def __init__(
        self,
        cfg: DataConfig,
    ):
        """
        Args:
            cfg: The data configuration to use.
        """
        super().__init__(cfg)
        attributes_to_keep = cfg.html_parser_config.all_tags_rules.attributes_to_keep
        txt_max_chr_len = cfg.html_parser_config.all_tags_rules.txt_max_chr_len
        txt_min_chr_len = cfg.html_parser_config.all_tags_rules.txt_min_chr_len
        tags_exceptions = cfg.html_parser_config.all_tags_rules.tags_exceptions_to_txt_max_min_chr_len
        tags_to_remove_alone = [
            TagToRemove(tag=tag, txt_max_chr_len=txt_max_chr_len, txt_min_chr_len=txt_min_chr_len)
            for (tag, txt_max_chr_len, txt_min_chr_len) in zip(
                cfg.html_parser_config.tags_to_remove_alone_tag_name,
                cfg.html_parser_config.tags_to_remove_alone_txt_max_chr_len,
                cfg.html_parser_config.tags_to_remove_alone_txt_min_chr_len,
            )
        ]

        self._tag_filter = TagFilter(
            tags_to_remove_alone=tags_to_remove_alone,
            txt_min_chr_len=txt_min_chr_len,
            txt_max_chr_len=txt_max_chr_len,
            tags_exceptions=tags_exceptions,
        )
        self._attributes_to_keep = attributes_to_keep

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent a html tag `T` by enclosing the corresponding text span with "<T>" and "</T>".
        # Example: An <b>apple</b> is an edible fruit.
        if self._tag_filter.drop_tag(
            Metadata(
                char_start_idx=metadata_attrs["char_start_idx"],
                value=HtmlTag(tag=metadata_attrs["value"], attrs=metadata_attrs["html_attrs"]),
                char_end_idx=metadata_attrs["char_end_idx"],
                key=metadata_attrs["key"],
                type=metadata_attrs["type"],
            )
        ):
            return None

        attributes = " ".join(
            f'{attr}="{value}"'
            for attr, value in zip(metadata_attrs["value"]["attrs"]["attr"], metadata_attrs["value"]["attrs"]["value"])
            if (self._attributes_to_keep is None or attr in self._attributes_to_keep)
        )
        if attributes:
            attributes = " " + attributes
        return f"<{metadata_attrs['value']['tag']}{attributes}>", f"</{metadata_attrs['value']['tag']}>"
