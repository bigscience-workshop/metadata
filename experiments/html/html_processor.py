import datetime
from typing import Any, Dict, Optional, Tuple, List
from urllib.parse import unquote_plus
from dataclasses import dataclass

from bsmetadata.input_pipeline import DataConfig
from bsmetadata.metadata_processors import MetadataProcessor


@dataclass
class TagToRemove:
    tag: str
    content_min_char_length: int = 0
    content_max_char_length: int = float("inf")


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


class TagFilter:
    def __init__(
        self,
        content_max_char_length: Optional[float] = float("inf"),
        content_min_char_length: Optional[float] = 0,
        tags_exceptions: Optional[List[str]] = None,
        tags_to_remove_alone: Optional[List[TagToRemove]] = None,
    ):
        self.tags_to_remove_alone = (
            {tag_to_remove.tag: tag_to_remove for tag_to_remove in tags_to_remove_alone}
            if isinstance(tags_to_remove_alone, list)
            else {}
        )
        self.content_max_char_length = content_max_char_length
        self.content_min_char_length = content_min_char_length
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
                content_char_length <= tag_to_remove_characteristics.content_max_char_length
                and content_char_length >= tag_to_remove_characteristics.content_min_char_length
            ):
                drop_tag = True

        if tag not in self.tags_exceptions:
            if (
                content_char_length <= self.content_max_char_length
                and content_char_length >= self.content_min_char_length
            ):
                drop_tag = True

        # raise TypeError(f"tag need to be a string not a {type(tag)}")
        return drop_tag


class HtmlProcessor(MetadataProcessor):
    """An example metadata processor for HTMl tags."""

    def __init__(
        self,
        cfg: DataConfig,
        attributes_to_keep=None,
        content_max_char_length: Optional[float] = float("inf"),
        content_min_char_length: Optional[float] = 0,
        tags_exceptions: Optional[List[str]] = None,
        tags_to_remove_alone: Optional[List[TagToRemove]] = None,
    ):
        """
        Args:
            cfg: The data configuration to use.
        """
        super().__init__(cfg)
        self._tag_filter = TagFilter(
            tags_to_remove_alone=tags_to_remove_alone,
            content_min_char_length=content_min_char_length,
            content_max_char_length=content_max_char_length,
            tags_exceptions=tags_exceptions,
        )
        self._attributes_to_keep = attributes_to_keep

    def process_local(self, metadata_attrs: Dict[str, Any]) -> Optional[Tuple[str, str]]:
        # We represent a html tag `T` by enclosing the corresponding text span with "<T>" and "</T>".
        # Example: An <b>apple</b> is an edible fruit.
        if self._tag_filter.drop_tag(
            Metadata(
                char_start_idx=metadata_attrs["char_start_idx"],
                value=HtmlTag(tag=metadata_attrs["value"]["tag"], attrs=metadata_attrs["value"]["attrs"]),
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
