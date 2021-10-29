from dataclasses import dataclass
from typing import Optional, OrderedDict


@dataclass
class TagToRemove:
    tag: str
    content_min_char_length: float = 0
    content_max_char_length: float = float("inf")


@dataclass
class TagToRemoveWithContent:
    tag: str
    content_min_char_length: float = 0
    content_max_char_length: float = float("inf")
    method: str = "top-down"  # or "bottom-up"


@dataclass
class HtmlTag:
    tag: str
    attrs: dict


@dataclass
class Metadata:
    char_start_idx: int
    relative_start_pos: int
    value: HtmlTag
    char_end_idx: Optional[int] = None
    relative_end_pos: Optional[int] = None
    key: str = "html"
    type: str = "local"


def convert_html_metadata_dataclass_to_dict(metadata: Metadata):
    html_metadata_dict = OrderedDict(
        {
            "key": metadata.key,
            "type": metadata.type,
            "char_start_idx": metadata.char_start_idx,
            "relative_start_pos": metadata.relative_start_pos,
            "char_end_idx": metadata.char_end_idx,
            "relative_end_pos": metadata.relative_end_pos,
            # The information about the HTML tag is separated into two keys because the dictionary must have a stable
            # format between the different types of metadata
            "value": metadata.value.tag,
            "html_attrs": metadata.value.attrs,
        }
    )
    return html_metadata_dict
