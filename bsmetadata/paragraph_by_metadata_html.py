import re
from itertools import zip_longest
from operator import itemgetter
from typing import Any, Dict, List


# https://developer.mozilla.org/en-US/docs/Web/HTML/Element#text_content
TEXT_CONTENT_ELEMENTS = {
    "p",
    "div",
    "li",
    "ul",
    "ol",
    "hr",
    "pre",
    "blockquote",
    "dt",
    "dd",
    "dl",
    "menu",
    "figcaption",
    "figure",
}

# https://developer.mozilla.org/en-US/docs/Web/HTML/Element#content_sectioning
CONTENT_SECTIONING_ELEMENTS = {  # except "header" and "footer"
    "h3",
    "h2",
    "h1",
    "h4",
    "h5",
    "h6",
    "nav",
    "article",
    "main",
    "section",
    "address",
    "aside",
}

MARKERS = TEXT_CONTENT_ELEMENTS | CONTENT_SECTIONING_ELEMENTS


def _pairify(iterable):
    return (pair for pair in (zip_longest(*[iter(iterable)] * 2, fillvalue="")))


def _split_by_double_lf(text: str):
    return list("".join(pair) for pair in _pairify(re.split("(\n\n)", text)))


def _extract_paragraphs(text_chunk: str, marker: str, char_start_idx: int):
    paragraphs = []
    append_p = paragraphs.append

    txt_segs = list(filter(None, _split_by_double_lf(text_chunk)))
    final_marker = f"{marker}+lf" if len(txt_segs) > 1 else marker
    offset = 0
    for txt_seg in txt_segs:
        txt_seg_len = len(txt_seg)
        paragraph = {
            "char_end_idx": char_start_idx + offset + txt_seg_len,
            "char_start_idx": char_start_idx + offset,
            "key": "paragraph",
            "type": "local",
            "value": txt_seg,
            "marker": final_marker,
        }
        append_p(paragraph)
        offset += txt_seg_len

    return paragraphs


def get_paragraphs(html_metadata: List[Dict[str, Any]], text: str) -> List[Dict[str, Any]]:
    """Get paragraphs from metadata-html and text of features

    Args:
        html_metadata (List[Dict[str, Any]]): metadata-html of features
        text (str): text of Features

    Returns:
        List[Dict[str, Any]]: metadata-paragraph of features
    """
    paragraphs = []
    extend_p = paragraphs.extend

    tags = sorted(
        [tag for tag in html_metadata if tag["value"] in MARKERS],
        key=itemgetter("char_end_idx", "char_start_idx"),
    )

    char_start_idx = 0
    for tag in tags:
        char_end_idx = tag["char_end_idx"] + 1 if "p" == tag["value"] else tag["char_end_idx"]
        txt_blk = text[char_start_idx:char_end_idx]
        if txt_blk:
            extend_p(_extract_paragraphs(txt_blk, tag["value"], char_start_idx))
        char_start_idx = char_end_idx
    remaining_txt = text[char_start_idx:]
    if remaining_txt:
        extend_p(_extract_paragraphs(remaining_txt, "remainder", char_start_idx))

    return paragraphs
