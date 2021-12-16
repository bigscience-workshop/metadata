from typing import List, Optional

from bsmetadata.preprocessing_tools.html_parser.filters_and_cleaners import TextAndMetadataCleaner
from bsmetadata.preprocessing_tools.html_parser.objects import TagToRemoveWithContent


def get_clean_text_and_metadata(
    html_str,
    tags_to_remove_with_content: Optional[List[TagToRemoveWithContent]] = None,
    tags_to_remove_alone: Optional[List[str]] = None,
    attrs_to_keep: Optional[List[str]] = None,
    consecutive_tags_to_fold: Optional[List[str]] = None,
    convert_br_tag_to_breaking_line: Optional[bool] = False,
    txt_max_chr_len_alone: float = -float("inf"),
    txt_min_chr_len_alone: float = -float("inf"),
    tags_exceptions_to_txt_max_min_chr_len_alone: List[str] = None,
    txt_max_chr_len_with_content: float = -float("inf"),
    txt_min_chr_len_with_content: float = -float("inf"),
    tags_exceptions_to_txt_max_min_chr_len_with_content: List[str] = None,
):
    text_and_metadata_cleaner = TextAndMetadataCleaner(
        html_str=html_str,
        tags_to_remove_with_content=tags_to_remove_with_content,
        tags_to_remove_alone=tags_to_remove_alone,
        attrs_to_keep=attrs_to_keep,
        start_parsing_at_tag="body",
        consecutive_tags_to_fold=consecutive_tags_to_fold,
        convert_br_tag_to_breaking_line=convert_br_tag_to_breaking_line,
        txt_max_chr_len_alone=txt_max_chr_len_alone,
        txt_min_chr_len_alone=txt_min_chr_len_alone,
        tags_exceptions_to_txt_max_min_chr_len_alone=tags_exceptions_to_txt_max_min_chr_len_alone,
        txt_max_chr_len_with_content=txt_max_chr_len_with_content,
        txt_min_chr_len_with_content=txt_min_chr_len_with_content,
        tags_exceptions_to_txt_max_min_chr_len_with_content=tags_exceptions_to_txt_max_min_chr_len_with_content,
    )
    return text_and_metadata_cleaner.apply()
