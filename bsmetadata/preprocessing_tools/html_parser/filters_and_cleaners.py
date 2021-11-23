from typing import DefaultDict, List, Optional, Tuple

import htmlmin
from lxml import etree
from lxml.html import fromstring

from bsmetadata.preprocessing_tools.html_parser.objects import HtmlTag, Metadata, TagToRemove, TagToRemoveWithContent
from bsmetadata.preprocessing_tools.html_parser.variables import (
    BLOCK_CONTENT_SEPARATOR,
    BLOCK_ELEMENTS,
    FAKE_TAG_BASIC,
    FAKE_TAG_BLOCK,
    FAKE_TAG_INLINE,
    INLINE_ELEMENTS_SPACING,
    PLAIN_TEXT_SEPARATOR,
    PRE_TAG,
)


class AttributeCleaner:
    def __init__(self, attrs_to_keep: Optional[List[str]]):
        self.attrs_to_keep = attrs_to_keep

    def _test(self, attr):
        return self.attrs_to_keep is None or attr in self.attrs_to_keep

    def __call__(self, attrs: List[Tuple[str]]):
        if isinstance(attrs, list):
            attrbs = [attr for attr, _ in attrs if self._test(attr)]
            values = [value for attr, value in attrs if self._test(attr)]
            return {
                "attrs": attrbs,
                "values": values,
            }
        else:
            attrs = dict(attrs)

            attrbs = [attr for attr, value in attrs.items() if self._test(attr)]
            values = [value for attr, value in attrs.items() if self._test(attr)]
            return {
                "attrs": attrbs,
                "values": values,
            }


class TagFilter:
    def __init__(
        self,
        tags_to_remove_alone: Optional[List[TagToRemove]],
        tags_to_remove_with_content: Optional[List[TagToRemoveWithContent]],
        txt_max_chr_len_alone: Optional[float] = -float("inf"),
        txt_min_chr_len_alone: Optional[float] = -float("inf"),
        tags_exceptions_alone: Optional[List[str]] = None,
        txt_max_chr_len_with_content: Optional[float] = -float("inf"),
        txt_min_chr_len_with_content: Optional[float] = -float("inf"),
        tags_exceptions_with_content: Optional[List[str]] = None,
    ):
        self.txt_max_chr_len_alone = txt_max_chr_len_alone
        self.txt_min_chr_len_alone = txt_min_chr_len_alone
        self.tags_exceptions_alone = tags_exceptions_alone if tags_exceptions_alone is not None else []
        self.txt_max_chr_len_with_content = txt_max_chr_len_with_content
        self.txt_min_chr_len_with_content = txt_min_chr_len_with_content
        self.tags_exceptions_with_content = (
            tags_exceptions_with_content if tags_exceptions_with_content is not None else []
        )

        self.tags_to_remove_alone = (
            {tag_to_remove.tag: tag_to_remove for tag_to_remove in tags_to_remove_alone}
            if isinstance(tags_to_remove_alone, list)
            else {}
        )
        self.tags_to_remove_with_content = (
            {tag_to_remove.tag: tag_to_remove for tag_to_remove in tags_to_remove_with_content}
            if isinstance(tags_to_remove_with_content, list)
            else {}
        )

        for tag_to_remove_characteristics in self.tags_to_remove_with_content.values():
            if tag_to_remove_characteristics.method not in ["top-down", "bottom-up"]:
                raise ValueError(
                    f"You have requested to remove {tag_to_remove_characteristics.tag} tags and their content if the "
                    f"content has a size between {tag_to_remove_characteristics.content_min_char_length} and "
                    f"{tag_to_remove_characteristics.content_max_char_length} with an invalid method "
                    f"({tag_to_remove_characteristics.method}). Valid methods are 'top_down' and 'bottom_up'."
                )
        # todo sanitize tags_to_remove_with_content

    def drop_tag(self, metadata_node):
        tag = str(metadata_node.value.tag)

        drop_tag = False
        content_char_length = (
            metadata_node.char_end_idx - metadata_node.char_start_idx if metadata_node.char_end_idx is not None else 0
        )

        if (
            tag in self.tags_to_remove_alone
            and content_char_length <= self.tags_to_remove_alone[tag].content_max_char_length
            and content_char_length >= self.tags_to_remove_alone[tag].content_min_char_length
        ):
            drop_tag = True

        if tag not in self.tags_exceptions_alone:
            if content_char_length <= self.txt_max_chr_len_alone and content_char_length >= self.txt_min_chr_len_alone:
                drop_tag = True

        # raise TypeError(f"tag need to be a string not a {type(tag)}")
        return drop_tag

    def drop_tag_and_content_top_down(self, tag: str, text: str):
        if tag in self.tags_to_remove_with_content and self.tags_to_remove_with_content[tag].method != "top-down":
            return False

        drop_tag = False
        content_char_length = len(text)
        if (
            tag in self.tags_to_remove_with_content
            and content_char_length <= self.tags_to_remove_with_content[tag].content_max_char_length
            and content_char_length >= self.tags_to_remove_with_content[tag].content_min_char_length
        ):
            drop_tag = True

        if tag not in self.tags_exceptions_with_content:
            if (
                content_char_length <= self.txt_max_chr_len_with_content
                and content_char_length >= self.txt_min_chr_len_with_content
            ):
                drop_tag = True
        return drop_tag

    def drop_tag_and_content_bottom_up(self, tag: str, text: str):
        if tag not in self.tags_to_remove_with_content:
            return False

        tag_to_remove_characteristics = self.tags_to_remove_with_content[tag]
        if tag_to_remove_characteristics.method != "bottom-up":
            return False

        content_char_length = len(text)
        if (
            content_char_length <= tag_to_remove_characteristics.content_max_char_length
            and content_char_length >= tag_to_remove_characteristics.content_min_char_length
        ):
            return True

        return False


class ConsecutiveTagCleaner:
    def __init__(
        self,
        block_elements: list,
        consecutive_tags_to_fold: Optional[List[str]],
    ):
        self.consecutive_tags_to_fold = consecutive_tags_to_fold if isinstance(consecutive_tags_to_fold, list) else []
        self.fake_tag_block = FAKE_TAG_BLOCK
        self.fake_tag_inline = FAKE_TAG_INLINE
        self.fake_tag_basic = FAKE_TAG_BASIC
        self.attrib_separator = " "
        self.block_elements = block_elements

    def __call__(self, root):
        tag = root.tag
        if (tag in self.consecutive_tags_to_fold and len(root) == 1 and root[0].tag == tag) or (
            tag in [FAKE_TAG_BLOCK, FAKE_TAG_INLINE, FAKE_TAG_BASIC]
            and len(root) == 1
            and "previous_tag" in root.attrib
            and root[0].tag == root.attrib["previous_tag"]
        ):  # has 1 child

            if tag in self.block_elements:
                root[0].tag = self.fake_tag_block
            elif tag in INLINE_ELEMENTS_SPACING:
                root[0].tag = self.fake_tag_inline
            else:
                root[0].tag = self.fake_tag_basic

            parent_root = root
            while parent_root.tag in [FAKE_TAG_BLOCK, FAKE_TAG_INLINE, FAKE_TAG_BASIC]:
                parent_root = parent_root.getparent()

            for key, value in root[0].attrib.items():
                if key in parent_root.attrib:
                    parent_root.attrib[key] += self.attrib_separator + value
                else:
                    parent_root.attrib[key] = value
            root[0].attrib["previous_tag"] = tag


def _remove_keeping_tail(element):
    """Safe the tail text and then delete the element"""
    _preserve_tail_before_delete(element)
    element.getparent().remove(element)


def _preserve_tail_before_delete(node):
    if node.tail:  # preserve the tail
        previous = node.getprevious()
        if previous is not None:  # if there is a previous sibling it will get the tail
            if previous.tail is None:
                previous.tail = node.tail
            elif (
                previous.text
                and not previous.text.endswith(PLAIN_TEXT_SEPARATOR)
                and not node.tail.startswith(PLAIN_TEXT_SEPARATOR)
            ):
                previous.text = previous.text + PLAIN_TEXT_SEPARATOR + node.tail
            elif (
                previous.text
                and previous.text.endswith(PLAIN_TEXT_SEPARATOR)
                and node.tail.startswith(PLAIN_TEXT_SEPARATOR)
            ):
                # Don't accumulate too much spaces
                previous.text = previous.text[: -len(PLAIN_TEXT_SEPARATOR)] + node.tail
            elif (
                previous.tail
                and not previous.tail.endswith(PLAIN_TEXT_SEPARATOR)
                and not node.tail.startswith(PLAIN_TEXT_SEPARATOR)
            ):
                previous.tail = previous.tail + PLAIN_TEXT_SEPARATOR + node.tail
            else:
                previous.tail = previous.tail + node.tail
        else:  # The parent get the tail as text
            parent = node.getparent()
            if parent.text is None:
                parent.text = node.tail
            elif not parent.text.endswith(PLAIN_TEXT_SEPARATOR) and not node.tail.startswith(PLAIN_TEXT_SEPARATOR):
                parent.text = parent.text + PLAIN_TEXT_SEPARATOR + node.tail
            elif parent.text.endswith(PLAIN_TEXT_SEPARATOR) and node.tail.startswith(PLAIN_TEXT_SEPARATOR):
                # Don't accumulate too much spaces
                parent.text = parent.text[: -len(PLAIN_TEXT_SEPARATOR)] + node.tail
            else:
                parent.text = parent.text + node.tail


class TextAndMetadataCleaner:
    def __init__(
        self,
        html_str,
        tags_to_remove_with_content: Optional[List[TagToRemoveWithContent]] = None,
        tags_to_remove_alone: Optional[List[TagToRemove]] = None,
        attrs_to_keep: Optional[List[str]] = None,
        start_parsing_at_tag: Optional[str] = "body",
        consecutive_tags_to_fold: Optional[List[str]] = None,
        convert_br_tag_to_breaking_line: Optional[bool] = False,
        txt_max_chr_len_alone: float = -float("inf"),
        txt_min_chr_len_alone: float = -float("inf"),
        tags_exceptions_to_txt_max_min_chr_len_alone: List[str] = None,
        txt_max_chr_len_with_content: float = -float("inf"),
        txt_min_chr_len_with_content: float = -float("inf"),
        tags_exceptions_to_txt_max_min_chr_len_with_content: List[str] = None,
    ):
        self.html_str = html_str
        self.tags_to_remove_with_content = tags_to_remove_with_content
        self.tags_to_remove_alone = tags_to_remove_alone
        self.attrs_to_keep = attrs_to_keep
        self.start_parsing_at_tag = start_parsing_at_tag
        self.convert_br_tag_to_breaking_line = convert_br_tag_to_breaking_line

        if self.tags_to_remove_alone is None:
            self.tags_to_remove_alone = []
        self.tags_to_remove_alone.extend(
            [
                TagToRemove(FAKE_TAG_BLOCK),
                TagToRemove(FAKE_TAG_INLINE),
                TagToRemove(FAKE_TAG_BASIC),
            ]
        )

        self.block_elements = BLOCK_ELEMENTS.copy()
        if self.convert_br_tag_to_breaking_line:
            self.block_elements.remove("br")
            self.tags_to_remove_alone.append(TagToRemove("br"))

        self.consecutive_tag_cleaner = ConsecutiveTagCleaner(
            block_elements=self.block_elements,
            consecutive_tags_to_fold=consecutive_tags_to_fold,
        )

        self.attribute_cleaner = AttributeCleaner(attrs_to_keep=attrs_to_keep)
        self.tag_filter = TagFilter(
            txt_max_chr_len_alone=txt_max_chr_len_alone,
            txt_min_chr_len_alone=txt_min_chr_len_alone,
            tags_exceptions_alone=tags_exceptions_to_txt_max_min_chr_len_alone,
            txt_max_chr_len_with_content=txt_max_chr_len_with_content,
            txt_min_chr_len_with_content=txt_min_chr_len_with_content,
            tags_exceptions_with_content=tags_exceptions_to_txt_max_min_chr_len_with_content,
            tags_to_remove_alone=self.tags_to_remove_alone,
            tags_to_remove_with_content=tags_to_remove_with_content,
        )

    def apply(self):
        html_str = self.html_str
        # Traitement n°1: start the parsing at a special tags (mostly tested with <body>)
        if self.start_parsing_at_tag is not None:
            root = fromstring(html_str)
            find = etree.XPath(f"//{self.start_parsing_at_tag}")
            try:
                new_etree = find(root)[0]
            except IndexError:
                raise ValueError(
                    f"You have asked to start parsing at the {self.start_parsing_at_tag} tag but the current example "
                    "does not contain this tag"
                )
            html_str = etree.tostring(new_etree, method="html", encoding="UTF-8", pretty_print=False).decode("UTF-8")
            if not html_str.startswith("<html>"):
                self.tag_filter.tags_to_remove_alone.update({"html": TagToRemove("html")})

                # need to re-add html tag otherwise the fromstring` do something strange
                html_str = f"<html>{html_str}</html>"

        # Traitement n°2: [all treatments impacting the chr_idx] we removes sub-trees from the HTML + we minify the html
        html_str = htmlmin.minify(html_str, remove_comments=True, keep_pre=True)

        new_etree = fromstring(html_str)

        self._clean_etree(new_etree)

        html_str = etree.tostring(new_etree, method="html", encoding="UTF-8", pretty_print=False).decode("UTF-8")
        html_str = htmlmin.minify(html_str, keep_pre=True)

        # Traitement n°3: we separate the text from the list of metadata json that we keep
        self.metadata = []
        self._current_char_idx = 0
        self._current_num_metadata_by_idx = DefaultDict(lambda: 0)
        self.text = ""
        self.last_tag = None

        plain_text = self._get_text_and_metadata(new_etree)

        self._clean_relative_pos(self.metadata)

        return plain_text, self.metadata

    def _br_conversion(self, tag):
        if tag == "br":
            self.text += "\n"

    def _clean_relative_pos(self, metadata):
        metadata_dict_idx = DefaultDict(dict)
        for metadata_node in metadata:
            metadata_dict_idx[metadata_node.char_start_idx][metadata_node.relative_start_pos] = (
                "start",
                metadata_node,
            )
            metadata_dict_idx[metadata_node.char_end_idx][metadata_node.relative_end_pos] = ("end", metadata_node)

        for absolute_idx, value in metadata_dict_idx.items():
            pos_sorted = sorted(list(value.keys()))
            idx = 0
            for pos in pos_sorted:
                if metadata_dict_idx[absolute_idx][pos][0] == "start":
                    metadata_dict_idx[absolute_idx][pos][1].relative_start_pos = idx
                    idx += 1

                if metadata_dict_idx[absolute_idx][pos][0] == "end":
                    metadata_dict_idx[absolute_idx][pos][1].relative_end_pos = idx
                    idx += 1

    def _add_text(self, tag, new_text):
        if tag in self.block_elements:
            self.text = self._append_block_separator(self.text)
        elif tag in INLINE_ELEMENTS_SPACING:
            self.text = self._append_inline_element_separator(self.text)

        if new_text:
            self._append_text_content(new_text)

        self._current_char_idx = len(self.text)

    def _append_text_content(self, txt):
        if self.current_tag == PRE_TAG:
            self.text += txt
        else:
            txt = txt.replace("\u00a0", " ")

            c = " "
            if len(self.text) > 0:
                c = self.text[-1]
            for i in range(len(txt)):
                c2 = txt[i]
                if c2 == "\r" or c2 == "\n":
                    c2 = " "
                if not c.isspace() or not c2.isspace():
                    self.text += c2
                c = c2

    def _append_block_separator(self, sb):
        length = len(sb)
        if length > 0:
            # remove white space before paragraph break
            # if self.last_tag != PRE_TAG:
            #     while (length > 0 and sb[-1] == PLAIN_TEXT_SEPARATOR):
            #         sb = sb[:-len(PLAIN_TEXT_SEPARATOR)]
            if length > 0 and sb[-1] == PLAIN_TEXT_SEPARATOR:
                sb = sb[:-1] + BLOCK_CONTENT_SEPARATOR
            elif length > 0 and sb[-1] != BLOCK_CONTENT_SEPARATOR:
                sb += BLOCK_CONTENT_SEPARATOR
        return sb

    def _append_inline_element_separator(self, sb):
        length = len(sb)
        if length > 0:
            last_buffer_char = sb[-1]
            if last_buffer_char != PLAIN_TEXT_SEPARATOR and last_buffer_char != BLOCK_CONTENT_SEPARATOR:
                sb += PLAIN_TEXT_SEPARATOR
        return sb

    def _get_text_and_metadata(self, root):
        self.current_tag = root.tag

        metadata_node = Metadata(
            char_start_idx=self._current_char_idx,
            relative_start_pos=self._current_num_metadata_by_idx[self._current_char_idx],
            value=HtmlTag(tag=root.tag, attrs=self.attribute_cleaner(root.attrib)),
        )

        self._current_num_metadata_by_idx[self._current_char_idx] += 1

        if self.convert_br_tag_to_breaking_line:
            self._br_conversion(root.tag)

        self._add_text(root.tag, root.text)
        for idx, child in enumerate(root):
            _ = self._get_text_and_metadata(child)

        self.current_tag = root.tag

        metadata_node.char_end_idx = self._current_char_idx
        metadata_node.relative_end_pos = self._current_num_metadata_by_idx[self._current_char_idx]
        self._current_num_metadata_by_idx[self._current_char_idx] += 1

        self._add_text(root.tag, root.tail)

        if not self.tag_filter.drop_tag(metadata_node=metadata_node):
            self.metadata.append(metadata_node)

        return self.text

    def _clean_etree(
        self,
        root,
    ):
        self.consecutive_tag_cleaner(root)

        # Top-Down deletion
        plain_text = etree.tostring(root, method="text", encoding="UTF-8", pretty_print=False).decode("UTF-8")
        text = plain_text[: -len(root.tail)] if root.tail else plain_text
        if self.tag_filter.drop_tag_and_content_top_down(tag=root.tag, text=text):
            _remove_keeping_tail(root)
            return

        for idx, child in enumerate(root):
            self._clean_etree(child)

        # Bottom-UP deletion
        plain_text = etree.tostring(root, method="text", encoding="UTF-8", pretty_print=False).decode("UTF-8")
        text = plain_text[: -len(root.tail)] if root.tail else plain_text
        if self.tag_filter.drop_tag_and_content_bottom_up(tag=root.tag, text=text):
            _remove_keeping_tail(root)
