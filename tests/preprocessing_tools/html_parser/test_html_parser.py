from typing import DefaultDict

from bsmetadata.preprocessing_tools.html_parser import get_clean_text_and_metadata
from bsmetadata.preprocessing_tools.html_parser.objects import TagToRemove, TagToRemoveWithContent


def check_content_parsing(target_content_plain_text: str, target_metadata_tags, metadata, plain_text):
    target_list_tags = []
    for target_tag in target_content_plain_text.keys():
        target_list_tags.extend([target_tag] * len(target_content_plain_text[target_tag]))

    for target_tag in target_list_tags:
        assert target_tag in target_metadata_tags
        target_metadata_tags.remove(target_tag)
        find = False
        for metadata_node in metadata:
            if (
                metadata_node.value.tag == target_tag
                and plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx]
                in target_content_plain_text[target_tag]
            ):
                find = True
                target_content_plain_text[target_tag].remove(
                    plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx]
                )
                if not target_content_plain_text[target_tag]:
                    target_content_plain_text.pop(target_tag)
                break

        error_msg = f"Plain text not found for the tag '{target_tag}'"
        if not find:
            retrived_plain_text = "\n ".join(
                [
                    f"{metadata_node.value.tag}: {repr(plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx])}"
                    for metadata_node in metadata
                ]
            )
            error_msg = f"{error_msg}\nThe plain text associated with each tags are:\n {retrived_plain_text} \nand the text to match with:\n{repr(plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx])}"
        assert find, error_msg

    assert not target_content_plain_text
    assert not target_metadata_tags


def check_content_parsing_and_metadata(target_content_plain_text: str, target_metadata_tags, metadata, plain_text):
    target_list_tags = []
    for target_tag in target_content_plain_text.keys():
        target_list_tags.extend([target_tag] * len(target_content_plain_text[target_tag]))
    for target_tag in target_list_tags:
        assert target_tag in target_metadata_tags
        target_metadata_tags.remove(target_tag)
        find = False
        for metadata_node in metadata:
            if (
                metadata_node.value.tag == target_tag
                and metadata_node.value.attrs in [item[1] for item in target_content_plain_text[target_tag]]
                and plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx]
                in [item[0] for item in target_content_plain_text[target_tag]]
            ):
                find = True
                target_content_plain_text[target_tag].remove(
                    (
                        plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx],
                        metadata_node.value.attrs,
                    )
                )
                if not target_content_plain_text[target_tag]:
                    target_content_plain_text.pop(target_tag)
                break

        error_msg = f"Plain text not found for the tag '{target_tag}'"
        if not find:
            retrived_plain_text = "\n ".join(
                [
                    f"{metadata_node.value.tag}: {repr(plain_text[metadata_node.char_start_idx : metadata_node.char_end_idx])}  {metadata_node.value.attrs}"
                    for metadata_node in metadata
                ]
            )
            error_msg = f"{error_msg}\nThe plain text associated with each tags are:\n {retrived_plain_text}"
        assert find, error_msg

    assert not target_content_plain_text
    assert not target_metadata_tags


def test_parse_simple_html():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    </body>
    </html>
"""
    plain_text, metadata = get_clean_text_and_metadata(html)
    assert plain_text == "This is a title\n"  # the space are doe to the block contents

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]
    assert len(metadata) == 2
    assert "html" not in metadata_tags
    assert "head" not in metadata_tags
    assert "body" in metadata_tags
    assert "h1" in metadata_tags

    for metadata_node in metadata:
        if metadata_node.value.tag == "h1":
            metadata_h1 = metadata_node
            break
    assert plain_text[metadata_h1.char_start_idx : metadata_h1.char_end_idx] == "This is a title"
    return (plain_text, metadata)


def test_parse_html_remove_tag_alone():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    </body>
    </html>
"""
    tags_to_remove_alone = [TagToRemove("body")]
    plain_text, metadata = get_clean_text_and_metadata(html, tags_to_remove_alone=tags_to_remove_alone)
    assert plain_text == "This is a title\n"

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]
    assert len(metadata) == 1
    assert "html" not in metadata_tags
    assert "head" not in metadata_tags
    assert "body" not in metadata_tags
    assert "h1" in metadata_tags

    for metadata_node in metadata:
        if metadata_node.value.tag == "h1":
            metadata_h1 = metadata_node
            break
    assert plain_text[metadata_h1.char_start_idx : metadata_h1.char_end_idx] == "This is a title"
    return (plain_text, metadata)


def test_parse_html_remove_tag_and_content():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    <div>
    <p>This is a first paragraph in div</p>
    <p>This is a second paragraph in div</p>
    </div>
    <p>This is a paragraph not in div</p>
    </body>
    </html>
"""
    tags_to_remove_with_content = [TagToRemoveWithContent(tag="div")]
    plain_text, metadata = get_clean_text_and_metadata(html, tags_to_remove_with_content=tags_to_remove_with_content)
    assert plain_text == (
        """This is a title
This is a paragraph not in div
"""
    )  # the space are doe to the block contents

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 3
    assert "html" not in metadata_tags
    assert "head" not in metadata_tags
    assert "body" in metadata_tags
    assert "h1" in metadata_tags
    assert "p" in metadata_tags

    for metadata_node in metadata:
        if metadata_node.value.tag == "h1":
            metadata_h1 = metadata_node
            break
    assert plain_text[metadata_h1.char_start_idx : metadata_h1.char_end_idx] == "This is a title"

    for metadata_node in metadata:
        if metadata_node.value.tag == "p":
            metadata_p = metadata_node
            break
    assert plain_text[metadata_p.char_start_idx : metadata_p.char_end_idx] == "This is a paragraph not in div"
    return (plain_text, metadata)


def test_parse_html_nested_example():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    <div>
    <div>This is a first sub-div in div</div>
    <div>This is a second sub-div in div</div>
    </div>
    <p>This is a paragraph not in div</p>
    </body>
    </html>
"""
    plain_text, metadata = get_clean_text_and_metadata(html)
    assert plain_text == (
        """This is a title
This is a first sub-div in div
This is a second sub-div in div
This is a paragraph not in div
"""
    )

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 6

    target_content_plain_text = {
        "body": [
            """This is a title
This is a first sub-div in div
This is a second sub-div in div
This is a paragraph not in div
"""
        ],
        "h1": ["This is a title"],
        "p": ["This is a paragraph not in div"],
        "div": [
            "This is a first sub-div in div",
            "This is a second sub-div in div",
            "This is a first sub-div in div\nThis is a second sub-div in div\n",
        ],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_parse_html_nested_example_2():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    <div>
    <div>This is a <div>first</div> sub-div in div</div>
    <div>This is a <div>second</div> sub-div in div</div>
    </div>
    <p>This is a paragraph not in div</p>
    </body>
    </html>
"""
    plain_text, metadata = get_clean_text_and_metadata(html)
    assert (
        plain_text
        == """This is a title
This is a
first
sub-div in div
This is a
second
sub-div in div
This is a paragraph not in div
"""
    )

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 8

    target_content_plain_text = {
        "body": [
            """This is a title
This is a
first
sub-div in div
This is a
second
sub-div in div
This is a paragraph not in div
"""
        ],
        "h1": ["This is a title"],
        "p": ["This is a paragraph not in div"],
        "div": [
            "first",
            "second",
            "This is a\nfirst\nsub-div in div",
            "This is a\nsecond\nsub-div in div",
            "This is a\nfirst\nsub-div in div\nThis is a\nsecond\nsub-div in div\n",
        ],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_parse_html_nested_example_max_length():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    <div>
    <div>This is a <div>first</div> sub-div in div</div>
    <div>This is a <div>second</div> sub-div in div</div>
    </div>
    <p>This is a paragraph not in div</p>
    </body>
    </html>
"""
    tags_to_remove_with_content = [TagToRemoveWithContent(tag="div", content_max_char_length=6)]
    plain_text, metadata = get_clean_text_and_metadata(html, tags_to_remove_with_content=tags_to_remove_with_content)
    assert plain_text == (
        "This is a title\n"
        "This is a sub-div in div\n"
        "This is a sub-div in div\n"
        "This is a paragraph not in div\n"
    )

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 6

    target_content_plain_text = {
        "body": [
            (
                "This is a title\n"
                "This is a sub-div in div\n"
                "This is a sub-div in div\n"
                "This is a paragraph not in div\n"
            )
        ],
        "h1": ["This is a title"],
        "p": ["This is a paragraph not in div"],
        "div": [
            "This is a sub-div in div",
            "This is a sub-div in div",
            ("This is a sub-div in div\n" "This is a sub-div in div\n"),
        ],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_parse_html_nested_example_min_length():
    html = """
    <html>
    <head>
    </head>
    <body>
    <h1>This is a title</h1>
    <div>small</div>
    <div>
    <div>This is a <div>first</div> sub-div in div</div>
    <div>This is a <div>second</div> sub-div in div</div>
    </div>
    <p>This is a paragraph not in div</p>
    </body>
    </html>
"""
    tags_to_remove_with_content = [TagToRemoveWithContent(tag="div", content_min_char_length=7, method="top-down")]
    plain_text, metadata = get_clean_text_and_metadata(html, tags_to_remove_with_content=tags_to_remove_with_content)
    assert plain_text == ("This is a title\n" "small\n" "This is a paragraph not in div\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 4

    target_content_plain_text = {
        "body": [("This is a title\n" "small\n" "This is a paragraph not in div\n")],
        "h1": ["This is a title"],
        "p": ["This is a paragraph not in div"],
        "div": ["small"],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_remove_all_table():
    html = """<html><caption>
</caption>
<tbody><tr>
<th>&nbsp;</th>
<th colspan="4"><b><a href="/wiki/Jeux_olympiques_d%27%C3%A9t%C3%A9" title="">Jeux olympiques d'été</a></b>
</th>
<th>&nbsp;</th>
<th colspan="3"><b><a href="/wiki/Jeux_olympiques_d%27hiver" title="Jeux olympiques d'hiver">Jeux olympiques d'hiver</a></b>
</th></tr>
<tr>
<td>2032</td>
<td><a href="/wiki/Jeux_olympiques_d%27%C3%A9t%C3%A9_de_2032" title="Jeux olympiques d'été de 2032">XXXV</a></td>
<td><a href="/wiki/Brisbane" title="Brisbane">Brisbane</a> (1)</td>
<td><span class="datasortkey" data-sort-value="Australie"><span class="flagicon"><a href="//commons.wikimedia.org/wiki/File:Flag_of_Australia.svg?uselang=fr" class="image" title="Drapeau de l'Australie"><img alt="Drapeau de l'Australie" src="//upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_Australia.svg/20px-Flag_of_Australia.svg.png" decoding="async" class="noviewer thumbborder" srcset="//upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_Australia.svg/30px-Flag_of_Australia.svg.png 1.5x, //upload.wikimedia.org/wikipedia/commons/thumb/b/b9/Flag_of_Australia.svg/40px-Flag_of_Australia.svg.png 2x" data-file-width="1280" data-file-height="640" width="20" height="10"></a> </span><a href="/wiki/Australie" title="Australie">Australie</a></span> (3)</td>
<td><a href="/wiki/Oc%C3%A9anie" title="Océanie">Océanie</a> (3)</td>
<td></td>
<td></td>
<td></td>
<td>
</td></tr></tbody></html>"""
    tags_to_remove_with_content = [
        TagToRemoveWithContent(tag="tbody"),
        TagToRemoveWithContent(tag="td"),
    ]
    attrs_to_keep = ["class", "id"]
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        tags_to_remove_with_content=tags_to_remove_with_content,
        attrs_to_keep=attrs_to_keep,
    )
    assert plain_text == ""

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 2

    target_content_plain_text = {
        "body": [""],
        "caption": [""],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_table():
    html = """<html><table>
    <thead>
        <tr>
            <th colspan="2">The table header</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>The table body</td>
            <td>with two columns</td>
        </tr>
    </tbody>
</table></html>"""
    tags_to_remove_with_content = [
        TagToRemoveWithContent(tag="table", content_min_char_length=54),
    ]
    attrs_to_keep = ["class", "id"]
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        tags_to_remove_with_content=tags_to_remove_with_content,
        attrs_to_keep=attrs_to_keep,
    )
    assert plain_text == ""

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 1

    target_content_plain_text = {
        "body": [""],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_table_keep_everything():
    html = """<html><body><table>
    <thead>
        <tr>
            <th colspan="2">The table header</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>The table body</td>
            <td>with two columns</td>
        </tr>
    </tbody>
</table></body></html>"""
    plain_text, metadata = get_clean_text_and_metadata(
        html,
    )
    assert plain_text == "The table header\nThe table body with two columns\n"

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 9

    target_content_plain_text = {
        "table": ["The table header\nThe table body with two columns\n"],
        "thead": ["The table header\n"],
        "tr": ["The table header\n", "The table body with two columns\n"],
        "th": ["The table header"],
        "tbody": ["The table body with two columns\n"],
        "td": ["The table body", "with two columns"],
        "body": ["The table header\nThe table body with two columns\n"],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_behavior_on_corrupt_examples():
    # Corrupt 1: missing end tag value
    html = """<p> test </>"""
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        # start_parsing_at_tag=None,
    )
    assert plain_text == "test >\n"

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 2

    target_content_plain_text = {
        "p": ["test >"],
        "body": ["test >\n"],
    }

    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )
    assert metadata[0].value.attrs == {"attrs": [], "values": []}

    # Corrupt 2: unnecessary "
    html = """<a href="http://example.com""> test </a>"""
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        # start_parsing_at_tag=None,
    )
    assert plain_text == "test\n"

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 2

    target_content_plain_text = {
        "a": ["test\n"],
        "body": ["test\n"],
    }
    check_content_parsing(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )
    assert metadata[0].value.attrs == {
        "attrs": ["href"],
        "values": ["http://example.com"],
    }


def test_attribs():
    html = (
        "<html><body>"
        "<h1>this is a title that we keep</h1>"
        '<div class="div-level-1">blablabla<div class="div-level-2">tidi tidi</div></div>'
        "</body></html>"
    )
    plain_text, metadata = get_clean_text_and_metadata(
        html,
    )
    assert plain_text == ("this is a title that we keep\n" "blablabla\n" "tidi tidi\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 4

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla\n" "tidi tidi\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [("this is a title that we keep", {"attrs": [], "values": []})],
        "div": [
            ("blablabla\ntidi tidi\n", {"attrs": ["class"], "values": ["div-level-1"]}),
            ("\ntidi tidi", {"attrs": ["class"], "values": ["div-level-2"]}),
        ],
    }

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_remove_consecutive_tag():
    html = (
        "<html><body>"
        "<h1>this is a title that we keep</h1>"
        '<div class="div-level-1" id=1>blablabla<div class="div-level-2" href="http">tidi tidi</div></div>'
        "</body></html>"
    )
    consecutive_tags_to_fold = ["div"]
    plain_text, metadata = get_clean_text_and_metadata(html, consecutive_tags_to_fold=consecutive_tags_to_fold)
    assert plain_text == ("this is a title that we keep\n" "blablabla\n" "tidi tidi\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 3

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla\n" "tidi tidi\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [("this is a title that we keep", {"attrs": [], "values": []})],
        "div": [
            (
                "blablabla\ntidi tidi\n",
                {
                    "attrs": ["class", "id", "href"],
                    "values": ["div-level-1 div-level-2", "1", "http"],
                },
            ),
        ],
    }

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_remove_consecutive_tag_with_tag_to_remove():
    html = (
        "<html><body>"
        "<h1 id=title>this is a title that we keep</h1>"
        '<div class="div-level-1" id=1>blablabla<div class="div-level-2" href="http">tidi <span>tidi</span></div></div>'
        "</body></html>"
    )
    consecutive_tags_to_fold = ["div"]
    tags_to_remove_alone = [TagToRemove("span")]
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        consecutive_tags_to_fold=consecutive_tags_to_fold,
        tags_to_remove_alone=tags_to_remove_alone,
    )
    assert plain_text == ("this is a title that we keep\n" "blablabla\n" "tidi tidi\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 3

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla\n" "tidi tidi\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [("this is a title that we keep", {"attrs": ["id"], "values": ["title"]})],
        "div": [
            (
                "blablabla\ntidi tidi\n",
                {
                    "attrs": ["class", "id", "href"],
                    "values": ["div-level-1 div-level-2", "1", "http"],
                },
            ),
        ],
    }

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_remove_consecutive_tag_very_nested():
    html = (
        "<html><body>"
        "<h1 id=title>this is a title that we keep</h1>"
        '<div class="div-level-1" id=1>blablabla<div class="div-level-2" href="http">tidi <div id=3>tidi2</div></div></div>'
        "</body></html>"
    )
    consecutive_tags_to_fold = ["div"]
    tags_to_remove_alone = [TagToRemove("span")]
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        consecutive_tags_to_fold=consecutive_tags_to_fold,
        tags_to_remove_alone=tags_to_remove_alone,
    )
    assert plain_text == ("this is a title that we keep\n" "blablabla\n" "tidi\ntidi2\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 3

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla\n" "tidi\ntidi2\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [("this is a title that we keep", {"attrs": ["id"], "values": ["title"]})],
        "div": [
            (
                "blablabla\ntidi\ntidi2\n",
                {
                    "attrs": ["class", "id", "href"],
                    "values": ["div-level-1 div-level-2", "1 3", "http"],
                },
            ),
        ],
    }

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_min_len_to_include_tag():
    html = (
        "<html><body>"
        "<h1 id=title>this is a title that we keep</h1>"
        '<div class="div-level-1" id=1>blablabla<div class="div-level-2" href="http">tidi <span id=3>tidi2</span> <span id=3>this one keep his tag</span></div></div>'
        "</body></html>"
    )
    consecutive_tags_to_fold = ["div"]
    tags_to_remove_alone = [TagToRemove("span", content_max_char_length=5)]
    plain_text, metadata = get_clean_text_and_metadata(
        html,
        consecutive_tags_to_fold=consecutive_tags_to_fold,
        tags_to_remove_alone=tags_to_remove_alone,
    )
    assert plain_text == ("this is a title that we keep\n" "blablabla\n" "tidi tidi2 this one keep his tag\n")

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    assert len(metadata) == 4

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla\n" "tidi tidi2 this one keep his tag\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [("this is a title that we keep", {"attrs": ["id"], "values": ["title"]})],
        "div": [
            (
                "blablabla\ntidi tidi2 this one keep his tag\n",
                {
                    "attrs": ["class", "id", "href"],
                    "values": ["div-level-1 div-level-2", "1", "http"],
                },
            ),
        ],
        "span": [("this one keep his tag", {"attrs": ["id"], "values": ["3"]})],
    }

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_idx_order():
    html = (
        "<html><body>"
        "<h1 id=title>this is a title that we keep</h1>"
        '<div class="div-level-1" id=1><div class="div-level-2" href="http"><div class="div-level-3"> blablabla tidi <span id=3>tidi2</span></div><span id=2>this one keep his tag</span></div></div>'
        "</body></html>"
    )
    plain_text, metadata = get_clean_text_and_metadata(
        html,
    )

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla tidi tidi2\n" "this one keep his tag\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [
            (
                "this is a title that we keep",
                {"attrs": ["id"], "values": ["title"]},
            )
        ],
        "div": [
            (
                "blablabla tidi tidi2\nthis one keep his tag",
                {"attrs": ["class", "href"], "values": ["div-level-2", "http"]},
            ),
            (
                "blablabla tidi tidi2",
                {"attrs": ["class"], "values": ["div-level-3"]},
            ),
            (
                "blablabla tidi tidi2\nthis one keep his tag\n",
                {"attrs": ["class", "id"], "values": ["div-level-1", "1"]},
            ),
        ],
        "span": [
            (
                "this one keep his tag",
                {"attrs": ["id"], "values": ["2"]},
            ),
            ("tidi2", {"attrs": ["id"], "values": ["3"]}),
        ],
    }
    metadata_sorted_by_start_idx = DefaultDict(list)
    metadata_sorted_by_end_idx = DefaultDict(list)

    metadata_dict_start_idx = DefaultDict(dict)
    metadata_dict_end_idx = DefaultDict(dict)
    for metadata_node in metadata:
        metadata_dict_start_idx[metadata_node.char_start_idx][metadata_node.relative_start_pos] = metadata_node
        metadata_dict_end_idx[metadata_node.char_end_idx][metadata_node.relative_end_pos] = metadata_node

    for key, value in metadata_dict_start_idx.items():
        pos_sorted = sorted(list(value.keys()))
        metadata_sorted_by_start_idx[key] = [value[pos] for pos in pos_sorted]

    for key, value in metadata_dict_end_idx.items():
        pos_sorted = sorted(list(value.keys()))
        metadata_sorted_by_end_idx[key] = [value[pos] for pos in pos_sorted]

    metadata_sorted_by_start_idx_simplify = dict()
    metadata_sorted_by_end_idx_simplify = dict()
    for key, value in metadata_sorted_by_start_idx.items():
        metadata_sorted_by_start_idx_simplify[key] = [
            (metadata_node.value.tag, metadata_node.value.attrs) for metadata_node in value
        ]

    for key, value in metadata_sorted_by_end_idx.items():
        metadata_sorted_by_end_idx_simplify[key] = [
            (metadata_node.value.tag, metadata_node.value.attrs) for metadata_node in value
        ]

    metadata_sorted_by_start_idx_simplify_true = {
        0: [
            ("body", {"attrs": [], "values": []}),
            ("h1", {"attrs": ["id"], "values": ["title"]}),
        ],
        29: [
            ("div", {"attrs": ["class", "id"], "values": ["div-level-1", "1"]}),
            ("div", {"attrs": ["class", "href"], "values": ["div-level-2", "http"]}),
            ("div", {"attrs": ["class"], "values": ["div-level-3"]}),
        ],
        44: [("span", {"attrs": ["id"], "values": ["3"]})],
        50: [("span", {"attrs": ["id"], "values": ["2"]})],
    }

    metadata_sorted_by_end_idx_simplify_true = {
        28: [("h1", {"attrs": ["id"], "values": ["title"]})],
        49: [
            ("span", {"attrs": ["id"], "values": ["3"]}),
            ("div", {"attrs": ["class"], "values": ["div-level-3"]}),
        ],
        71: [
            ("span", {"attrs": ["id"], "values": ["2"]}),
            ("div", {"attrs": ["class", "href"], "values": ["div-level-2", "http"]}),
        ],
        72: [
            ("div", {"attrs": ["class", "id"], "values": ["div-level-1", "1"]}),
            ("body", {"attrs": [], "values": []}),
        ],
    }

    assert metadata_sorted_by_start_idx_simplify_true == metadata_sorted_by_start_idx_simplify
    assert metadata_sorted_by_end_idx_simplify_true == metadata_sorted_by_end_idx_simplify

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_idx_order():
    html = (
        "<html><body>"
        "<h1 id=title>this is a title that we keep</h1>"
        '<br></br><div class="div-level-1" id=1><div class="div-level-2" href="http"><div class="div-level-3"><br> blablabla tidi <span id=3>tidi2</span></div><span id=2>this one keep his tag</span></div></div>'
        "</body></html>"
    )
    plain_text, metadata = get_clean_text_and_metadata(
        html,
    )

    metadata_tags = [metadata_node.value.tag for metadata_node in metadata]

    target_content_plain_text = {
        "body": [
            (
                "this is a title that we keep\n" "blablabla tidi tidi2\n" "this one keep his tag\n",
                {"attrs": [], "values": []},
            )
        ],
        "h1": [
            (
                "this is a title that we keep",
                {"attrs": ["id"], "values": ["title"]},
            )
        ],
        "div": [
            (
                "blablabla tidi tidi2\nthis one keep his tag",
                {"attrs": ["class", "href"], "values": ["div-level-2", "http"]},
            ),
            (
                "blablabla tidi tidi2",
                {"attrs": ["class"], "values": ["div-level-3"]},
            ),
            (
                "blablabla tidi tidi2\nthis one keep his tag\n",
                {"attrs": ["class", "id"], "values": ["div-level-1", "1"]},
            ),
        ],
        "span": [
            (
                "this one keep his tag",
                {"attrs": ["id"], "values": ["2"]},
            ),
            ("tidi2", {"attrs": ["id"], "values": ["3"]}),
        ],
        "br": [
            (
                "",
                {"attrs": [], "values": []},
            ),
            (
                "",
                {"attrs": [], "values": []},
            ),
        ],
    }

    metadata_dict_start_idx = DefaultDict(dict)
    metadata_dict_end_idx = DefaultDict(dict)
    for metadata_node in metadata:
        metadata_dict_start_idx[metadata_node.char_start_idx][metadata_node.relative_start_pos] = (
            metadata_node.value.tag,
            metadata_node.value.attrs,
        )
        metadata_dict_end_idx[metadata_node.char_end_idx][metadata_node.relative_end_pos] = (
            metadata_node.value.tag,
            metadata_node.value.attrs,
        )

    metadata_sorted_by_start_idx_simplify_true = {
        0: {
            0: ("body", {"attrs": [], "values": []}),
            1: ("h1", {"attrs": ["id"], "values": ["title"]}),
        },
        29: {
            0: ("br", {"attrs": [], "values": []}),
            2: ("div", {"attrs": ["class", "id"], "values": ["div-level-1", "1"]}),
            3: ("div", {"attrs": ["class", "href"], "values": ["div-level-2", "http"]}),
            4: ("div", {"attrs": ["class"], "values": ["div-level-3"]}),
            5: ("br", {"attrs": [], "values": []}),
        },
        44: {0: ("span", {"attrs": ["id"], "values": ["3"]})},
        50: {0: ("span", {"attrs": ["id"], "values": ["2"]})},
    }

    metadata_sorted_by_end_idx_simplify_true = {
        28: {0: ("h1", {"attrs": ["id"], "values": ["title"]})},
        29: {
            1: ("br", {"attrs": [], "values": []}),
            6: ("br", {"attrs": [], "values": []}),
        },
        49: {
            0: ("span", {"attrs": ["id"], "values": ["3"]}),
            1: ("div", {"attrs": ["class"], "values": ["div-level-3"]}),
        },
        71: {
            0: ("span", {"attrs": ["id"], "values": ["2"]}),
            1: ("div", {"attrs": ["class", "href"], "values": ["div-level-2", "http"]}),
        },
        72: {
            0: ("div", {"attrs": ["class", "id"], "values": ["div-level-1", "1"]}),
            1: ("body", {"attrs": [], "values": []}),
        },
    }

    assert metadata_sorted_by_start_idx_simplify_true == metadata_dict_start_idx
    assert metadata_sorted_by_end_idx_simplify_true == metadata_dict_end_idx

    check_content_parsing_and_metadata(
        target_content_plain_text=target_content_plain_text,
        target_metadata_tags=metadata_tags,
        metadata=metadata,
        plain_text=plain_text,
    )


def test_convert_br_tag():
    html = "<html><body>" "first line<br>" "second line" "</body></html>"
    plain_text, metadata = get_clean_text_and_metadata(html, convert_br_tag_to_breaking_line=True)
    assert plain_text == "first line\nsecond line\n"
    assert "br" not in [html_tag.value.tag for html_tag in metadata]

    html = "<html><body>" "first line<br><br><br>" "second line" "</body></html>"
    plain_text, metadata = get_clean_text_and_metadata(html, convert_br_tag_to_breaking_line=True)
    assert plain_text == "first line\n\n\nsecond line\n"
    assert "br" not in [html_tag.value.tag for html_tag in metadata]

    html = "<html><body>" "first line<br><br><br>" "second line" "</body></html>"
    plain_text, metadata = get_clean_text_and_metadata(
        html,
    )
    assert plain_text == "first line\nsecond line\n"
    assert "br" in [html_tag.value.tag for html_tag in metadata]

    html = "<html><body>" "first line<br />" "second line" "</body></html>"
    plain_text, metadata = get_clean_text_and_metadata(html, convert_br_tag_to_breaking_line=True)
    assert plain_text == "first line\nsecond line\n"
    assert "br" not in [html_tag.value.tag for html_tag in metadata]
