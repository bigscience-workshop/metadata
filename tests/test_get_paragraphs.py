from test_preprocessing_utils import HtmlToyData

from bsmetadata.paragraph_by_metadata_html import get_paragraphs


class TestToyData:
    """The test cases are shared with :py:class:`~tests.test_preprocessing_utils.HtmlPreprocessorTester`."""

    TEXTS = HtmlToyData.target_texts
    METADATA_HTMLS = HtmlToyData.target_metadata

    def test_get_paragraph_by_one_h1(self):
        """test_get_paragraph_by_one_h1

        Example:
            html = (
                '\n    <html>\n    <head><meta charset="utf-8"><title>My test page</title><head>\n'
                "    <body>\n    <h1>This is a title</h1>\n"
                "   with some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada </body>"
                ' <footer><p>Author: Hege Refsnes</p><p><a href="mailto:hege@example.com">hege@example.com</a></p></footer>'
                "<footer><p>Author: Anonymouss</p></footer></html>\n"
            )
        """
        expected_metadata_paras = [
            {
                "char_end_idx": 15,
                "char_start_idx": 0,
                "key": "paragraph",
                "marker": "h1",
                "type": "local",
                "value": "This is a title",
            },
            {
                "char_end_idx": 101,
                "char_start_idx": 15,
                "key": "paragraph",
                "marker": "remainder",
                "type": "local",
                "value": "\nwith some additional text to reach 64 characters tidi tadada tidi tadada tidi tadada\n",
            },
        ]
        metadata_paras = get_paragraphs(TestToyData.METADATA_HTMLS[0], TestToyData.TEXTS[0])
        assert metadata_paras == expected_metadata_paras
        assert "".join(mtdt_p["value"] for mtdt_p in metadata_paras) == TestToyData.TEXTS[0]

    def test_get_paragraph_by_one_p(self):
        """test_mark_paragraph_by_one_p

        Example:
            html = (
                "<html><body><p>this is a simple paragraph with Obama and Merkel mentioned. "
                "tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada</p></body></html>"
            )
        """
        expected_metadata_paras = [
            {
                "char_end_idx": 120,
                "char_start_idx": 0,
                "key": "paragraph",
                "marker": "p",
                "type": "local",
                "value": TestToyData.TEXTS[1],
            },
        ]
        metadata_paras = get_paragraphs(TestToyData.METADATA_HTMLS[1], TestToyData.TEXTS[1])
        assert metadata_paras == expected_metadata_paras

    def test_get_paragraph_by_two_ps(self):
        """test_mark_paragraph_by_two_ps

        Example:
            html = (
                "<html><body><p id=1>paragraph 1 tidi tadada tidi tadada tidi tadada tidi tadada tidi tadada.</p>"
                "<p id=2>paragraph 2 is in Paris tidi tadada tidi tadada tidi tadada tidi tadada.</p></body></html>"
            )
        """
        expected_metadata_paras = [
            {
                "char_end_idx": 73,
                "char_start_idx": 0,
                "key": "paragraph",
                "marker": "p",
                "type": "local",
                "value": TestToyData.TEXTS[2].split("\n")[0] + "\n",
            },
            {
                "char_end_idx": 146,
                "char_start_idx": 73,
                "key": "paragraph",
                "marker": "p",
                "type": "local",
                "value": TestToyData.TEXTS[2].split("\n")[1] + "\n",
            },
        ]
        metadata_paras = get_paragraphs(TestToyData.METADATA_HTMLS[2], TestToyData.TEXTS[2])
        assert metadata_paras == expected_metadata_paras
        assert "".join(mtdt_p["value"] for mtdt_p in metadata_paras) == TestToyData.TEXTS[2]

    def test_get_paragraph_by_nested_and_merged_divs(self):
        """test_mark_paragraph_by_nested_divs

        Example:
            html = (
                '<html><body><div class="div-level-1">blablabla blablabla blablabla blablabla blablabla blablabla
                '<div class="div-level-2">tidi tidi tidi tidi</div></div></body></html>'
            )
        """
        expected_metadata_paras = [
            {
                "char_end_idx": 80,
                "char_start_idx": 0,
                "key": "paragraph",
                "marker": "div",
                "type": "local",
                "value": TestToyData.TEXTS[3],
            },
        ]
        metadata_paras = get_paragraphs(TestToyData.METADATA_HTMLS[3], TestToyData.TEXTS[3])
        assert metadata_paras == expected_metadata_paras


class TestRealData:
    """The test case is from a small portion of `c4-en-html_cc-main-2019-18_pq00-000`[11].

    With the second last item of the output, the test shows the spec of dangling double LF.

    Example:
        html = (
            "..."
            '<div class="container"><div class="sidebars row"><div class="col-xs-12 col-sm-3">'
            '<div class="sidebar sidebar-1">...<div class="tagcloud">'
            "<a ...>CITY MAP</a><a ...>CINQUE TERRE</a><a ...>CITY</a><a ...>HOW TO GET</a><a ...>ACCOMODATION</a>"
            "<a ...>TRANSPORTS AND BOATS</a><a ...>EVENTS</a><a ...>GULF</a><a ...>PORTOVENERE</a>"
            "<a ...>NATURAL PARKS</a><a ...>LERICI</a><a ...>THE DISTRICT</a><a ...>HIKING</a><a ...>FOOD CULTURE</a>"
            "<a ...>INFO CENTER</a>"
            "..."
            '<div class="sh_contact"> <span ...>Comune della Spezia</span><br>'
            "<span ...>Piazza Europa, 1 | 19124 La Spezia</span><br> "
            "<a ...>infocenterlia@comune.sp.it</a><br>"
            "..."
            '<div id="copyright" class="container-full">'
            "<div ...></div>"
            "<a ...>...</a><br>"  # Dangling double LF begins
            "<br>"                # Dangling double LF ends
            "<span ...>Comune della Spezia - Piazza Europa, 1 |&nbsp;19124 La Spezia - P. IVA 00211160114 | "
            "<a ...>Cookie Policy</a>&nbsp;| <a ...>...</a></p>"
            "..."
        )
    """

    TEXT = (
        "CITY MAP CINQUE TERRE CITY HOW TO GET ACCOMODATION "
        "TRANSPORTS AND BOATS EVENTS GULF PORTOVENERE "
        "NATURAL PARKS LERICI THE DISTRICT HIKING FOOD CULTURE "
        "INFO CENTER\n"
        "Comune della Spezia\n"
        "Piazza Europa, 1 | 19124 La Spezia\n"
        "infocenterlia@comune.sp.it\n"
        "\n\n"
        "Comune della Spezia - Piazza Europa, 1 | 19124 La Spezia - P. IVA "
        "00211160114 | Cookie Policy |\n"
    )
    METADATA_HTML = [
        {
            "char_end_idx": 162,
            "char_start_idx": 0,
            "html_attrs": {"attrs": ["class"], "values": ["tagcloud"]},
            "key": "html",
            "relative_end_pos": 0,
            "relative_start_pos": 3,
            "type": "local",
            "value": "div",
        },
        {
            "char_end_idx": 162,
            "char_start_idx": 0,
            "html_attrs": {"attrs": ["class"], "values": ["col-xs-12 col-sm-3 sidebar sidebar-1"]},
            "key": "html",
            "relative_end_pos": 1,
            "relative_start_pos": 2,
            "type": "local",
            "value": "div",
        },
        {
            "char_end_idx": 244,
            "char_start_idx": 0,
            "html_attrs": {"attrs": ["class"], "values": ["container sidebars row"]},
            "key": "html",
            "relative_end_pos": 2,
            "relative_start_pos": 1,
            "type": "local",
            "value": "div",
        },
        {
            "char_end_idx": 244,
            "char_start_idx": 162,
            "html_attrs": {"attrs": ["class"], "values": ["sh_contact"]},
            "key": "html",
            "relative_end_pos": 0,
            "relative_start_pos": 3,
            "type": "local",
            "value": "div",
        },
        {
            "char_end_idx": 244,
            "char_start_idx": 162,
            "html_attrs": {"attrs": ["class"], "values": ["col-xs-12 col-sm-3 sidebar sidebar-3"]},
            "key": "html",
            "relative_end_pos": 1,
            "relative_start_pos": 2,
            "type": "local",
            "value": "div",
        },
        {
            "char_end_idx": 342,
            "char_start_idx": 244,
            "html_attrs": {"attrs": ["id", "class"], "values": ["copyright", "container-full"]},
            "key": "html",
            "relative_end_pos": 5,
            "relative_start_pos": 3,
            "type": "local",
            "value": "div",
        },
    ]
    EXPECTED_METADATA_PARAS = [
        {
            "char_end_idx": 162,
            "char_start_idx": 0,
            "key": "paragraph",
            "type": "local",
            "value": (
                "CITY MAP CINQUE TERRE CITY HOW TO GET ACCOMODATION TRANSPORTS AND BOATS "
                "EVENTS GULF PORTOVENERE NATURAL PARKS LERICI THE DISTRICT HIKING FOOD CULTURE INFO CENTER\n"
            ),
            "marker": "div",
        },
        {
            "char_end_idx": 244,
            "char_start_idx": 162,
            "key": "paragraph",
            "type": "local",
            "value": "Comune della Spezia\nPiazza Europa, 1 | 19124 La Spezia\ninfocenterlia@comune.sp.it\n",
            "marker": "div",
        },
        {
            "char_end_idx": 246,
            "char_start_idx": 244,
            "key": "paragraph",
            "type": "local",
            "value": "\n\n",
            "marker": "div+lf",
        },
        {
            "char_end_idx": 342,
            "char_start_idx": 246,
            "key": "paragraph",
            "type": "local",
            "value": "Comune della Spezia - Piazza Europa, 1 | 19124 La Spezia - P. IVA 00211160114 | Cookie Policy |\n",
            "marker": "div+lf",
        },
    ]
    METADATA_PARAS = get_paragraphs(METADATA_HTML, TEXT)

    def test_get_paragraphs_with_the_last_container_of_the_12th_record(self):
        assert TestRealData.METADATA_PARAS == TestRealData.EXPECTED_METADATA_PARAS
        assert "".join(mtdt_p["value"] for mtdt_p in TestRealData.METADATA_PARAS) == TestRealData.TEXT
