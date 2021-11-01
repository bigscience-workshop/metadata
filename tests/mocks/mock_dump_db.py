from typing import List


class MockParagraph:
    def __init__(self, text):
        self.text = text


class MockDumpDB:
    def __init__(self, db_file) -> None:
        self.db_file = db_file

    def redirects(self) -> List[tuple]:
        return [("xyz.com", "XYZ"), ("test.com", "Test"), ("test_key", "Test Key")]

    def get_paragraphs(self, title: str):
        paragraphs_map = {
            "XYZ": [
                MockParagraph("XYZ is a U.S. based company."),
                MockParagraph("Test paragraph for the key XYZ."),
            ],
            "Test": [
                MockParagraph("Test is a U.S. based company."),
                MockParagraph("Test paragraph for the key Test."),
            ],
            "Sometitle": [
                MockParagraph("SomeTitle is a U.S. based company."),
                MockParagraph("Test paragraph for the key SomeTitle."),
            ],
        }
        return paragraphs_map[title]
