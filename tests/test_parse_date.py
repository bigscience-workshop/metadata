import unittest
from datetime import datetime as DateTime

from bsmetadata.preprocessing_utils import TimestampPreprocessor, parse_date


class TestDateutil(unittest.TestCase):
    def test_parse_works(self):
        date = parse_date("2021/01/23/random text and num 123 asdf")
        self.assertTrue(date == DateTime(2021, 1, 23))
        date = parse_date("2021/jan/23/random text and num 123 asdf")
        self.assertTrue(date == DateTime(2021, 1, 23))
        date = parse_date("2021/Jan/23/random text and num 123 asdf")
        self.assertTrue(date == DateTime(2021, 1, 23))
        date = parse_date("2021-jan-23 random text and num 123 asdf")
        self.assertTrue(date == DateTime(2021, 1, 23))
        date = parse_date("2021 jan 23 random text and num 123 asdf")
        self.assertTrue(date == DateTime(2021, 1, 23))

    def test_parse_fail_without_full_date(self):
        date = parse_date("12")
        self.assertTrue(date is None)
        date = parse_date("2021")
        self.assertTrue(date is None)
        date = parse_date("2021/01")
        self.assertTrue(date is None)
        date = parse_date("2021 jan")
        self.assertTrue(date is None)


class TestTimestampPreprocessor(unittest.TestCase):
    def test_extract_timestamp_from_url(self):
        processor = TimestampPreprocessor()
        url = "https://www.nytimes.com/1998/03/08/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html"
        date = processor._extract_timestamp_from_url(url)
        self.assertEqual(date, "1998-03-08 00:00:00")
        url = "https://www.nytimes.com/sports/on-pro-basketball-one-last-hurrah-for-the-bulls-reinsdorf-isn-t-quite-saying.html"
        date = processor._extract_timestamp_from_url(url)
        self.assertTrue(date is None)


if __name__ == "__main__":
    unittest.main()
