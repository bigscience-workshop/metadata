import unittest
from datetime import datetime as DateTime

from bsmetadata.preprocessing_utils import get_path_from_url, parse_date


class TestDateutl(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
