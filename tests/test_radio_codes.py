import pytest

from radio_codes import normalize_radio_codes


@pytest.mark.parametrize("text,expected", [
    ("10 4 copy", "10-4 copy"),
    ("ten four", "10-4"),
    ("10.23 at the scene", "10-23 at the scene"),
    ("Ten-Fifty-One", "10-51"),
    ("ten fifty PI at Main", "10-50PI at Main"),
    ("10 50 F", "10-50F"),
    ("10, 43 a", "10-43a"),
    ("10-33A", "10-33a"),
    ("signal twenty", "Signal 20"),
    ("10-24 code 6", "10-24 code 6"),
])
def test_known_codes_are_normalized(text, expected):
    assert normalize_radio_codes(text) == expected


@pytest.mark.parametrize("text", [
    "10 40th Street",      # not followed by a word boundary
    "ten thirty-seven",    # 10-37 is reserved, not a real code
    "Signal 21",           # not a Durham signal
    "1050 Main Street",    # an address, no separator
])
def test_other_numbers_are_left_alone(text):
    assert normalize_radio_codes(text) == text


def test_a_suffix_the_code_does_not_take_stays_a_word():
    assert normalize_radio_codes("10 50 for") == "10-50 for"
