import pytest

from tagc.cleaner import Cleaner


@pytest.fixture()
def cleaner():
    return Cleaner()


def test_clean_text(cleaner: Cleaner):
    text = "BM214-01              04-04-01     "
    assert cleaner._clean_text(text) == "BM214-01 04-04-01"


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "BMPARTICLES: BM214-01:04-04-01 COMMENT:04-04-01",
            ["BM", "PARTICLES: BM214-01:04-04-01 ", "COMMENT:04-04-01"],
        ),
    ],
)
def test_entry_divide(cleaner: Cleaner, text, expected):
    assert cleaner._entry_divide(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        ("PARTICLES :04-04-01 ", ("PARTICLES", "04-04-01")),
        ("PARTICLES BM214-01:04-04-01 ", None),
        ("COMMENT:04-04-01", ("COMMENT", "04-04-01")),
    ],
)
def test_extract_entry(cleaner: Cleaner, text, expected):
    assert cleaner._extract_entry(text) == expected


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            "BM214-01  04-04-01  PARTICLES: 04-04-01 COMMENT:04-04-01 ",
            {"PARTICLES": "04-04-01", "COMMENT": "04-04-01"},
        ),
    ],
)
def test_parse_text(cleaner: Cleaner, text, expected):
    assert cleaner._parse_text(text) == expected
