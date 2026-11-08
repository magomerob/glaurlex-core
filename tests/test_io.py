import pandas as pd
import pytest

from glaurlex.core.io import (
    INFORMANT_COLUMNS,
    TOPIC_COLUMNS,
    decode_bytes,
    detect_topic_codes,
    parse_salamanca,
    parse_stimulus_map,
    read_salamanca,
)

SAMPLE = """\
10012 001 01 run, jump , swim
10012 002 01 jump, run
10022 001 02  ,  ,

10022 003 02 house, roof, door, window
"""


def test_parse_salamanca_builds_both_frames():
    parsed = parse_salamanca(SAMPLE, stimulus_map={"1": "ACTIONS"}, level_labels={"12": "B1"})

    assert list(parsed.topics) == ["ACTIONS", "topic_02"]
    actions = parsed.topics["ACTIONS"]
    assert tuple(actions.columns) == TOPIC_COLUMNS
    assert actions.to_numpy().tolist() == [
        [1, 0, "run"],
        [1, 1, "jump"],
        [1, 2, "swim"],
        [2, 0, "jump"],
        [2, 1, "run"],
    ]
    assert parsed.topics["topic_02"]["informant_id"].unique().tolist() == [3]

    informants = parsed.informants
    assert tuple(informants.columns) == INFORMANT_COLUMNS
    assert informants.to_numpy().tolist() == [
        [1, "10012", "12", "B1"],
        [1, "10022", "22", "22"],
        [2, "10012", "12", "B1"],
        [3, "10022", "22", "22"],
    ]


def test_stimulus_names_fall_back_to_prefix_and_strip_leading_zeros():
    parsed = parse_salamanca(
        "10012 001 007 a, b\n10012 001 08 c\n", {"7": "SEVEN"}, topic_prefix="tema_"
    )

    assert list(parsed.topics) == ["SEVEN", "tema_08"]


@pytest.mark.parametrize(
    ("text", "message"),
    [
        ("", "empty"),
        ("10012 001 01 a\nnot a record\n", "line 2"),
        ("1 001 01 a\n", "invalid experiment id"),
        ("10012 001 01 , ,\n", "no responses"),
    ],
)
def test_parse_salamanca_rejects_malformed_content(text, message):
    with pytest.raises(ValueError, match=message):
        parse_salamanca(text)


def test_read_salamanca_matches_parse_of_bytes(tmp_path):
    path = tmp_path / "input.txt"
    path.write_bytes(SAMPLE.encode("cp1252"))
    from_file = read_salamanca(path)
    from_text = parse_salamanca(SAMPLE)

    pd.testing.assert_frame_equal(from_file.informants, from_text.informants)
    assert from_file.topics.keys() == from_text.topics.keys()
    with pytest.raises(FileNotFoundError):
        read_salamanca(tmp_path / "missing.txt")


def test_detect_topic_codes_skips_invalid_lines():
    content = "10012 001 02 a\nheader line\n10012 002 01 b\n10012 003 02 c\n"

    assert detect_topic_codes(content) == ["02", "01"]
    assert detect_topic_codes(content.encode("utf-8")) == ["02", "01"]


ACCENTED = "casa, niño, camión, pingüino, árbol, corazón, montaña"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (ACCENTED.encode(), ACCENTED),
        (b"\xef\xbb\xbf" + ACCENTED.encode(), ACCENTED),
        (ACCENTED.encode("cp1252"), ACCENTED),
        (b"ni\xf1o \x81", "niño \x81"),
    ],
    ids=["utf-8", "utf-8-bom", "cp1252", "latin-1"],
)
def test_decode_bytes_recovers_accented_text(raw, expected):
    assert decode_bytes(raw) == expected


def test_parse_stimulus_map():
    assert parse_stimulus_map(" 1 = HOUSE \n\n02=Body parts = limbs\n") == {
        "1": "HOUSE",
        "02": "Body parts = limbs",
    }
    for text, message in (
        ("1 HOUSE", "line 1: expected"),
        ("1=A\n=B", "line 2: empty code"),
        ("1=", "empty stimulus"),
    ):
        with pytest.raises(ValueError, match=message):
            parse_stimulus_map(text)
