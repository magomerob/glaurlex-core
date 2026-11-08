"""Parsing of Salamanca-format TXT files into the topic and informants frames.

Each non-empty line reads ``EXPID INDIV TOPIC word1, word2, ...``: an
experiment id whose last two digits encode the education level, an informant
number, a topic code and the comma-separated responses in production order.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Mapping
    from os import PathLike

_LINE_RE = re.compile(r"^\s*(\d+)\s+(\d+)\s+(\d+)\s*(.*?)\s*$")
_LEVEL_CODE_LEN = 2

INFORMANT_COLUMNS = ("informant_id", "experiment_id", "level_code", "level")
TOPIC_COLUMNS = ("informant_id", "pos", "type")


@dataclass(frozen=True)
class ParsedDataset:
    """Frames produced by :func:`parse_salamanca`.

    Attributes
    ----------
    informants : pandas.DataFrame
        Columns ``INFORMANT_COLUMNS``, one row per distinct informant and
        experiment, sorted by ``informant_id`` and ``experiment_id``.
    topics : dict of str to pandas.DataFrame
        ``topic name -> topic frame`` (``TOPIC_COLUMNS``), in order of first
        appearance, rows sorted by ``informant_id`` and ``pos``.
    """

    informants: pd.DataFrame
    topics: dict[str, pd.DataFrame]


def decode_bytes(raw: bytes) -> str:
    """Decode text of unknown encoding.

    Parameters
    ----------
    raw : bytes

    Returns
    -------
    str

    Notes
    -----
    Tries UTF-8 (with or without BOM), then cp1252, then latin-1, which
    accepts any byte sequence. Statistical detectors are deliberately not
    used: on short Spanish text they tend to pick code pages such as cp775 or
    cp1006, silently corrupting accented letters.
    """
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    try:
        return raw.decode("cp1252")
    except UnicodeDecodeError:
        return raw.decode("latin-1")


def _as_text(source: str | bytes) -> str:
    return decode_bytes(source) if isinstance(source, bytes) else source


def detect_topic_codes(source: str | bytes) -> list[str]:
    """List the topic codes of a Salamanca file without validating it.

    Parameters
    ----------
    source : str or bytes
        File content. Lines that do not match the format are skipped.

    Returns
    -------
    list of str
        Distinct codes in order of first appearance.
    """
    codes: dict[str, None] = {}
    for line in _as_text(source).splitlines():
        match = _LINE_RE.match(line.strip())
        if match:
            codes.setdefault(match.group(3), None)
    return list(codes)


def parse_stimulus_map(text: str) -> dict[str, str]:
    """Parse ``code=stimulus`` lines into a topic-name mapping.

    Parameters
    ----------
    text : str
        One ``code=stimulus`` pair per line; blank lines are ignored and
        surrounding whitespace is stripped.

    Returns
    -------
    dict of str to str

    Raises
    ------
    ValueError
        If a line has no ``=`` or an empty code or stimulus. The message
        starts with the 1-based line number.
    """
    mapping: dict[str, str] = {}
    for line_no, line in enumerate(text.splitlines(), start=1):
        row = line.strip()
        if not row:
            continue
        code, sep, stimulus = (part.strip() for part in row.partition("="))
        if not sep:
            msg = f"line {line_no}: expected 'code=stimulus'"
            raise ValueError(msg)
        if not code:
            msg = f"line {line_no}: empty code"
            raise ValueError(msg)
        if not stimulus:
            msg = f"line {line_no}: empty stimulus"
            raise ValueError(msg)
        mapping[code] = stimulus
    return mapping


def _topic_name(code: str, stimulus_map: Mapping[str, str] | None, prefix: str) -> str:
    if stimulus_map:
        if code in stimulus_map:
            return str(stimulus_map[code])
        stripped = code.lstrip("0")
        if stripped and stripped in stimulus_map:
            return str(stimulus_map[stripped])
    return f"{prefix}{code}"


def parse_salamanca(
    source: str | bytes,
    stimulus_map: Mapping[str, str] | None = None,
    level_labels: Mapping[str, str] | None = None,
    topic_prefix: str = "topic_",
) -> ParsedDataset:
    """Parse the content of a Salamanca-format file.

    Parameters
    ----------
    source : str or bytes
        File content; bytes are decoded with :func:`decode_bytes`.
    stimulus_map : mapping, optional
        ``topic code -> topic name``. Codes are also looked up without
        leading zeros. Unmapped codes are named ``topic_prefix + code``.
    level_labels : mapping, optional
        ``level code -> label`` for the last two digits of the experiment id.
        Unmapped codes are used as their own label.
    topic_prefix : str, default "topic_"
        Prefix of the names given to unmapped topic codes.

    Returns
    -------
    ParsedDataset

    Raises
    ------
    ValueError
        If the content is empty, a non-empty line does not match the format,
        an experiment id has fewer than two digits, or no responses are found.
    """
    lines = _as_text(source).splitlines()
    if not lines:
        msg = "the Salamanca file is empty"
        raise ValueError(msg)

    informant_rows: list[tuple[int, str, str, str]] = []
    topic_rows: dict[str, list[tuple[int, int, str]]] = {}
    for line_no, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line:
            continue
        match = _LINE_RE.match(line)
        if match is None:
            msg = f"line {line_no}: expected 'EXPID INDIV TOPIC word1, word2, ...'"
            raise ValueError(msg)
        experiment_id, individual, code, words = match.groups()
        if len(experiment_id) < _LEVEL_CODE_LEN:
            msg = f"line {line_no}: invalid experiment id {experiment_id!r}"
            raise ValueError(msg)
        level_code = experiment_id[-_LEVEL_CODE_LEN:]
        level = (level_labels or {}).get(level_code, level_code)
        informant_id = int(individual)
        informant_rows.append((informant_id, experiment_id, level_code, level))

        tokens = [token for token in (t.strip() for t in words.split(",")) if token]
        if tokens:
            rows = topic_rows.setdefault(_topic_name(code, stimulus_map, topic_prefix), [])
            rows.extend((informant_id, pos, token) for pos, token in enumerate(tokens))

    if not informant_rows:
        msg = "no valid lines found"
        raise ValueError(msg)
    if not topic_rows:
        msg = "no responses found"
        raise ValueError(msg)

    informants = (
        pd.DataFrame(informant_rows, columns=list(INFORMANT_COLUMNS))
        .drop_duplicates()
        .sort_values(["informant_id", "experiment_id"])
        .reset_index(drop=True)
    )
    topics = {
        name: pd.DataFrame(rows, columns=list(TOPIC_COLUMNS))
        .sort_values(["informant_id", "pos"])
        .reset_index(drop=True)
        for name, rows in topic_rows.items()
    }
    return ParsedDataset(informants=informants, topics=topics)


def read_salamanca(
    path: str | PathLike[str],
    stimulus_map: Mapping[str, str] | None = None,
    level_labels: Mapping[str, str] | None = None,
    topic_prefix: str = "topic_",
) -> ParsedDataset:
    """Read and parse a Salamanca-format file.

    Parameters
    ----------
    path : str or path-like
    stimulus_map, level_labels, topic_prefix
        See :func:`parse_salamanca`.

    Returns
    -------
    ParsedDataset

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        See :func:`parse_salamanca`.
    """
    with open(path, "rb") as handle:
        raw = handle.read()
    return parse_salamanca(raw, stimulus_map, level_labels, topic_prefix)
