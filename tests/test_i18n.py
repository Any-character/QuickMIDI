"""The lookup contract: identity in Chinese, table in English, fallback for gaps."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from qmidi.i18n import current_language, set_language, tr  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_chinese():
    yield
    set_language("zh")


def test_chinese_is_the_identity():
    set_language("zh")
    assert tr("打开") == "打开"


def test_english_comes_from_the_table():
    set_language("en")
    assert tr("打开") == "Open"
    assert tr("导出 MIDI") == "Export MIDI"
    assert tr("撤销") == "Undo"


def test_missing_entries_fall_back_to_chinese_instead_of_crashing():
    set_language("en")
    assert tr("这句话还没有翻译") == "这句话还没有翻译"


def test_unknown_language_codes_keep_chinese():
    set_language("fr")
    assert current_language() == "zh"
