"""TimelineCache: reuse a parsed event list between playbacks.

Parsing the preview costs ~99 ms per press of play.  The cache must never hand
back an event list built from *different* content, which is why the key carries
the project digest rather than a file timestamp (see ``qmidi.timeline_cache``).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.timeline_cache import TimelineCache  # noqa: E402


def _counting_builder(payload: list) -> tuple:
    calls = []

    def build() -> list:
        calls.append(1)
        return list(payload)

    return build, calls


def test_the_same_key_is_built_once_and_returned_identically():
    cache = TimelineCache()
    build, calls = _counting_builder([1, 2, 3])
    first = cache.get(("song.mid", "abc", frozenset({0})), build)
    second = cache.get(("song.mid", "abc", frozenset({0})), build)
    assert first is second
    assert len(calls) == 1


def test_a_new_digest_forces_a_rebuild():
    cache = TimelineCache()
    build, calls = _counting_builder([1])
    cache.get(("song.mid", "abc", frozenset({0})), build)
    cache.get(("song.mid", "def", frozenset({0})), build)
    assert len(calls) == 2


def test_a_different_track_filter_is_a_different_entry():
    cache = TimelineCache()
    build, calls = _counting_builder([1])
    cache.get(("song.mid", "abc", frozenset({0})), build)
    cache.get(("song.mid", "abc", frozenset({0, 1})), build)
    assert len(calls) == 2


def test_no_key_means_no_caching():
    """A request without a digest must rebuild every time, never reuse."""
    cache = TimelineCache()
    build, calls = _counting_builder([1])
    cache.get(None, build)
    cache.get(None, build)
    assert len(calls) == 2
    assert len(cache) == 0


def test_capacity_evicts_the_least_recently_used_entry():
    cache = TimelineCache(capacity=2)
    build, calls = _counting_builder([1])
    cache.get(("a",), build)
    cache.get(("b",), build)
    cache.get(("a",), build)  # 'a' becomes the most recent again
    cache.get(("c",), build)  # evicts 'b'
    assert len(cache) == 2
    cache.get(("a",), build)
    assert len(calls) == 3, "the recently used entry was evicted"
    cache.get(("b",), build)
    assert len(calls) == 4


def test_clear_forgets_everything():
    cache = TimelineCache()
    build, calls = _counting_builder([1])
    cache.get(("a",), build)
    cache.clear()
    cache.get(("a",), build)
    assert len(calls) == 2
