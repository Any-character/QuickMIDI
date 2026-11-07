"""EngineCache: one loaded instrument reused across playbacks.

The pool exists because rebuilding an engine costs 51 ms warm and 450 ms cold
(measured 2026-11-07), which is most of the wait after pressing play.  What must
hold is *ownership*: an engine handed to one worker cannot come out of the pool
again until that worker returns it, so two renderer threads can never drive one
instrument at the same time.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.engine import EngineCache  # noqa: E402


class _FakeEngine:
    """Stands in for SfzEngine; the cache never calls into it."""


def _counter():
    made = []

    def build() -> _FakeEngine:
        made.append(_FakeEngine())
        return made[-1]

    return build, made


def test_a_returned_engine_is_handed_out_again():
    cache = EngineCache()
    build, made = _counter()
    first = cache.acquire(("piano",), build)
    assert made == [first]
    cache.release(("piano",), first)
    assert cache.acquire(("piano",), build) is first
    assert len(made) == 1, "the instrument was loaded twice"


def test_an_engine_on_loan_is_not_handed_out_twice():
    cache = EngineCache()
    build, made = _counter()
    first = cache.acquire(("piano",), build)
    second = cache.acquire(("piano",), build)
    assert second is not first, "the same engine was lent to two callers"
    assert len(made) == 2


def test_a_different_key_builds_a_new_engine():
    cache = EngineCache()
    build, made = _counter()
    piano = cache.acquire(("piano",), build)
    cache.release(("piano",), piano)
    other = cache.acquire(("organ",), build)
    assert other is not piano
    assert len(made) == 2


def test_capacity_drops_the_oldest_idle_engine():
    cache = EngineCache(capacity=1)
    build, made = _counter()
    piano = cache.acquire(("piano",), build)
    cache.release(("piano",), piano)
    organ = cache.acquire(("organ",), build)
    cache.release(("organ",), organ)  # evicts the idle piano
    assert cache.idle == 1
    rebuilt = cache.acquire(("piano",), build)
    assert rebuilt is not piano
    assert len(made) == 3


def test_a_failed_build_leaves_the_pool_untouched():
    cache = EngineCache()
    survivor = _FakeEngine()
    cache.release(("piano",), survivor)

    def broken() -> _FakeEngine:
        raise RuntimeError("sfz refused to load")

    with pytest.raises(RuntimeError):
        cache.acquire(("organ",), broken)  # a miss, so ``broken`` really runs
    assert cache.acquire(("piano",), lambda: _FakeEngine()) is survivor


def test_release_is_safe_to_call_with_a_stale_key():
    """A worker that outlived its request still hands its engine back."""
    cache = EngineCache()
    engine = _FakeEngine()
    cache.release(("whatever",), engine)
    assert cache.idle == 1
