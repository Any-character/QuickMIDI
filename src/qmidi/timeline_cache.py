"""Reuse a built event timeline between playbacks.

``offline.build_timeline`` parses the whole MIDI file and flattens it into one
absolute-time event list -- measured on the 4065-note sample file, 99 ms, on
every single press of play.  The list is read-only downstream
(``BlockStreamer`` sorts its own copy), so keeping it alive is safe.

Why the key is a content digest and not the file's mtime
--------------------------------------------------------
Rewriting the preview changes velocities but not the file size, and Windows can
serve a cached timestamp for a file rewritten milliseconds ago -- that is the
trap that made ``test_preview_midi_is_rewritten_only_when_the_project_changes``
flake in October.  A cache keyed on mtime would therefore occasionally play the
previous velocities, which is the one failure this editor cannot have.  Callers
pass ``PreviewRequest.timeline_key``, which carries the project digest.

The cache is deliberately duck-typed (no import of ``offline`` or ``engine``):
that keeps it testable without a MIDI file or an instrument, and keeps the
dependency arrow pointing one way.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Callable, Generic, Optional, TypeVar

__all__ = ["TimelineCache"]

T = TypeVar("T")


class TimelineCache(Generic[T]):
    """Least-recently-used store of built timelines, safe across threads.

    Only one entry is needed in practice (one preview file at a time); the
    capacity exists so a track-filter toggle does not immediately evict the
    previous view.
    """

    def __init__(self, capacity: int = 2) -> None:
        self._lock = threading.Lock()
        self._entries: "OrderedDict[tuple, T]" = OrderedDict()
        self._capacity = max(1, int(capacity))

    def get(self, key: Optional[tuple], build: Callable[[], T]) -> T:
        """The timeline for ``key``, building it on a miss.

        ``key=None`` means "do not cache" (used when the caller has no digest
        to trust); ``build()`` then runs every time.
        """
        if key is None:
            return build()
        with self._lock:
            hit = self._entries.get(key)
            if hit is not None:
                self._entries.move_to_end(key)
                return hit
        built = build()
        with self._lock:
            self._entries[key] = built
            self._entries.move_to_end(key)
            while len(self._entries) > self._capacity:
                self._entries.popitem(last=False)
        return built

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._entries)
