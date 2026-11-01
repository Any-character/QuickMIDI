"""The sustain pedal as an editable object.

MIDI stores the pedal as CC64 events -- a stream, not a shape.  What a user
actually edits is a **span**: pedal down at tick A, up at tick B.  Spans are
what you draw, drag and delete, and they turn back into exactly two CC64
messages at export.

Why replace instead of append
-----------------------------
A track that already carries 144 CC64 events cannot be improved by appending
more: two overlapping pedal streams fight, and the second one wins at random.
So for any track the user has touched, the model is the whole truth -- export
removes that track's CC64 messages and writes the spans instead.

Reading is therefore lossless about the two things that matter (where the pedal
goes down and up) and deliberately lossy about a re-press while the pedal is
already down, which changes nothing about the sound.

Nothing here imports Qt or pysfizz (AGENTS.md §5.1): the pedal model runs
headless, like the velocity segments it sits next to.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mido

__all__ = [
    "CC_SUSTAIN",
    "PEDAL_DOWN",
    "PedalSpan",
    "add_span",
    "normalise",
    "read_pedal",
    "remove_at",
    "span_at",
    "spans_as_cc",
]

#: MIDI continuous controller number of the damper/sustain pedal.
CC_SUSTAIN = 64

#: CC64 >= this value counts as "pedal down" (the spec's halfway point).  Lower
#: values are half-pedal on instruments that implement it; for the down/up model
#: used here they all mean "lifting".
PEDAL_DOWN = 64


@dataclass(frozen=True)
class PedalSpan:
    """Pedal held from ``start_tick`` to ``end_tick`` (half-open)."""

    start_tick: int
    end_tick: int
    value: int = 127

    def __post_init__(self) -> None:
        start, end, value = int(self.start_tick), int(self.end_tick), int(self.value)
        object.__setattr__(self, "start_tick", start)
        object.__setattr__(self, "end_tick", end)
        object.__setattr__(self, "value", value)
        if start < 0:
            raise ValueError(f"pedal start must be >= 0, got {start}")
        if end <= start:
            raise ValueError(f"pedal span [{start}, {end}) is empty or inverted")
        if not PEDAL_DOWN <= value <= 127:
            raise ValueError(
                f"pedal value must be {PEDAL_DOWN}..127 (below that is a lift), got {value}"
            )

    @property
    def duration(self) -> int:
        return self.end_tick - self.start_tick

    def contains(self, tick: int) -> bool:
        return self.start_tick <= tick < self.end_tick

    def moved_to(self, start_tick: int) -> "PedalSpan":
        return PedalSpan(start_tick, start_tick + self.duration, self.value)

    def to_dict(self) -> dict:
        return {"start_tick": self.start_tick, "end_tick": self.end_tick,
                "value": self.value}

    @classmethod
    def from_dict(cls, data: dict) -> "PedalSpan":
        return cls(int(data["start_tick"]), int(data["end_tick"]),
                   int(data.get("value", 127)))


def read_pedal(midi_path: Path | str) -> dict[int, list[PedalSpan]]:
    """``{track index: spans}`` for every track that carries CC64.

    A track with no CC64 is absent rather than present-and-empty: export reads
    "absent" as "leave this track's pedal exactly as the file wrote it".
    """
    midi = mido.MidiFile(str(midi_path))
    out: dict[int, list[PedalSpan]] = {}
    for index, track in enumerate(midi.tracks):
        tick = 0
        track_end = 0
        spans: list[PedalSpan] = []
        open_at: Optional[int] = None
        open_value = 127
        for msg in track:
            tick += msg.time
            track_end = max(track_end, tick)
            if msg.is_meta or msg.type != "control_change" or msg.control != CC_SUSTAIN:
                continue
            if msg.value >= PEDAL_DOWN:
                if open_at is None:
                    open_at, open_value = tick, int(msg.value)
            elif open_at is not None:
                if tick > open_at:
                    spans.append(PedalSpan(open_at, tick, open_value))
                open_at = None
        if open_at is not None and track_end > open_at:
            spans.append(PedalSpan(open_at, track_end, open_value))
        if spans:
            out[index] = spans
    return out


def normalise(spans: Iterable[PedalSpan]) -> list[PedalSpan]:
    """Sort and merge overlapping spans.

    Two overlapping spans are one pedal press in the model -- leaving them
    separate would emit a down/down/up/up sequence, and the second "up" would
    lift the pedal earlier than the user drew it.
    """
    ordered = sorted(spans, key=lambda span: (span.start_tick, span.end_tick))
    merged: list[PedalSpan] = []
    for span in ordered:
        if merged and span.start_tick <= merged[-1].end_tick:
            previous = merged[-1]
            if span.end_tick > previous.end_tick:
                merged[-1] = PedalSpan(previous.start_tick, span.end_tick,
                                       previous.value)
        else:
            merged.append(span)
    return merged


def add_span(spans: Sequence[PedalSpan], start_tick: int, end_tick: int,
             value: int = 127) -> list[PedalSpan]:
    """A new span unioned into the existing ones (drawing over is additive)."""
    start, end = int(start_tick), int(end_tick)
    if end < start:
        start, end = end, start
    if end - start < 1:
        end = start + 1
    return normalise([*spans, PedalSpan(max(0, start), max(1, end), value)])


def remove_at(spans: Sequence[PedalSpan], tick: int) -> tuple[list[PedalSpan], bool]:
    """Drop the span containing ``tick``; returns ``(spans, removed)``."""
    kept = [span for span in spans if not span.contains(int(tick))]
    return kept, len(kept) != len(spans)


def span_at(spans: Sequence[PedalSpan], tick: int) -> Optional[PedalSpan]:
    for span in spans:
        if span.contains(int(tick)):
            return span
    return None


def spans_as_cc(spans: Sequence[PedalSpan]) -> list[tuple[int, int]]:
    """``[(tick, cc value), ...]`` -- a down then an up for every span."""
    events: list[tuple[int, int]] = []
    for span in spans:
        events.append((span.start_tick, span.value))
        events.append((span.end_tick, 0))
    events.sort()
    return events
