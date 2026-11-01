"""Tests for qmidi.pedal_auto -- the harmonic pedal generator.

The rules are musical, so the tests state them as music: a chord that changes
is a pedal change, a chord that repeats is not, and the guards (min/max span)
are what stop the generator from turning a trill into a stutter.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmidi.pedal_auto import (  # noqa: E402
    AutoPedalOptions,
    generate_spans,
    harmonic_change_points,
)
from qmidi.segments import Note  # noqa: E402

TPB = 480
BEAT = TPB


def _chord(start_beat, pitches, *, length_beats=1, first_index=0):
    return [
        Note(index=first_index + i, channel=0, pitch=pitch, velocity=80,
             start_tick=int(start_beat * BEAT),
             end_tick=int((start_beat + length_beats) * BEAT))
        for i, pitch in enumerate(pitches)
    ]


def _melody(*specs):
    """``(start_beat, pitches)`` -> one flat note list with stable indices."""
    notes: list[Note] = []
    for start, pitches in specs:
        notes.extend(_chord(start, pitches, first_index=len(notes)))
    return notes


# --------------------------------------------------------------------------- #
# the musical rule
# --------------------------------------------------------------------------- #
def test_no_notes_means_no_pedal():
    assert generate_spans([], ticks_per_beat=TPB) == []


def test_pedal_starts_at_the_first_note_not_at_tick_zero():
    notes = _melody((2, (60, 64, 67)))
    spans = generate_spans(notes, ticks_per_beat=TPB)
    assert len(spans) == 1
    assert spans[0].start_tick == 2 * BEAT


def test_each_chord_change_gets_its_own_pedal():
    """C - F - G, one beat apart: three chords, three pedal spans."""
    notes = _melody((0, (48, 52, 55)), (1, (53, 57, 60)), (2, (55, 59, 62)))
    # One chord per beat: the shipped 2-beat floor would (correctly) merge them.
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=3 * BEAT,
                           options=AutoPedalOptions(min_span_beats=0.75))
    assert len(spans) == 3
    assert spans[0].start_tick == 0
    assert spans[1].start_tick > spans[0].end_tick  # the lift happens before the press
    assert spans[-1].end_tick == 3 * BEAT


def test_repeating_the_same_chord_does_not_re_take_the_pedal():
    notes = _melody((0, (48, 52, 55)), (1, (48, 52, 55)))
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=2 * BEAT)
    assert len(spans) == 1


def test_a_new_top_note_on_the_same_bass_still_changes_the_pedal():
    """C-E-G -> C-E-A: the bass is identical, the harmony is not.

    Two pitch classes change here, so the threshold has to be lowered: the
    shipped default of 4 is tuned for dense orchestral textures, where a
    two-note difference is an inner voice moving, not a new chord.
    """
    notes = _melody((0, (48, 52, 55)), (1, (48, 52, 57)))
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=2 * BEAT,
                           options=AutoPedalOptions(harmony_change=2,
                                                    min_span_beats=0.75))
    assert len(spans) == 2


def test_an_inversion_is_not_a_pedal_change():
    """C-E-G -> E-G-C: same three pitch classes, the pedal keeps holding."""
    notes = _melody((0, (48, 52, 55)), (1, (52, 55, 60)))
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=2 * BEAT,
                           options=AutoPedalOptions(harmony_change=2,
                                                    min_span_beats=0.75))
    assert len(spans) == 1


# --------------------------------------------------------------------------- #
# the guards
# --------------------------------------------------------------------------- #
def test_fast_changes_are_merged_by_the_minimum_span():
    """Four chords in one beat is a trill, not a pedal exercise."""
    notes = _melody((0, (60, 64, 67)), (0.125, (62, 65, 69)),
                    (0.25, (64, 67, 71)), (0.5, (65, 69, 72)))
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=BEAT,
                           options=AutoPedalOptions(min_span_beats=0.75))
    assert len(spans) == 1


def test_a_long_held_chord_is_split_by_the_maximum_span():
    notes = _chord(0, (48, 52, 55), length_beats=8)
    spans = generate_spans(
        notes, ticks_per_beat=TPB, end_tick=8 * BEAT,
        options=AutoPedalOptions(min_span_beats=0.5, max_span_beats=2.0),
    )
    assert len(spans) >= 4
    assert all(span.duration <= 2 * BEAT + 1 for span in spans)


def test_silence_between_phrases_does_not_extend_a_span_forever():
    notes = _chord(0, (60, 64, 67), length_beats=1) + _chord(8, (62, 65, 69),
                                                             length_beats=1,
                                                             first_index=3)
    spans = generate_spans(notes, ticks_per_beat=TPB)
    # No single span may cover the silence: the max-span ceiling re-takes the
    # pedal during the rest instead of holding one wash through it.
    assert all(s.end_tick - s.start_tick <= 6 * BEAT for s in spans)
    # The new phrase is still re-taken at its own onset, a hair after the change
    # point -- the lift has to come before the press or both CC64 messages would
    # land on one tick.
    starts = [s.start_tick for s in spans]
    assert any(8 * BEAT <= start <= 8 * BEAT + 40 for start in starts)


# --------------------------------------------------------------------------- #
# structural invariants the exporter relies on
# --------------------------------------------------------------------------- #
def test_spans_are_sorted_and_non_overlapping():
    notes = _melody((0, (48, 52, 55)), (1.5, (50, 53, 57)), (3, (43, 47, 50)))
    spans = generate_spans(notes, ticks_per_beat=TPB, end_tick=5 * BEAT)
    assert spans == sorted(spans, key=lambda s: s.start_tick)
    for earlier, later in zip(spans, spans[1:]):
        assert earlier.end_tick <= later.start_tick


def test_change_points_are_ascending_and_within_the_piece():
    notes = _melody((0, (48, 52, 55)), (2, (53, 57, 60)))
    points = harmonic_change_points(notes, ticks_per_beat=TPB, end_tick=4 * BEAT)
    assert points == sorted(points)
    assert points[0] == 0
    assert all(0 <= point <= 4 * BEAT for point in points)


@pytest.mark.parametrize("kwargs", [
    {"grid_beats": 0},
    {"max_span_beats": 0.5, "min_span_beats": 1.0},
    {"harmony_change": 0},
    {"lift_beats": -1},
])
def test_nonsense_options_are_rejected(kwargs):
    with pytest.raises(ValueError):
        AutoPedalOptions(**kwargs)
