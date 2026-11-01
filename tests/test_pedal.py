"""Tests for qmidi.pedal -- the sustain pedal as editable spans.

The model is small but the consequences are not: replacing a track's CC64 is
the difference between "the pedal I drew" and "two pedals fighting", and the
order of a pedal-down against a note-off at the same tick decides whether a
note is caught or released.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mido
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.exporters import export_midi  # noqa: E402
from qmidi.pedal import (  # noqa: E402
    CC_SUSTAIN,
    PedalSpan,
    add_span,
    normalise,
    read_pedal,
    remove_at,
    span_at,
    spans_as_cc,
)
from qmidi.segments import SegmentProject, build_tracks  # noqa: E402

TPB = 480


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _cc(tick: int, value: int, channel: int = 0):
    return (tick, mido.Message("control_change", channel=channel,
                               control=CC_SUSTAIN, value=value))


def _on(tick: int, note: int, velocity: int = 80, channel: int = 0):
    return (tick, mido.Message("note_on", channel=channel, note=note,
                               velocity=velocity))


def _off(tick: int, note: int, channel: int = 0):
    return (tick, mido.Message("note_on", channel=channel, note=note, velocity=0))


def _build(path: Path, tracks: list[list[tuple]]) -> SegmentProject:
    midi = mido.MidiFile(ticks_per_beat=TPB, type=1)
    for events in tracks:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name="Piano", time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=600000, time=0))
        previous = 0
        for tick, msg in sorted(events, key=lambda item: item[0]):
            track.append(msg.copy(time=tick - previous))
            previous = tick
        midi.tracks.append(track)
    midi.save(str(path))
    seg_tracks = build_tracks(path, ticks_per_beat=TPB)
    return SegmentProject(source=path, ticks_per_beat=TPB, tracks=seg_tracks)


def _cc_events(path: Path, track_index: int) -> list[tuple[int, int]]:
    """``[(absolute tick, value), ...]`` for CC64 in one exported track."""
    track = mido.MidiFile(str(path)).tracks[track_index]
    tick = 0
    out = []
    for msg in track:
        tick += msg.time
        if msg.type == "control_change" and msg.control == CC_SUSTAIN:
            out.append((tick, msg.value))
    return out


# --------------------------------------------------------------------------- #
# the span model
# --------------------------------------------------------------------------- #
def test_span_rejects_empty_inverted_and_low_values():
    with pytest.raises(ValueError):
        PedalSpan(100, 100)
    with pytest.raises(ValueError):
        PedalSpan(200, 100)
    with pytest.raises(ValueError):
        PedalSpan(-1, 100)
    with pytest.raises(ValueError):
        PedalSpan(0, 100, value=63)  # below the down threshold is a lift


def test_read_pairs_down_and_up_per_track(tmp_path: Path):
    path = tmp_path / "a.mid"
    _build(path, [[_on(0, 60), _off(960, 60), _cc(0, 127), _cc(480, 0),
                   _cc(960, 127), _cc(1440, 0)]])
    spans = read_pedal(path)
    assert spans[0] == [PedalSpan(0, 480), PedalSpan(960, 1440)]


def test_read_closes_an_unreleased_pedal_at_the_end_of_the_track(tmp_path: Path):
    path = tmp_path / "a.mid"
    _build(path, [[_on(0, 60), _off(1920, 60), _cc(480, 127)]])
    assert read_pedal(path)[0] == [PedalSpan(480, 1920)]


def test_read_ignores_tracks_without_a_pedal(tmp_path: Path):
    path = tmp_path / "a.mid"
    _build(path, [[_on(0, 60), _off(960, 60)], [_on(0, 62), _off(960, 62)]])
    assert read_pedal(path) == {}


@pytest.mark.skipif(
    not (ROOT / "example" / "midi" / "Unchained.mid").is_file(),
    reason="example MIDI is not part of the public distribution",
)
def test_reading_the_sample_file_finds_the_pedal_tracks():
    """Only tracks 1 and 3 carry CC64 in Unchained.mid (see AGENTS.md §9.2)."""
    spans = read_pedal(ROOT / "example" / "midi" / "Unchained.mid")
    assert set(spans) == {1, 3}
    assert all(span.end_tick > span.start_tick for group in spans.values()
               for span in group)


def test_normalise_merges_overlaps_and_keeps_the_first_value():
    merged = normalise([PedalSpan(0, 480, 127), PedalSpan(240, 960, 100)])
    assert merged == [PedalSpan(0, 960, 127)]


def test_normalise_keeps_disjoint_spans_separate():
    assert normalise([PedalSpan(480, 960), PedalSpan(0, 240)]) == [
        PedalSpan(0, 240), PedalSpan(480, 960)
    ]


def test_normalise_merges_adjacent_spans():
    """A lift and an immediate re-press is one continuous pedal in the sound."""
    assert normalise([PedalSpan(0, 480), PedalSpan(480, 960)]) == [PedalSpan(0, 960)]


def test_add_span_unions_with_what_is_there():
    spans = add_span([PedalSpan(0, 480)], 240, 960)
    assert spans == [PedalSpan(0, 960)]


def test_add_span_normalises_a_backwards_drag():
    assert add_span([], 960, 480) == [PedalSpan(480, 960)]


def test_remove_at_drops_the_containing_span_only():
    spans = [PedalSpan(0, 480), PedalSpan(960, 1440)]
    kept, removed = remove_at(spans, 1000)
    assert removed and kept == [PedalSpan(0, 480)]
    kept, removed = remove_at(spans, 5000)
    assert not removed and kept == spans


def test_span_at_is_half_open():
    spans = [PedalSpan(0, 480)]
    assert span_at(spans, 0) == spans[0]
    assert span_at(spans, 479) == spans[0]
    assert span_at(spans, 480) is None


def test_spans_as_cc_emits_down_then_up():
    assert spans_as_cc([PedalSpan(0, 480, 127), PedalSpan(960, 1440)]) == [
        (0, 127), (480, 0), (960, 127), (1440, 0)
    ]


# --------------------------------------------------------------------------- #
# sidecar
# --------------------------------------------------------------------------- #
def test_pedal_survives_a_sidecar_round_trip(tmp_path: Path):
    project = _build(tmp_path / "a.mid", [[_on(0, 60), _off(960, 60)]])
    project.set_pedal_spans(0, [PedalSpan(0, 480), PedalSpan(600, 900)])
    again = SegmentProject.from_dict(project.to_dict(), project.source,
                                     notes_by_track=[t.notes for t in project.tracks])
    assert again.pedal_spans(0) == [PedalSpan(0, 480), PedalSpan(600, 900)]


def test_an_empty_pedal_list_is_remembered_as_an_edit(tmp_path: Path):
    """"no pedal on this track" must survive a save/load, so it can erase CC64."""
    project = _build(tmp_path / "a.mid", [[_on(0, 60), _off(960, 60), _cc(0, 127)]])
    project.set_pedal_spans(0, [])
    again = SegmentProject.from_dict(project.to_dict(), project.source,
                                     notes_by_track=[t.notes for t in project.tracks])
    assert 0 in again.pedals and again.pedal_spans(0) == []


# --------------------------------------------------------------------------- #
# export
# --------------------------------------------------------------------------- #
def test_export_replaces_the_tracks_cc64_with_the_spans(tmp_path: Path):
    project = _build(tmp_path / "a.mid", [[
        _on(0, 60), _off(480, 60), _on(480, 62), _off(960, 62),
        _cc(0, 127), _cc(240, 0), _cc(480, 127), _cc(960, 0),
    ]])
    project.set_pedal_spans(0, [PedalSpan(0, 720)])
    out = tmp_path / "out.mid"
    export_midi(project, out)
    assert _cc_events(out, 0) == [(0, 127), (720, 0)]


def test_export_erases_the_pedal_when_the_spans_are_empty(tmp_path: Path):
    project = _build(tmp_path / "a.mid", [[
        _on(0, 60), _off(960, 60), _cc(0, 127), _cc(480, 0),
    ]])
    project.set_pedal_spans(0, [])
    out = tmp_path / "out.mid"
    export_midi(project, out)
    assert _cc_events(out, 0) == []


def test_export_leaves_untouched_tracks_pedal_alone(tmp_path: Path):
    project = _build(tmp_path / "a.mid", [
        [_on(0, 60), _off(960, 60), _cc(0, 127), _cc(960, 0)],
        [_on(0, 62), _off(960, 62), _cc(0, 127), _cc(480, 0)],
    ])
    project.set_pedal_spans(0, [PedalSpan(0, 240)])
    out = tmp_path / "out.mid"
    export_midi(project, out)
    assert _cc_events(out, 0) == [(0, 127), (240, 0)]
    assert _cc_events(out, 1) == [(0, 127), (480, 0)]


def test_pedal_down_is_written_before_a_note_off_at_the_same_tick(tmp_path: Path):
    """Otherwise the note is released instead of caught by the pedal."""
    project = _build(tmp_path / "a.mid", [[_on(0, 60), _off(960, 60)]])
    project.set_pedal_spans(0, [PedalSpan(960, 1920)])
    out = tmp_path / "out.mid"
    export_midi(project, out)

    tick = 0
    order = []
    for msg in mido.MidiFile(str(out)).tracks[0]:
        tick += msg.time
        if tick != 960:
            continue
        if msg.type == "control_change" and msg.control == CC_SUSTAIN:
            order.append("pedal")
        elif msg.type == "note_on":
            order.append("note_off" if msg.velocity == 0 else "note_on")
    assert order == ["pedal", "note_off"]


def test_pedal_up_is_written_after_a_note_off_at_the_same_tick(tmp_path: Path):
    project = _build(tmp_path / "a.mid", [[_on(0, 60), _off(480, 60)]])
    project.set_pedal_spans(0, [PedalSpan(0, 480)])
    out = tmp_path / "out.mid"
    export_midi(project, out)

    tick = 0
    order = []
    for msg in mido.MidiFile(str(out)).tracks[0]:
        tick += msg.time
        if tick != 480:
            continue
        if msg.type == "control_change" and msg.control == CC_SUSTAIN:
            order.append("pedal_up")
        elif msg.type == "note_on" and msg.velocity == 0:
            order.append("note_off")
    assert order == ["note_off", "pedal_up"]
