"""Tests for qmidi.segments -- the velocity-segment model.

Two flavours of fixture:

* synthetic notes (and tiny synthetic .mid files) for the algorithm's edge
  cases -- overlapping notes, jitter, periodic figures, mid-note splits;
* the real Unchained.mid for the invariants that must hold on a real take
  (no note split, every note covered, JSON round-trip).
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import mido
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.segments import (  # noqa: E402
    REASON_CYCLE,
    REASON_MANUAL,
    REASON_MERGED,
    REASON_SAME,
    AutoSegmentOptions,
    MergeConflictError,
    Note,
    SegmentError,
    SegmentedTrack,
    SegmentProject,
    SplitInsideNoteError,
    VelocitySegment,
    auto_segment,
    build_tracks,
    clamp_velocity,
    load_project,
    read_notes,
    sidecar_path,
)
from qmidi.segments import _snap_cuts  # noqa: E402  (unit under test)

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():  # example/ exists on the local copy only
    pytest.skip("example/midi/Unchained.mid is not part of the public distribution",
                allow_module_level=True)
TPB = 480


# --------------------------------------------------------------------------- #
# fixtures
# --------------------------------------------------------------------------- #
def _n(start: int, end: int, vel: int, pitch: int = 60, channel: int = 0) -> Note:
    return Note(index=0, channel=channel, pitch=pitch, velocity=vel,
                start_tick=start, end_tick=end)


def _seq(*pairs: tuple[int, int, int]) -> list[Note]:
    """``_seq((start, dur, vel), ...)`` -> non-overlapping notes with stable indices."""
    return [Note(i, 0, 60 + i, vel, start, start + dur)
            for i, (start, dur, vel) in enumerate(pairs)]


def _melody(velocities: list[int], *, beat: int = 240, start: int = 0) -> list[Note]:
    """One note per entry, back to back -- the classic 80-70-60 case."""
    return [Note(i, 0, 60 + i, v, start + i * beat, start + (i + 1) * beat)
            for i, v in enumerate(velocities)]


def _track(notes: list[Note], **kwargs) -> SegmentedTrack:
    opts = kwargs.pop("options", None)
    segments = auto_segment(notes, track=0, ticks_per_beat=TPB, options=opts)
    return SegmentedTrack(track=0, notes=list(notes), segments=segments)


def _write_midi(path: Path, track_events, tpb: int = TPB) -> Path:
    mid = mido.MidiFile(ticks_per_beat=tpb)
    tr = mido.MidiTrack()
    tr.append(mido.MetaMessage("track_name", name="test", time=0))
    tick = 0
    for kind, *rest in track_events:
        if kind == "note":
            start, dur, pitch, vel = rest
            tr.append(mido.Message("note_on", note=pitch, velocity=vel, time=start - tick))
            tick = start
            tr.append(mido.Message("note_on", note=pitch, velocity=0, time=dur))
            tick += dur
        elif kind == "hang":  # note_on with no matching off
            start, pitch, vel = rest
            tr.append(mido.Message("note_on", note=pitch, velocity=vel, time=start - tick))
            tick = start
        elif kind == "raw":  # raw message with an explicit delta
            msg, delta = rest
            tr.append(msg.copy(time=delta))
            tick += delta
        elif kind == "wait":
            tr.append(mido.MetaMessage("marker", text=str(rest[0]), time=rest[1]))
            tick += rest[1]
    mid.tracks.append(tr)
    mid.save(str(path))
    return path


# --------------------------------------------------------------------------- #
# reading notes
# --------------------------------------------------------------------------- #
def test_read_notes_folds_velocity_zero_note_on(tmp_path):
    """A vel-0 note_on is a note-off; trusting msg.type leaves notes ringing."""
    path = _write_midi(tmp_path / "v0.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 480, 240, 62, 90),
    ])
    notes = read_notes(path)[0]
    assert [(n.pitch, n.velocity, n.start_tick, n.end_tick) for n in notes] == [
        (60, 80, 0, 240),
        (62, 90, 480, 720),
    ]


def test_read_notes_matches_the_real_file_note_count():
    """Unchained.mid: 4065 note_ons, all with vel>0, and zero note_off messages."""
    tracks = read_notes(MIDI)
    raw_ons = 0
    raw_offs = 0
    for track in mido.MidiFile(str(MIDI)).tracks:
        for msg in track:
            if msg.type == "note_on" and msg.velocity > 0:
                raw_ons += 1
            elif msg.type == "note_off":
                raw_offs += 1
    assert raw_offs == 0
    assert sum(len(t) for t in tracks) == raw_ons == 4065


def test_read_notes_closes_a_dangling_note_at_the_track_end(tmp_path):
    path = _write_midi(tmp_path / "dangle.mid", [
        ("hang", 0, 60, 80),
        ("wait", "end", 1000),
    ])
    notes = read_notes(path)[0]
    assert notes[0].end_tick == 1000, "an unterminated note must not vanish"


def test_read_notes_keeps_a_zero_length_note_as_a_one_tick_attack(tmp_path):
    """A note whose off is on the same tick is still an attack; dropping it loses music."""
    path = _write_midi(tmp_path / "staccato.mid", [
        ("note", 0, 0, 60, 80),
        ("note", 240, 240, 62, 90),
    ])
    notes = read_notes(path)[0]
    assert [n.duration for n in notes] == [1, 240]
    assert notes[0].contains_tick(0) is False, "a 1-tick note can never be split"


def test_read_notes_pairs_overlapping_pitches_fifo(tmp_path):
    """The first release closes the *oldest* voice, as offline.SustainState does."""
    # _write_midi takes deltas separately, so every message carries time=0:
    # note_on@0, note_on@240, note_off@480, note_off@720.
    path = _write_midi(tmp_path / "overlap.mid", [
        ("raw", mido.Message("note_on", channel=0, note=60, velocity=40, time=0), 0),
        ("raw", mido.Message("note_on", channel=0, note=60, velocity=90, time=0), 240),
        ("raw", mido.Message("note_off", channel=0, note=60, velocity=64, time=0), 240),
        ("raw", mido.Message("note_off", channel=0, note=60, velocity=64, time=0), 240),
    ])
    notes = read_notes(path)[0]
    assert [(n.velocity, n.start_tick, n.end_tick) for n in notes] == [
        (40, 0, 480),      # the first note_on owns the first note_off
        (90, 240, 720),
    ]


def test_read_notes_drops_nothing_from_the_real_file():
    """4065 attacks in, 4065 notes out -- including the 33 zero-length ones."""
    tracks = read_notes(MIDI)
    total = sum(len(t) for t in tracks)
    assert total == 4065
    assert sum(1 for t in tracks for n in t if n.duration == 1) == 33


# --------------------------------------------------------------------------- #
# rule 2a: same value -> one segment
# --------------------------------------------------------------------------- #
def test_same_value_runs_become_one_segment_each():
    notes = _seq((0, 240, 60), (240, 240, 60), (480, 240, 80), (720, 240, 80))
    segs = _track(notes).segments
    assert [(s.start_tick, s.end_tick) for s in segs] == [(0, 480), (480, 960)]
    assert [s.label for s in segs] == ["同值60", "同值80"]
    assert all(s.reason == REASON_SAME for s in segs)


def test_velocity_tolerance_absorbs_performance_jitter():
    """+-2 on a repeated gesture must not make one segment per note."""
    notes = _seq((0, 240, 80), (240, 240, 82), (480, 240, 79), (720, 240, 81))
    strict = _track(notes, options=AutoSegmentOptions(velocity_tolerance=0))
    loose = _track(notes, options=AutoSegmentOptions(velocity_tolerance=4))
    assert len(strict.segments) == 4
    assert len(loose.segments) == 1


def test_tolerance_zero_reproduces_literal_equality():
    notes = _seq((0, 240, 60), (240, 240, 61), (480, 240, 60))
    segs = _track(notes, options=AutoSegmentOptions(velocity_tolerance=0)).segments
    assert len(segs) == 3


# --------------------------------------------------------------------------- #
# rule 2b: a repeating figure is ONE segment
# --------------------------------------------------------------------------- #
def test_repeating_cycle_is_not_cut_per_note():
    """The headline case: 80-70-60 over and over is one gesture, not 6 segments."""
    pattern = [80, 70, 60] * 2  # 6 notes = 6 beats, inside the 8-beat window
    segs = _track(_melody(pattern)).segments
    assert len(segs) == 1, [(s.start_tick, s.end_tick, s.label) for s in segs]
    assert segs[0].reason == REASON_CYCLE
    assert segs[0].reason_values == (80, 70, 60)
    assert segs[0].label == "周期80-70-60"
    assert segs[0].is_periodic
    assert len(segs[0].note_indices) == 6


def test_a_long_ostinato_stays_one_segment():
    """A repeating figure is one gesture however long it runs.

    It used to be chunked at ``period_beats``, which produced one long drag
    handle per chunk -- and, on the real file, chopped a single repeated phrase
    into ten segments.  The window now only gates *starting* a cycle; once one is
    found it extends for as long as it keeps repeating.
    """
    segs = _track(_melody([80, 70, 60] * 8)).segments
    assert len(segs) == 1
    assert segs[0].is_periodic
    assert len(segs[0].note_indices) == 24


def test_two_cycle_values_stay_one_segment():
    segs = _track(_melody([127, 96] * 2)).segments
    assert len(segs) == 1 and segs[0].reason_values == (127, 96)


def test_a_single_repetition_is_not_treated_as_periodic():
    """min_cycles=2: 80-70-60-80-70 happens in real music without being an ostinato."""
    segs = _track(_melody([80, 70, 60, 80, 70])).segments
    assert len(segs) == 5
    assert not any(s.is_periodic for s in segs)


def test_irregular_alternation_is_still_cut():
    """80/70/60/100/70/60 has no cycle, so it must not be swallowed."""
    segs = _track(_melody([80, 70, 60, 100, 70, 60])).segments
    assert len(segs) > 1
    assert not any(s.is_periodic for s in segs)


def test_period_window_is_the_merging_parameter():
    """Same notes, wider window -> fewer, longer segments.  This is the tunable knob."""
    notes = _melody([80, 70, 60] * 4 + [40] + [80, 70, 60] * 4)
    narrow = _track(notes, options=AutoSegmentOptions(period_beats=2.0))
    wide = _track(notes, options=AutoSegmentOptions(period_beats=32.0))
    assert len(narrow.segments) > len(wide.segments)
    assert wide.segments[0].is_periodic


def test_max_cycle_len_bounds_what_counts_as_a_figure():
    """A 5-note ostinato is longer than max_cycle_len=4 -> stays cut."""
    notes = _melody([80, 70, 60, 50, 40] * 3)
    bounded = _track(notes, options=AutoSegmentOptions(max_cycle_len=4))
    assert not any(s.is_periodic for s in bounded.segments)


def test_min_cycles_raises_the_evidence_bar():
    """Same7 notes: two repetitions prove a figure, one repetition is coincidence."""
    notes = _melody([80, 70, 60] * 2 + [40])
    two = _track(notes, options=AutoSegmentOptions(min_cycles=2)).segments
    three = _track(notes, options=AutoSegmentOptions(min_cycles=3)).segments
    assert [s.is_periodic for s in two] == [True, False]
    assert len(three) == 7 and not any(s.is_periodic for s in three)


def test_cycles_do_not_run_past_the_window():
    """period_beats=4 with a figure that spans 6 beats -> no cycle."""
    notes = _melody([80, 70, 60, 80, 70, 60], beat=480)  # 4 beats per note
    segs = _track(notes, options=AutoSegmentOptions(period_beats=4.0)).segments
    assert not any(s.is_periodic for s in segs)


def test_bad_options_are_rejected():
    for opts in (
        AutoSegmentOptions(velocity_tolerance=-1),
        AutoSegmentOptions(period_beats=0.0),
        AutoSegmentOptions(min_cycles=1),
        AutoSegmentOptions(max_cycle_len=1),
    ):
        with pytest.raises(SegmentError):
            auto_segment(_melody([80, 70]), options=opts)


# --------------------------------------------------------------------------- #
# rule 2c: a boundary never lands inside a note
# --------------------------------------------------------------------------- #
def test_boundary_never_splits_a_note_when_a_long_note_is_held():
    """The realistic case: a held bass under a moving melody."""
    notes = [
        Note(0, 0, 36, 40, 0, 4000),       # held the whole time
        Note(1, 0, 60, 80, 0, 240),
        Note(2, 0, 62, 60, 240, 480),
        Note(3, 0, 64, 80, 480, 720),
    ]
    segs = auto_segment(notes, ticks_per_beat=TPB)
    for seg in segs:
        for note in notes:
            inside = note.start_tick < seg.start_tick < note.end_tick
            assert not inside, f"cut {seg.start_tick} splits note {note.index}"
    track = SegmentedTrack(track=0, notes=notes, segments=segs)
    track.validate()


def test_cut_moves_to_the_nearest_note_edge():
    """A cut requested inside a sounding note lands on that note's nearest edge.

    A note is held across 0..1600 while the velocity changes at 200 and 1100 --
    both candidate cuts fall inside it.  The cut at 200 is nearer the note's
    start (200 vs 1400) and snaps back to 0; the cut at 1100 is nearer the note's
    end (500 vs 1100) and snaps forward to 1600.  Cutting *both* ways is the
    point: a rule that only walked backwards would have merged 12 beats of
    melody into the previous segment.
    """
    notes = [
        Note(0, 0, 36, 80, 0, 1600),       # held under everything else
        Note(1, 0, 60, 60, 200, 1000),
        Note(2, 0, 62, 80, 1100, 1400),
    ]
    assert _snap_cuts(notes, [200, 1100]) == [0, 1600]


def test_auto_cut_never_loses_a_note_while_snapping():
    """Whatever the snap decides, every note still lands in exactly one segment."""
    notes = [
        Note(0, 0, 36, 80, 0, 1600),
        Note(1, 0, 60, 60, 200, 1000),
        Note(2, 0, 62, 80, 1100, 1400),
        Note(3, 0, 64, 100, 1700, 1940),
    ]
    segs = auto_segment(notes, ticks_per_beat=TPB)
    covered = [i for s in segs for i in s.note_indices]
    assert sorted(covered) == [0, 1, 2, 3]
    SegmentedTrack(track=0, notes=notes, segments=segs).validate()


def test_every_note_belongs_to_exactly_one_segment_on_the_real_file():
    tracks = build_tracks(MIDI)
    for t in tracks:
        t.validate()
        seen = [i for s in t.segments for i in s.note_indices]
        assert sorted(seen) == [n.index for n in t.notes]
        assert len(seen) == len(set(seen))


def test_segments_partition_the_track_without_gaps():
    for t in build_tracks(MIDI):
        assert t.segments[0].start_tick == 0
        for a, b in zip(t.segments, t.segments[1:]):
            assert a.end_tick == b.start_tick
        assert t.segments[-1].end_tick == t.end_tick


def test_segment_count_is_much_smaller_than_the_note_count():
    """A cut that produced one segment per note would be useless to drag."""
    tracks = build_tracks(MIDI)
    assert all(len(t.segments) < len(t.notes) / 5 for t in tracks if t.notes)


# --------------------------------------------------------------------------- #
# segment operations
# --------------------------------------------------------------------------- #
def test_apply_delta_moves_a_whole_segment():
    track = _track(_seq((0, 240, 60), (240, 240, 60), (480, 240, 80), (720, 240, 80)))
    track.apply_delta(0, -20)
    track.apply_delta(1, 20)
    out = {n.pitch: n.velocity for n in track.resolved_notes()}
    assert out == {60: 40, 61: 40, 62: 100, 63: 100}
    # repeated nudges accumulate rather than overwrite
    track.apply_delta(0, -5)
    assert track.segments[0].delta == -25


def test_delta_is_clamped_at_resolution_not_stored_out_of_range():
    track = _track(_seq((0, 240, 5), (240, 240, 120)))
    track.apply_delta(0, -100)
    track.apply_delta(1, 100)
    assert track.resolved_notes()[0].velocity == 1
    assert track.resolved_notes()[1].velocity == 127


def test_set_value_pins_a_segment():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    track.set_value(0, 33)
    assert [n.velocity for n in track.resolved_notes()] == [33, 100]
    assert track.segments[0].target == 33


def test_set_value_out_of_range_is_rejected():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    with pytest.raises(SegmentError):
        track.set_value(0, 200)
    with pytest.raises(SegmentError):
        track.set_value(0, 0)


def test_target_wins_over_delta():
    seg = VelocitySegment(0, 0, 100, delta=10, target=50)
    assert seg.velocity_for(_n(0, 100, 60)) == 50


def test_clear_adjustment_restores_the_original_velocities():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    track.apply_delta(0, -10)
    track.clear_adjustment(0)
    assert [n.velocity for n in track.resolved_notes()] == [60, 100]


def test_merge_adjacent_unions_two_segments():
    track = _track(_seq((0, 240, 60), (240, 240, 60), (480, 240, 80), (720, 240, 80)))
    track.apply_delta(0, -10)
    track.apply_delta(1, 5)
    merged = track.merge_adjacent(0)
    assert len(track.segments) == 1
    assert merged.delta == -5, "deltas must add up, not overwrite"
    assert merged.reason == REASON_MERGED
    assert merged.note_indices == (0, 1, 2, 3)
    assert len(track.resolved_notes()) == 4


def test_merge_adjacent_refuses_conflicting_pinned_velocities():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    track.set_value(0, 33)
    track.set_value(1, 99)
    with pytest.raises(MergeConflictError):
        track.merge_adjacent(0)
    assert len(track.segments) == 2


def test_merge_adjacent_at_the_end_is_an_error():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    with pytest.raises(SegmentError):
        track.merge_adjacent(1)


def test_merge_all_collapses_the_whole_track():
    track = _track(_seq(*[(i * 240, 240, 40 + i * 20) for i in range(6)]))
    assert len(track.segments) == 6
    assert len(track.merge_all()) == 1
    track.validate()


# --------------------------------------------------------------------------- #
# manual splitting
# --------------------------------------------------------------------------- #
def test_split_at_tick_cuts_on_a_note_edge():
    track = _track(_seq((0, 480, 60), (480, 480, 60)))
    assert len(track.segments) == 1
    right = track.split_at_tick(480)
    assert len(track.segments) == 2
    assert right.start_tick == 480 and right.reason == REASON_MANUAL
    assert track.segments[0].end_tick == 480
    track.validate()


def test_split_children_inherit_the_adjustment():
    """A user who nudged a passage then cuts it expects both halves to stay nudged."""
    track = _track(_seq((0, 480, 60), (480, 480, 60), (960, 480, 60)))
    track.apply_delta(0, -15)
    track.split_at_tick(480)
    assert [s.delta for s in track.segments] == [-15, -15]
    assert {n.velocity for n in track.resolved_notes()} == {45}


def test_split_inside_a_note_is_rejected():
    """Spec rule 2c: a cut through a sounding note is an error, not a silent split."""
    track = _track(_seq((0, 480, 60), (480, 480, 100)))
    before = len(track.segments)
    with pytest.raises(SplitInsideNoteError) as excinfo:
        track.split_at_tick(700)
    assert "inside note" in str(excinfo.value)
    assert len(track.segments) == before, "the rejected split must not have mutated state"
    track.validate()


def test_split_at_an_existing_boundary_is_rejected():
    track = _track(_seq((0, 480, 60), (480, 480, 100)))
    with pytest.raises(SegmentError):
        track.split_at_tick(0)
    with pytest.raises(SegmentError):
        track.split_at_tick(480)


def test_split_outside_every_segment_is_rejected():
    track = _track(_seq((0, 480, 60), (480, 480, 100)))
    with pytest.raises(SegmentError):
        track.split_at_tick(99999)


# --------------------------------------------------------------------------- #
# model validation
# --------------------------------------------------------------------------- #
def test_empty_or_inverted_segment_is_rejected():
    with pytest.raises(SegmentError):
        VelocitySegment(0, 100, 100)
    with pytest.raises(SegmentError):
        VelocitySegment(0, 200, 100)
    with pytest.raises(SegmentError):
        VelocitySegment(0, -1, 100)


def test_segment_index_out_of_range_is_rejected():
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    with pytest.raises(SegmentError):
        track.apply_delta(7, 10)


def test_validate_catches_a_split_note_and_a_missing_one():
    notes = _seq((0, 480, 60), (480, 480, 100))
    track = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 960, note_indices=(0, 1)),
    ])
    track.validate()  # a hand-built but legal layout is fine
    broken = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 700, note_indices=(0,)),
    ])
    with pytest.raises(SegmentError):
        broken.validate()


def test_clamp_velocity_bounds():
    assert clamp_velocity(-5) == 1
    assert clamp_velocity(999) == 127
    assert clamp_velocity(64) == 64


# --------------------------------------------------------------------------- #
# sidecar JSON
# --------------------------------------------------------------------------- #
def test_sidecar_path_never_overwrites_the_midi():
    p = sidecar_path(MIDI)
    assert p.name == "Unchained.mid.qmidi.json"
    assert p != MIDI


def test_sidecar_round_trip_is_exactly_equal(tmp_path):
    midi_copy = tmp_path / MIDI.name
    midi_copy.write_bytes(MIDI.read_bytes())
    tracks = build_tracks(midi_copy)
    project = SegmentProject(source=midi_copy, ticks_per_beat=TPB, tracks=tracks)
    path = project.save()
    assert path == sidecar_path(midi_copy)
    back = SegmentProject.load(path)
    assert back.to_dict() == project.to_dict()
    assert back.ticks_per_beat == project.ticks_per_beat
    assert back.source == midi_copy
    for a, b in zip(project.tracks, back.tracks):
        assert a.segments == b.segments
        assert a.notes == b.notes


def test_sidecar_round_trip_preserves_adjustments(tmp_path):
    tracks = build_tracks(MIDI)
    tracks[0].apply_delta(2, -18)
    tracks[1].set_value(0, 55)
    tracks[2].merge_adjacent(0)
    tracks[3].split_at_tick(
        next(n.start_tick for n in tracks[3].notes if n.start_tick > 0)
    )
    project = SegmentProject(source=MIDI, ticks_per_beat=TPB, tracks=tracks)
    path = project.save(tmp_path / "s.qmidi.json")
    back = SegmentProject.load(path, notes_by_track=read_notes(MIDI))
    for a, b in zip(project.tracks, back.tracks):
        assert a.segments == b.segments
    assert back.tracks[0].segments[2].delta == -18
    assert back.tracks[1].segments[0].target == 55


def test_sidecar_file_is_valid_json_and_never_pollutes_the_midi(tmp_path):
    midi_copy = tmp_path / "song.mid"
    midi_copy.write_bytes(MIDI.read_bytes())
    before = midi_copy.read_bytes()
    project = SegmentProject(source=midi_copy, ticks_per_beat=TPB,
                             tracks=build_tracks(midi_copy))
    project.save()
    assert sidecar_path(midi_copy).is_file()
    assert midi_copy.read_bytes() == before, "the original MIDI was modified"
    data = json.loads(sidecar_path(midi_copy).read_text(encoding="utf-8"))
    assert data["format"] == "qmidi-segments"
    assert data["ticks_per_beat"] == TPB


def test_load_project_prefers_an_existing_sidecar(tmp_path):
    midi_copy = tmp_path / "song.mid"
    midi_copy.write_bytes(MIDI.read_bytes())
    first = load_project(midi_copy)
    first.tracks[0].apply_delta(0, -30)
    first.save()

    second = load_project(midi_copy)
    assert second.tracks[0].segments[0].delta == -30
    assert second.segment_count == first.segment_count


def test_a_foreign_or_newer_sidecar_is_refused(tmp_path):
    path = tmp_path / "bad.qmidi.json"
    path.write_text(json.dumps({"format": "something-else"}), encoding="utf-8")
    with pytest.raises(SegmentError):
        SegmentProject.load(path)
    path.write_text(json.dumps({"format": "qmidi-segments", "version": 999}),
                    encoding="utf-8")
    with pytest.raises(SegmentError):
        SegmentProject.load(path)


# --------------------------------------------------------------------------- #
# snapping must never strand a note
# --------------------------------------------------------------------------- #
def test_snap_cuts_never_leaves_a_cut_inside_a_note_when_edges_oscillate():
    """Three overlapping notes make the nearest edge point back at itself.

    Nearest-edge distance from 107 goes 107 -> 130 -> 140 -> 107 -> ..., and the
    old guard gave up in the middle of that cycle and *returned* 140 -- a cut
    straight through two notes, which is the one thing this function promises
    never to return.
    """
    notes = [
        Note(0, 0, 60, 80, 0, 69),
        Note(1, 0, 61, 80, 29, 130),
        Note(2, 0, 62, 80, 33, 140),
        Note(3, 0, 63, 80, 107, 161),
        Note(4, 0, 64, 80, 130, 178),
        Note(5, 0, 65, 80, 151, 228),
        Note(6, 0, 66, 80, 164, 262),
        Note(7, 0, 67, 80, 187, 383),
    ]
    out = _snap_cuts(notes, [107])
    assert out, "the cut must not be thrown away entirely"
    for cut in out:
        for note in notes:
            assert not note.contains_tick(cut), f"cut {cut} splits note {note.index}"


def test_auto_cut_keeps_a_note_a_snapped_boundary_would_strand():
    """The note that starts before the previous segment's end still belongs somewhere.

    Snapping pushed the first boundary to 78 while a note started at 56 and
    ended at 230: it was too late for the window before 78 and too early for
    every window after it, so it silently ended up in no segment at all and
    validate() refused the whole track.
    """
    notes = [
        Note(0, 0, 60, 100, 56, 230),
        Note(1, 0, 61, 40, 26, 78),
        Note(2, 0, 62, 100, 290, 367),
    ]
    segments = auto_segment(notes, ticks_per_beat=TPB)
    covered = sorted(i for s in segments for i in s.note_indices)
    assert covered == [0, 1, 2], "every note must land in exactly one segment"
    SegmentedTrack(track=0, notes=notes, segments=segments).validate()


def test_two_cuts_pulled_to_the_same_held_note_become_one_segment():
    """One held note can swallow several cuts; the notes must not go with them."""
    notes = [
        Note(0, 0, 36, 40, 0, 4000),        # held under the whole melody
        Note(1, 0, 60, 80, 200, 400),
        Note(2, 0, 62, 60, 600, 800),
        Note(3, 0, 64, 80, 1000, 1200),
        Note(4, 0, 65, 60, 1400, 1600),
    ]
    segments = auto_segment(notes, ticks_per_beat=TPB)
    assert len(segments) == 1, [(s.start_tick, s.end_tick) for s in segments]
    assert sorted(i for s in segments for i in s.note_indices) == [0, 1, 2, 3, 4]
    SegmentedTrack(track=0, notes=notes, segments=segments).validate()


def test_a_note_left_ringing_at_end_of_track_does_not_strand_the_others(tmp_path):
    """read_notes closes a dangling note at the track end; that long span must
    not swallow the notes that follow it, and they must still be partitioned."""
    path = _write_midi(tmp_path / "dangling.mid", [
        ("hang", 0, 60, 80),
        ("note", 480, 240, 62, 60),
        ("note", 960, 240, 64, 80),
        ("wait", "end", 4000),
    ])
    tracks = build_tracks(path, ticks_per_beat=TPB)
    for t in tracks:
        t.validate()
        assert sorted(i for s in t.segments for i in s.note_indices) == \
               [n.index for n in t.notes]


# --------------------------------------------------------------------------- #
# auto_segment(start_tick=...)
# --------------------------------------------------------------------------- #
def test_auto_segment_rejects_a_start_tick_that_would_orphan_notes():
    """Silently dropping the notes before start_tick is data loss, so refuse it."""
    notes = [Note(i, 0, 60 + i, 80, i * 240, (i + 1) * 240) for i in range(6)]
    with pytest.raises(SegmentError) as excinfo:
        auto_segment(notes, ticks_per_beat=TPB, start_tick=1000)
    assert "outside the partition" in str(excinfo.value)


def test_auto_segment_cuts_only_the_notes_it_was_given():
    notes = [Note(i, 0, 60 + i, 80 + i * 10, i * 240, (i + 1) * 240) for i in range(6)]
    later = notes[1:]
    segments = auto_segment(later, ticks_per_beat=TPB, start_tick=240)
    assert segments[0].start_tick == 240
    assert segments[-1].end_tick == 1440
    assert sorted(i for s in segments for i in s.note_indices) == [1, 2, 3, 4, 5]
    SegmentedTrack(track=0, notes=later, segments=segments).validate()
    with pytest.raises(SegmentError):
        auto_segment(notes, ticks_per_beat=TPB, start_tick=-1)


# --------------------------------------------------------------------------- #
# adjustments vs. what a segment *is*
# --------------------------------------------------------------------------- #
def test_clear_adjustment_keeps_the_segments_reason():
    """Clearing a nudge says nothing about how the segment came to exist."""
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    before = track.segments[0].reason
    assert before == REASON_SAME
    track.apply_delta(0, -10)
    track.clear_adjustment(0)
    assert track.segments[0].reason == before, "the reason tag must not become 'merged'"
    assert track.segments[0].reason_values == track.segments[0].reason_values
    assert track.segments[0].delta == 0 and track.segments[0].target is None


def test_clear_adjustment_keeps_a_periodic_label():
    track = _track(_melody([80, 70, 60] * 2))
    assert track.segments[0].is_periodic
    track.set_value(0, 55)
    track.clear_adjustment(0)
    assert track.segments[0].is_periodic and track.segments[0].label == "周期80-70-60"


def test_merge_adjacent_adds_opposite_deltas():
    """Documenting the contract: merging is additive, +50/-50 undoes the nudges.

    Not a conflict -- a merge the user asked for is the way back to the
    original velocities, and refusing it would break merge_all after any pair of
    opposite nudges.  Conflicting *pinned* targets stay refused.
    """
    track = _track(_seq((0, 240, 60), (240, 240, 100)))
    track.apply_delta(0, 50)
    track.apply_delta(1, -50)
    merged = track.merge_adjacent(0)
    assert merged.delta == 0
    assert [n.velocity for n in track.resolved_notes()] == [60, 100]
    assert merged.reason == REASON_MERGED


# --------------------------------------------------------------------------- #
# manual splitting in dead spots
# --------------------------------------------------------------------------- #
def test_split_into_a_silent_lead_in_snaps_to_the_nearest_usable_edge():
    """Cutting inside a span no note covers must not create an undraggable half."""
    notes = [Note(0, 0, 60, 80, 9600, 9840),
             Note(1, 0, 62, 90, 9840, 10080)]
    track = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 10080, note_indices=(0, 1)),
    ])
    right = track.split_at_tick(9600)
    assert right.start_tick == 9840, "snapped to the first edge that really splits"
    assert track.segments[0].note_indices == (0,)
    assert track.segments[1].note_indices == (1,)
    assert all(s.note_indices for s in track.segments), "no dead handles"
    track.validate()


def test_split_with_no_usable_edge_at_all_is_rejected():
    """A hand-built segment whose note_indices disagree with its span has no cut."""
    notes = _seq((0, 240, 60), (240, 240, 100))
    track = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 480, note_indices=(0,)),
    ])
    with pytest.raises(SegmentError) as excinfo:
        track.split_at_tick(240)
    assert "no note edge" in str(excinfo.value)
    assert len(track.segments) == 1


# --------------------------------------------------------------------------- #
# validation the model was missing
# --------------------------------------------------------------------------- #
def test_velocity_segment_rejects_junk_from_a_hand_edited_sidecar():
    with pytest.raises(SegmentError):
        VelocitySegment(0, 0, 100, reason="whatever")
    with pytest.raises(SegmentError):
        VelocitySegment(0, 0, 100, note_indices=(-1,))
    with pytest.raises(SegmentError):
        VelocitySegment(0, 0, 100, note_indices=(3, 3))


def test_validate_rejects_a_segment_reaching_past_the_last_note():
    notes = _seq((0, 240, 60), (240, 240, 100))
    ok = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 480, note_indices=(0, 1)),
    ])
    ok.validate()
    too_far = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 4800, note_indices=(0, 1)),
    ])
    with pytest.raises(SegmentError) as excinfo:
        too_far.validate()
    assert "past the last note" in str(excinfo.value)


def test_validate_rejects_a_segment_referring_to_a_note_the_track_lacks():
    notes = _seq((0, 240, 60))
    track = SegmentedTrack(track=0, notes=notes, segments=[
        VelocitySegment(0, 0, 240, note_indices=(0, 7)),
    ])
    with pytest.raises(SegmentError) as excinfo:
        track.validate()
    assert "does not hold" in str(excinfo.value)


def test_a_segment_covering_two_figures_is_not_labelled_periodic():
    """A held note glued two ostinatos into one segment; calling it one figure
    would tell the user a single repeating gesture is there when it is not."""
    notes = [Note(0, 0, 36, 40, 0, 2000)]
    notes += [Note(i, 0, 60, v, 100 + (i - 1) * 200, 300 + (i - 1) * 200)
              for i, v in enumerate((80, 70, 60, 80, 70, 60), start=1)]
    notes += [Note(i, 0, 62, v, 1200 + (i - 7) * 200, 1400 + (i - 7) * 200)
              for i, v in enumerate((100, 50, 100, 50, 100, 50, 100, 50), start=7)]
    segments = auto_segment(notes, ticks_per_beat=TPB)
    first = segments[0]
    assert len(first.note_indices) > 6, "the two figures really did get glued"
    assert first.reason != REASON_CYCLE
    assert set(first.reason_values) == {40, 80, 70, 60, 100, 50}
    assert segments[-1].is_periodic, "a segment with one figure may still be periodic"
    SegmentedTrack(track=0, notes=notes, segments=segments).validate()


# --------------------------------------------------------------------------- #
# a sidecar whose MIDI moved on underneath it
# --------------------------------------------------------------------------- #
def test_a_sidecar_warns_when_the_midi_gained_notes(tmp_path):
    midi = _write_midi(tmp_path / "song.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 240, 240, 62, 90),
    ])
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    project.save()
    _write_midi(tmp_path / "song.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 240, 240, 62, 90),
        ("note", 480, 240, 64, 70),
    ])
    back = SegmentProject.load(sidecar_path(midi))
    assert back.warnings, "a stale sidecar must not load silently"
    assert any("belong to no segment" in w for w in back.warnings), back.warnings
    assert back.verify_notes()  # the same check on demand


def test_a_sidecar_warns_when_the_midi_lost_notes(tmp_path):
    midi = _write_midi(tmp_path / "song.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 240, 240, 62, 90),
        ("note", 480, 240, 64, 70),
    ])
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    project.save()
    _write_midi(tmp_path / "song.mid", [
        ("note", 0, 240, 60, 80),
    ])
    back = SegmentProject.load(sidecar_path(midi))
    assert any("does not hold" in w for w in back.warnings), back.warnings


def test_a_fresh_sidecar_raises_no_warnings(tmp_path):
    midi = _write_midi(tmp_path / "song.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 240, 240, 62, 90),
    ])
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    project.save()
    back = SegmentProject.load(sidecar_path(midi))
    assert back.warnings == []
    assert back.verify_notes() == []


def test_a_moved_sidecar_finds_the_midi_it_recorded(tmp_path):
    """`source` is stored by name, so a renamed sidecar resolves via that name.

    The old code trusted ``Path.suffix``, which for ``moved.qmidi.json`` made
    the project point at the sidecar file itself -- a MIDI file that is not there.
    """
    home = tmp_path / "home"
    home.mkdir()
    midi = _write_midi(home / "song.mid", [
        ("note", 0, 240, 60, 80),
        ("note", 240, 240, 62, 90),
    ])
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    (elsewhere / "song.mid").write_bytes(midi.read_bytes())   # folder moved as a unit
    moved = project.save(elsewhere / "moved.qmidi.json")
    back = SegmentProject.load(moved)
    assert back.source == elsewhere / "song.mid"
    assert [n.velocity for n in back.tracks[0].resolved_notes()] == [80, 90]


def test_a_sidecar_whose_midi_is_missing_still_loads(tmp_path):
    midi = tmp_path / "gone.mid"
    midi.write_bytes(MIDI.read_bytes())
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    path = project.save(tmp_path / "kept.qmidi.json")
    midi.unlink()
    back = SegmentProject.load(path)
    assert back.segment_count == project.segment_count
    assert any("no notes" in w for w in back.warnings), back.warnings


# --------------------------------------------------------------------------- #
# the two floors that keep the auto cut coarse enough to be usable
# --------------------------------------------------------------------------- #
def _note(index: int, velocity: int, start: int, length: int = 200,
          pitch: int = 60) -> Note:
    return Note(index=index, channel=0, pitch=pitch, velocity=velocity,
                start_tick=start, end_tick=start + length)


def test_a_repeated_figure_survives_a_one_unit_velocity_drift():
    """96,91 played four times with +-1 jitter is one gesture, not eight segments.

    Exact equality used to break the cycle on the first deviation, which is how
    one four-bar phrase became ten segments on the real file.
    """
    pattern = (96, 91, 96, 91)
    notes = []
    for i in range(16):
        velocity = pattern[i % 4] + (1 if (i // 4) % 2 else 0)
        notes.append(_note(i, velocity, i * 240))
    segments = auto_segment(notes, ticks_per_beat=TPB,
                            options=AutoSegmentOptions(velocity_tolerance=4))
    assert len(segments) == 1, [len(s.note_indices) for s in segments]


def test_a_lone_accent_is_folded_into_its_neighbour():
    """A single loud note inside a quiet passage must not own a segment."""
    notes = [_note(i, 120 if i == 6 else 60, i * TPB) for i in range(12)]
    options = AutoSegmentOptions(min_segment_notes=8, min_segment_beats=4.0)
    segments = auto_segment(notes, ticks_per_beat=TPB, options=options)
    assert len(segments) == 2, [len(s.note_indices) for s in segments]
    values = [notes[i].velocity for i in segments[0].note_indices]
    assert 120 in values and len(values) == 7


def test_the_floor_does_not_collapse_a_small_piece_into_one_segment():
    """Both floors must be violated: a 4-note block of 4 beats stays a segment."""
    notes = [_note(i, 60, i * TPB, length=TPB) for i in range(4)]
    notes += [_note(4 + i, 100, (4 + i) * TPB, length=TPB) for i in range(4)]
    options = AutoSegmentOptions(min_segment_notes=8, min_segment_beats=4.0)
    segments = auto_segment(notes, ticks_per_beat=TPB, options=options)
    assert len(segments) == 2


def test_a_proportional_scale_survives_the_sidecar(tmp_path):
    from dataclasses import replace as dc_replace

    midi = tmp_path / "scale.mid"
    midi.write_bytes(MIDI.read_bytes())
    project = SegmentProject(source=midi, ticks_per_beat=TPB, tracks=build_tracks(midi))
    segment = project.tracks[0].segments[0]
    project.tracks[0].segments[0] = dc_replace(segment, scale=0.5)
    path = project.save(tmp_path / "scaled.qmidi.json")

    back = SegmentProject.load(path)
    assert back.tracks[0].segments[0].scale == pytest.approx(0.5)


# --------------------------------------------------------------------------- #
# pedal display for channel siblings
# --------------------------------------------------------------------------- #
def test_a_channel_sibling_shows_the_carriers_pedal():
    """One piano split over two tracks: both lanes show the same pedal.

    The CC64 stream is per channel and must stay single -- the sibling track
    only *displays* it, editing still lands on the carrier alone.
    """
    from qmidi.pedal import PedalSpan
    from qmidi.segments import Note, SegmentedTrack

    def _notes(channel: int, base: int):
        return [Note(index=i, channel=channel, pitch=base + i, velocity=80,
                     start_tick=i * 100, end_tick=i * 100 + 50) for i in range(4)]

    project = SegmentProject(
        source=Path("unused.mid"), ticks_per_beat=TPB,
        tracks=[SegmentedTrack(track=0, notes=_notes(0, 60)),
                SegmentedTrack(track=1, notes=_notes(0, 70)),
                SegmentedTrack(track=2, notes=_notes(1, 72))],
    )
    spans = [PedalSpan(0, 400)]
    project.set_pedal_spans(1, spans)

    assert project.pedal_spans(1) == spans      # the carrier
    assert project.pedal_spans(0) == spans      # same-channel sibling sees it
    assert project.pedal_spans(2) == []         # another channel does not

    # editing still goes to the carrier only
    project.set_pedal_spans(0, [PedalSpan(100, 500)])
    assert project.pedal_spans(0) == [PedalSpan(100, 500)]   # own edits win
    assert project.pedal_spans(1) == spans                   # carrier untouched