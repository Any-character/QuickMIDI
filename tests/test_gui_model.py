"""Tests for the Qt-free half of the editor (``qmidi.gui.model``).

These are the rules that decide what the user hears: which segment is selected,
what the loop region is, what gets muted, and what undo restores.  A widget bug
shows up as a wrong picture; a bug in here shows up as the wrong audio, or as
lost work.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mido
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.exporters import export_midi  # noqa: E402
from qmidi.gui.model import Session, TrackStats  # noqa: E402
from qmidi.offline import TickClock  # noqa: E402
from qmidi.segments import SegmentProject  # noqa: E402
from qmidi.gui.render_plan import (  # noqa: E402
    PreviewMidi,
    PreviewRequest,
    file_request,
    loop_request,
    play_request,
    project_digest,
)
from qmidi.pedal import PedalSpan, read_pedal  # noqa: E402

TPB = 480


def _track(notes, name="Piano") -> mido.MidiTrack:
    """notes: (pitch, velocity, start_tick, duration_ticks)"""
    events = []
    for pitch, velocity, start, duration in notes:
        events.append((start, 1, pitch, velocity))
        events.append((start + duration, 0, pitch, 0))
    events.sort()
    track = mido.MidiTrack([mido.MetaMessage("track_name", name=name)])
    last = 0
    for tick, is_on, pitch, velocity in events:
        track.append(mido.Message("note_on", note=pitch,
                                  velocity=velocity if is_on else 0,
                                  time=tick - last))
        last = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    return track


@pytest.fixture()
def midi_path(tmp_path: Path) -> Path:
    """Two tracks, two velocity blocks each -- so auto-segmentation has work."""
    # Eight notes per velocity block on purpose: the auto cut now folds
    # segments that are both short and sparse, so a four-note block would be
    # merged away and the "two segments" premise would silently disappear.
    path = tmp_path / "song.mid"
    midi = mido.MidiFile(ticks_per_beat=TPB)
    step = TPB // 2
    first = [(60 + i % 12, 60, i * step, step - 10) for i in range(8)]
    first += [(72 + i % 12, 90, (8 + i) * step, step - 10) for i in range(8)]
    second = [(48 + i % 12, 70, i * step, step - 10) for i in range(8)]
    second += [(58 + i % 12, 100, (8 + i) * step, step - 10) for i in range(8)]
    midi.tracks.append(_track(first, "Piano"))
    midi.tracks.append(_track(second, "Piano 2"))
    midi.save(str(path))
    return path


@pytest.fixture()
def session(midi_path: Path) -> Session:
    return Session.open(midi_path)


# --------------------------------------------------------------------------- #
# opening
# --------------------------------------------------------------------------- #
def test_open_builds_segments_and_selects_the_first(session: Session):
    assert session.track_count == 2
    assert session.project.ticks_per_beat == TPB
    assert session.selection[session.active_track] == 0
    assert session.selected_segment() is not None


def test_open_reads_the_sidecar_when_one_exists(session: Session):
    session.set_delta(-7)
    written = session.save()
    assert written.is_file()
    again = Session.open(session.project.source)
    assert again.selected_segment().delta == -7


def test_open_reports_a_missing_file(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        Session.open(tmp_path / "nope.mid")


def test_stats_describe_each_track(session: Session):
    stats = session.stats()
    assert len(stats) == 2
    assert all(isinstance(s, TrackStats) for s in stats)
    assert stats[0].notes == 16
    assert stats[0].segments >= 2
    assert stats[0].velocity_low <= stats[0].velocity_high


# --------------------------------------------------------------------------- #
# selection
# --------------------------------------------------------------------------- #
def test_step_selection_clamps_at_both_ends(session: Session):
    session.select(0, 0)
    assert session.step_selection(-1) == 0
    last = len(session.active().segments) - 1
    session.select(0, last)
    assert session.step_selection(1) == last


def test_select_at_tick_picks_the_covering_segment(session: Session):
    segments = session.active().segments
    index = session.select_at_tick(segments[-1].start_tick + 1)
    assert index == len(segments) - 1


def test_selection_is_remembered_per_track(session: Session):
    session.select(0, 1)
    session.select(1, 0)
    session.active_track = 0
    assert session.selected_index() == 1


def test_selecting_a_segment_arms_it_as_the_loop(session: Session):
    session.set_manual_loop(0, TPB)
    assert session.loop_ticks() == (0, TPB)
    session.select(0, 1)
    segment = session.selected_segment()
    assert session.loop_ticks() == (segment.start_tick, segment.end_tick)


def test_loop_seconds_use_the_tempo_map(session: Session):
    session.select(0, 0)
    start, end = session.loop_seconds()
    segment = session.selected_segment()
    assert end > start
    assert end - start == pytest.approx(
        session.clock.seconds(segment.end_tick) - session.clock.seconds(segment.start_tick)
    )


def test_manual_loop_normalises_reversed_and_degenerate_drags(session: Session):
    session.set_manual_loop(5000, 1000)
    assert session.loop_ticks() == (1000, 5000)
    session.set_manual_loop(700, 700)
    assert session.loop_ticks() == (700, 701)


# --------------------------------------------------------------------------- #
# tempo
# --------------------------------------------------------------------------- #
def test_a_scaled_clock_runs_the_same_map_faster(session: Session):
    base = session._base_clock
    assert base.bpm == pytest.approx(120.0)
    faster = base.scaled(120.0 / 240.0)
    assert faster.bpm == pytest.approx(240.0, abs=0.5)
    assert faster.seconds(4 * TPB) == pytest.approx(base.seconds(4 * TPB) / 2, rel=1e-3)


def test_setting_bpm_shortens_the_piece(session: Session):
    slow = session.clock.seconds(session.total_end_tick())
    assert session.set_bpm(240)
    fast = session.clock.seconds(session.total_end_tick())
    assert fast == pytest.approx(slow / 2, rel=1e-3)
    assert session.bpm == pytest.approx(240.0, abs=0.5)


def test_setting_bpm_is_undoable(session: Session):
    before = session.bpm
    assert session.set_bpm(90)
    assert session.undo()
    assert session.bpm == pytest.approx(before)


def test_setting_the_files_own_tempo_clears_the_override(session: Session):
    assert session.set_bpm(90)
    assert session.project.bpm == pytest.approx(90.0)
    assert session.set_bpm(120)      # the file's own tempo
    assert session.project.bpm is None, "typing the original value must not pin it"


def test_bpm_out_of_range_is_refused(session: Session):
    with pytest.raises(ValueError):
        session.set_bpm(5)
    with pytest.raises(ValueError):
        session.set_bpm(999)


def test_bpm_survives_a_sidecar_round_trip(session: Session, tmp_path: Path):
    assert session.set_bpm(90)
    path = session.project.save(tmp_path / "slow.qmidi.json")
    assert SegmentProject.load(path).bpm == pytest.approx(90.0)


def test_an_old_sidecar_without_bpm_still_loads(session: Session, tmp_path: Path):
    payload = session.project.to_dict()
    payload.pop("bpm", None)
    path = tmp_path / "old.qmidi.json"
    path.write_text(__import__("json").dumps(payload), encoding="utf-8")
    assert SegmentProject.load(path).bpm is None


def test_exported_midi_carries_the_new_tempo(session: Session, tmp_path: Path):
    assert session.set_bpm(240)
    result = export_midi(session.project, tmp_path / "fast.mid")
    assert TickClock.from_midi(result.path).bpm == pytest.approx(240.0, abs=0.5)


def test_exporting_without_a_bpm_leaves_the_tempo_alone(session: Session,
                                                        tmp_path: Path):
    result = export_midi(session.project, tmp_path / "same.mid")
    assert TickClock.from_midi(result.path).bpm == pytest.approx(120.0, abs=0.5)


# --------------------------------------------------------------------------- #
# playhead snap
# --------------------------------------------------------------------------- #
def test_snap_tick_lands_on_sixteenth_notes(session: Session):
    session.snap_division = 4
    step = TPB // 4
    assert session.snap_tick(0) == 0
    assert session.snap_tick(step // 2 - 1) == 0        # just under half a cell
    assert session.snap_tick(step // 2 + 1) == step    # just over half
    assert session.snap_tick(step * 3 + step // 2) == step * 4


def test_snap_tick_can_be_thirty_seconds(session: Session):
    session.snap_division = 8
    step = TPB // 8
    assert session.snap_tick(step + 1) == step
    assert session.snap_tick(step * 2 - 1) == step * 2


def test_snap_tick_off_places_the_cursor_anywhere(session: Session):
    session.snap_division = 0
    assert session.snap_tick(12_345) == 12_345


def test_snap_tick_never_goes_negative(session: Session):
    session.snap_division = 4
    assert session.snap_tick(-500) == 0


# --------------------------------------------------------------------------- #
# transport: loop off by default, play from the cursor
# --------------------------------------------------------------------------- #
def test_loop_is_off_by_default_and_play_runs_from_the_cursor(session: Session):
    assert not session.loop_enabled
    session.cursor_tick = 2 * TPB
    start, end = session.playback_span_ticks()
    assert start == 2 * TPB
    assert end == session.total_end_tick()


def test_loop_mode_plays_the_selected_segment(session: Session):
    session.loop_enabled = True
    segment = session.selected_segment()
    assert session.playback_span_ticks() == (segment.start_tick, segment.end_tick)


def test_playback_span_seconds_follow_the_tempo_map(session: Session):
    session.cursor_tick = TPB
    start, end = session.playback_span_seconds()
    assert start == pytest.approx(session.clock.seconds(TPB))
    assert end > start


def test_a_cursor_past_the_end_still_yields_a_playable_span(session: Session):
    session.cursor_tick = session.total_end_tick() + 10_000
    start, end = session.playback_span_ticks()
    assert end > start


def test_timecode_is_minutes_and_tenths():
    assert Session.timecode(0) == "0:00.0"
    assert Session.timecode(83.4) == "1:23.4"
    assert Session.timecode(-5) == "0:00.0"


def test_bar_beat_label_is_one_based(session: Session):
    assert session.bar_beat_label(0) == "第 1 小节 · 1 拍"
    assert session.bar_beat_label(TPB * 4) == "第 2 小节 · 1 拍"
    assert session.bar_beat_label(TPB * 5) == "第 2 小节 · 2 拍"


def test_position_label_carries_both_units(session: Session):
    label = session.position_label(0.0)
    assert "第 1 小节" in label
    assert "0:00.0" in label
    assert "/" in label


# --------------------------------------------------------------------------- #
# velocity lane drag
# --------------------------------------------------------------------------- #
def test_selected_velocity_range_reports_min_and_max(session: Session):
    session.select(0, 0)
    span = session.selected_velocity_range()
    assert span is not None
    low, high = span
    assert 1 <= low <= high <= 127


def test_selected_velocity_range_follows_a_nudge(session: Session):
    session.select(0, 0)
    low, high = session.selected_velocity_range()
    assert session.set_delta(-9)
    assert session.selected_velocity_range() == (max(1, low - 9), max(1, high - 9))


def test_selected_velocity_range_needs_a_selection(session: Session):
    session.select(0, None)
    assert session.selected_velocity_range() is None


def test_a_velocity_drag_is_absolute_and_undoes_in_one_step(session: Session):
    session.select(0, 0)
    grabbed = session.active().notes[0].velocity  # 60 in the fixture
    assert session.begin_velocity_drag(grabbed)
    assert session.drag_velocity(-6)
    assert session.selected_segment().scale == pytest.approx((grabbed - 6) / grabbed)
    assert session.selected_segment().delta == 0
    assert session.drag_velocity(-3)  # absolute, not cumulative
    assert session.selected_segment().scale == pytest.approx((grabbed - 3) / grabbed)
    assert session.end_velocity_drag()
    assert session.undo()
    assert session.selected_segment().scale == pytest.approx(1.0)


def test_a_drag_scales_the_whole_segment_proportionally():
    """The user's example: 100 -> 50 must put 90 at 45, not 40."""
    from qmidi.segments import Note, VelocitySegment

    segment = VelocitySegment(track=0, start_tick=0, end_tick=100, scale=0.5)
    loud = Note(index=0, channel=0, pitch=60, velocity=100,
                start_tick=0, end_tick=10)
    softer = Note(index=1, channel=0, pitch=64, velocity=90,
                  start_tick=0, end_tick=10)
    assert segment.velocity_for(loud) == 50
    assert segment.velocity_for(softer) == 45


def test_a_drag_that_moved_nothing_leaves_no_undo_step(session: Session):
    session.select(0, 0)
    assert session.begin_velocity_drag(60)
    assert not session.end_velocity_drag()
    assert not session.can_undo()


def test_a_drag_moves_a_pinned_segment_by_the_same_units(session: Session):
    session.select(0, 1)
    session.set_target(60)
    assert session.begin_velocity_drag(90)
    session.drag_velocity(-2)
    assert session.selected_segment().target == 58
    assert session.end_velocity_drag()


def test_a_drag_cannot_push_a_pin_out_of_range(session: Session):
    session.select(0, 1)
    session.set_target(120)
    assert session.begin_velocity_drag(90)
    session.drag_velocity(10)
    assert session.selected_segment().target == 127
    session.end_velocity_drag()


def test_a_drag_without_a_selection_is_a_no_op(session: Session):
    session.select(0, None)
    assert not session.begin_velocity_drag(60)
    assert not session.drag_velocity(1)


# --------------------------------------------------------------------------- #
# renaming tracks
# --------------------------------------------------------------------------- #
def test_rename_track_changes_the_name_and_is_undoable(session: Session):
    before = session.project.track(0).name
    assert session.rename_track(0, "右手")
    assert session.project.track(0).name == "右手"
    assert session.dirty
    assert session.undo()
    assert session.project.track(0).name == before


def test_renaming_to_blank_falls_back_to_the_file_name(session: Session):
    assert session.rename_track(0, "临时")
    assert session.rename_track(0, "   ")
    assert session.project.track(0).name == "Piano", "never leave a nameless row"


def test_renaming_to_the_same_name_is_a_no_op(session: Session):
    current = session.project.track(0).name
    assert not session.rename_track(0, current)
    assert not session.can_undo()


def test_a_rename_survives_a_save_and_reload(session: Session, tmp_path: Path):
    assert session.rename_track(0, "右手旋律")
    path = session.project.save(tmp_path / "renamed.qmidi.json")
    assert SegmentProject.load(path).track(0).name == "右手旋律"


def test_resegment_keeps_a_renamed_track(session: Session):
    assert session.rename_track(0, "右手旋律")
    assert session.resegment()
    assert session.project.track(0).name == "右手旋律"


# --------------------------------------------------------------------------- #
# resegmentation
# --------------------------------------------------------------------------- #
def test_resegment_rebuilds_the_cut_and_is_undoable(session: Session):
    before = session.project.segment_count
    assert session.set_delta(-9)  # an adjustment the re-cut will drop
    assert session.resegment()
    assert session.project.segment_count >= 1
    assert session.can_undo()
    assert session.undo()
    assert session.project.segment_count == before
    assert session.selected_segment() is not None


def test_resegment_keeps_the_pedal_edits(session: Session):
    session.add_pedal(0, TPB)
    assert session.project.pedals
    assert session.resegment()
    assert session.project.pedals, "re-cutting segments must not touch the pedal"


# --------------------------------------------------------------------------- #
# edits and undo
# --------------------------------------------------------------------------- #
def test_delta_shifts_every_note_of_the_segment(session: Session):
    session.select(0, 0)
    segment = session.selected_segment()
    notes = session.active().notes_in(segment)
    session.set_delta(-10)
    resolved = session.resolved_velocity_map()
    assert all(resolved[(0, n.index)] == max(1, n.velocity - 10) for n in notes)


def test_target_pins_the_whole_segment(session: Session):
    session.select(0, 1)
    session.set_target(33)
    segment = session.selected_segment()
    assert segment.target == 33
    resolved = session.resolved_velocity_map()
    assert all(resolved[(0, i)] == 33 for i in segment.note_indices)
    # ... and only that segment: the neighbouring one keeps its own value.
    other = session.active().segments[0]
    assert all(resolved[(0, i)] != 33 for i in other.note_indices)


def test_target_is_clamped_to_the_legal_range(session: Session):
    session.select(0, 1)
    session.set_target(999)
    assert session.selected_segment().target == 127


def test_undo_restores_the_previous_delta_and_redo_reapplies_it(session: Session):
    session.select(0, 0)
    session.set_delta(-5)
    assert session.selected_segment().delta == -5
    assert session.undo()
    assert session.selected_segment().delta == 0
    assert session.redo()
    assert session.selected_segment().delta == -5


def test_undo_survives_a_structural_edit(session: Session):
    """A split changes the segment list; the snapshot must still restore."""
    session.select(0, 1)
    before = len(session.active().segments)
    segment = session.selected_segment()
    session.cursor_tick = segment.start_tick + TPB
    if session.split_at_cursor():
        assert len(session.active().segments) == before + 1
        assert session.undo()
        assert len(session.active().segments) == before


def test_edit_marks_the_session_dirty_and_save_clears_it(session: Session):
    assert not session.dirty
    session.set_delta(1)
    assert session.dirty
    session.save()
    assert not session.dirty


def test_edits_are_no_ops_without_a_selection(session: Session):
    session.select(0, None)
    assert not session.set_delta(5)
    assert not session.can_undo()


# --------------------------------------------------------------------------- #
# mute / solo
# --------------------------------------------------------------------------- #
def test_mute_and_solo_feed_the_render_filter(session: Session):
    assert session.visible_tracks() == frozenset({0, 1})
    session.toggle_mute(1)
    assert session.visible_tracks() == frozenset({0})
    session.toggle_solo(1)
    assert session.visible_tracks() == frozenset({1})  # solo beats mute
    session.toggle_solo(1)
    session.toggle_mute(1)
    assert session.visible_tracks() == frozenset({0, 1})


# --------------------------------------------------------------------------- #
# render plan
# --------------------------------------------------------------------------- #
def test_preview_midi_is_rewritten_only_when_the_project_changes(
        session: Session, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    import qmidi.gui.render_plan as render_plan

    # Count export calls instead of watching the file's mtime: NTFS can serve
    # a cached timestamp when a tiny file is rewritten within milliseconds,
    # which made the mtime-based assertion flake on Windows.
    calls: list[Path] = []
    real_export = render_plan.export_midi

    def counting_export(project, target, **kwargs):
        calls.append(Path(target))
        return real_export(project, target, **kwargs)

    monkeypatch.setattr(render_plan, "export_midi", counting_export)

    cache = PreviewMidi(tmp_path / "preview")
    first = cache.ensure(session.project)
    assert cache.ensure(session.project) == first
    assert len(calls) == 1, "unchanged project must not rewrite the preview"

    digest = project_digest(session.project)
    session.set_delta(-3)
    assert project_digest(session.project) != digest
    cache.ensure(session.project)
    assert len(calls) == 2, "a changed project must rewrite the preview"


def test_a_request_carries_the_digest_the_timeline_cache_keys_on(session: Session,
                                                                tmp_path: Path):
    """Caching the event list is only safe when the key follows the content."""
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    first = play_request(session, midi, loop=False)
    assert first.digest == project_digest(session.project)
    assert first.timeline_key == (str(midi), first.digest, session.visible_tracks())

    session.set_delta(-3)
    second = play_request(session, midi, loop=False)
    assert second.digest != first.digest
    assert second.timeline_key != first.timeline_key
    # The instrument did not change, so the engine may be reused.
    assert second.engine_key == first.engine_key


def test_a_request_without_a_digest_opts_out_of_the_timeline_cache():
    """An empty digest must mean 'rebuild every time', never 'reuse the last'."""
    request = PreviewRequest(midi_path=Path("song.mid"), tracks=frozenset({0}),
                             start_sec=0.0, end_sec=1.0, loop=False, label="x")
    assert request.timeline_key is None


def test_loop_request_carries_the_visible_tracks_and_the_segment_span(session: Session,
                                                                    tmp_path: Path):
    session.select(0, 0)
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    request = loop_request(session, midi)
    assert request is not None
    assert request.loop
    assert request.tracks == frozenset({0, 1})
    assert request.end_sec > request.start_sec


def test_loop_request_is_none_when_everything_is_muted(session: Session, tmp_path: Path):
    session.toggle_mute(0)
    session.toggle_mute(1)
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    assert loop_request(session, midi) is None


def test_file_request_covers_the_whole_file(session: Session, tmp_path: Path):
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    request = file_request(session, midi)
    assert not request.loop
    assert request.start_sec == 0
    assert request.end_sec >= session.clock.seconds(session.total_end_tick()) - 1e-6


def test_play_request_starts_at_the_cursor_when_not_looping(session: Session,
                                                           tmp_path: Path):
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    session.cursor_tick = 2 * TPB
    request = play_request(session, midi, loop=False)
    assert request is not None
    assert not request.loop
    assert request.start_sec == pytest.approx(session.clock.seconds(2 * TPB))


def _pedal_end_midi(path: Path) -> Path:
    """A two-note phrase pedalled throughout, lifted exactly on the last note."""
    midi = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano")])
    track.append(mido.Message("control_change", control=64, value=127, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("note_on", note=60, velocity=0, time=TPB))
    track.append(mido.Message("note_on", note=64, velocity=80, time=0))
    track.append(mido.Message("note_on", note=64, velocity=0, time=TPB))
    track.append(mido.Message("control_change", control=64, value=0, time=0))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(str(path))
    return path


def _pedal_releases(path: Path) -> list[int]:
    releases, tick = [], 0
    for message in mido.MidiFile(str(path)).tracks[0]:
        tick += message.time
        if message.type == "control_change" and message.control == 64 and message.value == 0:
            releases.append(tick)
    return releases


def test_a_pedal_lift_on_the_final_note_is_dropped(tmp_path: Path):
    """Notation software writes "pedal up" at the last note; that chops the ring."""
    from qmidi.gui.render_plan import _hold_final_pedal

    path = _pedal_end_midi(tmp_path / "on.mid")
    assert _pedal_releases(path), "the fixture must start with a lift"
    _hold_final_pedal(path)
    assert _pedal_releases(path) == [], "the mechanical lift should be gone"


def test_a_pedal_lift_before_the_last_note_is_kept(tmp_path: Path):
    """A short final chord with the pedal already up is a played articulation."""
    from qmidi.gui.render_plan import _hold_final_pedal

    midi = mido.MidiFile(ticks_per_beat=TPB)
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano")])
    track.append(mido.Message("control_change", control=64, value=127, time=0))
    track.append(mido.Message("note_on", note=60, velocity=80, time=0))
    track.append(mido.Message("control_change", control=64, value=0, time=TPB))
    track.append(mido.Message("note_on", note=60, velocity=0, time=TPB * 3))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    path = tmp_path / "early.mid"
    midi.save(str(path))

    before = _pedal_releases(path)
    _hold_final_pedal(path)
    assert _pedal_releases(path) == before


def test_play_request_keeps_a_release_tail(session: Session, tmp_path: Path):
    """Playback must not cut the final chord at the last note-off."""
    from qmidi.gui.render_plan import PLAY_TAIL_SEC

    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    session.cursor_tick = 0
    request = play_request(session, midi, loop=False)
    content_end = session.clock.seconds(session.total_end_tick())
    assert request.end_sec - content_end == pytest.approx(PLAY_TAIL_SEC, abs=1e-6)


def test_looping_does_not_add_a_tail(session: Session, tmp_path: Path):
    """A loop is exactly the segment -- a tail would smear it into the next one."""
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    session.loop_enabled = True
    request = play_request(session, midi, loop=True)
    segment = session.selected_segment()
    assert request.end_sec == pytest.approx(session.clock.seconds(segment.end_tick))


def test_play_request_loops_the_selected_segment(session: Session, tmp_path: Path):
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    session.loop_enabled = True
    request = play_request(session, midi, loop=True)
    assert request is not None
    assert request.loop
    segment = session.selected_segment()
    assert request.start_sec == pytest.approx(session.clock.seconds(segment.start_tick))


def test_play_request_is_none_when_everything_is_muted(session: Session,
                                                       tmp_path: Path):
    session.toggle_mute(0)
    session.toggle_mute(1)
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    assert play_request(session, midi, loop=False) is None


# --------------------------------------------------------------------------- #
# sustain pedal in the session
# --------------------------------------------------------------------------- #
@pytest.fixture()
def pedal_midi(tmp_path: Path) -> Path:
    """One track, four notes, and a pedal from tick 0 to 960."""
    path = tmp_path / "pedal.mid"
    midi = mido.MidiFile(ticks_per_beat=TPB)
    events = [(0, mido.Message("control_change", control=64, value=127))]
    for i in range(4):
        start = i * TPB
        events.append((start, mido.Message("note_on", note=60 + i, velocity=70)))
        events.append((start + TPB - 10,
                       mido.Message("note_on", note=60 + i, velocity=0)))
    events.append((960, mido.Message("control_change", control=64, value=0)))
    track = mido.MidiTrack([mido.MetaMessage("track_name", name="Piano")])
    previous = 0
    for tick, msg in sorted(events, key=lambda item: item[0]):
        track.append(msg.copy(time=tick - previous))
        previous = tick
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.tracks.append(track)
    midi.save(str(path))
    return path


def test_the_lane_shows_the_files_own_pedal_until_it_is_edited(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.pedal_spans() == [PedalSpan(0, 960)]
    assert not session.pedal_edited()


def test_drawing_extends_the_existing_pedal_instead_of_replacing_it(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.add_pedal(1200, 1600)
    assert session.pedal_edited()
    assert session.pedal_spans() == [PedalSpan(0, 960), PedalSpan(1200, 1600)]


def test_erasing_and_clearing_the_pedal_are_undoable(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.erase_pedal(100)
    assert session.pedal_spans() == []
    assert session.undo()
    assert session.pedal_spans() == [PedalSpan(0, 960)]

    assert session.clear_pedal()
    assert session.pedal_spans() == []
    assert session.undo()
    assert session.pedal_spans() == [PedalSpan(0, 960)]


def test_a_press_holds_until_the_next_press(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.clear_pedal()
    assert session.pedal_spans() == []
    assert session.toggle_pedal_at(TPB)      # foot down at beat 2
    assert session.pedal_spans() == [PedalSpan(TPB, session.total_end_tick())]
    assert session.toggle_pedal_at(2 * TPB)  # foot up at beat 3
    assert session.pedal_spans() == [PedalSpan(TPB, 2 * TPB)]


def test_a_channel_sibling_shares_and_clears_the_pedal(pedal_midi: Path):
    """One piano over two tracks: both lanes show the pedal, clear clears the carrier."""
    from qmidi.pedal import PedalSpan
    from qmidi.segments import Note, SegmentedTrack

    def _notes(channel, base):
        return [Note(index=i, channel=channel, pitch=base + i, velocity=80,
                     start_tick=i * 100, end_tick=i * 100 + 50) for i in range(4)]

    project = SegmentProject(
        source=pedal_midi, ticks_per_beat=TPB,
        tracks=[SegmentedTrack(track=0, notes=_notes(0, 60)),
                SegmentedTrack(track=1, notes=_notes(0, 70))])
    session = Session(project, TickClock([(0, 500000)], TPB))
    session.active_track = 0
    session.project.set_pedal_spans(1, [PedalSpan(0, 400)])

    assert session.pedal_spans() == [PedalSpan(0, 400)], "shared display"
    assert session.clear_pedal()
    assert not session.project.pedal_spans(1), "the carrier edit is gone"
    # With the edits cleared, the lane falls back to the pedal the FILE carries
    # on this track -- the same precedence the preview has always used.
    assert session.pedal_spans() == [PedalSpan(0, 960)]


def test_a_press_inside_a_held_span_lifts_it_there(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.pedal_spans() == [PedalSpan(0, 960)]
    assert session.toggle_pedal_at(400)
    assert session.pedal_spans() == [PedalSpan(0, 400)]


def test_a_press_on_the_start_of_a_span_cancels_it(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.toggle_pedal_at(0)
    assert session.pedal_spans() == []


def test_a_pedal_press_is_undoable(pedal_midi: Path):
    session = Session.open(pedal_midi)
    assert session.clear_pedal()
    assert session.toggle_pedal_at(TPB)
    assert session.undo()
    assert session.pedal_spans() == []


def test_pedal_edits_reach_the_preview_midi(pedal_midi: Path, tmp_path: Path):
    """The export is what the renderer plays, so the edit must be in the file."""
    session = Session.open(pedal_midi)
    session.add_pedal(1200, 1600)
    midi = PreviewMidi(tmp_path / "p").ensure(session.project)
    spans = read_pedal(midi)
    assert spans[0] == [PedalSpan(0, 960), PedalSpan(1200, 1600)]
