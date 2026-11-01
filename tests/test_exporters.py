"""Tests for qmidi.exporters -- the two files the user actually ships.

The MIDI tests need nothing but mido.  The PCM/MP3 encoder tests need lameenc
but no instrument.  The one real render test is skipped when no .sfz is
reachable, exactly like ``tests/test_offline.py``.
"""
from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import mido
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi import exporters  # noqa: E402
from qmidi.engine import EngineError, resolve_sfz_path  # noqa: E402
from qmidi.exporters import (  # noqa: E402
    BITRATES,
    AudioExportResult,
    ExportError,
    default_midi_output,
    encode_mp3,
    export_mp3,
    export_midi,
    export_project,
    to_pcm16,
)
from qmidi.offline import RenderResult  # noqa: E402
from qmidi.segments import (  # noqa: E402
    SegmentProject,
    SegmentedTrack,
    VelocitySegment,
    build_tracks,
    clamp_velocity,
    read_notes,
)

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():  # example/ exists on the local copy only
    pytest.skip("example/midi/Unchained.mid is not part of the public distribution",
                allow_module_level=True)


def _instrument_available() -> bool:
    try:
        resolve_sfz_path()
        return True
    except Exception:
        return False


def _need_sfz():
    try:
        return resolve_sfz_path()
    except EngineError:
        pytest.skip("no .sfz instrument reachable")


# --------------------------------------------------------------------------- #
# helpers: a hand-built MIDI file, so a test can state its ticks exactly
# --------------------------------------------------------------------------- #
def _on(tick, note, velocity, channel=0):
    return (tick, mido.Message("note_on", channel=channel, note=note,
                               velocity=velocity))


def _off(tick, note, channel=0):
    """Release in Unchained.mid's convention: a velocity-0 ``note_on``."""
    return (tick, mido.Message("note_on", channel=channel, note=note, velocity=0))


def _write_midi(path: Path, track_notes: list[tuple[str, list[tuple]]]) -> Path:
    """``track_notes`` is ``[(name, [(tick, message), ...]), ...]``.

    Tick deltas are computed here so a test can talk in absolute ticks without
    the reader having to do mental arithmetic.
    """
    midi = mido.MidiFile(ticks_per_beat=480, type=1)
    for name, events in track_notes:
        track = mido.MidiTrack()
        track.append(mido.MetaMessage("track_name", name=name, time=0))
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        previous = 0
        for tick, msg in events:
            track.append(msg.copy(time=tick - previous))
            previous = tick
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.tracks.append(track)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(path))
    return path


def _at(path: Path, tick: int, note: int, channel: int = 0) -> list[mido.Message]:
    """Every note message for ``note`` sitting exactly on ``tick``."""
    out = []
    for track in mido.MidiFile(str(path)).tracks:
        at = 0
        for msg in track:
            at += msg.time
            if at == tick and not msg.is_meta and getattr(msg, "note", None) == note \
                    and msg.channel == channel:
                out.append(msg)
    return out


def _is_release(msg: mido.Message) -> bool:
    return msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0)


def _project_for(path: Path, segments_by_track: list[list[VelocitySegment]],
                 ) -> SegmentProject:
    """A project over ``path`` with hand-picked segments (no auto cut)."""
    notes_by_track = read_notes(path)
    tracks = [
        SegmentedTrack(
            track=i,
            notes=list(notes_by_track[i]),
            segments=list(segments_by_track[i]),
        )
        for i in range(len(notes_by_track))
    ]
    return SegmentProject(source=path, ticks_per_beat=480, tracks=tracks)


def _seg(track, start, end, indices, *, delta=0, target=None):
    return VelocitySegment(track=track, start_tick=start, end_tick=end,
                           delta=delta, target=target, note_indices=tuple(indices))


# --------------------------------------------------------------------------- #
# MIDI export -- fidelity
# --------------------------------------------------------------------------- #
def test_export_without_adjustments_is_a_faithful_copy(tmp_path):
    """No segments touched -> the exported file must carry the original notes."""
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    out = tmp_path / "copy.mid"
    result = export_midi(project, out)

    assert result.notes_rewritten == 0
    assert result.retriggers == 0
    assert result.tracks_touched == 0

    before, after = read_notes(MIDI), read_notes(out)
    assert len(before) == len(after)
    for track_before, track_after in zip(before, after):
        assert [(n.channel, n.pitch, n.velocity, n.start_tick, n.end_tick)
                for n in track_after] == [
            (n.channel, n.pitch, n.velocity, n.start_tick, n.end_tick)
            for n in track_before]


def test_export_preserves_track_structure_names_and_meta(tmp_path):
    """Spec 1a: the multi-track skeleton must survive, not be flattened."""
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    out = tmp_path / "struct.mid"
    export_midi(project, out)

    before, after = mido.MidiFile(str(MIDI)), mido.MidiFile(str(out))
    assert after.ticks_per_beat == before.ticks_per_beat == 480
    assert after.type == before.type == 1
    assert len(after.tracks) == len(before.tracks) == 4

    def meta(midi, kind):
        return [[(m.name, m.time) if kind == "track_name" else m.type
                 for m in t if m.is_meta and m.type == kind]
                for t in midi.tracks]

    for kind in ("track_name", "set_tempo", "time_signature", "key_signature",
                 "end_of_track"):
        assert meta(after, kind) == meta(before, kind), kind
    # Non-note channel traffic (pedal, program change) is untouched too.
    def cc(midi):
        return [[(m.control, m.value) for m in t
                 if not m.is_meta and m.type == "control_change"] for t in midi.tracks]
    assert cc(after) == cc(before)


# --------------------------------------------------------------------------- #
# MIDI export -- the round-trip invariant
# --------------------------------------------------------------------------- #
def test_round_trip_velocities_equal_resolved_notes(tmp_path):
    """Spec 1d: re-reading the export must reproduce ``resolved_notes()``.

    This is the test that locks the whole feature down.  Every track gets a
    mix of deltas and one pinned target, then the export is re-read with mido
    and compared note-for-note against what the segment model says the note
    should be.
    """
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    for t in project.tracks:
        t.validate()
        for i in range(len(t.segments)):
            t.apply_delta(i, (i % 5) - 2)
    project.track(1).set_value(2, 64)

    out = tmp_path / "roundtrip.mid"
    result = export_midi(project, out)
    assert result.notes_rewritten > 1000
    assert result.warnings == []

    reread = read_notes(out)
    for index, track in enumerate(project.tracks):
        expected = {n.index: n.velocity for n in track.resolved_notes()}
        actual = {n.index: n.velocity for n in reread[index]}
        assert actual == expected, f"track {index} velocities drifted"
        # Spans are structural: a retake must not move or shorten a note.
        assert [(n.pitch, n.start_tick, n.end_tick) for n in reread[index]] == [
            (n.pitch, n.start_tick, n.end_tick) for n in track.notes]


def test_target_segment_overrides_delta(tmp_path):
    """``target`` wins over ``delta`` and the export honours that."""
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    track = project.track(0)
    track.apply_delta(0, -10)
    track.set_value(0, 64)
    out = tmp_path / "target.mid"
    export_midi(project, out)

    reread = read_notes(out)[0]
    expected = {n.index: n.velocity for n in track.resolved_notes()}
    assert {n.index: n.velocity for n in reread} == expected

    pinned = set(track.segments[0].note_indices)
    assert pinned
    assert {expected[i] for i in pinned} == {64}
    # The delta on that same segment is inert because `target` wins.
    assert track.segments[0].delta == -10
    assert {expected[i] for i in pinned} == {
        clamp_velocity(track.segments[0].target)}


def test_velocity_is_clamped_into_the_legal_range(tmp_path):
    """Spec 1c: a huge negative delta must bottom out at 1, never 0 or negative."""
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    track = project.track(2)
    for i in range(len(track.segments)):
        track.apply_delta(i, -200)
    out = tmp_path / "clamped.mid"
    export_midi(project, out)

    reread = read_notes(out)[2]
    assert reread, "track 2 has notes"
    assert all(1 <= n.velocity <= 127 for n in reread)
    assert {n.velocity for n in reread} == {1}
    # And it agrees with what the model resolved, not just with the range check.
    assert {n.index: n.velocity for n in reread} == {
        n.index: n.velocity for n in track.resolved_notes()}


# --------------------------------------------------------------------------- #
# MIDI export -- the retake
# --------------------------------------------------------------------------- #
def test_velocity_change_retakes_the_pitch(tmp_path):
    """A re-attack written before its own release gets the release moved first.

    ``on A | on B | off A | off B`` on one pitch is the ordering that makes some
    samplers cancel B: the note-off for A arrives after B's attack.  The export
    must turn it into ``on A | off A | on B | off B`` with no note shortened.
    """
    src = _write_midi(tmp_path / "retake.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _on(480, 60, 100, 0),
        _off(480, 60, 0),
        _off(960, 60, 0),
    ])])
    project = _project_for(src, [[
        _seg(0, 0, 480, [0]),
        _seg(0, 480, 960, [1], delta=-20),
    ]])
    out = tmp_path / "retake_out.mid"
    result = export_midi(project, out)

    assert result.retriggers == 1
    assert result.notes_rewritten == 1
    assert result.warnings == []

    boundary = _at(out, 480, 60)
    assert [_is_release(m) for m in boundary] == [True, False], boundary
    assert boundary[-1].velocity == 80

    # Spans untouched: this was a relocation, not a cut.
    assert [(n.start_tick, n.end_tick, n.velocity) for n in read_notes(out)[0]] == [
        (0, 480, 100), (480, 960, 80)]


def test_release_written_before_the_attack_is_left_alone(tmp_path):
    """``on A | off A | on B`` already re-attacks; adding a retake would be churn."""
    src = _write_midi(tmp_path / "clean.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _off(480, 60, 0),
        _on(480, 60, 100, 0),
        _off(960, 60, 0),
    ])])
    project = _project_for(src, [[
        _seg(0, 0, 480, [0]),
        _seg(0, 480, 960, [1], delta=-20),
    ]])
    out = tmp_path / "clean_out.mid"
    result = export_midi(project, out)

    assert result.retriggers == 0
    boundary = _at(out, 480, 60)
    assert [_is_release(m) for m in boundary] == [True, False]
    assert boundary[-1].velocity == 80


def test_unchanged_velocity_never_inserts_a_retake(tmp_path):
    """A boundary with no velocity change must not reorder the file."""
    src = _write_midi(tmp_path / "same.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _on(480, 60, 100, 0),
        _off(480, 60, 0),
        _off(960, 60, 0),
    ])])
    project = _project_for(src, [[_seg(0, 0, 960, [0, 1])]])
    out = tmp_path / "same_out.mid"
    result = export_midi(project, out)

    assert result.retriggers == 0
    assert [_is_release(m) for m in _at(out, 480, 60)] == [False, True]


def test_same_pitch_on_two_channels_is_not_retaken(tmp_path):
    """Two pianos on two channels are two voices; a retake would cross the wire."""
    src = _write_midi(tmp_path / "duo.mid", [
        ("Piano", [
            _on(0, 60, 100, 0),
            _on(480, 60, 100, 0),
            _off(480, 60, 0),
            _off(960, 60, 0),
        ]),
        ("Piano 2", [
            _on(0, 60, 100, 1),
            _on(480, 60, 100, 1),
            _off(480, 60, 1),
            _off(960, 60, 1),
        ]),
    ])
    project = _project_for(src, [
        [_seg(0, 0, 480, [0]), _seg(0, 480, 960, [1], delta=-20)],
        [_seg(1, 0, 960, [0])],
    ])
    out = tmp_path / "duo_out.mid"
    result = export_midi(project, out)

    # Channel 0's retake must not release channel 1's held note.
    assert result.retriggers == 1
    assert [(n.channel, n.start_tick, n.end_tick) for n in read_notes(out)[1]] == [
        (1, 0, 480), (1, 480, 960)]


def test_straddling_note_is_never_cut_short(tmp_path):
    """Anti-voice-kill guard: a boundary forced through a note is reported, not applied.

    A hand-built (invalid) sidecar can claim a note belongs to two segments.  A
    retake here would release the held note early, so the exporter declines and
    says so instead of silently truncating someone's music.
    """
    src = _write_midi(tmp_path / "straddle.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _on(480, 60, 100, 0),
        _off(960, 60, 0),
        _off(960, 60, 0),
    ])])
    project = _project_for(src, [[
        _seg(0, 0, 480, [0], delta=-20),
        _seg(0, 480, 960, [0, 1], delta=-20),
    ]])
    out = tmp_path / "straddle_out.mid"
    result = export_midi(project, out)

    assert result.retriggers == 0
    assert any("left uncut" in w for w in result.warnings), result.warnings
    # The held note still lasts its full length in the exported file.
    spans = [(n.start_tick, n.end_tick) for n in read_notes(out)[0]]
    assert spans == [(0, 960), (480, 960)]


def test_note_without_a_segment_keeps_its_velocity(tmp_path):
    """Incomplete coverage is a warning, not a crash and not a silent change."""
    src = _write_midi(tmp_path / "partial.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _off(480, 60, 0),
        _on(480, 62, 100, 0),
        _off(960, 62, 0),
    ])])
    project = _project_for(src, [[_seg(0, 0, 480, [0], delta=-20)]])
    out = tmp_path / "partial_out.mid"
    result = export_midi(project, out)

    assert result.notes_rewritten == 1
    assert any("no segment" in w for w in result.warnings), result.warnings
    assert {n.velocity for n in read_notes(out)[0]} == {80, 100}


# --------------------------------------------------------------------------- #
# MIDI export -- refusing bad input
# --------------------------------------------------------------------------- #
def test_export_refuses_a_missing_source(tmp_path):
    project = SegmentProject(source=tmp_path / "nope.mid", ticks_per_beat=480, tracks=[])
    with pytest.raises(ExportError, match="source MIDI not found"):
        export_midi(project, tmp_path / "out.mid")


def test_export_refuses_a_track_count_mismatch(tmp_path):
    src = _write_midi(tmp_path / "two.mid", [
        ("a", [_on(0, 60, 1, 0)]),
        ("b", [_on(0, 60, 1, 0)]),
    ])
    project = _project_for(src, [[_seg(0, 0, 480, [0])], [_seg(1, 0, 480, [0])]])
    project.tracks = project.tracks[:1]
    with pytest.raises(ExportError, match="tracks but the project"):
        export_midi(project, tmp_path / "out.mid")


def test_export_refuses_a_sidecar_from_a_different_file(tmp_path):
    """Index alignment is verified, not assumed (see ``_verify_indices``)."""
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    track = project.track(0)
    track.notes = [replace(n, velocity=n.velocity + 7) if n.index == 5 else n
                   for n in track.notes]
    with pytest.raises(ExportError, match="does not match the project's note"):
        export_midi(project, tmp_path / "stale.mid")
    # ...and the check is switchable for a caller who knows better.
    assert export_midi(project, tmp_path / "stale.mid", verify=False).path.is_file()


def test_default_midi_output_never_overwrites_the_source():
    project = SegmentProject(source=Path("song.mid"), ticks_per_beat=480, tracks=[])
    assert default_midi_output(project).name == "song_qmidi.mid"
    assert default_midi_output(project) != Path("song.mid")


# --------------------------------------------------------------------------- #
# PCM conversion (no instrument, no encoder needed)
# --------------------------------------------------------------------------- #
def test_to_pcm16_only_attenuates_and_never_amplifies():
    quiet = np.full((2, 100), 0.25, dtype=np.float32)
    pcm, gain = to_pcm16(quiet)
    assert gain == 1.0
    assert pcm.shape == (100, 2)
    assert pcm.dtype == np.dtype("<i2")

    loud = np.full((2, 100), 2.0, dtype=np.float32)   # sfizz bus does this
    pcm, gain = to_pcm16(loud, peak_target=0.97)
    assert 0 < gain < 1
    assert np.max(np.abs(pcm.astype(np.float32) / 32767)) <= 0.97 + 1e-4


def test_to_pcm16_preserves_relative_dynamics():
    """One global gain: a loud sample stays loud *relative to* a soft one."""
    audio = np.array([[0.25, 1.0]], dtype=np.float32)
    pcm, gain = to_pcm16(audio, peak_target=0.5)
    values = pcm[:, 0].astype(np.float32) / 32767
    assert gain == pytest.approx(0.5)
    assert values[0] / values[1] == pytest.approx(0.25, rel=1e-3)


def test_to_pcm16_refuses_to_clip_when_normalisation_is_off():
    with pytest.raises(ExportError, match="normalize=False"):
        to_pcm16(np.full((2, 10), 3.0, dtype=np.float32), normalize=False)
    # A render that already fits is fine either way.
    _, gain = to_pcm16(np.full((2, 10), 0.5, dtype=np.float32), normalize=False)
    assert gain == 1.0


def test_to_pcm16_rejects_nan_and_bad_shapes():
    with pytest.raises(ExportError, match="NaN or inf"):
        to_pcm16(np.array([[0.1, np.nan]], dtype=np.float32))
    with pytest.raises(ExportError, match="channel-first"):
        to_pcm16(np.zeros((3, 10), dtype=np.float32))
    with pytest.raises(ExportError, match="channel-first"):
        to_pcm16(np.zeros(10, dtype=np.float32).reshape(1, 1, 10))


def test_to_pcm16_accepts_mono():
    pcm, _ = to_pcm16(np.zeros((1, 8), dtype=np.float32))
    assert pcm.shape == (8, 1)


# --------------------------------------------------------------------------- #
# MP3 encoding (lameenc only, no instrument)
# --------------------------------------------------------------------------- #
def _tone(seconds=0.5, rate=48000, channels=2, amp=0.3):
    t = np.arange(int(seconds * rate)) / rate
    mono = (amp * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    return np.repeat(mono[np.newaxis, :], channels, axis=0)


def test_encode_mp3_writes_a_real_frame_stream():
    pcm, _ = to_pcm16(_tone())
    blob = encode_mp3(pcm, 48000, 192)
    assert blob[:2] == b"\xff\xfb", "not an MPEG audio frame header"
    assert len(blob) > 1000


@pytest.mark.parametrize("bitrate", BITRATES)
def test_every_offered_bitrate_encodes(bitrate):
    blob = encode_mp3(to_pcm16(_tone(0.25))[0], 48000, bitrate)
    assert blob[:2] == b"\xff\xfb"


def test_higher_bitrate_is_a_bigger_file():
    pcm, _ = to_pcm16(_tone(1.0))
    sizes = [len(encode_mp3(pcm, 48000, b)) for b in BITRATES]
    assert sizes == sorted(sizes), sizes


def test_encode_mp3_rejects_unknown_bitrate_and_bad_pcm():
    pcm, _ = to_pcm16(_tone(0.1))
    with pytest.raises(ExportError, match="bitrate must be one of"):
        encode_mp3(pcm, 48000, 256)
    with pytest.raises(ExportError, match="bitrate must be one of"):
        encode_mp3(pcm, 48000, 64)
    with pytest.raises(ExportError, match="interleaved"):
        encode_mp3(np.zeros((10, 5), dtype="<i2"), 48000, 192)


def test_export_mp3_validates_before_it_renders(tmp_path):
    """Bad arguments must fail without ever touching the instrument."""
    src = _write_midi(tmp_path / "tiny.mid", [("a", [
        _on(0, 60, 100, 0)])])
    with pytest.raises(ExportError, match="MIDI not found"):
        export_mp3(tmp_path / "absent.mid", tmp_path / "o.mp3")
    with pytest.raises(ExportError, match="bitrate"):
        export_mp3(src, tmp_path / "o.mp3", bitrate=99)
    with pytest.raises(ExportError, match="peak_target"):
        export_mp3(src, tmp_path / "o.mp3", peak_target=1.5)
    with pytest.raises(ExportError, match="duration_sec"):
        export_mp3(src, tmp_path / "o.mp3", duration_sec=float("nan"))
    with pytest.raises(ExportError, match="duration_sec"):
        export_mp3(src, tmp_path / "o.mp3", duration_sec=0.0)


def test_whole_file_duration_defaults_to_file_length_plus_tail(tmp_path):
    """No duration -> the whole file, with room for the last release."""
    src = _write_midi(tmp_path / "len.mid", [("a", [
        _on(0, 60, 100, 0),
        _off(1920, 60, 0),
    ])])
    # 1920 ticks at 480 tpb and 120 BPM = 4 beats = 2.0 s.
    assert exporters._resolve_duration(src, None, 2.0) == pytest.approx(4.0)
    assert exporters._resolve_duration(src, None, 0.0) == pytest.approx(2.0)
    assert exporters._resolve_duration(src, 1.5, 2.0) == pytest.approx(1.5)


def test_export_project_renders_from_the_exported_midi(tmp_path, monkeypatch):
    """The MP3 must contain the user's edits, so it must read the *new* file."""
    src = _write_midi(tmp_path / "src.mid", [("Piano", [
        _on(0, 60, 100, 0),
        _off(480, 60, 0),
        _on(480, 62, 100, 0),
        _off(960, 62, 0),
    ])])
    project = _project_for(src, [[
        _seg(0, 0, 480, [0]),
        _seg(0, 480, 960, [1], delta=-20),
    ]])

    rendered_from = {}

    def fake_render(midi_path, offset_sec, duration_sec, **kwargs):
        rendered_from["path"] = Path(midi_path)
        rendered_from["offset"] = offset_sec
        rendered_from["duration"] = duration_sec
        frames = int(duration_sec * kwargs.get("sample_rate", 48000))
        return RenderResult(
            audio=np.zeros((2, frames), dtype=np.float32),
            tail=np.zeros((2, 0), dtype=np.float32),
            offset_sec=offset_sec, duration_sec=duration_sec, held=[],
            pedal_down=False, sfz_path=None,
            sample_rate=kwargs.get("sample_rate", 48000),
        )

    monkeypatch.setattr(exporters, "render_from", fake_render)
    midi_out = tmp_path / "out.mid"
    mp3_out = tmp_path / "out.mp3"
    bundle = export_project(project, midi_out, mp3_out, duration_sec=1.0, tail_sec=0.0)

    assert rendered_from["path"] == midi_out
    assert rendered_from["duration"] == 1.0
    assert {n.velocity for n in read_notes(rendered_from["path"])[0]} == {100, 80}
    assert {n.velocity for n in read_notes(src)[0]} == {100, 100}
    assert mp3_out.read_bytes()[:2] == b"\xff\xfb"
    assert isinstance(bundle.audio, AudioExportResult)


def test_export_project_without_mp3_only_writes_midi(tmp_path):
    project = SegmentProject(
        source=MIDI, ticks_per_beat=480, tracks=build_tracks(MIDI, ticks_per_beat=480))
    bundle = export_project(project, tmp_path / "only.mid")
    assert bundle.audio is None
    assert (tmp_path / "only.mid").is_file()


def test_export_mp3_keeps_a_wav_intermediate_when_asked(tmp_path, monkeypatch):
    src = _write_midi(tmp_path / "wav.mid", [("a", [
        _on(0, 60, 100, 0)])])

    def fake_render(midi_path, offset_sec, duration_sec, **kwargs):
        frames = int(duration_sec * 48000)
        return RenderResult(
            audio=np.full((2, frames), 1.5, dtype=np.float32),
            tail=np.zeros((2, 0), dtype=np.float32),
            offset_sec=offset_sec, duration_sec=duration_sec, held=[],
            pedal_down=False, sfz_path=None, sample_rate=48000,
        )

    monkeypatch.setattr(exporters, "render_from", fake_render)
    out = tmp_path / "clip.mp3"
    result = export_mp3(src, out, duration_sec=0.5, keep_wav=True)

    assert result.wav_path == out.with_suffix(".wav")
    assert result.wav_path.is_file()
    assert result.gain < 1.0, "a 1.5-peak render must have been attenuated"
    assert result.peak_before == pytest.approx(1.5)
    assert result.peak_after <= 0.97 + 1e-6
    assert result.channels == 2 and result.frames == 24000

    import soundfile as sf
    data, rate = sf.read(str(result.wav_path), always_2d=True)
    assert rate == 48000
    assert np.max(np.abs(data)) <= 0.97 + 1e-6


# --------------------------------------------------------------------------- #
# one real render (needs the instrument)
# --------------------------------------------------------------------------- #
def test_real_export_mp3_round_trip(tmp_path):
    _need_sfz()
    out = tmp_path / "snippet.mp3"
    result = export_mp3(MIDI, out, bitrate=128, offset_sec=72.801, duration_sec=1.0,
                        tail_sec=0.0)

    assert out.is_file()
    blob = out.read_bytes()
    assert blob[:2] == b"\xff\xfb"
    assert len(blob) > 5000, "a second of 128 kbps stereo is ~16 kB"
    assert result.peak_before > 0.0
    assert result.peak_after <= 0.97 + 1e-6
    assert result.wav_path is None

    # 320 kbps really is asked for, not silently ignored.
    bigger = export_mp3(MIDI, tmp_path / "hi.mp3", bitrate=320, offset_sec=72.801,
                        duration_sec=1.0, tail_sec=0.0)
    assert bigger.path.stat().st_size > out.stat().st_size


# --------------------------------------------------------------------------- #
# pan and mastering (2026-11-01)
# --------------------------------------------------------------------------- #
def _cc(path: Path, control: int) -> list[tuple[int, int, int]]:
    """``[(channel, value, tick), ...]`` for one controller across the file."""
    out = []
    for track in mido.MidiFile(str(path)).tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "control_change" and msg.control == control:
                out.append((msg.channel, msg.value, tick))
    return out


def test_pan_is_written_once_per_channel_at_tick_zero(tmp_path):
    """Two tracks per piano must not each write their own CC10."""
    src = _write_midi(tmp_path / "duo.mid", [
        ("Piano", [_on(0, 60, 100, 0), _off(480, 60, 0)]),
        ("Piano pedal", [_on(0, 64, 90, 0), _off(480, 64, 0)]),
        ("Piano 2", [_on(0, 67, 100, 1), _off(480, 67, 0)]),
    ])
    project = SegmentProject(
        source=src, ticks_per_beat=480,
        tracks=build_tracks(src, ticks_per_beat=480),
    )
    project.set_pan(0, 30)
    project.set_pan(1, 98)

    result = export_midi(project, tmp_path / "duo_out.mid")
    pans = _cc(tmp_path / "duo_out.mid", exporters.CC_PAN)

    assert result.pans_written == 2
    assert pans == [(0, 30, 0), (1, 98, 0)]


def test_pan_replaces_an_existing_controller_instead_of_stacking(tmp_path):
    src = _write_midi(tmp_path / "old.mid", [("a", [
        (0, mido.Message("control_change", channel=0, control=10, value=64)),
        _on(0, 60, 100, 0),
        _off(480, 60, 0),
    ])])
    project = SegmentProject(source=src, ticks_per_beat=480,
                             tracks=build_tracks(src, ticks_per_beat=480))
    project.set_pan(0, 20)
    export_midi(project, tmp_path / "out.mid")
    assert _cc(tmp_path / "out.mid", exporters.CC_PAN) == [(0, 20, 0)]


def test_panning_a_channel_the_file_does_not_use_warns(tmp_path):
    src = _write_midi(tmp_path / "mono.mid", [("a", [_on(0, 60, 100, 0)])])
    project = SegmentProject(source=src, ticks_per_beat=480,
                             tracks=build_tracks(src, ticks_per_beat=480))
    project.set_pan(5, 20)
    result = export_midi(project, tmp_path / "out.mid")
    assert result.pans_written == 0
    assert any("channel 5" in w for w in result.warnings)


def test_pan_gains_are_constant_power():
    left, right = exporters.pan_gains(0)
    assert (round(left, 6), round(right, 6)) == (1.0, 0.0)
    centre = exporters.pan_gains(64)
    assert sum(g ** 2 for g in centre) == pytest.approx(1.0, abs=1e-6)
    far_right = exporters.pan_gains(127)
    assert far_right[1] == pytest.approx(1.0)


def test_a_panned_channel_really_moves_in_the_rendered_file(tmp_path, monkeypatch):
    """The MIDI says CC10=20, so the audio must be louder on the left."""
    src = _write_midi(tmp_path / "p.mid", [("a", [
        _on(0, 60, 100, 0), _off(480, 60, 0)])])

    def fake_render(midi_path, offset_sec, duration_sec, **kwargs):
        frames = int(duration_sec * 48000)
        return RenderResult(
            audio=np.full((2, frames), 0.5, dtype=np.float32),
            tail=np.zeros((2, 0), dtype=np.float32),
            offset_sec=offset_sec, duration_sec=duration_sec, held=[],
            pedal_down=False, sfz_path=tmp_path / "fake.sfz", sample_rate=48000,
        )

    monkeypatch.setattr(exporters, "render_from", fake_render)
    out = tmp_path / "p.mp3"
    export_mp3(src, out, duration_sec=0.2, keep_wav=True, pans={0: 20})

    import soundfile as sf
    data, _ = sf.read(str(out.with_suffix(".wav")), always_2d=True)
    left = np.sqrt(np.mean(data[:, 0] ** 2))
    right = np.sqrt(np.mean(data[:, 1] ** 2))
    assert left > right * 2.0, "pan 20 should sit clearly on the left"


def test_master_settings_reach_the_mp3_and_report_loudness(tmp_path, monkeypatch):
    src = _write_midi(tmp_path / "m.mid", [("a", [
        _on(0, 60, 100, 0), _off(480, 60, 0)])])
    t = np.arange(int(48000 * 0.5)) / 48000
    sine = (np.sin(2 * np.pi * 440 * t) * 0.2).astype(np.float32)

    def fake_render(midi_path, offset_sec, duration_sec, **kwargs):
        frames = int(duration_sec * 48000)
        block = np.stack([sine, sine])[:, :frames]
        if block.shape[1] < frames:  # pad if the request is longer than the sine
            block = np.pad(block, ((0, 0), (0, frames - block.shape[1])))
        return RenderResult(
            audio=block.astype(np.float32),
            tail=np.zeros((2, 0), dtype=np.float32),
            offset_sec=offset_sec, duration_sec=duration_sec, held=[],
            pedal_down=False, sfz_path=tmp_path / "fake.sfz", sample_rate=48000,
        )

    monkeypatch.setattr(exporters, "render_from", fake_render)
    from qmidi.mastering import MasterSettings

    result = export_mp3(src, tmp_path / "m.mp3", duration_sec=0.5,
                        master=MasterSettings(enabled=True, target_lufs=-14.0))
    assert result.lufs_after == pytest.approx(-14.0, abs=1.0)
    assert result.lufs_before < result.lufs_after  # the render was quieter
    assert result.loudness_gain_db > 0.0


def test_mastering_off_restores_the_old_level_behaviour(tmp_path, monkeypatch):
    src = _write_midi(tmp_path / "off.mid", [("a", [
        _on(0, 60, 100, 0), _off(480, 60, 0)])])

    def fake_render(midi_path, offset_sec, duration_sec, **kwargs):
        frames = int(duration_sec * 48000)
        return RenderResult(
            audio=np.full((2, frames), 0.4, dtype=np.float32),
            tail=np.zeros((2, 0), dtype=np.float32),
            offset_sec=offset_sec, duration_sec=duration_sec, held=[],
            pedal_down=False, sfz_path=tmp_path / "fake.sfz", sample_rate=48000,
        )

    monkeypatch.setattr(exporters, "render_from", fake_render)
    from qmidi.mastering import MasterSettings

    result = export_mp3(src, tmp_path / "off.mp3", duration_sec=0.2,
                        master=MasterSettings(enabled=False))
    assert result.lufs_after == float("-inf")
    assert result.loudness_gain_db == 0.0


def test_pans_and_master_survive_the_sidecar(tmp_path):
    from qmidi.segments import SegmentProject, sidecar_path

    src = _write_midi(tmp_path / "s.mid", [("a", [_on(0, 60, 100, 0)])])
    project = SegmentProject(source=src, ticks_per_beat=480,
                             tracks=build_tracks(src, ticks_per_beat=480))
    project.set_pan(0, 33)
    project.set_master({"reverb_wet": 0.2, "target_lufs": -16.0})
    sidecar = project.save()

    again = SegmentProject.load(sidecar)
    assert again.pans == {0: 33}
    assert again.master == {"reverb_wet": 0.2, "target_lufs": -16.0}
    assert sidecar == sidecar_path(src)


def test_pan_out_of_range_is_rejected(tmp_path):
    src = _write_midi(tmp_path / "pan.mid", [("a", [_on(0, 60, 100, 0)])])
    project = SegmentProject(source=src, tracks=build_tracks(src))
    with pytest.raises(Exception):
        project.set_pan(0, 200)
    with pytest.raises(Exception):
        project.set_pan(16, 64)


@pytest.mark.skipif(not _instrument_available(), reason="no instrument installed")
def test_render_reports_progress_that_only_goes_forward():
    """The bar must be driven by the renderer, and end at exactly 1.0."""
    from qmidi.offline import render_from

    seen: list[float] = []
    render_from(MIDI, 0.0, 1.0, on_progress=seen.append)
    assert seen, "a callback that is never called is a spinner in disguise"
    assert seen[-1] == pytest.approx(1.0)
    assert all(0.0 <= value <= 1.0 for value in seen)
    assert seen == sorted(seen), "the bar must not jump backwards"


@pytest.mark.skipif(not _instrument_available(), reason="no instrument installed")
def test_mp3_export_reports_progress_all_the_way_to_done(tmp_path: Path):
    seen: list[float] = []
    export_mp3(MIDI, tmp_path / "p.mp3", duration_sec=0.4,
               on_progress=seen.append)
    assert seen[-1] == pytest.approx(1.0)
    assert seen[0] < 1.0, "even a short export must start below 100%"


# --------------------------------------------------------------------------- #
# channel x track matrix: two pianos (4 tracks / 2 channels) and smaller shapes
# --------------------------------------------------------------------------- #
def _piano_midi(tmp_path: Path, tracks: int, channels: list[int]) -> Path:
    """`tracks` note tracks; track i plays on channels[i]."""
    path = tmp_path / f"piano_{tracks}t.mid"
    track_notes = []
    for i in range(tracks):
        channel = channels[i]
        events = [_on(t, 60 + (i * 4) + (t // 240) % 4, 90, channel)
                  for t in range(0, 960, 240)]
        events += [_off(t + 200, 60 + (i * 4) + (t // 240) % 4, channel)
                   for t in range(0, 960, 240)]
        events.sort(key=lambda pair: pair[0])  # interleaved, or deltas go negative
        track_notes.append((f"P{i}", events))
    return _write_midi(path, track_notes)


def _cc10_per_channel(path: Path) -> dict[int, list[int]]:
    values: dict[int, list[int]] = {}
    for track in mido.MidiFile(str(path)).tracks:
        for msg in track:
            if msg.type == "control_change" and msg.control == 10:
                values.setdefault(msg.channel, []).append(msg.value)
    return values


def test_two_pianos_four_tracks_one_pan_per_channel(tmp_path: Path):
    """The canonical shape: 4 tracks, 2 channels, pan lands once per channel."""
    src = _piano_midi(tmp_path, 4, [0, 0, 1, 1])
    project = SegmentProject(source=src, tracks=build_tracks(src))
    project.set_pan(0, 40)
    project.set_pan(1, 88)
    result = export_midi(project, tmp_path / "out.mid")

    values = _cc10_per_channel(result.path)
    assert values == {0: [40], 1: [88]}, "exactly one CC10 per channel, tick 0"
    assert result.pans_written == 2


def test_two_tracks_one_channel_no_pans_no_cc10(tmp_path: Path):
    src = _piano_midi(tmp_path, 2, [0, 0])
    result = export_midi(
        SegmentProject(source=src, tracks=build_tracks(src)), tmp_path / "out.mid")
    assert _cc10_per_channel(result.path) == {}


def test_three_tracks_two_channels_pan_covers_both(tmp_path: Path):
    """Track 2 sits alone on channel 1; both channels still get their CC10."""
    src = _piano_midi(tmp_path, 3, [0, 0, 1])
    project = SegmentProject(source=src, tracks=build_tracks(src))
    project.set_pan(0, 52)
    project.set_pan(1, 76)
    result = export_midi(project, tmp_path / "out.mid")
    assert _cc10_per_channel(result.path) == {0: [52], 1: [76]}
    assert result.pans_written == 2


def test_single_track_single_channel_still_exports(tmp_path: Path):
    src = _piano_midi(tmp_path, 1, [0])
    result = export_midi(
        SegmentProject(source=src, tracks=build_tracks(src)), tmp_path / "out.mid")
    assert _cc10_per_channel(result.path) == {}
    assert result.path.is_file(), "the single-track shape must still export"


def test_panned_channels_without_notes_warn_instead_of_failing(tmp_path: Path):
    src = _piano_midi(tmp_path, 2, [0, 0])
    project = SegmentProject(source=src, tracks=build_tracks(src))
    project.set_pan(5, 30)          # channel 5 exists nowhere in the file
    result = export_midi(project, tmp_path / "out.mid")
    assert _cc10_per_channel(result.path) == {}
    assert any("channel 5" in w for w in result.warnings)


@pytest.mark.skipif(not _need_sfz(), reason="no .sfz instrument reachable")
def test_two_piano_mp3_with_mastering_round_trips(tmp_path: Path):
    """The full double-piano path: pan render, mix, mastering, encode."""
    src = _piano_midi(tmp_path, 4, [0, 0, 1, 1])
    project = SegmentProject(source=src, tracks=build_tracks(src))
    project.set_pan(0, 40)
    project.set_pan(1, 88)
    midi = export_midi(project, tmp_path / "out.mid").path
    from qmidi.mastering import MasterSettings

    result = export_mp3(midi, tmp_path / "out.mp3", duration_sec=0.5,
                        pans=project.pans,
                        master=MasterSettings(enabled=True, target_lufs=-14.0))
    assert result.frames > 0
    assert result.peak_after <= 1.0
    assert result.lufs_after == pytest.approx(-14.0, abs=1.5)
