"""Tests for qmidi.offline -- the seek-anywhere scheduling layer.

The fast tests need nothing but mido; the audio tests need a real .sfz and are
skipped when the instrument junction is not present.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mido
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.engine import EngineError, resolve_sfz_path  # noqa: E402
from qmidi.offline import (  # noqa: E402
    CC_SUSTAIN,
    KIND_CC,
    KIND_NOTE_OFF,
    KIND_NOTE_ON,
    KIND_PANIC,
    BlockStreamer,
    Event,
    TickClock,
    build_timeline,
    render_from,
    resolve_track_filter,
    scan_state,
)

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():  # example/ exists on the local copy only
    pytest.skip("example/midi/Unchained.mid is not part of the public distribution",
                allow_module_level=True)


def _ev(*events: Event) -> list[Event]:
    return list(events)


# --------------------------------------------------------------------------- #
# timeline
# --------------------------------------------------------------------------- #
def test_velocity_zero_note_on_becomes_note_off():
    """Unchained.mid has 4065 velocity-0 note-ons and zero note_off messages."""
    events = build_timeline(MIDI)
    assert events, "timeline is empty"
    assert not [e for e in events if e.kind == KIND_NOTE_ON and e.value == 0]
    on = sum(1 for e in events if e.kind == KIND_NOTE_ON)
    off = sum(1 for e in events if e.kind == KIND_NOTE_OFF)
    assert on == 4065 and off == 4065


def test_timeline_is_sorted_and_ends_at_file_length():
    events = build_timeline(MIDI)
    assert all(a.t <= b.t for a, b in zip(events, events[1:]))
    assert events[-1].t == pytest.approx(218.4025, abs=0.01)


def test_pedal_is_folded_into_one_global_controller():
    """Two channels each have a pedal; the adapter owns exactly one."""
    events = build_timeline(MIDI)
    pedals = [e for e in events if e.kind == KIND_CC and e.pitch == CC_SUSTAIN]
    assert pedals, "no pedal events"
    assert all(e.value in (0, 127) for e in pedals)
    raw = 0
    for track in mido.MidiFile(str(MIDI)).tracks:
        raw += sum(1 for m in track if m.type == "control_change" and m.control == CC_SUSTAIN)
    assert len(pedals) < raw, "pedal events were not folded"
    # The fold is order-dependent, so it must run in *time* order, not track
    # order -- otherwise the sorted stream contains redundant pedal writes.
    assert not [
        1 for a, b in zip(pedals, pedals[1:]) if a.value == b.value
    ], "redundant pedal write survived the fold"


def test_channel_filter_keeps_one_channel():
    everything = build_timeline(MIDI)
    ch0 = build_timeline(MIDI, channel=0)
    assert len(ch0) < len(everything)
    assert scan_state(ch0, 30.0).pedal_down is True


# --------------------------------------------------------------------------- #
# state rebuild
# --------------------------------------------------------------------------- #
def test_released_note_keeps_sounding_while_pedal_is_down():
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_NOTE_ON, 60, 100),
            Event(0.5, 1, KIND_NOTE_ON, 64, 90),
            Event(0.6, 2, KIND_CC, CC_SUSTAIN, 127),
            Event(1.0, 3, KIND_NOTE_OFF, 60, 0),  # released, but pedal holds it
        ),
        2.0,
    )
    assert state.pedal_down
    # 60 was released at 1.0 s but still sounds at 2.0 s -- "held" is not
    # "note_on without a note_off".
    assert state.held() == [(60, 100), (64, 90)]


def test_pedal_up_drops_the_deferred_releases():
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_NOTE_ON, 60, 100),
            Event(0.1, 1, KIND_NOTE_ON, 62, 100),
            Event(0.6, 2, KIND_CC, CC_SUSTAIN, 127),
            Event(1.0, 3, KIND_NOTE_OFF, 60, 0),
            Event(1.5, 4, KIND_CC, CC_SUSTAIN, 0),
        ),
        3.0,
    )
    assert not state.pedal_down
    assert state.held() == [(62, 100)]


def test_release_without_pedal_ends_the_note():
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_NOTE_ON, 60, 100),
            Event(0.6, 1, KIND_CC, CC_SUSTAIN, 127),
            Event(1.0, 2, KIND_CC, CC_SUSTAIN, 0),
            Event(1.5, 3, KIND_NOTE_OFF, 60, 0),
        ),
        3.0,
    )
    assert state.held() == []


def test_same_pitch_overlap_collapses_to_one_voice():
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_NOTE_ON, 60, 40),
            Event(0.5, 1, KIND_NOTE_ON, 60, 90),
            Event(1.0, 2, KIND_NOTE_OFF, 60, 0),
        ),
        2.0,
    )
    assert state.held() == [(60, 90)]


def test_panic_clears_everything():
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_NOTE_ON, 60, 100),
            Event(0.5, 1, "panic"),
        ),
        1.0,
    )
    assert state.held() == [] and not state.pedal_down


def test_panic_keeps_state_ccs_because_the_engine_keeps_them():
    """all_sound_off silences voices; it does not reset controllers (CC121 does).

    Clearing these on panic would make a restored state disagree with the
    engine, so the staleness is deliberate, not an oversight.
    """
    state = scan_state(
        _ev(
            Event(0.0, 0, KIND_CC, 7, 100),
            Event(0.1, 1, KIND_CC, 91, 40),
            Event(0.2, 2, KIND_NOTE_ON, 60, 90),
            Event(0.5, 3, KIND_PANIC),
        ),
        1.0,
    )
    assert state.held() == []
    assert state.ccs == {7: 100, 91: 40}


# --------------------------------------------------------------------------- #
# ordering + argument validation
# --------------------------------------------------------------------------- #
def test_unsorted_events_are_sorted_not_silently_truncated():
    """scan_state breaks at the first event past the offset, so order matters."""
    events = [Event(100.0, 0, KIND_NOTE_ON, 60, 100), Event(0.0, 1, KIND_NOTE_ON, 64, 90)]
    assert scan_state(events, 9.0).held() == [(64, 90)]


def test_nan_and_inf_arguments_are_rejected():
    for bad in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValueError, match="finite"):
            render_from(MIDI, bad, 1.0)
        with pytest.raises(ValueError, match="finite"):
            render_from(MIDI, 0.0, bad)
        with pytest.raises(ValueError, match="finite"):
            render_from(MIDI, 0.0, 1.0, settle_sec=bad)


def test_render_from_rejects_empty_timeline_cleanly():
    """No events at all is a valid (silent) fragment, not a crash."""
    r = render_from(MIDI, 0.0, 0.1, timeline=[])
    assert r.audio.shape == (2, 4800)
    assert float(np.max(np.abs(r.audio))) == 0.0
    assert r.held == []


# --------------------------------------------------------------------------- #
# rendering (needs an instrument)
# --------------------------------------------------------------------------- #
def _need_sfz():
    try:
        return resolve_sfz_path()
    except EngineError:
        pytest.skip("no .sfz instrument reachable")


def test_seek_from_mid_file_is_deterministic_and_not_stuck():
    _need_sfz()
    offset, dur = 72.801, 2.0
    events = build_timeline(MIDI)
    a = render_from(MIDI, offset, dur, settle_sec=0.5, timeline=events)
    b = render_from(MIDI, offset, dur, settle_sec=0.5, timeline=events)

    assert np.array_equal(a.audio, b.audio), "freewheeling render is not deterministic"
    assert np.array_equal(a.tail, b.tail)

    head = a.audio[:, : a.sample_rate // 2]
    assert np.sqrt(np.mean(np.square(head.astype(np.float64)))) > 1e-4, (
        "nothing is sounding at the seek point -- state rebuild failed"
    )
    assert float(np.max(np.abs(a.tail))) < 1e-4, "a voice survived all_sound_off"
    assert len(a.held) > 0, "test point should sit on a held chord"


def test_seek_at_zero_is_a_plain_timeline_render():
    """No restore to do, and the t=0 events must not be fired twice."""
    _need_sfz()
    events = build_timeline(MIDI)
    r = render_from(MIDI, 0.0, 0.5, timeline=events)
    assert r.held == [(55, 16), (60, 16)]  # the two notes that start the file
    assert float(np.max(np.abs(r.audio))) > 1e-4


def test_panic_event_during_render_does_not_raise():
    """all_sound_off() takes no args -- the uniform (pitch, value) dispatch
    raised TypeError on every MIDI containing CC120/CC123."""
    _need_sfz()
    events = [
        Event(0.0, 0, KIND_NOTE_ON, 60, 100),
        Event(0.02, 1, KIND_NOTE_ON, 64, 100),
        Event(0.05, 2, KIND_PANIC),
        Event(0.08, 3, KIND_NOTE_ON, 67, 100),
    ]
    r = render_from(MIDI, 0.0, 0.5, timeline=events, settle_sec=0.2)
    assert r.audio.shape == (2, 24000)
    # Something before the panic, then the post-panic note still sounds.
    assert float(np.max(np.abs(r.audio[:, : int(0.02 * 48000)]))) > 1e-4
    assert float(np.max(np.abs(r.audio[:, int(0.08 * 48000):]))) > 1e-4


def test_seek_after_a_panic_replays_cleanly():
    """A file whose panic precedes the offset must still seek without stuck notes."""
    _need_sfz()
    events = [
        Event(0.0, 0, KIND_NOTE_ON, 60, 100),
        Event(0.5, 1, KIND_PANIC),
        Event(1.0, 2, KIND_NOTE_ON, 67, 100),
    ]
    r = render_from(MIDI, 2.0, 0.5, timeline=events, settle_sec=0.2)
    assert r.held == [(67, 100)]
    assert float(np.max(np.abs(r.audio))) > 1e-4
    assert float(np.max(np.abs(r.tail))) < 1e-4


def test_write_wav_round_trips(tmp_path):
    _need_sfz()
    from qmidi.offline import write_wav

    r = render_from(MIDI, 0.0, 0.2, timeline=build_timeline(MIDI))
    path = write_wav(tmp_path / "nested" / "frag.wav", r.audio, r.sample_rate)
    assert path.is_file()

    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    assert sr == r.sample_rate
    assert data.shape == r.audio.T.shape
    assert np.allclose(data.T, r.audio, atol=1e-6)


def test_write_wav_rejects_wrong_shapes(tmp_path):
    from qmidi.offline import write_wav

    with pytest.raises(ValueError, match="channel-first"):
        write_wav(tmp_path / "bad.wav", np.zeros(4800), 48000)          # 1-D
    with pytest.raises(ValueError, match="channel-first"):
        write_wav(tmp_path / "bad.wav", np.zeros((3, 100)), 48000)      # 3 channels


def test_write_wav_accepts_mono_and_empty_tail(tmp_path):
    """settle_sec=0 yields a (2, 0) tail -- an empty array must still write."""
    from qmidi.offline import write_wav

    path = write_wav(tmp_path / "empty.wav", np.zeros((2, 0), dtype=np.float32), 48000)
    assert path.is_file()
    path = write_wav(tmp_path / "mono.wav", np.zeros((1, 128), dtype=np.float32), 48000)
    assert path.is_file()

# --------------------------------------------------------------------------- #
# track filter (the mute / solo buttons)
# --------------------------------------------------------------------------- #
def _track_note_ons(index: int) -> int:
    track = mido.MidiFile(str(MIDI)).tracks[index]
    return sum(1 for m in track if m.type == "note_on" and m.velocity > 0)


def test_track_filter_keeps_exactly_one_track():
    everything = build_timeline(MIDI)
    only_zero = build_timeline(MIDI, tracks=[0])
    assert len(only_zero) < len(everything)
    assert sum(1 for e in only_zero if e.kind == KIND_NOTE_ON) == _track_note_ons(0)
    assert all(a.t <= b.t for a, b in zip(only_zero, only_zero[1:]))


def test_muting_the_pedal_tracks_lifts_the_pedal():
    """Tracks 1 and 3 are the only ones carrying CC64 in this file."""
    without_pedal = build_timeline(MIDI, tracks=[0, 2])
    assert not [e for e in without_pedal if e.kind == KIND_CC and e.pitch == CC_SUSTAIN]
    with_pedal = build_timeline(MIDI, tracks=[1, 3])
    assert [e for e in with_pedal if e.kind == KIND_CC and e.pitch == CC_SUSTAIN]


def test_solo_wins_over_mute():
    """A muted-and-soloed track still plays; otherwise the solo button lies."""
    assert resolve_track_filter(4, muted=[1], solo=[1]) == frozenset({1})
    assert resolve_track_filter(4, muted=[0, 1, 2, 3], solo=[2]) == frozenset({2})


def test_solo_is_exclusive_but_may_hold_several_tracks():
    assert resolve_track_filter(4, muted=[], solo=[0, 3]) == frozenset({0, 3})


def test_muting_everything_yields_an_empty_filter():
    assert resolve_track_filter(3, muted=[0, 1, 2]) == frozenset()
    assert resolve_track_filter(3) == frozenset({0, 1, 2})


def test_track_filter_rejects_junk_and_out_of_range():
    with pytest.raises(ValueError):
        resolve_track_filter(4, muted=[4])
    with pytest.raises(ValueError):
        resolve_track_filter(4, solo=[-1])
    with pytest.raises(ValueError):
        build_timeline(MIDI, tracks=[99])
    with pytest.raises(ValueError):
        resolve_track_filter(3, muted=[True])
    with pytest.raises(ValueError):
        resolve_track_filter(-1)


def test_render_from_refuses_timeline_plus_tracks():
    """An event list has lost its track tags; guessing would mute the wrong part."""
    events = build_timeline(MIDI, tracks=[0])
    with pytest.raises(ValueError, match="not both"):
        render_from(MIDI, 0.0, 1.0, timeline=events, tracks=[0])


# --------------------------------------------------------------------------- #
# tick <-> second clock
# --------------------------------------------------------------------------- #
def test_tick_clock_matches_the_timeline():
    """The clock and build_timeline must agree, or the playhead drifts."""
    clock = TickClock.from_midi(MIDI)
    midi = mido.MidiFile(str(MIDI))
    assert clock.ticks_per_beat == midi.ticks_per_beat
    assert clock.seconds(0) == pytest.approx(0.0)
    # Whatever this file's tempo is, a constant-tempo file must be linear in
    # beats.  (Unchained.mid is 100 bpm, not the 120 the default would imply,
    # which is exactly why the assertion is relational instead of literal.)
    one = clock.seconds(midi.ticks_per_beat)
    assert one > 0
    assert clock.seconds(midi.ticks_per_beat * 4) == pytest.approx(4 * one)


def test_tick_clock_round_trips_through_a_tempo_change():
    clock = TickClock([(0, 500_000), (1920, 250_000)], 480)  # 120 -> 240 bpm at beat 4
    assert clock.seconds(1920) == pytest.approx(2.0)
    assert clock.seconds(2880) == pytest.approx(2.5)  # one beat at 240 bpm
    for tick in (0, 1, 480, 1919, 1920, 1921, 5000):
        assert clock.tick(clock.seconds(tick)) == tick


def test_tick_clock_inverse_is_reachable_past_the_last_tempo_event():
    clock = TickClock([(0, 500_000), (1920, 250_000)], 480)
    assert clock.tick(60.0) > 1920
    assert clock.seconds(clock.tick(60.0)) == pytest.approx(60.0, abs=1e-3)


def test_tick_clock_rejects_junk():
    with pytest.raises(ValueError):
        TickClock([], 480)
    with pytest.raises(ValueError):
        TickClock([(0, 500_000)], 0)
    with pytest.raises(ValueError):
        TickClock([(0, 0)], 480)
    with pytest.raises(ValueError):
        TickClock([(0, 500_000)], 480).tick(float("nan"))


# --------------------------------------------------------------------------- #
# BlockStreamer (shared by the offline renderer and the GUI preview)
# --------------------------------------------------------------------------- #
def test_streamer_seek_rewinds_the_clock_and_keeps_rendering():
    _need_sfz()
    from qmidi.engine import SfzEngine

    eng = SfzEngine(block_size=256)
    eng.load_sfz()
    streamer = BlockStreamer(eng, build_timeline(MIDI))
    streamer.seek(100.0)
    for _ in range(8):
        streamer.next_block()
    assert streamer.position > 100.0
    streamer.seek(50.0)
    assert streamer.position == pytest.approx(50.0)
    block = streamer.next_block()
    assert block.shape == (2, 256)
    assert np.all(np.isfinite(block))
    assert streamer.position == pytest.approx(50.0 + 256 / 48000)


def test_streamer_seek_rejects_junk():
    _need_sfz()
    from qmidi.engine import SfzEngine

    eng = SfzEngine(block_size=256)
    eng.load_sfz()
    streamer = BlockStreamer(eng, [])
    with pytest.raises(ValueError):
        streamer.seek(-1.0)
    with pytest.raises(ValueError):
        streamer.seek(float("inf"))


# --------------------------------------------------------------------------- #
# BlockStreamer (shared by the offline renderer and the GUI preview)
# --------------------------------------------------------------------------- #
def test_streamer_seek_rewinds_the_clock_and_keeps_rendering():
    _need_sfz()
    from qmidi.engine import SfzEngine

    eng = SfzEngine(block_size=256)
    eng.load_sfz()
    streamer = BlockStreamer(eng, build_timeline(MIDI))
    streamer.seek(100.0)
    for _ in range(8):
        streamer.next_block()
    assert streamer.position > 100.0
    streamer.seek(50.0)
    assert streamer.position == pytest.approx(50.0)
    block = streamer.next_block()
    assert block.shape == (2, 256)
    assert np.all(np.isfinite(block))
    assert streamer.position == pytest.approx(50.0 + 256 / 48000)


def test_streamer_seek_rejects_junk():
    _need_sfz()
    from qmidi.engine import SfzEngine

    eng = SfzEngine(block_size=256)
    eng.load_sfz()
    streamer = BlockStreamer(eng, [])
    with pytest.raises(ValueError):
        streamer.seek(-1.0)
    with pytest.raises(ValueError):
        streamer.seek(float("inf"))
