"""Offline rendering: MIDI timeline -> block stream, with seek-anywhere.

This module owns the *scheduling* half of the audio pipeline; the synthesis
half stays behind :class:`qmidi.engine.SfzEngine` (six actions, no pysfizz).

Responsibilities
----------------
1. :func:`build_timeline` -- read a standard MIDI file with mido and flatten
   all tracks into one absolute-time event stream (the event-timeline idea that
   lived in ``scripts/verify_sfizz_render.py``, promoted to importable code).
2. :func:`SustainState.scan` -- replay that stream up to a timestamp and report
   the two pieces of state a seek must rebuild: pedal position and the notes
   still physically sounding.
3. :func:`render_from` -- ``all_sound_off``, rebuild that state, then stream
   blocks from ``offset_sec`` onward.

Three things a naive port from the scripts gets wrong
----------------------------------------------------
*Note-offs are usually note-ons.*  Unchained.mid has 4065 ``note_on``
messages with ``velocity == 0`` and *zero* ``note_off`` messages.  mido does
not rewrite them, so a ``msg.type == "note_on"`` dispatch would attack with
velocity 0 and the notes would never stop.  :func:`build_timeline` folds
velocity-0 note-ons into note-offs.

*The pedal holds notes that already ended.*  "Held at the offset" is not
"note_on without a note_off": with CC64 down, a note released ten seconds
ago is still ringing.  :class:`SustainState` models exactly that (and exactly
sfizz's rule: a release is deferred until the pedal lifts).

*The adapter has one pedal, the file has two.*  :meth:`SfzEngine.cc` has no
channel argument -- neither does pysfizz's binding.  Unchained.mid runs two
pianos on channels 0 and 1, each with its own sustain.  The timeline folds
the per-channel pedals into one global pedal (see
:data:`PEDAL_FOLD_POLICY`), which makes the rebuilt state agree with what the
engine will actually do for the rest of the fragment.  Consistency beats
musical fidelity here: a restore model that disagrees with the engine would
make the "no stuck notes" assertion meaningless.  A real 2-channel render
needs one engine instance per channel and a summed stereo bus -- deliberately
out of scope for this step.

*Mute/solo is per track, not per channel.*  The unit a user mutes in the GUI is
the *mido track*, because that is what the file shows and what the segment
model is keyed on.  In ``Unchained.mid`` the pedal lives on tracks 1 and 3,
each of which also holds notes -- so muting "the pedal track" is a real user
action, and it must be able to silence a pedal that a channel filter would
happily keep.  :func:`resolve_track_filter` turns the two GUI toggles into the
``tracks`` argument both :func:`build_timeline` and :func:`render_from` accept.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import mido
import numpy as np

from .engine import SfzEngine
from .pedal import CC_SUSTAIN, PEDAL_DOWN

__all__ = [
    "CC_SUSTAIN",
    "KIND_CC",
    "KIND_NOTE_OFF",
    "KIND_NOTE_ON",
    "KIND_PANIC",
    "PEDAL_FOLD_POLICY",
    "STATE_CCS",
    "BlockStreamer",
    "Event",
    "RenderResult",
    "SustainState",
    "TickClock",
    "build_timeline",
    "render_from",
    "resolve_track_filter",
    "write_wav",
]

#: MIDI controller number for the damper/sustain pedal (defined once, in
#: :mod:`qmidi.pedal`, and re-exported here because this module predates it).
KIND_CC = "cc"
KIND_NOTE_ON = "on"
KIND_NOTE_OFF = "off"
#: CC120/CC123 -- a panic, mapped onto the engine's ``all_sound_off``.
KIND_PANIC = "panic"

#: "any channel's pedal down == the pedal is down".  The engine owns exactly
#: one pedal, so a single channel lifting its foot cannot lift it.
PEDAL_FOLD_POLICY = "any-down"

#: Controllers restored at the seek point in addition to the pedal.  These are
#: the ones sfizz binds by default and that change how loud/bright the *next*
#: attack sounds, so a seek that ignores them audibly jumps.
STATE_CCS = (7, 10, 11, 91, 93)

#: CC >= this value means "pedal down" (the spec's 64 halfway point).
_PEDAL_DOWN = PEDAL_DOWN

#: Event kind -> engine method taking ``(pitch, value)``.  ``KIND_PANIC`` is
#: deliberately absent: :meth:`SfzEngine.all_sound_off` takes no arguments, so
#: routing it through the uniform two-argument dispatch raised ``TypeError`` on
#: every MIDI file containing CC120/CC123.  It gets its own call site instead.
_DISPATCH = {
    KIND_CC: "cc",
    KIND_NOTE_ON: "note_on",
    KIND_NOTE_OFF: "note_off",
}


@dataclass(frozen=True)
class Event:
    """One thing to do to the engine at absolute time ``t`` seconds."""

    t: float
    seq: int
    kind: str
    pitch: int = 0
    value: int = 0

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if self.kind == KIND_CC:
            return f"<cc{self.pitch}={self.value} @{self.t:.4f}s>"
        return f"<{self.kind} {self.pitch} v{self.value} @{self.t:.4f}s>"


# --------------------------------------------------------------------------- #
# timeline
# --------------------------------------------------------------------------- #
def _tempo_map(midi: mido.MidiFile) -> list[tuple[int, int]]:
    """``[(tick, us_per_beat), ...]`` sorted by tick, always starting at 0."""
    events: list[tuple[int, int]] = []
    for track in midi.tracks:
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.type == "set_tempo":
                events.append((tick, msg.tempo))
    events.sort()
    if not events or events[0][0] != 0:
        events.insert(0, (0, 500000))
    return events


class TickClock:
    """Tick <-> second conversion for one MIDI file's tempo map.

    The editor needs both directions and they must agree: a segment's tick span
    becomes a loop region in seconds for the renderer, and the playhead's
    seconds become a tick again for the cursor.  Two independent
    implementations would drift apart at every tempo change, so both directions
    here come from the same precomputed cumulative table.

    Ticks are the musical unit (segments live in ticks); seconds are the audio
    unit (``render_from`` lives in seconds).  Nothing in this class touches the
    file after construction.
    """

    __slots__ = ("_tempos", "_tpb", "_cumulative", "_ticks")

    def __init__(self, tempos: Sequence[tuple[int, int]], ticks_per_beat: int) -> None:
        if ticks_per_beat <= 0:
            raise ValueError(f"ticks_per_beat must be positive, got {ticks_per_beat!r}")
        if not tempos:
            raise ValueError("a tempo map needs at least one entry")
        ordered = sorted((int(t), int(us)) for t, us in tempos)
        if ordered[0][0] != 0:
            ordered.insert(0, (0, 500000))
        if any(t < 0 for t, _ in ordered) or any(us <= 0 for _, us in ordered):
            raise ValueError(f"invalid tempo map entry in {ordered!r}")
        cumulative: list[float] = [0.0]
        for (t0, us0), (t1, _) in zip(ordered, ordered[1:]):
            cumulative.append(cumulative[-1] + (t1 - t0) * us0 / (ticks_per_beat * 1e6))
        self._tempos = tuple(ordered)
        self._tpb = int(ticks_per_beat)
        self._cumulative = tuple(cumulative)
        self._ticks = tuple(t for t, _ in ordered)

    @classmethod
    def from_midi(cls, midi_path: Path | str) -> "TickClock":
        midi = mido.MidiFile(str(midi_path))
        return cls(_tempo_map(midi), midi.ticks_per_beat)

    @property
    def ticks_per_beat(self) -> int:
        return self._tpb

    @property
    def bpm(self) -> float:
        """Tempo at tick 0, in beats per minute -- the number the user edits."""
        return 60_000_000.0 / self._tempos[0][1]

    def scaled(self, factor: float) -> "TickClock":
        """A copy whose whole tempo map runs ``factor`` times slower.

        Scaling rather than replacing keeps whatever the file did musically: a
        ritardando stays a ritardando, just at the new speed.  Only the first
        tempo is exposed as "BPM", because that is the one a musician reads off
        the score.
        """
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"factor must be a positive finite number, got {factor!r}")
        if abs(factor - 1.0) < 1e-9:
            return self
        return TickClock(
            [(t, max(1, int(round(us * factor)))) for t, us in self._tempos],
            self._tpb,
        )

    def seconds(self, tick: float) -> float:
        """Seconds elapsed at ``tick`` (negative ticks clamp to 0)."""
        tick = max(0.0, float(tick))
        i = bisect.bisect_right(self._ticks, tick) - 1
        i = max(0, i)
        t0, us = self._tempos[i]
        return self._cumulative[i] + (tick - t0) * us / (self._tpb * 1e6)

    def tick(self, seconds: float) -> int:
        """Inverse of :meth:`seconds`, rounded to the nearest tick.

        A time past the last tempo change is extended at the final tempo, which
        is what a DAW does too -- the alternative (clamping) would freeze the
        playhead at the last tempo event.
        """
        if not math.isfinite(seconds):
            raise ValueError(f"seconds must be finite, got {seconds!r}")
        target = max(0.0, float(seconds))
        i = bisect.bisect_right(self._cumulative, target) - 1
        i = max(0, i)
        t0, us = self._tempos[i]
        return int(round(t0 + (target - self._cumulative[i]) * self._tpb * 1e6 / us))


def _track_indices(
    values: Optional[Sequence[int]], track_count: int, what: str
) -> Optional[set[int]]:
    """Validate a set of track indices, or ``None`` for "all tracks".

    ``bool`` is rejected (``True`` is not track 1) and out-of-range indices
    raise instead of being dropped, because a silently ignored index is how a
    mute button stops working without anybody noticing.
    """
    if values is None:
        return None
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{what} must be a sequence of track indices, got {values!r}")
    out: set[int] = set()
    for value in values:
        if isinstance(value, bool):
            raise ValueError(f"{what} must hold track indices, got bool {value!r}")
        try:
            index = int(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{what} must hold track indices, got {value!r}") from exc
        if index != value:
            raise ValueError(f"{what} must hold whole track indices, got {value!r}")
        if not 0 <= index < track_count:
            raise ValueError(
                f"{what} names track {index}, but the file holds "
                f"{track_count} track(s) (0..{track_count - 1})"
            )
        out.add(index)
    return out


def resolve_track_filter(
    track_count: int,
    muted: Sequence[int] = (),
    solo: Sequence[int] = (),
) -> frozenset[int]:
    """Turn the GUI's mute/solo toggles into a ``tracks`` argument.

    Two rules, both taken from every DAW the user has seen:

    * **Solo wins over mute.**  With at least one solo active, only the soloed
      tracks play -- a track that is both muted and soloed still plays, because
      otherwise the solo button would appear broken.
    * **Solo is exclusive.**  Two soloed tracks play together, which is how a
      user checks "these two parts against each other".

    Returns an empty frozenset when every track is muted; callers must treat
    that as "nothing to render" rather than passing it to
    :func:`render_from`, which rejects an empty timeline.
    """
    if isinstance(track_count, bool) or int(track_count) != track_count or track_count < 0:
        raise ValueError(f"track_count must be a non-negative integer, got {track_count!r}")
    count = int(track_count)
    muted_set = _track_indices(muted, count, "muted") or set()
    solo_set = _track_indices(solo, count, "solo") or set()
    if solo_set:
        return frozenset(solo_set)
    return frozenset(range(count)) - muted_set


def build_timeline(
    midi_path: Path | str,
    channel: Optional[int] = None,
    tracks: Optional[Sequence[int]] = None,
) -> list[Event]:
    """Flatten a (multi-track) MIDI file into one absolute-time event list.

    ``channel`` restricts note/CC traffic to one MIDI channel; ``None`` (the
    default) merges every channel, folding the pedals with
    :data:`PEDAL_FOLD_POLICY`.

    ``tracks`` restricts to a set of *mido track indices* (that is what the
    mute/solo buttons act on; see :func:`resolve_track_filter`).  ``None``
    means every track.  The filter is applied before the pedal fold, so muting
    a track that carries a pedal really does lift that pedal -- folding first
    and filtering afterwards would leave the pedal held down by a track the
    user cannot hear.

    The result is sorted by ``(t, seq)`` so the render loop can consume it with
    a single forward pointer.
    """
    midi = mido.MidiFile(str(midi_path))
    tpb = midi.ticks_per_beat
    clock = TickClock(_tempo_map(midi), tpb)
    keep = _track_indices(tracks, len(midi.tracks), "tracks")

    events: list[Event] = []
    #: (t, seq, channel, value_or_None) -- pedal writes and panics, folded
    #: *after* the sort.  Folding while walking tracks would be wrong: the
    #: fold is order-dependent and track order is not time order.
    pedal_stream: list[tuple[float, int, int, Optional[int]]] = []
    seq = 0

    for track_no, track in enumerate(midi.tracks):
        if keep is not None and track_no not in keep:
            continue
        tick = 0
        for msg in track:
            tick += msg.time
            if msg.is_meta or not hasattr(msg, "channel"):
                continue
            if channel is not None and msg.channel != channel:
                continue
            t = clock.seconds(tick)

            if msg.type == "note_on":
                if msg.velocity > 0:
                    events.append(Event(t, seq, KIND_NOTE_ON, msg.note, msg.velocity))
                else:
                    # Velocity-0 note-on IS a note-off (Unchained.mid: 4065 of them).
                    events.append(Event(t, seq, KIND_NOTE_OFF, msg.note, 0))
            elif msg.type == "note_off":
                events.append(Event(t, seq, KIND_NOTE_OFF, msg.note, msg.velocity))
            elif msg.type == "control_change":
                num, val = msg.control, msg.value
                if num == CC_SUSTAIN:
                    pedal_stream.append((t, seq, msg.channel, val))
                elif num in (120, 123):
                    events.append(Event(t, seq, KIND_PANIC))
                    pedal_stream.append((t, seq, msg.channel, None))  # None == panic
                elif num in STATE_CCS:
                    events.append(Event(t, seq, KIND_CC, num, val))
            seq += 1

    events.extend(_fold_pedal(pedal_stream, seq))
    events.sort(key=lambda e: (e.t, e.seq))
    return events


def _fold_pedal(stream: list[tuple[float, int, int, Optional[int]]], seq: int) -> list[Event]:
    """Collapse per-channel CC64 traffic into one global pedal (:data:`PEDAL_FOLD_POLICY`)."""
    stream.sort(key=lambda item: (item[0], item[1]))
    per_channel: dict[int, bool] = {}
    global_down = False
    out: list[Event] = []
    for t, s, ch, val in stream:
        if val is None:  # panic: every pedal is released by definition
            per_channel.clear()
            new_down = False
        else:
            per_channel[ch] = val >= _PEDAL_DOWN
            new_down = any(per_channel.values())  # "any-down"
        if new_down != global_down:
            global_down = new_down
            out.append(Event(t, s, KIND_CC, CC_SUSTAIN, 127 if global_down else 0))
            seq += 1
    return out


# --------------------------------------------------------------------------- #
# state at an arbitrary time
# --------------------------------------------------------------------------- #
@dataclass
class SustainState:
    """Which notes still sound at a given instant, and where the pedal is.

    Mirrors sfizz's sustain rule: a release while the pedal is down is
    remembered, not executed; lifting the pedal releases every remembered
    note at once.
    """

    pedal_down: bool = False
    ccs: dict[int, int] = field(default_factory=dict)
    #: pitch -> list of live voices ``(velocity, released)``, oldest first.
    voices: dict[int, list[list]] = field(default_factory=dict)

    def apply(self, ev: Event) -> None:
        """Advance the state machine by one event."""
        if ev.kind == KIND_NOTE_ON:
            self.voices.setdefault(ev.pitch, []).append([ev.value, False])
        elif ev.kind == KIND_NOTE_OFF:
            live = self.voices.get(ev.pitch)
            if live:
                for voice in live:  # oldest unreleased voice owns the release
                    if not voice[1]:
                        voice[1] = True
                        break
                if not self.pedal_down:
                    # Only the released voice dies; a later overlapping voice
                    # of the same pitch is still being held down.
                    kept = [v for v in live if not v[1]]
                    if kept:
                        self.voices[ev.pitch] = kept
                    else:
                        del self.voices[ev.pitch]
        elif ev.kind == KIND_CC and ev.pitch == CC_SUSTAIN:
            self.pedal_down = bool(ev.value)
            self.ccs[CC_SUSTAIN] = ev.value
            if not self.pedal_down:
                for pitch, live in list(self.voices.items()):
                    kept = [v for v in live if not v[1]]
                    if kept:
                        self.voices[pitch] = kept
                    else:
                        del self.voices[pitch]
        elif ev.kind == KIND_CC:
            self.ccs[ev.pitch] = ev.value
        elif ev.kind == KIND_PANIC:
            self.voices.clear()
            self.pedal_down = False
            self.ccs.pop(CC_SUSTAIN, None)
            # The remaining STATE_CCS deliberately survive a panic.  Measured on
            # Concert Grand: cc(7,127) -> all_sound_off() -> replay the same
            # note still sounds at 2.25x the RMS of a replay that forces cc7
            # back to 64, so sfizz's all_sound_off does *not* reset
            # controllers.  Clearing them here would make a restored state
            # disagree with what the engine actually does next -- the very
            # disagreement this module exists to avoid.  The MIDI spec agrees:
            # CC120/CC123 are "stop sounding", CC121 is "reset controllers".

    def held(self) -> list[tuple[int, int]]:
        """``[(pitch, velocity), ...]`` currently sounding, one entry per pitch.

        Several voices of one pitch cannot be re-attacked distinctly by the
        engine, so they collapse to the loudest one.
        """
        out = []
        for pitch, live in self.voices.items():
            if live:
                out.append((pitch, max(vel for vel, _ in live)))
        out.sort()
        return out


def _ordered(events: Sequence[Event]) -> list[Event]:
    """Return ``events`` as a list sorted by ``(t, seq)``.

    :func:`scan_state` stops at the first event past the offset and
    :func:`_first_event_after` binary-searches, so both are only correct on a
    time-ordered stream.  :func:`build_timeline` always returns one, so this is
    a no-op for the normal path (already sorted -> Timsort is linear); it only
    pays for itself when a caller hands over a hand-built list.
    """
    return sorted(events, key=lambda e: (e.t, e.seq))


def scan_state(events: Sequence[Event], offset_sec: float) -> SustainState:
    """Rebuild the engine state implied by every event at ``t <= offset_sec``.

    ``events`` must be ordered by time; out-of-order input is sorted here
    rather than silently truncated, because an unordered stream would make the
    early ``break`` drop every later-but-earlier event.
    """
    state = SustainState()
    for ev in _ordered(events):
        if ev.t <= offset_sec:
            state.apply(ev)
        else:
            break
    return state


# --------------------------------------------------------------------------- #
# streaming one engine over a timeline
# --------------------------------------------------------------------------- #
class BlockStreamer:
    """Drive one engine over a timeline: seek, then render blocks in order.

    Both the offline renderer and the GUI's live preview need the same three
    things -- rebuild the engine state at an arbitrary time, fire the events
    that fall inside each block, advance the clock.  Two copies of that
    dispatch would drift apart (one would learn about a new event kind before
    the other), so there is exactly one copy and it lives here.

    The clock is *audio* time, not wall time: :meth:`next_block` advances by
    ``block_size / sample_rate`` regardless of how long the call took, which is
    what makes a faster-than-real-time render possible at all (see the
    freewheeling note in :mod:`qmidi.engine`).

    ``seek()`` may be called at any time, including mid-stream: that is how the
    preview loops a segment without rebuilding the instrument.
    """

    __slots__ = ("_engine", "_events", "_offset", "_cursor", "_t", "_block_sec")

    def __init__(self, engine: SfzEngine, events: Sequence[Event]) -> None:
        self._engine = engine
        self._events = _ordered(events)
        self._block_sec = engine.block_size / engine.sample_rate
        self._offset = 0.0
        self._cursor = 0
        self._t = 0.0

    @property
    def position(self) -> float:
        """Start time of the block :meth:`next_block` will render next."""
        return self._t

    @property
    def block_seconds(self) -> float:
        return self._block_sec

    def seek(self, offset_sec: float) -> SustainState:
        """Rebuild the engine state implied by the events at or before the offset.

        Panic first, then controllers, then the pedal, then the still-sounding
        notes with their original attack velocities -- the order matters,
        because a pedal write after a note-on would hold a note that was never
        released.
        """
        if not math.isfinite(offset_sec) or offset_sec < 0:
            raise ValueError(f"offset_sec must be a finite number >= 0, got {offset_sec!r}")
        state = scan_state(self._events, offset_sec)
        eng = self._engine
        eng.all_sound_off()
        for num in STATE_CCS:
            if num in state.ccs:
                eng.cc(num, state.ccs[num])
        eng.cc(CC_SUSTAIN, 127 if state.pedal_down else 0)
        for pitch, vel in state.held():
            eng.note_on(pitch, vel)
        self._offset = float(offset_sec)
        self._t = float(offset_sec)
        self._cursor = _first_event_after(self._events, offset_sec)
        return state

    def next_block(self) -> np.ndarray:
        """Render the next ``block_size`` frames, firing everything inside it.

        Events at exactly the seek point are skipped: :meth:`seek` already
        applied them, and applying a note-on twice would double-attack.
        """
        block_end = self._t + self._block_sec
        events = self._events
        while self._cursor < len(events) and events[self._cursor].t < block_end:
            ev = events[self._cursor]
            if ev.t > self._offset:
                self._dispatch(ev)
            self._cursor += 1
        self._t = block_end
        return self._engine.render_block()

    def _dispatch(self, ev: Event) -> None:
        if ev.kind == KIND_PANIC:
            # Takes no arguments, so it cannot ride the (pitch, value) dispatch.
            self._engine.all_sound_off()
            return
        getattr(self._engine, _DISPATCH[ev.kind])(ev.pitch, ev.value)


# --------------------------------------------------------------------------- #
# rendering
# --------------------------------------------------------------------------- #
@dataclass
class RenderResult:
    audio: np.ndarray  # (2, n) float32 -- the requested fragment
    tail: np.ndarray  # (2, m) float32 -- rendered after all-sound-off
    offset_sec: float
    duration_sec: float
    held: list[tuple[int, int]]
    pedal_down: bool
    sfz_path: Path
    sample_rate: int

    @property
    def peak(self) -> float:
        return float(np.max(np.abs(self.audio))) if self.audio.size else 0.0


def render_from(
    midi_path: Path | str,
    offset_sec: float,
    duration_sec: float,
    sfz_path: Path | str | None = None,
    sample_rate: int = 48000,
    block_size: int = 1024,
    channel: Optional[int] = None,
    settle_sec: float = 0.0,
    timeline: Optional[Sequence[Event]] = None,
    tracks: Optional[Sequence[int]] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    sample_quality: Optional[int] = None,
) -> RenderResult:
    """Render ``duration_sec`` of the file starting at ``offset_sec``.

    Seek procedure, in order:

    1. ``all_sound_off()`` -- drop anything the engine still holds from before.
    2. Restore the pedal and the state controllers to their values at the
       offset, then ``note_on`` every still-sounding pitch **with its original
       attack velocity**.  Velocities matter: a restored note at velocity 100
       and one at velocity 20 are not the same instrument state.
    3. Stream blocks, firing timeline events in ``(offset, block_end]``.

    ``settle_sec > 0`` additionally renders that long a tail after
    ``all_sound_off()`` + pedal-up, which is what proves nothing is stuck.

    ``tracks`` filters the timeline to those mido tracks (see
    :func:`resolve_track_filter`).  It only applies when ``timeline`` is
    omitted: an event list has already lost which track each event came from,
    so honouring both would be a guess.  Passing both raises instead of
    silently rendering the muted tracks.

    Caveat, stated plainly: a restored note re-attacks.  Its waveform phase and
    envelope restart at the seek point, which is not bit-identical to a
    continuous render through that point.  The alternative -- rendering from
    0 and slicing -- is bit-exact but O(file length); for a 218 s file that is
    exactly the latency this whole step exists to avoid.

    ``offset_sec``, ``duration_sec`` and ``settle_sec`` must all be finite:
    ``float('nan')`` compares False against every bound, so the sign checks
    below cannot see it, and it would otherwise reach ``np.ceil``/``int()``
    (or worse, silently produce an all-zero fragment).
    """
    for name, value in (("offset_sec", offset_sec), ("duration_sec", duration_sec),
                        ("settle_sec", settle_sec)):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number, got {value!r}")
    if offset_sec < 0 or duration_sec <= 0 or settle_sec < 0:
        raise ValueError("offset_sec >= 0, duration_sec > 0, settle_sec >= 0 required")
    if timeline is not None and tracks is not None:
        raise ValueError(
            "pass either timeline or tracks, not both: a prebuilt event list no "
            "longer knows which track each event came from"
        )

    events = list(timeline) if timeline is not None else build_timeline(
        midi_path, channel, tracks
    )
    # scan_state and _first_event_after both rely on the stream being ordered
    # by t (the scan breaks at the first event past the offset).  A caller
    # passing a hand-built list would otherwise lose events silently, so make
    # the invariant explicit rather than trusting it.
    events = _ordered(events)

    eng = SfzEngine(sample_rate=sample_rate, block_size=block_size, freewheeling=True,
                    sample_quality=sample_quality)
    sfz = eng.load_sfz(sfz_path)
    sr, bs = eng.sample_rate, eng.block_size

    # ---- 1 + 2 + 3: panic, rebuild, stream ------------------------------- #
    streamer = BlockStreamer(eng, events)
    state = streamer.seek(offset_sec)

    blocks = int(np.ceil(duration_sec * sr / bs))
    out = np.empty((2, blocks * bs), dtype=np.float32)
    # Roughly 200 updates across the render: enough for a bar that moves every
    # frame, few enough that the callback is never the cost.
    report_every = max(1, blocks // 200)
    for b in range(blocks):
        out[:, b * bs : (b + 1) * bs] = streamer.next_block()
        if on_progress is not None and (b % report_every == 0 or b == blocks - 1):
            on_progress((b + 1) / blocks)

    n = min(int(round(duration_sec * sr)), out.shape[1])
    audio = np.ascontiguousarray(out[:, :n])

    tail = np.zeros((2, 0), dtype=np.float32)
    if settle_sec > 0:
        eng.all_sound_off()
        eng.cc(CC_SUSTAIN, 0)
        settle_blocks = int(np.ceil(settle_sec * sr / bs))
        tail_out = np.empty((2, settle_blocks * bs), dtype=np.float32)
        for b in range(settle_blocks):
            tail_out[:, b * bs : (b + 1) * bs] = eng.render_block()
        tail = np.ascontiguousarray(tail_out[:, : int(settle_sec * sr)])
    else:
        eng.all_sound_off()

    return RenderResult(
        audio=audio,
        tail=tail,
        offset_sec=offset_sec,
        duration_sec=duration_sec,
        held=state.held(),
        pedal_down=state.pedal_down,
        sfz_path=sfz,
        sample_rate=sr,
    )


def _first_event_after(events: Sequence[Event], offset_sec: float) -> int:
    lo, hi = 0, len(events)
    while lo < hi:
        mid = (lo + hi) // 2
        if events[mid].t <= offset_sec:
            lo = mid + 1
        else:
            hi = mid
    return lo


def write_wav(path: Path | str, audio: np.ndarray, sample_rate: int) -> Path:
    """Write channel-first ``(2, n)`` (or ``(1, n)``) audio as 32-bit float."""
    import soundfile as sf

    data = np.asarray(audio)
    if data.ndim != 2 or data.shape[0] not in (1, 2):
        raise ValueError(f"expected channel-first (1|2, n) audio, got {data.shape}")
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out), data.T, sample_rate, subtype="FLOAT")
    return out