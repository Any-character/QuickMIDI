"""Turn a :class:`~qmidi.segments.SegmentProject` into the two files a user
actually hands to somebody else: a **standard MIDI file** and an **MP3**.

Two very different jobs, two very different contracts
------------------------------------------------------

*MIDI export is surgical.*  The sidecar's velocity segments are not a MIDI
concept, so the only thing that may change in the output file is the attack
velocity of notes (plus the release/re-attack pairs needed to express a
velocity change mid-pitch).  Track count, track names, tempo map, time
signatures, key signatures, controllers, program changes, the channel each
piano sits on, ``ticks_per_beat`` and the file type all survive untouched: we
import the original :class:`mido.MidiFile`, replace the *message list* of each
track, and save.  Nothing is flattened onto one track, because flattening is
exactly the thing that makes a shared file worse than the original.

*MP3 export is a straight line.*  Render with :func:`qmidi.offline.render_from`,
convert to 16-bit PCM, hand the bytes to lameenc (LGPL), write the file.  It
reads a **.mid path**, not a project, on purpose: audio must be rendered from
the *exported* file so that what the user hears is what they are about to send.
:func:`export_project` sequences the two so that ordering is never a
responsibility of the caller.

Why the release/re-attack pair is safe here
-------------------------------------------
The retake rule sounds dangerous -- cutting a sounding voice to force a fresh
attack can silently truncate a held note -- so here is the argument that it
cannot, in this model:

1. Segments of a track are contiguous and every note lies wholly inside one
   segment (that is :meth:`SegmentedTrack.validate`'s invariant, guaranteed at
   construction by ``_snap_cuts``).
2. A velocity actually changed (``resolved != original``) only when the note
   sits in a segment that carries a ``delta`` or a ``target``.  So a boundary
   tick is what separates the old value from the new one.
3. Consider a ``note_on`` at boundary tick ``T`` on ``(channel, pitch)`` and
   any voice of that key still open at ``T``.  Its segment ends at or before
   ``T`` (point 1) and it is still sounding at ``T``, therefore its segment ends
   exactly at ``T`` and its real note-off is at ``T`` too.

So the message we move in front of the attack is *that note's own release*,
merely relocated inside the same tick.  Nothing is cut short, no other part is
silenced, and the file keeps whatever release convention it already used
(Unchained.mid releases with velocity-0 ``note_on``, which is preserved because
we move the original message rather than synthesising a new one).

The guard for a *malformed* project (a boundary forced through the middle of a
note, which ``validate()`` would reject but a hand-built sidecar can still
contain) is the ``end_tick <= T`` condition in :func:`_rewrite_track`: a voice
that would have to be cut *early* is left alone and reported in
:attr:`ExportResult.warnings` instead of being truncated.

Anti-clipping policy for MP3
----------------------------
sfizz's bus sums voices without limiting, so ``render_from`` can and does hand
back ``|x| > 1``.  :func:`to_pcm16` applies **one global gain**,
``min(1, peak_target / peak)``: it can only ever attenuate, never amplify, and a
single gain preserves the dynamics exactly -- no compression, no distortion, no
per-sample limiting that would smear the note attacks this whole app is about.
The default ``peak_target`` is 0.97 (~-0.26 dBFS), which leaves a sliver of
headroom for MP3 encoder overs.  Asking for ``normalize=False`` on a clipping
render is an error rather than a silent hard clip: a user who turned
normalisation off deserves to be told why the file cannot be written.

Neither import is at module scope: ``lameenc`` is imported inside
:func:`encode_mp3` (mirroring how :func:`qmidi.offline.write_wav` imports
soundfile lazily), so this module still imports on a machine that has neither
lameenc nor an instrument.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import mido
import numpy as np

from .mastering import MasterSettings, master as master_audio
from .offline import build_timeline, render_from, write_wav
from .pedal import CC_SUSTAIN, PedalSpan
from .segments import SegmentProject, clamp_velocity

__all__ = [
    "AUDIO_FORMATS",
    "AudioFormat",
    "BITRATES",
    "DEFAULT_BITRATE",
    "DEFAULT_BIT_DEPTH",
    "DEFAULT_PEAK_TARGET",
    "DEFAULT_SAMPLE_RATE",
    "SAMPLE_RATES",
    "AudioExportResult",
    "ExportError",
    "ExportResult",
    "MidiExportResult",
    "audio_format",
    "compose_export_path",
    "default_audio_output",
    "default_mp3_output",
    "default_midi_output",
    "encode_mp3",
    "export_audio",
    "export_mp3",
    "export_midi",
    "export_project",
    "to_pcm16",
    "write_lossless",
]


class ExportError(RuntimeError):
    """An export cannot be written as asked."""


#: The three bit rates offered to the user.  128 is "send it over chat",
#: 192 is the transparent-enough default, 320 is "give me the tape".
BITRATES: tuple[int, ...] = (128, 192, 320)
#: 320 is the default because the render is already done at this point: the
#: encoder is the only remaining lossy step, and the file is a keepsake, not a
#: preview.  A 7-minute piece is ~17 MB instead of ~10 MB.
DEFAULT_BITRATE = 320


@dataclass(frozen=True)
class AudioFormat:
    """One container the export dialog can write.

    ``bit_depths`` is empty for MP3: it is a lossy encoder with a bit rate, not
    a PCM container, and offering "16 bit MP3" would be a lie the user could
    then compare against a WAV and be confused by.
    """

    key: str
    extension: str
    lossless: bool
    bit_depths: tuple[int, ...]
    default_bit_depth: int
    #: Rates the encoder can actually write.  MP3 stops at 48 kHz (LAME cannot
    #: encode 96 kHz); offering it in the dialog and failing at export would be
    #: the worst of both worlds.
    sample_rates: tuple[int, ...] = (44100, 48000, 96000)


#: Ordered as the dialog shows them: the lossy chat format last, because the
#: common case is "save something that sounds exactly like what I heard".
AUDIO_FORMATS: tuple[AudioFormat, ...] = (
    AudioFormat("wav", ".wav", True, (16, 24, 32), 24),
    AudioFormat("flac", ".flac", True, (16, 24), 24),
    AudioFormat("mp3", ".mp3", False, (), 0, (44100, 48000)),
)


def audio_format(key: str) -> AudioFormat:
    """Look one format up; an unknown key is a programming error, not a dialog."""
    for fmt in AUDIO_FORMATS:
        if fmt.key == key:
            return fmt
    raise ExportError(f"unknown audio format {key!r}; expected one of "
                      f"{[f.key for f in AUDIO_FORMATS]}")


#: Offered sample rates.  48 kHz is the default because it is what the preview
#: renders at and what the device runs at: picking anything else resamples every
#: sample twice for no audible benefit (this is a piano, not a spectrum test).
SAMPLE_RATES: tuple[int, ...] = (44100, 48000, 96000)
DEFAULT_SAMPLE_RATE = 48000

#: 24 bit keeps a decaying piano tail well clear of the quantisation floor and
#: still halves against 32-bit float.  16 bit is offered for CD compatibility,
#: not because it is a better default.
DEFAULT_BIT_DEPTH = 24

#: ~-0.26 dBFS.  Just under full scale so MP3 encoder overs does not wrap.
DEFAULT_PEAK_TARGET = 0.97

#: Seconds of decay appended after the requested span so a render does not end
#: on a hard cut of the last chord's release.  Five seconds, at the user's
#: request (2026-11-01): a piano release can ring that long, and preview uses the
#: same constant so both end at the same moment.
DEFAULT_TAIL_SEC = 5.0

#: MIDI continuous controller number for stereo position (pan).
CC_PAN = 10

#: Audio block size for the export render.  512, measured on a 20 s full mix:
#: at quality 6 it runs 3.26x real time against 1.98x for the 128 the preview
#: was tuned to -- the preview's number was measured with GIL contention in the
#: loop, which does not exist here.  512 is >= 128 at every quality level.
EXPORT_BLOCK_SIZE = 512


# --------------------------------------------------------------------------- #
# MIDI export
# --------------------------------------------------------------------------- #
@dataclass
class MidiExportResult:
    """What one MIDI export did, in numbers the GUI can show."""

    path: Path
    notes_rewritten: int = 0
    retriggers: int = 0
    tracks_touched: int = 0
    #: How many tracks had their CC10 rewritten from the project's channel pans.
    pans_written: int = 0
    warnings: list[str] = field(default_factory=list)


@dataclass
class _Voice:
    """One note found while walking a track's raw message list.

    ``on_index``/``off_index`` point into that message list, which is how the
    rewrite pass can move a release in front of an attack without having to
    re-derive anything.
    """

    index: int = -1
    channel: int = 0
    pitch: int = 0
    velocity: int = 0
    start_tick: int = 0
    end_tick: int = 0
    on_index: int = -1
    off_index: int = -1


def _scan_track(messages: Sequence[mido.Message]) -> list[_Voice]:
    """Recover the notes of one raw track, indexed exactly as ``read_notes`` does.

    This mirrors :func:`qmidi.segments.read_notes` deliberately -- velocity-0
    ``note_on`` folded to a release, first-in-first-out matching of releases to
    attacks, dangling notes closed at end of track, and finally the *stable*
    sort by ``(start, end, channel, pitch)`` that decides the ``Note.index``
    every segment refers to.  Getting that last step wrong would silently apply
    the wrong adjustment to the wrong note, so the indices are re-verified
    against the project's own notes by :func:`_verify_indices`.
    """
    tick = 0
    track_end = 0
    pending: dict[tuple[int, int], list[_Voice]] = {}
    completed: list[_Voice] = []

    for i, msg in enumerate(messages):
        tick += msg.time
        track_end = max(track_end, tick)
        if msg.is_meta or msg.type not in ("note_on", "note_off"):
            continue
        key = (msg.channel, msg.note)
        if msg.type == "note_on":
            if msg.velocity > 0:
                voice = _Voice(-1, msg.channel, msg.note, msg.velocity,
                               tick, tick, i, -1)
                pending.setdefault(key, []).append(voice)
            else:
                stack = pending.get(key)
                if stack:
                    voice = stack.pop(0)
                    voice.end_tick = tick
                    voice.off_index = i
                    completed.append(voice)
        elif msg.type == "note_off":
            stack = pending.get(key)
            if stack:
                voice = stack.pop(0)
                voice.end_tick = tick
                voice.off_index = i
                completed.append(voice)

    for stack in pending.values():
        for voice in stack:
            voice.end_tick = max(track_end, voice.start_tick)
            completed.append(voice)

    # Stable sort on purpose: `read_notes` builds `raw` in *completion* order and
    # relies on sort stability to break ties between two notes that share all
    # four key fields.
    completed.sort(key=lambda v: (v.start_tick, v.end_tick, v.channel, v.pitch))
    for i, voice in enumerate(completed):
        voice.index = i
    return completed


def _verify_indices(voices: Sequence[_Voice], notes: Sequence) -> Optional[str]:
    """Cross-check the freshly scanned notes against the project's own notes.

    Returns ``None`` when they agree (or when the project carries no notes to
    compare against) and a human-readable reason when they do not.  A track
    whose note *count* differs is a mismatch too: comparing only what zipped
    together would let a truncated sidecar pass.
    """
    if not notes:
        return None
    if len(notes) != len(voices):
        return (f"the file holds {len(voices)} note(s) on this track but the "
                f"sidecar describes {len(notes)}; it was saved against a "
                f"different version of the MIDI file")
    for voice, note in zip(voices, notes):
        end = max(voice.end_tick, voice.start_tick + 1)
        if (voice.channel, voice.pitch, voice.velocity, voice.start_tick, end) != (
            note.channel, note.pitch, note.velocity, note.start_tick, note.end_tick
        ):
            return (
                f"scanned note {voice.index} "
                f"(ch{voice.channel} pitch{voice.pitch} v{voice.velocity} "
                f"[{voice.start_tick},{end})) does not match the project's note "
                f"(ch{note.channel} pitch{note.pitch} v{note.velocity} "
                f"[{note.start_tick},{note.end_tick})); the sidecar was built "
                f"against a different version of the MIDI file"
            )
    return None


def _release_message(messages: Sequence[mido.Message], voice: _Voice) -> mido.Message:
    """The release event for ``voice``, reusing the file's own message."""
    if 0 <= voice.off_index < len(messages):
        return messages[voice.off_index]
    # No release in the file (a note left ringing at end of track).  Synthesise
    # one in the convention `read_notes`/`build_timeline` both assume: a
    # velocity-0 note_on *is* a note-off.
    return mido.Message("note_on", channel=voice.channel, note=voice.pitch,
                        velocity=0)


def _pedal_channel(messages: Sequence[mido.Message]) -> int:
    """Which channel the track's pedal belongs on.

    Prefer the channel the file already used for CC64 (that is the pedal's own
    channel even when the track holds several), then any note's channel, then 0.
    """
    for msg in messages:
        if msg.type == "control_change" and msg.control == CC_SUSTAIN and not msg.is_meta:
            return msg.channel
    for msg in messages:
        if msg.type in ("note_on", "note_off"):
            return msg.channel
    return 0


def _track_channel(messages: Sequence[mido.Message]) -> int:
    """First channel a track actually plays on (pan has to address a channel)."""
    for msg in messages:
        if not msg.is_meta and msg.type in ("note_on", "note_off"):
            return int(msg.channel)
    return 0


def _rewrite_track(
    messages: Sequence[mido.Message],
    voices: Sequence[_Voice],
    resolved: dict[int, int],
    *,
    result: MidiExportResult,
    track_no: int,
    pedal: Optional[Sequence[PedalSpan]] = None,
    pan: Optional[int] = None,
) -> list[mido.Message]:
    """One track's message list with velocities applied and retakes inserted.

    ``out`` collects ``(absolute_tick, rank, message)`` triples and only at the
    very end are the delta times recomputed, which is what lets a release be
    moved inside a tick without any arithmetic on deltas.

    ``rank`` decides the order of messages that share a tick, and it exists for
    one reason: a pedal-down must be written *before* a note-off at the same
    tick, or the note is released instead of being caught by the pedal (the
    exact moment a pedal change is supposed to hold the note).  A pedal-up goes
    last for symmetry.  Notes keep rank 1 and therefore their insertion order,
    which is what the retake relocation depends on.

    ``pedal`` is ``None`` for "leave this track's CC64 alone".  A list (even an
    empty one) replaces every CC64 in the track with the spans -- see
    :mod:`qmidi.pedal` for why appending cannot work.
    """
    open_by_key: dict[tuple[int, int], list[_Voice]] = {}
    #: message indices already written in front of a retake; they must not be
    #: emitted again, and -- just as importantly -- must not pop the *new*
    #: attack off ``open_by_key`` when the walk finally reaches them.
    suppressed: set[int] = set()
    voice_of_on = {v.on_index: v for v in voices}
    out: list[tuple[int, int, mido.Message]] = []
    tick = 0
    changed = 0
    replacing_pedal = pedal is not None
    replacing_pan = pan is not None
    rank_note = 1

    for i, msg in enumerate(messages):
        tick += msg.time
        if msg.is_meta:
            out.append((tick, rank_note, msg))
            continue
        if msg.type not in ("note_on", "note_off"):
            if msg.type == "control_change":
                if replacing_pedal and msg.control == CC_SUSTAIN:
                    continue  # superseded by the edited spans
                if replacing_pan and msg.control == CC_PAN:
                    continue  # superseded by the edited position
            out.append((tick, rank_note, msg))
            continue
        key = (msg.channel, msg.note)
        is_release = (msg.type == "note_off" or msg.velocity == 0)

        if is_release:
            if i in suppressed:
                continue  # already relocated; leave open_by_key alone
            stack = open_by_key.get(key)
            if stack:
                open_by_key[key] = stack[1:]
            out.append((tick, rank_note, msg))
            continue

        # An attack.  Find the voice this very message started.
        voice = voice_of_on.get(i)
        if voice is None:
            out.append((tick, rank_note, msg))
            continue

        new_velocity = clamp_velocity(resolved.get(voice.index, voice.velocity))
        if new_velocity != voice.velocity:
            result.notes_rewritten += 1
            changed += 1

            # Retake: release every voice of this key whose *own* release is at
            # or before this tick, then attack.  See the module docstring for
            # why that is a relocation and not a cut.
            for held in list(open_by_key.get(key, ())):
                if held.end_tick > tick:
                    result.warnings.append(
                        f"track {track_no}: pitch {voice.pitch} keeps the note "
                        f"[{held.start_tick},{held.end_tick}) sounding across a "
                        f"velocity change at tick {tick}; left uncut"
                    )
                    continue
                if held.off_index >= 0:
                    suppressed.add(held.off_index)
                out.append((tick, rank_note, _release_message(messages, held)))
                result.retriggers += 1
            open_by_key[key] = []

        open_by_key.setdefault(key, []).append(voice)
        if new_velocity != msg.velocity:
            msg = msg.copy(velocity=new_velocity)
        out.append((tick, rank_note, msg))

    if pan is not None:
        # Tick 0, so the position is in force before the first note.  Written
        # once per *channel* (see export_midi), never per track, or the two
        # tracks of one piano would each fight over the same controller.
        out.append((0, 0, mido.Message(
            "control_change", channel=_track_channel(messages), control=CC_PAN,
            value=int(pan), time=0)))

    if pedal is not None:
        channel = _pedal_channel(messages)
        for span in pedal:
            out.append((span.start_tick, 0, mido.Message(
                "control_change", channel=channel, control=CC_SUSTAIN,
                value=span.value, time=0)))
            out.append((span.end_tick, 2, mido.Message(
                "control_change", channel=channel, control=CC_SUSTAIN,
                value=0, time=0)))

    if changed:
        result.tracks_touched += 1

    # Sorting on (tick, rank) is stable, and insertion order *is* the intended
    # order inside the same tick: a relocated release was appended before the
    # attack it belongs to, so it stays there.
    new_messages: list[mido.Message] = []
    previous = 0
    for at, _rank, msg in sorted(out, key=lambda item: (item[0], item[1])):
        new_messages.append(msg.copy(time=at - previous))
        previous = at
    return new_messages


def default_midi_output(project: SegmentProject) -> Path:
    """``song.mid`` -> ``song_qmidi.mid``, next to the source.

    Never returns the source path itself: exporting must not be able to eat the
    user's original file by accident.
    """
    source = Path(project.source)
    return source.with_name(f"{source.stem}_qmidi{source.suffix or '.mid'}")


def _apply_project_bpm(midi: mido.MidiFile, bpm: Optional[float]) -> None:
    """Make tick 0 play at ``bpm`` by scaling the file's whole tempo map.

    Scaling rather than replacing keeps whatever the file did musically: a
    ritardando stays a ritardando, only faster.  ``None`` leaves the file
    untouched -- a sidecar without a tempo must not change the piece.
    """
    if bpm is None or bpm <= 0:
        return
    first: Optional[mido.MetaMessage] = None
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                first = message
                break
        if first is not None:
            break

    if first is None:
        # No tempo event at all: the file means MIDI's default 120 BPM.  Write
        # the requested tempo explicitly, or the export would ignore the setting.
        midi.tracks[0].insert(0, mido.MetaMessage(
            "set_tempo", tempo=max(1, int(round(60_000_000.0 / bpm))), time=0))
        return

    current_bpm = 60_000_000.0 / first.tempo
    if abs(current_bpm - bpm) < 1e-6:
        return
    factor = current_bpm / bpm
    for track in midi.tracks:
        for message in track:
            if message.type == "set_tempo":
                message.tempo = max(1, int(round(message.tempo * factor)))


def export_midi(
    project: SegmentProject,
    out_path: Path | str | None = None,
    *,
    verify: bool = True,
) -> MidiExportResult:
    """Write ``project``'s velocities into a copy of the original MIDI file.

    Everything except note velocities is preserved: the file is imported with
    mido and each track's *message list* is swapped, so track count, track
    names, tempo, time/key signatures, controllers, program changes and
    ``ticks_per_beat`` are byte-for-byte the original ones.

    ``verify`` re-scans each track and cross-checks the note list against the
    project's, which catches a sidecar that was saved against a different
    revision of the MIDI instead of quietly exporting the wrong velocities.
    """
    source = Path(project.source)
    if not source.is_file():
        raise ExportError(f"source MIDI not found: {source}")
    midi = mido.MidiFile(str(source))
    _apply_project_bpm(midi, project.bpm)
    if len(midi.tracks) != len(project.tracks):
        raise ExportError(
            f"{source.name} has {len(midi.tracks)} tracks but the project "
            f"describes {len(project.tracks)}"
        )

    result = MidiExportResult(path=Path(out_path) if out_path is not None
                              else default_midi_output(project))

    # Pan is a channel property, the sidecar is keyed by track.  Resolve it once,
    # here, so exactly one track per channel receives the CC10 -- writing it on
    # both tracks of a two-track piano would be harmless in value but would make
    # the file carry a duplicate controller nobody can explain later.
    pan_tracks: dict[int, int] = {}
    for channel, pan in sorted((project.pans or {}).items()):
        for index, track in enumerate(midi.tracks):
            messages = list(track)
            if _track_channel(messages) != channel:
                continue
            if not any(m.type == "note_on" and m.velocity > 0 for m in messages):
                continue
            pan_tracks[index] = int(pan)
            break
        else:
            result.warnings.append(
                f"channel {channel} was panned to {pan} but no track in "
                f"{source.name} plays on that channel; the pan was not written"
            )

    for seg_track in sorted(project.tracks, key=lambda t: t.track):
        if not 0 <= seg_track.track < len(midi.tracks):
            raise ExportError(f"track {seg_track.track} is not in {source.name}")
        messages = list(midi.tracks[seg_track.track])
        voices = _scan_track(messages)
        if verify:
            problem = _verify_indices(voices, seg_track.notes)
            if problem:
                raise ExportError(problem)
        resolved = {n.index: n.velocity for n in seg_track.resolved_notes()}
        missing = len(voices) - len(resolved)
        if missing:
            result.warnings.append(
                f"track {seg_track.track}: {missing} note(s) belong to no segment "
                f"and keep their original velocity"
            )
        new_messages = _rewrite_track(messages, voices, resolved,
                                      result=result, track_no=seg_track.track,
                                      pedal=project.pedals.get(seg_track.track),
                                      pan=pan_tracks.get(seg_track.track))
        if seg_track.track in pan_tracks:
            result.pans_written += 1
        midi.tracks[seg_track.track] = new_messages

    # A track the sidecar does not describe (MuseScore sometimes writes a
    # note-less setup track) still carries a channel the user panned.
    described = {t.track for t in project.tracks}
    for track_no, pan in pan_tracks.items():
        if track_no in described:
            continue
        messages = list(midi.tracks[track_no])
        midi.tracks[track_no] = _rewrite_track(
            messages, _scan_track(messages), {}, result=result,
            track_no=track_no, pan=pan)
        result.pans_written += 1

    result.path.parent.mkdir(parents=True, exist_ok=True)
    midi.save(str(result.path))
    return result


# --------------------------------------------------------------------------- #
# PCM / MP3
# --------------------------------------------------------------------------- #
@dataclass
class AudioExportResult:
    """What one audio export produced."""

    path: Path
    wav_path: Optional[Path]
    bitrate: int
    sample_rate: int
    channels: int
    frames: int
    offset_sec: float
    duration_sec: float
    peak_before: float
    peak_after: float
    gain: float
    #: Loudness before/after the mastering chain (``-inf`` when it ran not at all)
    #: and the gain the loudness stage applied.  Printed by the status bar so a
    #: user can see that "-14 LUFS" really happened.
    lufs_before: float = float("-inf")
    lufs_after: float = float("-inf")
    loudness_gain_db: float = 0.0
    sfz_path: Optional[Path] = None
    #: Container key ("mp3" / "flac" / "wav") and the PCM depth actually written
    #: (0 for MP3, which has no depth).  The status bar prints these, so a user
    #: who picked 24-bit FLAC can see that 24-bit FLAC is what they got.
    format: str = "mp3"
    bit_depth: int = 16


def _apply_gain(
    audio: np.ndarray,
    *,
    peak_target: float = DEFAULT_PEAK_TARGET,
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Validate the render, apply one global attenuation, clip to [-1, 1].

    Anti-clipping is **one global attenuation**, ``min(1, peak_target/peak)``:
    it never amplifies, it never compresses, and it cannot smear an attack the
    way a per-sample limiter would.  See the module docstring.

    Returns ``(scaled (channels, n) float, gain)``.
    """
    data = np.asarray(audio)
    if data.ndim == 1:
        data = data[np.newaxis, :]
    if data.ndim != 2 or data.shape[0] not in (1, 2):
        raise ExportError(f"expected channel-first (1|2, n) audio, got {data.shape}")
    if not np.all(np.isfinite(data)):
        raise ExportError("render contains NaN or inf samples; refusing to encode")

    peak = float(np.max(np.abs(data))) if data.size else 0.0
    gain = 1.0
    if peak > 0.0 and peak > peak_target:
        if not normalize:
            raise ExportError(
                f"render peaks at {peak:.3f} (> {peak_target:.3f}) and "
                f"normalize=False; enable normalisation or render quieter"
            )
        gain = peak_target / peak
    scaled = data * gain
    # Unreachable by construction (gain <= peak_target/peak); kept as the last
    # line of defence against an integer-rounding surprise.
    np.clip(scaled, -1.0, 1.0, out=scaled)
    return scaled, gain


def to_pcm16(
    audio: np.ndarray,
    *,
    peak_target: float = DEFAULT_PEAK_TARGET,
    normalize: bool = True,
) -> tuple[np.ndarray, float]:
    """Channel-first float audio -> interleaved int16, plus the gain applied.

    Returns ``(int16 (n, channels), gain)``.  MP3 needs this; the lossless
    containers take the float signal and let libsndfile do the quantisation
    (see :func:`write_lossless`), because hand-rolling a 24-bit int would make
    the rounding policy ours instead of the reference implementation's.
    """
    scaled, gain = _apply_gain(audio, peak_target=peak_target, normalize=normalize)
    return np.round(scaled.T * 32767.0).astype("<i2"), gain


#: Bit depth -> libsndfile subtype.  32 is float, not int32: "32 bit WAV" is
#: overwhelmingly float in audio work, and int32 has no dynamic-range advantage
#: over float for a render that is already normalised.
_SUBTYPES = {16: "PCM_16", 24: "PCM_24", 32: "FLOAT"}


def write_lossless(
    path: Path | str,
    audio: np.ndarray,
    sample_rate: int,
    *,
    fmt: str,
    bit_depth: int = DEFAULT_BIT_DEPTH,
) -> Path:
    """Write channel-first float audio as WAV or FLAC at ``bit_depth``.

    The signal must already be scaled and clipped (:func:`_apply_gain`);
    libsndfile quantises it, and it clips silently, so the caller keeping that
    honest is what stops a too-hot render from wrapping.
    """
    import soundfile as sf  # lazy: the module must import without soundfile

    info = audio_format(fmt)
    if not info.lossless:
        raise ExportError(f"{fmt} is not a lossless container")
    if bit_depth not in info.bit_depths:
        raise ExportError(
            f"{fmt} supports {info.bit_depths} bit, got {bit_depth}")
    data = np.asarray(audio)
    if data.ndim != 2 or data.shape[0] not in (1, 2):
        raise ExportError(f"expected channel-first (1|2, n) audio, got {data.shape}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(target), data.T, int(sample_rate),
             format=fmt.upper(), subtype=_SUBTYPES[bit_depth])
    return target


def encode_mp3(pcm: np.ndarray, sample_rate: int, bitrate: int = DEFAULT_BITRATE) -> bytes:
    """Interleaved int16 PCM -> MP3 bytes, via lameenc (LGPL)."""
    if int(bitrate) not in BITRATES:
        raise ExportError(f"bitrate must be one of {BITRATES}, got {bitrate}")
    if sample_rate <= 0:
        raise ExportError(f"sample_rate must be positive, got {sample_rate}")
    data = np.asarray(pcm)
    if data.ndim != 2 or data.shape[1] not in (1, 2):
        raise ExportError(f"expected interleaved (n, 1|2) PCM, got {data.shape}")

    try:
        import lameenc
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise ExportError(
            "lameenc is not installed; MP3 export needs it (LGPL, pip install lameenc)"
        ) from exc

    channels = int(data.shape[1])
    encoder = lameenc.Encoder()
    encoder.set_bit_rate(int(bitrate))
    encoder.set_in_sample_rate(int(sample_rate))
    encoder.set_channels(channels)
    # 2 == "high quality, no psychoacoustic weirdness"; the default 7 makes a
    # 320 kbps file sound worse than a 192 kbps one, which users notice.
    encoder.set_quality(2)
    return bytes(encoder.encode(data.astype("<i2", copy=False).tobytes())
                 + encoder.flush())


def pan_gains(pan: int) -> tuple[float, float]:
    """Constant-power gains for a CC10 pan position (0..127, 64 = centre).

    Constant power rather than linear: a piano panned hard left then keeps the
    same perceived loudness as one in the middle, instead of dropping 3 dB and
    making the user re-balance the velocities they just finished editing.
    """
    position = max(0, min(127, int(pan))) / 127.0
    theta = position * (math.pi / 2.0)
    return math.cos(theta), math.sin(theta)


def note_channels(midi_path: Path | str) -> set[int]:
    """Every MIDI channel that actually plays a note somewhere in the file."""
    midi = mido.MidiFile(str(midi_path))
    out: set[int] = set()
    for track in midi.tracks:
        for msg in track:
            if not msg.is_meta and msg.type == "note_on" and msg.velocity > 0:
                out.add(int(msg.channel))
    return out


def _render_with_pans(
    midi_path: Path,
    offset_sec: float,
    span: float,
    pans: Optional[dict[int, int]],
    *,
    sfz_path,
    sample_rate: int,
    settle_sec: float,
    block_size: int,
    on_progress: Optional[Callable[[float], None]] = None,
    sample_quality: Optional[int] = None,
) -> tuple[np.ndarray, Path, int]:
    """Render the whole mix, placing each panned channel with its own gains.

    sfizz has a single stereo bus and no per-channel pan, so a channel can only
    be positioned by rendering it **alone** and panning the result.  That costs
    one engine start per panned channel (two for a two-piano piece) and is the
    only way the exported MP3 can actually contain the separation the MIDI file
    describes.  Unpanned channels are rendered together in one extra pass so a
    single-piano file still takes exactly as long as before.
    """
    channels = note_channels(midi_path)
    wanted = {int(ch): int(pan) for ch, pan in (pans or {}).items() if int(ch) in channels}
    wanted = {ch: pan for ch, pan in wanted.items() if pan != 64}

    if not wanted:
        render = render_from(midi_path, offset_sec, span, sfz_path=sfz_path,
                             sample_rate=sample_rate, settle_sec=settle_sec,
                             block_size=block_size, on_progress=on_progress,
                             sample_quality=sample_quality)
        return render.audio, render.sfz_path, render.sample_rate

    blocks: list[np.ndarray] = []
    sfz_resolved: Optional[Path] = None
    rate = sample_rate

    rest = sorted(channels - set(wanted))
    # One render pass per panned channel, plus one for everything unpanned: the
    # bar has to be told how many passes there are or it would reach 100% while
    # the second piano is still rendering.
    passes = len(wanted) + (1 if rest else 0)
    finished = [0]

    def part_progress(fraction: float) -> None:
        if on_progress is not None and passes:
            on_progress((finished[0] + fraction) / passes)

    for channel, pan in sorted(wanted.items()):
        timeline = build_timeline(midi_path, channel=channel)
        if not timeline:
            continue
        part = render_from(midi_path, offset_sec, span, timeline=timeline,
                           sfz_path=sfz_path, sample_rate=sample_rate,
                           settle_sec=settle_sec, block_size=block_size,
                           on_progress=part_progress,
                           sample_quality=sample_quality)
        left, right = pan_gains(pan)
        placed = part.audio.astype(np.float32, copy=True)
        placed[0] *= left
        placed[1] *= right
        blocks.append(placed)
        sfz_resolved, rate = part.sfz_path, part.sample_rate
        finished[0] += 1

    if rest:
        timeline = []
        for channel in rest:
            timeline.extend(build_timeline(midi_path, channel=channel))
        if timeline:
            part = render_from(midi_path, offset_sec, span, timeline=timeline,
                               sfz_path=sfz_path, sample_rate=sample_rate,
                               settle_sec=settle_sec, block_size=block_size,
                               on_progress=part_progress,
                               sample_quality=sample_quality)
            blocks.append(part.audio)
            sfz_resolved, rate = part.sfz_path, part.sample_rate

    if not blocks:
        raise ExportError(f"{Path(midi_path).name} has no playable notes to render")
    total = blocks[0].copy()
    for block in blocks[1:]:
        total += block
    if sfz_resolved is None:  # pragma: no cover - blocks non-empty implies a render
        raise ExportError("render produced audio without an instrument path")
    return total, sfz_resolved, rate


def default_audio_output(midi_path: Path | str, fmt: str = "mp3",
                         suffix: str = "_qmidi") -> Path:
    """``song.mid`` -> ``song_qmidi.<ext of fmt>`` beside it."""
    p = Path(midi_path)
    return p.with_name(f"{p.stem}{suffix}{audio_format(fmt).extension}")


#: Suffixes a user may paste into the dialog's name field, longest first so a
#: full ``.midi`` is not cut down to ``.mid`` plus a stray ``i``.
_NAME_EXTENSIONS = (".midi", ".mid", ".wav", ".flac", ".mp3")


def compose_export_path(directory: Path | str, name: str, extension: str) -> Path:
    """``directory`` + ``name`` (suffix stripped) + ``extension``.

    The dialog asks for a bare name, but people paste whole file names; typing
    ``song.mid`` and getting ``song.mid.wav`` is the paper cut that makes an
    export screen feel unfinished.  An empty name becomes ``untitled`` so the
    path is always writable.
    """
    stem = str(name).strip() or "untitled"
    lowered = stem.lower()
    for suffix in _NAME_EXTENSIONS:
        if lowered.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    return Path(directory) / f"{stem}{extension}"


def default_mp3_output(midi_path: Path | str, suffix: str = "_qmidi") -> Path:
    """``song.mid`` -> ``song_qmidi.mp3`` beside it."""
    return default_audio_output(midi_path, "mp3", suffix)


def _resolve_duration(midi_path: Path | str, duration_sec: Optional[float],
                      tail_sec: float) -> float:
    """Requested span, or the whole file plus a decay tail."""
    if duration_sec is not None:
        if not math.isfinite(duration_sec) or duration_sec <= 0:
            raise ExportError(f"duration_sec must be a positive finite number, "
                              f"got {duration_sec!r}")
        return float(duration_sec)
    length = float(mido.MidiFile(str(midi_path)).length)
    return max(length + max(0.0, tail_sec), tail_sec or 1.0)


def export_audio(
    midi_path: Path | str,
    out_path: Path | str | None = None,
    *,
    fmt: str = "mp3",
    bitrate: int = DEFAULT_BITRATE,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    bit_depth: Optional[int] = None,
    offset_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    tail_sec: float = DEFAULT_TAIL_SEC,
    sfz_path: Path | str | None = None,
    keep_wav: bool = False,
    wav_path: Path | str | None = None,
    peak_target: float = DEFAULT_PEAK_TARGET,
    normalize: bool = True,
    settle_sec: float = 0.0,
    pans: Optional[dict[int, int]] = None,
    master: Optional[MasterSettings] = None,
    block_size: int = EXPORT_BLOCK_SIZE,
    on_progress: Optional[Callable[[float], None]] = None,
    sample_quality: Optional[int] = None,
) -> AudioExportResult:
    """Render a span of a MIDI file and write it in ``fmt`` (MP3/WAV/FLAC).

    ``bit_depth=None`` means the container's own default (24 for WAV and FLAC;
    MP3 has no depth).  A depth or rate the container cannot store is refused
    rather than silently substituted -- 32-bit FLAC does not exist, and a user
    who chose 96 kHz should not receive a 48 kHz file with a matching name.

    ``duration_sec=None`` renders the whole file plus ``tail_sec`` so the last
    chord's release is not chopped off.  ``keep_wav`` additionally writes the
    float WAV intermediate next to the output (through
    :func:`qmidi.offline.write_wav`, so it is lossless) for auditioning.
    """
    midi_path = Path(midi_path)
    if not midi_path.is_file():
        raise ExportError(f"MIDI not found: {midi_path}")
    info = audio_format(fmt)
    if int(sample_rate) not in info.sample_rates:
        raise ExportError(
            f"{fmt} cannot write {sample_rate} Hz; choose one of {info.sample_rates}")
    if info.lossless:
        depth = info.default_bit_depth if bit_depth is None else int(bit_depth)
        if depth not in info.bit_depths:
            raise ExportError(
                f"{fmt} supports {info.bit_depths} bit, got {bit_depth}")
    else:
        if int(bitrate) not in BITRATES:
            raise ExportError(f"bitrate must be one of {BITRATES}, got {bitrate}")
        depth = 0
    if not 0 < float(peak_target) <= 1.0:
        raise ExportError(f"peak_target must be in (0, 1], got {peak_target}")

    def emit(fraction: float) -> None:
        if on_progress is not None:
            on_progress(min(1.0, max(0.0, float(fraction))))

    span = _resolve_duration(midi_path, duration_sec, tail_sec)
    # Rendering is ~85% of the wall time; mastering, conversion and the MP3
    # encoder share the rest.  The split only has to be close enough that the
    # bar does not stall at the end.
    audio, sfz_resolved, rate = _render_with_pans(
        midi_path, offset_sec, span, pans,
        sfz_path=sfz_path, sample_rate=sample_rate, settle_sec=settle_sec,
        block_size=block_size,
        on_progress=lambda fraction: emit(fraction * 0.85),
        sample_quality=sample_quality,
    )
    emit(0.86)
    peak_before = float(np.max(np.abs(audio))) if audio.size else 0.0

    report = None
    if master is not None and master.enabled:
        audio, report = master_audio(audio, rate, master)
        emit(0.90)

    scaled, gain = _apply_gain(audio, peak_target=peak_target, normalize=normalize)
    emit(0.93)

    target = (Path(out_path) if out_path is not None
              else default_audio_output(midi_path, fmt))
    if info.lossless:
        # libsndfile quantises the float signal itself; handing it pre-rounded
        # integers would make this module, not the reference decoder, decide
        # how 24-bit rounding behaves.
        write_lossless(target, scaled, rate, fmt=fmt, bit_depth=depth)
        channels, frames = int(scaled.shape[0]), int(scaled.shape[1])
    else:
        pcm = np.round(scaled.T * 32767.0).astype("<i2")
        blob = encode_mp3(pcm, rate, bitrate)
        if not blob:
            raise ExportError("lameenc produced no data")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        channels, frames = int(pcm.shape[1]), int(pcm.shape[0])
    emit(1.0)

    written_wav: Optional[Path] = None
    if keep_wav and not info.lossless:
        written_wav = Path(wav_path) if wav_path is not None else target.with_suffix(".wav")
        write_wav(written_wav, audio * gain, rate)

    return AudioExportResult(
        path=target,
        wav_path=written_wav,
        bitrate=int(bitrate) if not info.lossless else 0,
        sample_rate=int(rate),
        channels=channels,
        frames=frames,
        offset_sec=float(offset_sec),
        duration_sec=span,
        peak_before=peak_before,
        peak_after=float(np.max(np.abs(scaled))) if scaled.size else 0.0,
        gain=gain,
        lufs_before=report.lufs_before if report is not None else float("-inf"),
        lufs_after=report.lufs_after if report is not None else float("-inf"),
        loudness_gain_db=report.gain_db if report is not None else 0.0,
        sfz_path=sfz_resolved,
        format=fmt,
        bit_depth=depth,
    )


def export_mp3(
    midi_path: Path | str,
    out_path: Path | str | None = None,
    *,
    bitrate: int = DEFAULT_BITRATE,
    offset_sec: float = 0.0,
    duration_sec: Optional[float] = None,
    tail_sec: float = DEFAULT_TAIL_SEC,
    sfz_path: Path | str | None = None,
    sample_rate: int = DEFAULT_SAMPLE_RATE,
    keep_wav: bool = False,
    wav_path: Path | str | None = None,
    peak_target: float = DEFAULT_PEAK_TARGET,
    normalize: bool = True,
    settle_sec: float = 0.0,
    pans: Optional[dict[int, int]] = None,
    master: Optional[MasterSettings] = None,
    block_size: int = EXPORT_BLOCK_SIZE,
    on_progress: Optional[Callable[[float], None]] = None,
    sample_quality: Optional[int] = None,
) -> AudioExportResult:
    """MP3 convenience wrapper -- see :func:`export_audio`."""
    return export_audio(
        midi_path, out_path, fmt="mp3", bitrate=bitrate, sample_rate=sample_rate,
        offset_sec=offset_sec, duration_sec=duration_sec, tail_sec=tail_sec,
        sfz_path=sfz_path, keep_wav=keep_wav, wav_path=wav_path,
        peak_target=peak_target, normalize=normalize, settle_sec=settle_sec,
        pans=pans, master=master, block_size=block_size,
        on_progress=on_progress, sample_quality=sample_quality,
    )


# --------------------------------------------------------------------------- #
# both
# --------------------------------------------------------------------------- #
@dataclass
class ExportResult:
    """MIDI first, then audio **from that MIDI** -- never from the source."""

    midi: MidiExportResult
    audio: Optional[AudioExportResult] = None


def export_project(
    project: SegmentProject,
    midi_path: Path | str | None = None,
    mp3_path: Path | str | None = None,
    *,
    verify: bool = True,
    **audio_kwargs,
) -> ExportResult:
    """Export the project to MIDI and (optionally) to an MP3 of *that* MIDI.

    The ordering is the whole point: rendering the source file would produce an
    MP3 that does not contain a single one of the user's edits.
    """
    midi_result = export_midi(project, midi_path, verify=verify)
    audio_result = None
    if mp3_path is not None:
        audio_result = export_mp3(midi_result.path, mp3_path, **audio_kwargs)
    return ExportResult(midi=midi_result, audio=audio_result)
