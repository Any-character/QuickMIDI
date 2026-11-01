"""End-to-end export demo: the two files a user actually hands to somebody.

Run::

    python scripts/demo_export.py

Does the whole chain on ``example/midi/Unchained.mid``:

1. auto-segment every track into a :class:`~qmidi.segments.SegmentProject`;
2. apply real adjustments -- every segment of track 0 down 10, and one segment
   of track 2 pinned to velocity 64;
3. export a **standard MIDI file** to ``tmp/``, preserving the 4-track skeleton,
   and verify by re-reading it that every velocity equals ``resolved_notes()``;
4. render 10 s from that exported file and encode a **192 kbps MP3** to
   ``tmp/`` (plus the float WAV intermediate), verifying the decoded audio's
   length and peak.

Everything lands under ``tmp/``, which is git-ignored.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import mido  # noqa: E402
import numpy as np  # noqa: E402

from qmidi.exporters import export_project  # noqa: E402
from qmidi.segments import (  # noqa: E402
    SegmentProject,
    build_tracks,
    read_notes,
)

MIDI = ROOT / "example" / "midi" / "Unchained.mid"
if not MIDI.is_file():
    sys.exit(
        "example/midi/Unchained.mid not found -- the example folder is not part "
        "of the public distribution; edit MIDI in this script to point at your "
        "own .mid file."
    )
OUT_DIR = ROOT / "tmp"
TPB = 480
SNIPPET_SEC = 10.0
BITRATE = 192
#: Chosen because this is one of the loud spots of the file: the sfz bus sums
#: voices without limiting, so a render here peaks above 1.0 and the
#: anti-clipping gain in the MP3 export actually has something to do.
SNIPPET_OFFSET = 130.0


def _kb(path: Path) -> str:
    return f"{path.stat().st_size / 1024:,.1f} KB"


def _stacked_pitch_pairs(notes_by_track) -> int:
    """How many same-``(channel, pitch)`` pairs in a file touch or overlap.

    This is what decides whether a retake can happen at all: a retake only
    fires when a note's release and the next note's attack land on the same
    pitch in the same tick.  Unchained.mid has none, and saying so in the demo
    output is more useful than printing a bare ``retakes: 0``.
    """
    from collections import defaultdict

    total = 0
    for notes in notes_by_track:
        spans = defaultdict(list)
        for n in notes:
            spans[(n.channel, n.pitch)].append((n.start_tick, n.end_tick))
        for sequence in spans.values():
            sequence.sort()
            total += sum(1 for a, b in zip(sequence, sequence[1:])
                         if b[0] <= a[1])
    return total


def main(argv: list[str]) -> int:
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except AttributeError:  # pragma: no cover - very old interpreters
        pass

    # ---- 1. project ------------------------------------------------------ #
    tracks = build_tracks(MIDI, ticks_per_beat=TPB)
    project = SegmentProject(source=MIDI, ticks_per_beat=TPB, tracks=tracks)
    for t in project.tracks:
        t.validate()
    print(f"source : {MIDI.name}  {TPB} ticks/beat")
    print(f"segments: {project.segment_count} over {len(project.tracks)} tracks "
          f"({', '.join(f't{t.track}:{len(t.segments)}' for t in project.tracks)})")

    # ---- 2. adjustments -------------------------------------------------- #
    track0 = project.track(0)
    for i in range(len(track0.segments)):
        track0.apply_delta(i, -10)
    track2 = project.track(2)
    pinned = len(track2.segments) // 2
    track2.set_value(pinned, 64)
    nudged = track0.segments[0]
    print(f"\nadjustments:")
    print(f"  track 0: all {len(track0.segments)} segments delta=-10")
    print(f"          e.g. seg 0 [{nudged.start_tick},{nudged.end_tick}) "
          f"{len(nudged.note_indices)} notes")
    pinned_seg = track2.segments[pinned]
    print(f"  track 2: segment {pinned} [{pinned_seg.start_tick},"
          f"{pinned_seg.end_tick}) target=64  ({len(pinned_seg.note_indices)} notes)")

    # ---- 3 + 4. export, MP3 rendered from the *exported* MIDI ------------ #
    midi_out = OUT_DIR / "Unchained_qmidi_demo.mid"
    mp3_out = OUT_DIR / "Unchained_snippet_demo.mp3"
    bundle = export_project(
        project, midi_out, mp3_out,
        bitrate=BITRATE, offset_sec=SNIPPET_OFFSET, duration_sec=SNIPPET_SEC,
        keep_wav=True, sample_rate=48000,
    )
    midi_result, audio_result = bundle.midi, bundle.audio

    print(f"\nMIDI export: {midi_result.path}")
    print(f"  {_kb(midi_result.path)}   notes rewritten: {midi_result.notes_rewritten}"
          f"   retakes: {midi_result.retriggers}"
          f"   tracks touched: {midi_result.tracks_touched}")
    if midi_result.warnings:
        for w in midi_result.warnings[:5]:
            print(f"  ! {w}")
    stacked = _stacked_pitch_pairs(read_notes(MIDI))
    print(f"  retake note: {stacked} same-pitch note pair(s) abut or overlap here "
          f"(the only places a retake can happen); "
          f"{midi_result.retriggers} release(s) had to move")

    # ---- verification: mido re-reads the export -------------------------- #
    before, after = mido.MidiFile(str(MIDI)), mido.MidiFile(str(midi_result.path))
    reread = read_notes(midi_result.path)
    mismatched = []
    for index, track in enumerate(project.tracks):
        expected = {n.index: n.velocity for n in track.resolved_notes()}
        actual = {n.index: n.velocity for n in reread[index]}
        if expected != actual:
            mismatched.append(index)
        span_ok = ([(n.pitch, n.start_tick, n.end_tick) for n in reread[index]]
                   == [(n.pitch, n.start_tick, n.end_tick) for n in track.notes])
        if not span_ok:
            mismatched.append(f"{index}:spans")

    def names(midi):
        return [m.name for t in midi.tracks for m in t
                if m.is_meta and m.type == "track_name"]

    print(f"  mido re-read : {len(after.tracks)} tracks (was {len(before.tracks)}), "
          f"ticks_per_beat {after.ticks_per_beat}, type {after.type}")
    print(f"  track names  : {names(after)}")
    print(f"  velocity == resolved_notes() on every track: {not mismatched}")
    lo = min(n.velocity for t in reread for n in t)
    hi = max(n.velocity for t in reread for n in t)
    print(f"  velocity range: {lo}..{hi} (legal 1..127: {1 <= lo and hi <= 127})")

    # ---- verification: audio --------------------------------------------- #
    print(f"\nMP3 export : {audio_result.path}")
    print(f"  {_kb(audio_result.path)}   {audio_result.bitrate} kbps, "
          f"{audio_result.sample_rate} Hz, {audio_result.channels} ch, "
          f"{audio_result.frames} frames")
    print(f"  span      : offset {audio_result.offset_sec:g}s + "
          f"{audio_result.duration_sec:g}s")
    print(f"  clipping  : render peak {audio_result.peak_before:.3f} -> "
          f"{audio_result.peak_after:.3f} (gain {audio_result.gain:.4f})")

    import soundfile as sf

    wav = audio_result.wav_path
    data, rate = sf.read(str(wav), always_2d=True)
    blob = mp3_out.read_bytes()
    valid = blob[:2] == b"\xff\xfb"
    print(f"  WAV mid   : {wav}")
    print(f"  {_kb(wav)}   soundfile reads {data.shape[0]/rate:.3f}s at {rate} Hz, "
          f"peak {np.max(np.abs(data)):.3f}, rms {np.sqrt((data**2).mean()):.4f}")
    print(f"  MP3 bytes : header {blob[:2].hex()}, "
          f"{'valid MPEG frame sync' if valid else 'BAD HEADER'}")
    try:
        decoded, dec_rate = sf.read(str(mp3_out))
        print(f"  MP3 decode: soundfile decoded {decoded.shape[0]/dec_rate:.3f}s at "
              f"{dec_rate} Hz, peak {np.max(np.abs(decoded)):.3f}")
    except Exception as exc:  # pragma: no cover - libsndfile without mp3
        print(f"  MP3 decode: unavailable ({type(exc).__name__}: {exc})")

    print(f"\nboth files are under {OUT_DIR} (git-ignored).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
