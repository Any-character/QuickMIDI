"""What to render, and from which file.

The renderer (``qmidi.offline``) reads a **MIDI file**, but the user's edits
live in the session's segment model.  So a preview is always two steps: write
the edited velocities into a throwaway ``.mid``, then stream that file.

Keeping the temp file in sync is the only interesting part.  Re-exporting on
every play is ~60 ms for the sample file and buys correctness for free; a cache
keyed on the project digest avoids even that when nothing changed.

This module is Qt-free so the "what would play if I hit space" question is
answerable in a test.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mido

from ..exporters import DEFAULT_TAIL_SEC, export_midi
from ..i18n import tr
from ..segments import SegmentProject

#: Preview block size.  Measured on this machine: 128 renders ~1.7x real time
#: against 0.97x for the historical 1024 default (AGENTS.md §4.1), and it
#: quantises note attacks to 2.7 ms instead of 21 ms.
#: 512 rather than 128: Python round-trips per block dominate once the render is
#: already close to real time, and 512 still quantises attacks to 10.7 ms.
PREVIEW_BLOCK_SIZE = 512
PREVIEW_SAMPLE_RATE = 48000

#: sfizz's sample interpolation quality (0..10; 10 is the default maximum).
#: Measured on this machine, dense passage, 3 runs best-of: quality 10 renders at
#: 1.85x real time, quality 6 at 4.07x -- with the identical voice count.  That
#: 2.2x is the difference between a preview that starves and one that does not.
#: Export deliberately keeps the maximum: it is offline, and the file is the
#: product.
PREVIEW_SAMPLE_QUALITY = 6

#: Voices the preview may use.  Dense passages genuinely need ~74 while sfizz
#: defaults to 64, so it was cutting sounding notes off -- audible as clicks,
#: and unlike quality it costs almost nothing to raise (3.96x vs 4.07x at 128).
PREVIEW_NUM_VOICES = 128


@dataclass(frozen=True)
class PreviewRequest:
    """One playback job: a file, a track filter, and a time span."""

    midi_path: Path
    tracks: frozenset[int]
    start_sec: float
    end_sec: float
    loop: bool
    label: str
    #: Instrument chosen in the settings screen; ``None`` means "let the engine
    #: resolve it" (SFZ_PATH, else the repo default).
    sfz_path: Optional[Path] = None

    @property
    def duration_sec(self) -> float:
        return max(0.0, self.end_sec - self.start_sec)


def project_digest(project: SegmentProject) -> str:
    """Short stable hash of everything that affects the rendered audio."""
    payload = json.dumps(project.to_dict(), sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def _hold_final_pedal(path: Path) -> None:
    """Delete a CC64=0 that lands on the final note-off.

    Notation software writes "pedal up" at exactly the last note, which chokes
    the release the instant the piece ends -- that is why the ending sounded
    clipped while the pedal clearly spans to the last beat.  A pianist lifts a
    second or two later, which is the whole reason a grand-piano ending rings.

    Only the preview copy is touched: "export MIDI" still writes the file the
    user gave us, pedal and all.
    """
    midi = mido.MidiFile(str(path))
    tpb = max(1, midi.ticks_per_beat)
    changed = False

    for track in midi.tracks:
        tick = 0
        last_off = 0
        releases: list[tuple[int, int]] = []
        for index, message in enumerate(track):
            tick += message.time
            if message.type == "note_on" and message.velocity == 0:
                last_off = max(last_off, tick)
            elif message.type == "note_off":
                last_off = max(last_off, tick)
            elif (message.type == "control_change" and message.control == 64
                    and message.value == 0):
                releases.append((index, tick))
        if not releases:
            continue
        index, at = releases[-1]
        # Keep the event when the pedal was lifted clearly *before* the last
        # note -- that is a played articulation.  Drop it when the lift happens
        # at or after the final note-off, which is the notation software's
        # mechanical "pedal up" and never a musical gesture.
        if at < last_off - max(1, tpb // 8):
            continue
        # Deleting an event must not swallow its delta time, or every later
        # message shifts earlier by that amount.
        if index + 1 < len(track):
            track[index + 1].time += track[index].time
        del track[index]
        changed = True

    if changed:
        midi.save(str(path))


class PreviewMidi:
    """A temp ``.mid`` that follows the project's current velocities."""

    def __init__(self, workdir: Path | str | None = None) -> None:
        self._dir = Path(workdir) if workdir is not None else Path(
            tempfile.mkdtemp(prefix="qmidi-preview-")
        )
        self._dir.mkdir(parents=True, exist_ok=True)
        self._digest: Optional[str] = None
        self._path: Optional[Path] = None

    @property
    def path(self) -> Optional[Path]:
        return self._path

    def ensure(self, project: SegmentProject) -> Path:
        """Write the preview file if the project changed; return its path."""
        digest = project_digest(project)
        target = self._dir / "preview.mid"
        if digest != self._digest or not target.is_file():
            export_midi(project, target, verify=True)
            _hold_final_pedal(target)
            self._digest = digest
            self._path = target
        return target

    def invalidate(self) -> None:
        self._digest = None


def _file_end_sec(session) -> float:
    return max(0.05, session.clock.seconds(session.total_end_tick()))


def loop_request(session, midi_path: Path | str,
                 sfz_path: Optional[Path] = None) -> Optional[PreviewRequest]:
    """The selected segment (or the dragged region) as a looping preview.

    Returns ``None`` when there is nothing audible: a segment with no notes, or
    every track muted.  The caller turns that into a status message rather than
    starting an engine over an empty timeline.
    """
    tracks = session.visible_tracks()
    if not tracks:
        return None
    span = session.loop_seconds()
    if span is None:
        return None
    start, end = span
    if end - start < 0.02:
        return None
    return PreviewRequest(
        midi_path=Path(midi_path),
        tracks=tracks,
        start_sec=start,
        end_sec=end,
        loop=True,
        label=tr("循环 {a}–{b}").format(a=_mmss(start), b=_mmss(end)),
        sfz_path=sfz_path,
    )


#: Seconds of release kept after the last note-off when previewing.  Deliberately
#: the export constant rather than a copy of its value: preview and MP3 have to
#: end together, and two numbers drift apart the first time one is retuned.
PLAY_TAIL_SEC = DEFAULT_TAIL_SEC


def play_request(session, midi_path: Path | str, *, loop: bool,
                 sfz_path: Optional[Path] = None) -> Optional[PreviewRequest]:
    """What the transport plays right now.

    Loop on -> the selected segment repeats.  Loop off -> from the cursor to the
    end of the file, once.  This is the only place that decision is made, so the
    button, the space bar and the overview bar cannot disagree.
    """
    if not session.visible_tracks():
        return None
    if loop:
        return loop_request(session, midi_path, sfz_path=sfz_path)
    start = max(0.0, session.clock.seconds(max(0, int(session.cursor_tick))))
    # Keep rendering after the last note-off: sfizz voices keep ringing for a
    # couple of seconds and cutting at the last note chops the final chord --
    # the exact moment a pianist is listening to.  ``export_mp3`` has always had
    # a tail; playback did not.
    end = _file_end_sec(session) + PLAY_TAIL_SEC
    end = max(end, start + 0.05)
    return PreviewRequest(
        midi_path=Path(midi_path),
        tracks=session.visible_tracks(),
        start_sec=start,
        end_sec=end,
        loop=False,
        label=tr("播放 {a}–{b}").format(a=_mmss(start), b=_mmss(end)),
        sfz_path=sfz_path,
    )


def file_request(session, midi_path: Path | str, start_sec: float = 0.0) -> PreviewRequest:
    """The whole file from ``start_sec``, played once (no loop)."""
    tracks = session.visible_tracks()
    start = max(0.0, float(start_sec))
    end = max(_file_end_sec(session), start + 0.05)
    return PreviewRequest(
        midi_path=Path(midi_path),
        tracks=tracks,
        start_sec=start,
        end_sec=end,
        loop=False,
        label=tr("整曲 {a}–{b}").format(a=_mmss(start), b=_mmss(end)),
    )


def _mmss(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    return f"{total // 60}:{total % 60:02d}"
