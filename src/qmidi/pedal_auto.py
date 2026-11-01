"""Generate sustain-pedal spans from the notes themselves.

Why this exists
---------------
A transcribed orchestral MIDI usually arrives with one of two pedals:

* **none at all** -- the transcriber was not a pianist and only cared about notes;
* **a mechanical one** -- down at the start of a bar, up four or five beats
  later, no matter what the harmony does underneath.

The second kind is worse than it looks: five beats of a dense orchestral
reduction piled on one pedal is a wash of overtones, and no amount of velocity
editing fixes it.  A pianist changes the pedal when the *bass* or the *harmony*
changes, which is a rule that can be stated exactly and therefore automated.

What the rule is
----------------
The piece is walked in small steps (:attr:`AutoPedalOptions.grid_beats`).  A
pedal change is placed at the start of a step when either of these holds:

1. at least ``harmony_change`` pitch classes differ from what the pedal is
   already holding, counted as a *symmetric difference* (so swapping the fifth
   for a sixth counts as two, while a melody note landing on a held chord
   counts as one).  The bass needs no separate rule: it is part of the set, and
   a mere inversion (C-E-G -> E-G-C) is *not* a harmony change -- which is
   exactly when a pianist keeps the pedal down;
2. the current pedal has been held for ``max_span_beats`` already (a hard
   ceiling, so a slow chorale cannot become one long pedal wash).

An explicit "the bass note moved" trigger was tried and removed: on real
material the bass of a transcription changes on almost every sixteenth, and it
produced one pedal change every half second.  Pitch-class difference plus the
two guards below made the difference between 805 changes and 300.

Two guards keep the result playable rather than literal:

* ``min_span_beats`` -- a change closer than this is skipped, otherwise a fast
  Alberti bass would produce one pedal per note and the result would chatter;
* ``lift_beats`` -- the pedal is lifted a hair *before* the next change and
  pressed again just after it.  Writing the lift and the press on the same tick
  leaves both CC64 messages at one instant, which every player resolves
  differently; a couple of hundredths of a beat is inaudible but unambiguous.

This module is deliberately Qt-free and sidecar-free: it takes notes, returns
spans, and the caller decides whether to keep them (``SegmentProject.set_pedal_spans``).
Nothing here writes a file.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

from .pedal import PedalSpan, normalise
from .segments import Note

__all__ = [
    "AutoPedalOptions",
    "generate_spans",
    "harmonic_change_points",
]


@dataclass(frozen=True)
class AutoPedalOptions:
    """Knobs for :func:`generate_spans`, all in beats except the counts.

    Tuned on a 7-minute two-piano transcription at ~82 BPM, measured per track:

    * ``harmony_change=4`` was the original guess and it **never fired** in a
      dense texture -- the sounding set already holds 7-9 pitch classes, so a
      four-class symmetric difference hardly ever happens.  70% of the changes
      landed on the max-span ceiling, i.e. the result was a mechanical
      every-5-beats pedal, exactly what this module exists to replace.  At 2,
      the ceiling accounts for only 16-28%.
    * ``min_span_beats=3.0`` keeps the total change count where it was (~150 per
      track, median 2.2-2.9 s) while letting the harmonic rule drive the timing
      instead of the ceiling.
    * ``max_span_beats=6.0`` gives the 3-beat floor room for two changes before
      the backstop forces one anyway.
    """

    grid_beats: float = 0.5
    min_span_beats: float = 3.0
    max_span_beats: float = 6.0
    lift_beats: float = 0.04
    harmony_change: int = 2

    def __post_init__(self) -> None:
        if self.grid_beats <= 0:
            raise ValueError(f"grid_beats must be positive, got {self.grid_beats!r}")
        if self.min_span_beats < 0 or self.max_span_beats <= 0:
            raise ValueError("min_span_beats >= 0 and max_span_beats > 0 required")
        if self.max_span_beats < self.min_span_beats:
            raise ValueError(
                f"max_span_beats ({self.max_span_beats}) must be >= "
                f"min_span_beats ({self.min_span_beats})"
            )
        if self.lift_beats < 0:
            raise ValueError(f"lift_beats must be >= 0, got {self.lift_beats!r}")
        if self.harmony_change < 1:
            raise ValueError(f"harmony_change must be >= 1, got {self.harmony_change!r}")


def harmonic_change_points(
    notes: Sequence[Note],
    *,
    ticks_per_beat: int,
    end_tick: Optional[int] = None,
    options: AutoPedalOptions = AutoPedalOptions(),
) -> list[int]:
    """Tick positions where the pedal should be re-taken, in ascending order.

    The first entry is the tick of the first note (or 0 when there is nothing
    to play), so the caller can always treat ``points[0]`` as "start pedalling
    here" without a special case.
    """
    if ticks_per_beat <= 0:
        raise ValueError(f"ticks_per_beat must be positive, got {ticks_per_beat!r}")
    ordered = sorted((int(n.start_tick), int(n.pitch)) for n in notes)
    if not ordered:
        return []
    step = max(1, int(round(options.grid_beats * ticks_per_beat)))
    min_span = int(round(options.min_span_beats * ticks_per_beat))
    max_span = int(round(options.max_span_beats * ticks_per_beat))

    last_tick = max(int(n.end_tick) for n in notes)
    end = max(last_tick, int(end_tick) if end_tick is not None else last_tick)

    #: What is *sounding* right now, counted per pitch class.  Keeping the count
    #: (not a bare set) is what lets a released note be removed again: an earlier
    #: version only ever added, so after a minute the "current harmony" held every
    #: pitch class and no chord could count as a change -- the pedal then fell
    #: back to the max-span ceiling on 70% of its changes, which is the mechanical
    #: pedalling this module exists to replace.
    sounding: dict[int, int] = {}
    endings = sorted((int(n.end_tick), int(n.pitch) % 12) for n in notes)
    end_cursor = 0

    points: list[int] = []
    active_pcs: set[int] = set()
    index = 0
    cell = ordered[0][0]
    #: True when the previous cell had nothing sounding.  A phrase that starts
    #: after silence always re-takes the pedal, whatever ``min_span`` says: the
    #: minimum exists to stop a trill from chattering, not to keep a pedal held
    #: across a rest and into the next phrase.
    was_silent = False

    while cell <= end:
        cell_end = cell + step
        # Retire everything that has finished by the start of this cell.
        while end_cursor < len(endings) and endings[end_cursor][0] <= cell:
            pc = endings[end_cursor][1]
            remaining = sounding.get(pc, 0) - 1
            if remaining > 0:
                sounding[pc] = remaining
            else:
                sounding.pop(pc, None)
            end_cursor += 1

        started: list[int] = []
        while index < len(ordered) and ordered[index][0] < cell_end:
            started.append(ordered[index][1])
            index += 1
        for pitch in started:
            pc = pitch % 12
            sounding[pc] = sounding.get(pc, 0) + 1

        if not sounding:
            # Silence: the pedal cannot be held forever, so a long rest still
            # gets re-taken at the ceiling.
            if points and cell - points[-1] >= max_span:
                points.append(cell)
            was_silent = True
            cell = cell_end
            continue

        pcs = set(sounding)
        if not points:
            points.append(cell)
            active_pcs = pcs
            cell = cell_end
            continue

        held = cell - points[-1]
        if was_silent:
            # A new phrase: re-take regardless of how recently the last change
            # was.  Silently skipping it here is what made a rest-then-chord
            # figure play entirely on the previous phrase's pedal.
            points.append(cell)
            active_pcs = pcs
            was_silent = False
            cell = cell_end
            continue
        if held < min_span:
            cell = cell_end
            continue

        # Symmetric difference, not "how many notes are new": swapping the
        # fifth for a sixth (C-E-G -> C-E-A) is two pitch classes in and out, a
        # real chord change, while a melody note landing on a held chord adds one
        # and is *not*.  Counting both directions tells them apart.
        changed = len(pcs ^ active_pcs)
        harmonic = changed >= options.harmony_change
        forced = held >= max_span

        if harmonic or forced:
            points.append(cell)
            active_pcs = pcs
        cell = cell_end

    return points


def generate_spans(
    notes: Sequence[Note],
    *,
    ticks_per_beat: int,
    end_tick: Optional[int] = None,
    options: AutoPedalOptions = AutoPedalOptions(),
) -> list[PedalSpan]:
    """Turn one part's notes into pedal spans (half-open ``[down, up)``).

    ``notes`` is the note list of **one hand/one part** (``segments.read_notes``
    gives one list per track); mixing two pianos into one call would make the
    bass of the second piano suppress every change of the first.

    Spans are inclusive of the first change point and the last one runs to
    ``end_tick`` (default: the last note-off), so a final chord keeps ringing.
    """
    points = harmonic_change_points(
        notes, ticks_per_beat=ticks_per_beat, end_tick=end_tick, options=options
    )
    if not points:
        return []
    lift = int(round(options.lift_beats * ticks_per_beat))
    last = max(int(n.end_tick) for n in notes)
    tail = max(last, int(end_tick) if end_tick is not None else last)

    spans: list[PedalSpan] = []
    for i, point in enumerate(points):
        start = point if i == 0 else point + lift
        end = points[i + 1] if i + 1 < len(points) else tail
        if end <= start:
            continue
        spans.append(PedalSpan(start, end))
    return normalise(spans)
