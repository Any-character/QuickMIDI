"""Velocity segments: the software's own concept, layered on top of a MIDI file.

A "velocity segment" (力度段) does not exist in the MIDI format.  It is the unit
a performer wants to grab when they would otherwise cut a region, drag it to a
temporary track, nudge every velocity and drag it back.  This module owns that
concept end to end:

* :func:`read_notes` -- per-track note spans straight out of mido (ticks, no
  tempo conversion: segment boundaries are a musical/tick question, not a wall
  clock one).
* :func:`auto_segment` -- the auto cut: same-value runs, but a *repeating*
  figure stays one segment so the user never gets one segment per note.
* :class:`SegmentedTrack` -- the operations (nudge, set, merge, split) that
  must be unit-testable without a GUI.
* :class:`SegmentProject` -- the sidecar JSON, because a velocity segment is
  not a MIDI event and must never be written back into the user's .mid.

Nothing here imports Qt or pysfizz (see AGENTS.md): the whole model runs
headless, and the GUI is a thin shell over it.

Two design points that override the naive reading of the spec
-----------------------------------------------------------

*Boundaries are snapped, never cut through a note.*  A human performance
overlaps notes constantly (a held accompaniment note under a moving melody),
so a boundary derived purely from "where the velocity changed" can easily land
in the middle of a still-sounding note.  :func:`_snap_cuts` walks each candidate
boundary back to the start of any note that straddles it, which is what makes
"a segment either fully contains a note or does not contain it at all" true by
construction rather than by luck.

*Equality has a tolerance.*  Real takes jitter by +-1..3 between repetitions of
the same gesture.  With literal ``==`` every repeated note becomes its own
segment, which is exactly the failure mode this feature exists to remove.
:class:`AutoSegmentOptions.velocity_tolerance` absorbs that jitter; set it to 0
for the literal behaviour.
"""

from __future__ import annotations

import bisect
import json
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Iterable, Optional, Sequence

import mido

from .i18n import tr
from .pedal import PedalSpan, normalise as normalise_pedal

__all__ = [
    "AutoSegmentOptions",
    "MergeConflictError",
    "Note",
    "REASON_CYCLE",
    "REASON_MANUAL",
    "REASON_MERGED",
    "REASON_SAME",
    "SIDECAR_FORMAT",
    "SIDECAR_VERSION",
    "SIDECAR_SUFFIX",
    "SegmentError",
    "SegmentedTrack",
    "SegmentProject",
    "SplitInsideNoteError",
    "VEL_MAX",
    "VEL_MIN",
    "VelocitySegment",
    "auto_segment",
    "build_tracks",
    "load_project",
    "read_notes",
    "sidecar_path",
]

VEL_MIN = 1
VEL_MAX = 127

#: Upper bound for a segment's proportional scale.  4x turns a 30 into 120;
#: beyond that it is a data-entry mistake, not a musical intent, and the sidecar
#: should refuse it rather than render a wall of 127s.
_MAX_SCALE = 4.0

#: Reason tags.  Kept as machine codes so the GUI can localize them;
#: :meth:`VelocitySegment.label` renders the Chinese default label.
REASON_SAME = "same"
REASON_CYCLE = "cycle"
REASON_MANUAL = "manual"
REASON_MERGED = "merged"

_REASON_TAGS = frozenset(
    {REASON_SAME, REASON_CYCLE, REASON_MANUAL, REASON_MERGED}
)

SIDECAR_FORMAT = "qmidi-segments"
SIDECAR_VERSION = 1
SIDECAR_SUFFIX = ".qmidi.json"


class SegmentError(ValueError):
    """Base class for every rejected segment operation."""


class SplitInsideNoteError(SegmentError):
    """A cut was requested in the middle of a sounding note (spec rule 2c)."""


class MergeConflictError(SegmentError):
    """Two adjacent segments cannot be merged without losing information."""


# --------------------------------------------------------------------------- #
# notes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Note:
    """One note, in ticks, with the velocity exactly as it appears in the file.

    ``index`` is the note's position inside its own track's note list and is
    stable for a given file; segments refer to notes by that index, so the
    sidecar stays small and a re-read of the same MIDI resolves identically.
    """

    index: int
    channel: int
    pitch: int
    velocity: int
    start_tick: int
    end_tick: int

    @property
    def duration(self) -> int:
        return self.end_tick - self.start_tick

    @property
    def sort_key(self) -> tuple:
        return (self.start_tick, self.end_tick, self.pitch, self.channel)

    def contains_tick(self, tick: int) -> bool:
        """True if ``tick`` falls strictly inside the note (half-open span)."""
        return self.start_tick < tick < self.end_tick


def read_notes(midi_path: Path | str) -> list[list[Note]]:
    """Read every track's notes out of a MIDI file.

    Returns one note list per track, sorted by ``(start_tick, end_tick, pitch,
    channel)``.  Four details that a naive reader gets wrong:

    * A velocity-0 ``note_on`` **is** a note-off.  Unchained.mid has 4065 of
      them and zero ``note_off`` messages, so trusting ``msg.type`` alone
      leaves every note ringing forever.
    * Overlapping notes of one pitch are matched **first-in-first-out**, the
      same rule :class:`qmidi.offline.SustainState` uses when it hands a
      release to "the oldest unreleased voice".
    * A note with no matching off is closed at the end of its track rather than
      dropped, so nothing silently vanishes from a segment.
    * A note whose off lands on the same tick as its on (Unchained.mid has 33 of
      them in track 2) keeps a **one-tick** span instead of being discarded.  A
      zero-length note is still an attack the user will hear, and dropping it
      would quietly remove music from the segment model.  One tick is also the
      smallest possible span, so such a note can never have a cut inside it.
    """
    midi = mido.MidiFile(str(midi_path))
    tracks: list[list[Note]] = []
    for track in midi.tracks:
        tick = 0
        track_end = 0
        #: (channel, pitch) -> stack of (start_tick, velocity), oldest first
        pending: dict[tuple[int, int], list[tuple[int, int]]] = {}
        raw: list[tuple[int, int, int, int, int]] = []  # start, end, ch, pitch, vel
        for msg in track:
            tick += msg.time
            # The track's length includes its meta messages: a dangling note is
            # closed at end-of-track, which is itself a meta message.
            track_end = max(track_end, tick)
            if msg.is_meta:
                continue
            if msg.type == "note_on":
                key = (msg.channel, msg.note)
                if msg.velocity > 0:
                    pending.setdefault(key, []).append((tick, msg.velocity))
                else:
                    stack = pending.get(key)
                    if stack:
                        start, vel = stack.pop(0)
                        raw.append((start, tick, key[0], key[1], vel))
            elif msg.type == "note_off":
                stack = pending.get((msg.channel, msg.note))
                if stack:
                    start, vel = stack.pop(0)
                    raw.append((start, tick, msg.channel, msg.note, vel))
        for (channel, pitch), stack in pending.items():
            for start, vel in stack:
                raw.append((start, max(track_end, start), channel, pitch, vel))
        raw.sort(key=lambda r: (r[0], r[1], r[3], r[2]))
        tracks.append(
            [Note(i, ch, pitch, vel, start, max(end, start + 1))
             for i, (start, end, ch, pitch, vel) in enumerate(raw)]
        )
    return tracks


# --------------------------------------------------------------------------- #
# segment model
# --------------------------------------------------------------------------- #
@dataclass
class VelocitySegment:
    """A contiguous span of one track whose notes move together.

    ``start_tick``/``end_tick`` form a half-open span ``[start, end)``; together
    the segments of a track partition the track, so a segment is dragged as a
    whole and never has to grow.

    Three adjustments, and the order they apply:

    ``target`` (absolute velocity) wins over everything -- "this section is p,
    always".  Otherwise a note's velocity becomes ``round(velocity * scale) +
    delta``: ``scale`` is the *proportional* handle (dragging a velocity bar by
    100 -> 50 puts a 90 at 45, keeping the passage's internal balance) and
    ``delta`` is the *additive* one (the ±3 nudge from the keyboard).

    A segment is one gesture, so both adjustments are per segment, never per note.
    """

    track: int
    start_tick: int
    end_tick: int
    delta: int = 0
    target: Optional[int] = None
    scale: float = 1.0
    reason: str = REASON_MANUAL
    reason_values: tuple[int, ...] = ()
    note_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        self.reason_values = tuple(int(v) for v in self.reason_values)
        self.note_indices = tuple(int(i) for i in self.note_indices)
        self.delta = int(self.delta)
        self.scale = float(self.scale)
        self.start_tick = int(self.start_tick)
        self.end_tick = int(self.end_tick)
        if not 0.0 < self.scale <= _MAX_SCALE:
            raise SegmentError(
                f"scale must be in (0, {_MAX_SCALE}], got {self.scale!r}"
            )
        if self.start_tick < 0:
            raise SegmentError(f"start_tick must be >= 0, got {self.start_tick}")
        if self.end_tick <= self.start_tick:
            raise SegmentError(
                f"empty or inverted segment [{self.start_tick}, {self.end_tick})"
            )
        if self.target is not None:
            self.target = int(self.target)
            if not VEL_MIN <= self.target <= VEL_MAX:
                raise SegmentError(
                    f"target velocity must be {VEL_MIN}..{VEL_MAX}, got {self.target}"
                )
        # The tick *upper* bound and "is this note really inside this segment"
        # need the track's notes, so they live in SegmentedTrack.validate.  What
        # a lone segment can still police is that its own fields are not junk:
        # bad data used to flow straight out of a hand-edited sidecar.
        if self.reason not in _REASON_TAGS:
            raise SegmentError(
                f"unknown reason {self.reason!r}; expected one of "
                f"{sorted(_REASON_TAGS)}"
            )
        if any(i < 0 for i in self.note_indices):
            raise SegmentError(f"note_indices must be >= 0, got {self.note_indices}")
        if len(set(self.note_indices)) != len(self.note_indices):
            raise SegmentError(f"note_indices must be unique, got {self.note_indices}")

    # -- presentation ------------------------------------------------------ #
    @property
    def label(self) -> str:
        """Short human label for the GUI (Chinese original, tr() on display)."""
        if self.reason == REASON_CYCLE:
            return tr("周期{values}").format(values="-".join(str(v) for v in self.reason_values))
        if self.reason == REASON_SAME:
            values = self.reason_values or (0,)
            if len(values) == 1:
                return tr("同值{value}").format(value=values[0])
            return tr("同段{values}").format(values="/".join(str(v) for v in values))
        if self.reason == REASON_MERGED:
            return tr("合并")
        return tr("手动")

    @property
    def is_periodic(self) -> bool:
        return self.reason == REASON_CYCLE

    @property
    def duration(self) -> int:
        return self.end_tick - self.start_tick

    # -- resolution -------------------------------------------------------- #
    def velocity_for(self, note: Note) -> int:
        """Velocity this segment gives ``note``, clamped to the legal range."""
        if self.target is not None:
            return clamp_velocity(self.target)
        scaled = note.velocity * self.scale + self.delta
        return clamp_velocity(int(round(scaled)))

    def with_delta(self, delta: int) -> "VelocitySegment":
        """A copy with ``delta`` *added* (nudging never clobbers the old value)."""
        return replace(self, delta=self.delta + int(delta))

    def with_target(self, velocity: int) -> "VelocitySegment":
        return replace(self, target=int(velocity))

    def with_scale(self, scale: float) -> "VelocitySegment":
        """A copy rescaled proportionally; ``delta`` is cleared by the caller.

        Moving a whole segment while keeping the *additive* offset would make the
        two handles fight (rescale, then nudge, then rescale again drifts).  The
        GUI zeroes ``delta`` when it drags a bar, so a drag is always absolute.
        """
        return replace(self, scale=float(scale))

    def copy(self) -> "VelocitySegment":
        return replace(self)


def clamp_velocity(value: int) -> int:
    """Clamp to the legal MIDI velocity range (0 is silence, not a segment)."""
    return max(VEL_MIN, min(VEL_MAX, int(value)))


# --------------------------------------------------------------------------- #
# auto segmentation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class AutoSegmentOptions:
    """Knobs for :func:`auto_segment`.

    ``velocity_tolerance``  same-value test band, in velocity units (default 4
        covers typical performance jitter of +-2; 0 = literal equality).
    ``period_beats``       how wide a repeating figure may span and still count
        as *one* gesture.  This is the "N 拍内出现的规律变化归一组" parameter: a
        3-note figure that repeats every 4 beats stays one segment, a change
        40 beats later does not get glued to it.
    ``max_cycle_len``      longest repeating cycle considered (2..N).  Longer
        ostinatos are left as separate segments rather than guessed at.
    ``min_cycles``         how many back-to-back repetitions are required before
        a pattern is called periodic.  2 is the minimum that is not a
        coincidence.
    """

    velocity_tolerance: int = 4
    period_beats: float = 8.0
    max_cycle_len: int = 4
    min_cycles: int = 2
    #: Jitter allowed when matching a repeated figure.  ``None`` follows
    #: ``velocity_tolerance``.  Exact equality was too strict in practice: one
    #: note played 96 instead of 97 broke the cycle and the same four-bar phrase
    #: became five separate segments.
    cycle_tolerance: Optional[int] = None
    #: Floor for a segment: shorter than this many notes (or this many beats) and
    #: the segment is folded into the neighbour with the closest velocity range.
    #: 0 disables the rule, which is what the single-rule tests use.
    min_segment_notes: int = 0
    min_segment_beats: float = 0.0

    @property
    def effective_cycle_tolerance(self) -> int:
        if self.cycle_tolerance is None:
            return self.velocity_tolerance
        return max(0, int(self.cycle_tolerance))

    def validate(self, tpb: int) -> None:
        if tpb <= 0:
            raise SegmentError(f"ticks_per_beat must be positive, got {tpb}")
        if self.velocity_tolerance < 0:
            raise SegmentError("velocity_tolerance must be >= 0")
        if self.period_beats <= 0:
            raise SegmentError("period_beats must be > 0")
        if self.min_cycles < 2:
            raise SegmentError("min_cycles must be >= 2 (one repetition is a coincidence)")
        if self.max_cycle_len < 2:
            raise SegmentError("max_cycle_len must be >= 2")
        if self.min_segment_notes < 0:
            raise SegmentError("min_segment_notes must be >= 0")
        if self.min_segment_beats < 0:
            raise SegmentError("min_segment_beats must be >= 0")


@dataclass
class _Run:
    """A maximal group of consecutive notes sharing one velocity class."""

    notes: list[Note]
    value: int                 # representative velocity of the class
    start_tick: int
    end_tick: int
    cycle_len: int = 0         # >0 when this run belongs to a detected cycle
    cycle_values: tuple[int, ...] = ()

    @property
    def first_index(self) -> int:
        return self.notes[0].index


def _build_runs(notes: Sequence[Note], tol: int) -> list[_Run]:
    """Group consecutive (in time) notes whose velocity agrees within ``tol``."""
    runs: list[_Run] = []
    for note in notes:
        if runs and abs(note.velocity - runs[-1].value) <= tol:
            runs[-1].notes.append(note)
            runs[-1].end_tick = max(runs[-1].end_tick, note.end_tick)
        else:
            runs.append(_Run([note], note.velocity, note.start_tick, note.end_tick))
    return runs


def _detect_cycles(
    runs: list[_Run],
    opts: AutoSegmentOptions,
    window_ticks: int,
) -> set[int]:
    """Return the indices of runs that *continue* a repeating figure.

    A run is marked when, starting at it, some cycle length ``p`` in
    ``2..max_cycle_len`` makes the following values repeat ``min_cycles`` times
    back to back inside the window.  Marking only the *continuation* runs (never
    the first) is what keeps one segment per figure instead of one per note.

    Greedy left to right: the earliest evidence of a cycle wins, and the scan
    jumps past the whole figure, so an 80-70-60 ostinato becomes one group
    rather than overlapping 3-note windows.
    """
    values = [r.value for r in runs]
    n = len(runs)
    tol = opts.effective_cycle_tolerance
    suppressed: set[int] = set()
    i = 0
    while i < n:
        fit = _cycle_at(values, runs, i, opts, window_ticks)
        if fit is None:
            i += 1
            continue
        p, span = fit
        group_values = tuple(values[i : i + p])
        # Keep absorbing while the same cycle continues.  Without this the loop
        # restarted at ``i + span`` and that run became a cut again, so a phrase
        # repeated four times still produced four segments.
        last = i + span - 1
        while last + 1 < n and abs(values[last + 1] - values[last + 1 - p]) <= tol:
            last += 1
        for j in range(i, last + 1):
            runs[j].cycle_len = p
            runs[j].cycle_values = group_values
            if j > i:
                suppressed.add(j)
        i = last + 1
    return suppressed


def _cycle_at(
    values: list[int],
    runs: list[_Run],
    i: int,
    opts: AutoSegmentOptions,
    window_ticks: int,
) -> Optional[tuple[int, int]]:
    """Smallest ``(cycle_len, run_span)`` fitting a periodic figure at run ``i``.

    Matching allows ``effective_cycle_tolerance`` of jitter: a human repeating a
    four-bar figure does not hit the same velocity to the unit, and demanding
    exact equality turned every repetition into a new segment.
    """
    n = len(runs)
    tol = opts.effective_cycle_tolerance
    for p in range(2, opts.max_cycle_len + 1):
        span = p * opts.min_cycles
        if i + span > n:
            break
        if runs[i + span - 1].end_tick - runs[i].start_tick > window_ticks:
            # Longer cycles only get wider, so no need to keep trying.
            break
        if all(abs(values[i + k] - values[i + p + k]) <= tol for k in range(span - p)):
            return p, span
    return None


def _snap_cuts(notes: Sequence[Note], cuts: Sequence[int]) -> list[int]:
    """Move every cut off the middle of a note, to the *nearest* note edge.

    A velocity change happens where a note is struck, but that strike can land
    inside a still-sounding note from the previous layer -- a held bass under a
    moving melody is the normal case, not an exotic one.  Snapping to the
    closest edge (a note's start or its end) displaces the cut as little as
    possible and guarantees rule 2c by construction: with no cut strictly
    inside any note, every note falls wholly inside exactly one segment.

    Ties prefer the earlier edge, which keeps the delay to the new segment
    small.  Cuts are processed left to right with a floor, so a cut can never
    walk back past one already placed.

    Overlapping notes can make the nearest edge point straight back at a tick
    the cut has already visited (three overlapping notes give a 107 -> 130 ->
    140 -> 107 cycle).  Bailing out of that cycle used to leave the cut *inside*
    a note, which is exactly the corruption this function exists to prevent, so
    the second sign of oscillation switches to a pure backward walk: each step
    jumps to the earliest start among the straddling notes, which is strictly
    less than the current tick, so the walk terminates and ends on a tick no
    note straddles.
    """
    out: list[int] = []
    floor: Optional[int] = None
    for cut in cuts:
        t = int(cut)
        seen = {t}
        backward = False
        for _ in range(len(notes) * 4 + 8):  # guard; progress is guaranteed
            straddling = [n for n in notes if n.contains_tick(t)]
            if not straddling:
                break
            if backward:
                edge = min(n.start_tick for n in straddling)
            else:
                edges: list[tuple[int, int]] = []
                for n in straddling:
                    # distance to *each* edge -- using min() for both would make
                    # them tie and the tie-break would always take the earlier one.
                    edges.append((n.start_tick, t - n.start_tick))
                    edges.append((n.end_tick, n.end_tick - t))
                edge, _dist = min(edges, key=lambda e: (e[1], e[0]))
                if edge in seen:
                    backward = True
                    edge = min(n.start_tick for n in straddling)
            seen.add(edge)
            t = edge
        if floor is not None and t <= floor:
            continue
        out.append(t)
        floor = t
    return out


def auto_segment(
    notes: Sequence[Note],
    track: int = 0,
    *,
    ticks_per_beat: int = 480,
    options: Optional[AutoSegmentOptions] = None,
    start_tick: int = 0,
) -> list[VelocitySegment]:
    """Cut one track's notes into velocity segments.

    The two rules that shape the result:

    1. **Same value -> one segment.**  Consecutive notes agreeing within
       ``velocity_tolerance`` form a run.
    2. **A repeating figure is one segment.**  Runs whose values recur
       ``min_cycles`` times within ``period_beats`` are folded into a single
       periodic segment, so a 80-70-60 ostinato does not become three segments
       per bar.

    Boundaries between segments are then snapped so no note is ever split, and
    the segments partition ``[start_tick, end_tick)`` of the track.

    ``start_tick`` must not leave notes outside the partition: a note starting
    before it could belong to no segment at all, which would silently drop it
    (and make :meth:`SegmentedTrack.validate` fail).  Callers that want to cut
    only a later passage filter ``notes`` themselves.
    """
    opts = options or AutoSegmentOptions()
    opts.validate(ticks_per_beat)
    if not notes:
        return []
    if start_tick < 0:
        raise SegmentError(f"start_tick must be >= 0, got {start_tick}")
    notes = sorted(notes, key=lambda n: n.sort_key)
    early = [n for n in notes if n.start_tick < start_tick]
    if early:
        raise SegmentError(
            f"start_tick {start_tick} would leave {len(early)} note(s) outside the "
            f"partition (first starts at {early[0].start_tick}); filter them out first"
        )
    window_ticks = int(opts.period_beats * ticks_per_beat)

    runs = _build_runs(notes, opts.velocity_tolerance)
    suppressed = _detect_cycles(runs, opts, window_ticks)

    cuts = [r.start_tick for i, r in enumerate(runs) if i > 0 and i not in suppressed]
    end_tick = max(n.end_tick for n in notes)

    boundaries = [start_tick]
    boundaries.extend(_snap_cuts(notes, cuts))
    boundaries.append(end_tick)
    boundaries = [b for b in boundaries if b >= start_tick]
    # Snapping can collide two boundaries; drop empties, keep them sorted.
    deduped: list[int] = []
    for b in boundaries:
        if not deduped or b > deduped[-1]:
            deduped.append(b)

    value_of = {n.index: r.value for r in runs for n in r.notes}

    # With no boundary strictly inside a note, every note lies wholly inside
    # exactly one window -- so assign each note to the window holding its
    # *start* instead of filtering per window.  The filter version could strand
    # a note whose start sat before the previous non-empty window's end: it was
    # excluded from that window (start < its end) and from every window after it
    # (start < their start), so it silently belonged to no segment at all.
    buckets: list[list[Note]] = [[] for _ in range(max(len(deduped) - 1, 0))]
    for note in notes:
        i = bisect.bisect_right(deduped, note.start_tick) - 1
        buckets[min(max(i, 0), len(buckets) - 1)].append(note)

    segments: list[VelocitySegment] = []
    last_end = start_tick
    for i, inside in enumerate(buckets):
        hi = deduped[i + 1]
        if not inside:
            # A gap left by snapping (the cut moved to an edge no note starts
            # at) is absorbed into the *next* segment's front rather than
            # dropped: an empty segment would be an undraggable dead handle, and
            # dropping it outright would leave a hole in an otherwise gapless
            # partition, which every later operation assumes.
            continue
        reason, values = _reason_for(
            _pattern([value_of[n.index] for n in inside]), inside, runs)
        segments.append(
            VelocitySegment(
                track=track,
                start_tick=last_end,
                end_tick=hi,
                reason=reason,
                reason_values=values,
                note_indices=tuple(n.index for n in inside),
            )
        )
        last_end = hi
    if opts.min_segment_notes > 0 or opts.min_segment_beats > 0:
        segments = _merge_undersized(segments, notes, opts, ticks_per_beat)
    return segments


def _merge_undersized(
    segments: Sequence[VelocitySegment],
    notes: Sequence[Note],
    opts: AutoSegmentOptions,
    ticks_per_beat: int,
) -> list[VelocitySegment]:
    """Fold segments below the floor into the neighbour with the closest range.

    A single accented note inside a quiet passage *is* a velocity change, but it
    is not a separate gesture: as its own segment it becomes a 0.25-beat handle
    the user cannot usefully drag, and a 30-segment track where 15 would do.
    Folding towards the closest velocity range (not simply the previous segment)
    keeps an accent from being swallowed by a passage in a different dynamic.
    """
    if len(segments) <= 1:
        return list(segments)
    by_index = {note.index: note for note in notes}
    min_ticks = int(opts.min_segment_beats * ticks_per_beat)

    groups = [
        {"start": seg.start_tick, "end": seg.end_tick,
         "indices": list(seg.note_indices), "reason": seg.reason,
         "values": list(seg.reason_values)}
        for seg in segments
    ]

    def span_of(group) -> tuple[int, int]:
        values = [by_index[i].velocity for i in group["indices"] if i in by_index]
        return (min(values), max(values)) if values else (0, 0)

    changed = True
    while changed and len(groups) > 1:
        changed = False
        for index, group in enumerate(groups):
            # Both floors must be violated before folding.  "One note" alone is
            # not small (a held chord can be short but musically huge), and
            # "short" alone is not small (a dense sixteenth run is long enough to
            # matter note-count-wise).  Using OR collapsed entire little pieces
            # into a single segment.
            few_notes = len(group["indices"]) < opts.min_segment_notes
            short_time = group["end"] - group["start"] < min_ticks
            if not (few_notes and short_time):
                continue
            low, high = span_of(group)
            best: Optional[int] = None
            best_gap: Optional[int] = None
            for other in (index - 1, index + 1):
                if not 0 <= other < len(groups):
                    continue
                other_low, other_high = span_of(groups[other])
                gap = max(0, max(low, other_low) - min(high, other_high))
                if best_gap is None or gap < best_gap:
                    best, best_gap = other, gap
            if best is None:
                continue
            left, right = (best, index) if best < index else (index, best)
            merged = groups[left]
            merged["end"] = groups[right]["end"]
            merged["indices"] = merged["indices"] + groups[right]["indices"]
            merged["values"] = _pattern(merged["values"] + groups[right]["values"])
            merged["reason"] = REASON_MERGED
            del groups[right]
            changed = True
            break

    track = segments[0].track
    return [
        VelocitySegment(
            track=track,
            start_tick=group["start"],
            end_tick=group["end"],
            reason=group["reason"],
            reason_values=tuple(group["values"]),
            note_indices=tuple(sorted(group["indices"])),
        )
        for group in groups
    ]


def _pattern(values: Iterable[int]) -> list[int]:
    """Consecutive duplicates collapsed, in time order."""
    out: list[int] = []
    for v in values:
        if not out or out[-1] != v:
            out.append(v)
    return out


def _reason_for(
    pattern: list[int],
    inside: Sequence[Note],
    runs: Sequence[_Run],
) -> tuple[str, tuple[int, ...]]:
    """Pick the reason tag for one finished segment from its own notes.

    Read back from the notes that actually landed in the segment (rather than
    remembered from the run that produced it) so the label can never drift away
    from the data after boundary snapping reshuffled things.
    """
    if not pattern:
        return REASON_MANUAL, ()
    cycle = _cycle_of(inside, runs)
    if cycle is not None:
        return REASON_CYCLE, tuple(cycle)
    if len(pattern) == 1:
        return REASON_SAME, (pattern[0],)
    # Several velocity classes ended up inside one segment (boundary snapping
    # pulled a held note into its neighbour).  Report the whole pattern rather
    # than pretending it is single-valued.
    return REASON_SAME, tuple(pattern)


def _cycle_of(inside: Sequence[Note], runs: Sequence[_Run]) -> Optional[tuple[int, ...]]:
    """The cycle values of the periodic figure contributing to this segment.

    Only returned when every periodic run reaching into the segment agrees on
    the figure: returning the first one's values would label a segment
    "周期80-70-60" while half of it is a different ostinato.  ``None`` lets the
    caller fall back to reporting the actual velocity pattern.
    """
    if not inside:
        return None
    wanted = {n.index for n in inside}
    figures = {
        run.cycle_values
        for run in runs
        if run.cycle_len and any(n.index in wanted for n in run.notes)
    }
    figures.discard(())
    if len(figures) == 1:
        return next(iter(figures))
    return None


# --------------------------------------------------------------------------- #
# track operations
# --------------------------------------------------------------------------- #
@dataclass
class SegmentedTrack:
    """One MIDI track plus the velocity segments laid over it."""

    track: int
    name: str = ""
    notes: list[Note] = field(default_factory=list)
    segments: list[VelocitySegment] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.notes = sorted(self.notes, key=lambda n: n.sort_key)
        self.sort_segments()

    # -- structure --------------------------------------------------------- #
    def sort_segments(self) -> None:
        self.segments.sort(key=lambda s: (s.start_tick, s.end_tick))

    @property
    def end_tick(self) -> int:
        return max((n.end_tick for n in self.notes), default=0)

    @property
    def boundaries(self) -> list[int]:
        """The tick of every cut, including 0 (i.e. the segment start ticks)."""
        return [s.start_tick for s in self.segments]

    def index_of_tick(self, tick: int) -> Optional[int]:
        """Index of the segment containing ``tick`` (half-open), else ``None``."""
        for i, seg in enumerate(self.segments):
            if seg.start_tick <= tick < seg.end_tick:
                return i
        return None

    def segment_at(self, tick: int) -> VelocitySegment:
        i = self.index_of_tick(tick)
        if i is None:
            raise SegmentError(f"tick {tick} is outside every segment of track {self.track}")
        return self.segments[i]

    def note_by_index(self) -> dict[int, Note]:
        return {n.index: n for n in self.notes}

    def notes_in(self, seg: VelocitySegment) -> list[Note]:
        return self._notes_in(seg, self.note_by_index())

    @staticmethod
    def _notes_in(
        seg: VelocitySegment, lookup: dict[int, Note]
    ) -> list[Note]:
        """``notes_in`` with a caller-provided index, so hot loops build it once."""
        return [lookup[i] for i in seg.note_indices if i in lookup]

    def validate(self) -> None:
        """Check the invariants everything else relies on."""
        lookup = self.note_by_index()
        limit = self.end_tick
        prev_end = None
        for seg in self.segments:
            if seg.track != self.track:
                raise SegmentError(
                    f"segment track {seg.track} does not belong to track {self.track}"
                )
            if prev_end is not None and seg.start_tick != prev_end:
                raise SegmentError(
                    f"segments must be contiguous: {seg.start_tick} != {prev_end}"
                )
            if self.notes and seg.end_tick > limit:
                # The tick upper bound a lone segment cannot check: a segment
                # reaching past the last note would cover a span no note can
                # ever be in, which is how a stale sidecar shows up as silence.
                raise SegmentError(
                    f"segment ends at {seg.end_tick}, past the last note of track "
                    f"{self.track} at {limit}"
                )
            unknown = [i for i in seg.note_indices if i not in lookup]
            if unknown:
                raise SegmentError(
                    f"segment [{seg.start_tick},{seg.end_tick}) refers to note(s) "
                    f"{unknown} that track {self.track} does not hold"
                )
            prev_end = seg.end_tick
        for seg in self.segments:
            for note in self._notes_in(seg, lookup):
                if not (seg.start_tick <= note.start_tick and note.end_tick <= seg.end_tick):
                    raise SegmentError(
                        f"note {note.index} is split by segment [{seg.start_tick},"
                        f" {seg.end_tick})"
                    )
        covered: set[int] = set()
        for seg in self.segments:
            covered.update(seg.note_indices)
        missing = {n.index for n in self.notes} - covered
        if missing:
            raise SegmentError(f"{len(missing)} note(s) belong to no segment")

    # -- operations -------------------------------------------------------- #
    def apply_delta(self, index: int, delta: int) -> VelocitySegment:
        """Nudge one whole segment up (``delta > 0``) or down."""
        seg = self._get(index)
        new = seg.with_delta(delta)
        self.segments[index] = new
        return new

    def set_value(self, index: int, velocity: int) -> VelocitySegment:
        """Pin a whole segment to one absolute velocity."""
        seg = self._get(index)
        new = seg.with_target(velocity)
        self.segments[index] = new
        return new

    def clear_adjustment(self, index: int) -> VelocitySegment:
        """Undo a segment's nudge/pin but keep what the segment *is*.

        The reason tag describes how the segment came to exist (a same-value
        run, a repeating figure, a merge), not what adjustment it carries, so
        clearing the adjustment must not relabel it -- tagging it ``merged`` was
        both a lie and a permanent loss of the original classification.
        """
        seg = self._get(index)
        new = replace(seg, delta=0, target=None, scale=1.0)
        self.segments[index] = new
        return new

    def merge_adjacent(self, index: int) -> VelocitySegment:
        """Merge segment ``index`` with the one after it.

        Two segments are only mergeable if they are truly adjacent and agree on
        an absolute ``target``; merging deltas is additive, merging two
        different pinned velocities would have to silently drop one of them.

        Deltas adding up is the contract, not a loophole: +50 next to -50 merges
        back to the original velocities, which is how a user undoes a pair of
        opposite nudges.  A threshold that refused "too different" deltas would
        also break :meth:`merge_all` for any track where two neighbouring
        passages were nudged in opposite directions.
        """
        if not 0 <= index < len(self.segments) - 1:
            raise SegmentError(
                f"no segment after index {index} (track {self.track} has "
                f"{len(self.segments)})"
            )
        a, b = self.segments[index], self.segments[index + 1]
        if a.end_tick != b.start_tick:
            raise MergeConflictError(
                f"segments [{a.start_tick},{a.end_tick}) and [{b.start_tick},"
                f"{b.end_tick}) are not adjacent"
            )
        if a.target != b.target:
            raise MergeConflictError(
                f"cannot merge pinned velocities {a.target} and {b.target}"
            )
        if abs(a.scale - b.scale) > 1e-9:
            # Two different proportional scales cannot be expressed as one:
            # (v*a + d) * b is not v * (a*b).  Refusing keeps the merge honest.
            raise MergeConflictError(
                f"cannot merge segments scaled {a.scale:.3f} and {b.scale:.3f}; "
                f"clear one of them first"
            )
        merged = replace(
            a,
            end_tick=b.end_tick,
            delta=a.delta + b.delta,
            reason=REASON_MERGED,
            reason_values=_merge_values(a, b),
        )
        merged.note_indices = tuple(sorted(set(a.note_indices) | set(b.note_indices)))
        self.segments[index : index + 2] = [merged]
        return merged

    def merge_all(self) -> list[VelocitySegment]:
        """Undo every cut: collapse the track back to a single segment."""
        while len(self.segments) > 1:
            self.merge_adjacent(len(self.segments) - 2)
        return self.segments

    def split_at_tick(self, tick: int) -> VelocitySegment:
        """Cut a new boundary at ``tick``, which must be on a note edge.

        The children inherit the parent's adjustment: a user who nudged a whole
        passage and then cuts it in half expects both halves to stay nudged.

        A tick that would leave one child holding no note at all is **snapped** to
        the nearest note edge that does separate notes, matching the module's
        "boundaries are snapped, never placed in a dead spot" rule; a segment
        with no notes is an undraggable dead handle in the GUI.  Only a parent
        with no usable edge at all (a hand-built segment whose ``note_indices``
        do not match its span) raises.
        """
        if tick in self.boundaries:
            raise SegmentError(f"tick {tick} is already a segment boundary")
        index = self.index_of_tick(tick)
        if index is None:
            raise SegmentError(f"tick {tick} is outside every segment of track {self.track}")
        parent = self.segments[index]
        splitting = [n for n in self.notes if n.contains_tick(tick)]
        if splitting:
            note = min(splitting, key=lambda n: n.start_tick)
            raise SplitInsideNoteError(
                f"tick {tick} falls inside note {note.index} "
                f"(pitch {note.pitch}, [{note.start_tick},{note.end_tick}))"
            )
        parent_indices = set(parent.note_indices)
        halves = self._halves_at(parent, tick)
        if halves is None:
            tick = self._nearest_usable_edge(parent, tick)
            if tick is None:
                raise SegmentError(
                    f"no note edge inside segment [{parent.start_tick},"
                    f"{parent.end_tick}) separates its notes; split it by hand"
                )
            halves = self._halves_at(parent, tick)
        left_indices, right_indices = halves
        left = replace(parent, end_tick=tick, reason=REASON_MANUAL,
                       reason_values=parent.reason_values)
        left.note_indices = left_indices
        right = replace(parent, start_tick=tick, reason=REASON_MANUAL,
                        reason_values=parent.reason_values)
        right.note_indices = right_indices
        self.segments[index : index + 1] = [left, right]
        self.sort_segments()
        return right

    def _halves_at(
        self, parent: VelocitySegment, tick: int
    ) -> Optional[tuple[tuple[int, ...], tuple[int, ...]]]:
        """The note indices each half of ``parent`` would own at ``tick``.

        ``None`` when the tick would starve one half, i.e. when it is a dead
        spot: no note ends by then, or none starts from then, inside the parent.
        """
        owned = set(parent.note_indices)
        left = tuple(n.index for n in self.notes if n.end_tick <= tick and n.index in owned)
        right = tuple(n.index for n in self.notes if n.start_tick >= tick and n.index in owned)
        if not left or not right:
            return None
        return left, right

    def _nearest_usable_edge(
        self, parent: VelocitySegment, tick: int
    ) -> Optional[int]:
        """Closest tick to ``tick`` that really cuts ``parent`` in two."""
        owned = set(parent.note_indices)
        edges = {
            e
            for n in self.notes
            if n.index in owned
            for e in (n.start_tick, n.end_tick)
            if parent.start_tick < e < parent.end_tick
        }
        for edge in sorted(edges, key=lambda e: (abs(e - tick), e)):
            if self._halves_at(parent, edge) is not None:
                return edge
        return None

    def resolved_notes(self) -> list[Note]:
        """The track's notes with every segment's adjustment applied."""
        out: dict[int, Note] = {}
        lookup = self.note_by_index()
        for seg in self.segments:
            for note in self._notes_in(seg, lookup):
                out[note.index] = Note(
                    note.index, note.channel, note.pitch,
                    seg.velocity_for(note), note.start_tick, note.end_tick,
                )
        return sorted(out.values(), key=lambda n: n.sort_key)

    def _get(self, index: int) -> VelocitySegment:
        if not -len(self.segments) <= index < len(self.segments):
            raise SegmentError(f"segment index {index} out of range")
        return self.segments[index]


def _merge_values(a: VelocitySegment, b: VelocitySegment) -> tuple[int, ...]:
    values = _pattern(list(a.reason_values) + list(b.reason_values))
    return tuple(values)


def build_tracks(
    midi_path: Path | str,
    *,
    ticks_per_beat: Optional[int] = None,
    options: Optional[AutoSegmentOptions] = None,
) -> list[SegmentedTrack]:
    """Read a MIDI file and auto-segment every track.

    ``ticks_per_beat`` defaults to the file's own value -- the auto cut works in
    ticks, so a wrong default here silently scales the periodic window.
    """
    tpb = int(mido.MidiFile(str(midi_path)).ticks_per_beat) if ticks_per_beat is None \
        else int(ticks_per_beat)
    # The production defaults carry the two practical floors; ``auto_segment``
    # itself stays neutral so single-rule tests can isolate one behaviour at a
    # time.
    opts = options or AutoSegmentOptions(min_segment_notes=8, min_segment_beats=4.0)
    notes_by_track = read_notes(midi_path)
    out = []
    for i, notes in enumerate(notes_by_track):
        out.append(
            SegmentedTrack(
                track=i,
                segments=auto_segment(notes, i, ticks_per_beat=tpb, options=opts),
                notes=list(notes),
            )
        )
    return out


# --------------------------------------------------------------------------- #
# sidecar JSON
# --------------------------------------------------------------------------- #
@dataclass
class SegmentProject:
    """A MIDI file plus its velocity segments, as stored in the sidecar."""

    source: Path
    ticks_per_beat: int = 480
    tracks: list[SegmentedTrack] = field(default_factory=list)
    #: Non-fatal problems found while reading a sidecar (see :meth:`verify_notes`).
    #: Kept on the project instead of raised: the edits already in the file are
    #: still the user's, and silently discarding them would be worse.  The MIDI
    #: export does its own cross-check of the same kind and refuses to write.
    warnings: list[str] = field(default_factory=list)
    #: track index -> edited sustain-pedal spans.  A track that is *absent* keeps
    #: whatever CC64 the file already wrote; a track that is *present* (even with
    #: an empty list) has its CC64 replaced by the spans.  See qmidi.pedal for why
    #: "replace" is the only rule that does not leave two pedals fighting.
    pedals: dict[int, list[PedalSpan]] = field(default_factory=dict)
    #: Playback tempo in BPM.  ``None`` = use the file's own tempo map, which is
    #: the only honest default: a sidecar written before this field existed must
    #: not silently play the piece at a different speed.
    bpm: Optional[float] = None
    #: MIDI channel -> stereo position (CC10, 0..127; 64 is centre).  Keyed by
    #: *channel*, not by track, because that is what a pan message addresses: the
    #: two pianos of a two-piano reduction usually sit on two tracks each (one
    #: holding the notes, one the pedal) and panning them individually would
    #: split a single instrument in half.
    pans: dict[int, int] = field(default_factory=dict)
    #: Optional mastering settings for the MP3 export (see qmidi.mastering).
    #: Stored as a plain dict so this module keeps no audio dependency and an old
    #: reader can ignore the key.
    master: Optional[dict] = None

    def track(self, index: int) -> SegmentedTrack:
        for t in self.tracks:
            if t.track == index:
                return t
        raise SegmentError(f"no track {index} in project")

    def pedal_source(self, index: int) -> Optional[int]:
        """The track whose pedal spans sound for track ``index``.

        Own edits win; otherwise the pedal is a *channel* thing and a
        transcription that split one piano over two tracks (notes here, CC64
        there) holds the sibling track through its channel-mate's spans.  The
        display shows that shared pedal, while editing must keep landing on the
        single carrier: the export writes one CC64 stream per channel, and two
        streams on one channel would fight each other.
        """
        index = int(index)
        if index in self.pedals:
            # Own edits exist even when the list is empty: an explicit clear is
            # a statement ("no pedal here") that a channel-mate's pedal must not
            # override on screen.
            return index
        try:
            channels = {n.channel for n in self.track(index).notes}
        except SegmentError:
            return None
        for track_no, spans in self.pedals.items():
            if track_no == index or not spans:
                continue  # an empty sibling expresses nothing to share
            try:
                if {n.channel for n in self.track(track_no).notes} & channels:
                    return track_no
            except SegmentError:
                continue
        return None

    def pedal_spans(self, index: int) -> list[PedalSpan]:
        """The pedal sounding for track ``index`` (own, else the channel mate's)."""
        source = self.pedal_source(index)
        return list(self.pedals[source]) if source is not None else []

    def set_pedal_spans(self, index: int, spans: Iterable[PedalSpan]) -> list[PedalSpan]:
        """Replace the track's pedal edits with (normalised) ``spans``."""
        cleaned = normalise_pedal(spans)
        key = int(index)
        if cleaned:
            self.pedals[key] = cleaned
        else:
            # An empty list still marks the track as "edited": the file's own
            # CC64 must be removed, which is what "no pedal here" means.
            self.pedals[key] = []
        return cleaned

    def pan(self, channel: int) -> Optional[int]:
        """Stereo position of ``channel``, or ``None`` for "leave the file alone"."""
        return self.pans.get(int(channel))

    def set_pan(self, channel: int, value: Optional[int]) -> Optional[int]:
        """Set (or with ``None`` clear) one channel's stereo position.

        ``0`` is hard left, ``127`` hard right, ``64`` centre -- the raw CC10
        scale, because the GUI shows the same number the MIDI file will carry.
        """
        channel = int(channel)
        if not 0 <= channel <= 15:
            raise SegmentError(f"MIDI channel must be 0..15, got {channel}")
        if value is None:
            self.pans.pop(channel, None)
            return None
        pan = int(value)
        if not 0 <= pan <= 127:
            raise SegmentError(f"pan must be 0..127, got {pan}")
        self.pans[channel] = pan
        return pan

    def set_master(self, settings: Optional[dict]) -> None:
        """Store (or clear) the MP3 mastering settings."""
        self.master = None if settings is None else dict(settings)

    @property
    def segment_count(self) -> int:
        return sum(len(t.segments) for t in self.tracks)

    def verify_notes(self) -> list[str]:
        """Describe every way the sidecar stopped matching the MIDI it points at.

        A segment refers to notes by index, so a MIDI edited after the sidecar
        was saved does not *error* -- it silently shifts every index after the
        first added or removed note, applying the wrong adjustment to the wrong
        note.  That is exactly the failure the exporters' own index
        cross-check guards against on export; this is the same check, available
        at load time, on the loaded project.
        """
        problems = list(self.warnings)
        for t in self.tracks:
            covered = {i for s in t.segments for i in s.note_indices}
            unknown = sorted(covered - {n.index for n in t.notes})
            if unknown:
                problems.append(
                    f"track {t.track}: {len(unknown)} segment(s) refer to note "
                    f"index/indices {unknown[:8]} the MIDI file does not hold"
                )
            if not t.notes and t.segments:
                problems.append(
                    f"track {t.track}: the MIDI file holds no notes but the sidecar "
                    f"describes {len(t.segments)} segment(s)"
                )
            orphans = {n.index for n in t.notes} - covered
            if orphans:
                problems.append(
                    f"track {t.track}: {len(orphans)} note(s) of the MIDI file "
                    f"(index/indices {sorted(orphans)[:8]}) belong to no segment; "
                    f"the sidecar was saved against a different version of the file"
                )
        return problems

    # -- serialisation ----------------------------------------------------- #
    def to_dict(self) -> dict:
        return {
            "format": SIDECAR_FORMAT,
            "version": SIDECAR_VERSION,
            "source": self.source.name,
            "ticks_per_beat": self.ticks_per_beat,
            "bpm": self.bpm,
            "tracks": [
                {
                    "index": t.track,
                    "name": t.name,
                    "segments": [
                        {
                            "start_tick": s.start_tick,
                            "end_tick": s.end_tick,
                            "delta": s.delta,
                            "target": s.target,
                            "scale": s.scale,
                            "reason": s.reason,
                            "reason_values": list(s.reason_values),
                            "note_indices": list(s.note_indices),
                        }
                        for s in t.segments
                    ],
                }
                for t in self.tracks
            ],
            # Additive key, not a format version bump: a reader that predates
            # pedal editing ignores it, and a reader that knows it finds the key
            # missing in old files (which correctly means "no pedal edits").
            "pedals": {
                str(index): [span.to_dict() for span in spans]
                for index, spans in sorted(self.pedals.items())
            },
            # Additive keys, same rule as "pedals" above: a reader that predates
            # them ignores the key, an old file simply lacks it.
            "pans": {str(ch): int(value) for ch, value in sorted(self.pans.items())},
            "master": dict(self.master) if self.master else None,
        }

    @classmethod
    def from_dict(
        cls,
        data: dict,
        source: Path | str,
        notes_by_track: Optional[list[list[Note]]] = None,
    ) -> "SegmentProject":
        if data.get("format") != SIDECAR_FORMAT:
            raise SegmentError(f"not a {SIDECAR_FORMAT} sidecar: {data.get('format')!r}")
        version = data.get("version")
        if version != SIDECAR_VERSION:
            raise SegmentError(
                f"sidecar version {version!r} != supported {SIDECAR_VERSION}"
            )
        tpb = int(data.get("ticks_per_beat", 480))
        if notes_by_track is None:
            notes_by_track = read_notes(source)
        tracks = []
        warnings: list[str] = []
        for entry in data.get("tracks", []):
            index = int(entry["index"])
            if index >= len(notes_by_track):
                warnings.append(
                    f"the sidecar describes track {index} but the MIDI file only "
                    f"holds {len(notes_by_track)} track(s); it was saved against a "
                    f"different version of the file"
                )
                notes: list[Note] = []
            else:
                notes = list(notes_by_track[index])
            segments = [
                VelocitySegment(
                    track=index,
                    start_tick=int(s["start_tick"]),
                    end_tick=int(s["end_tick"]),
                    delta=int(s.get("delta", 0)),
                    target=None if s.get("target") is None else int(s["target"]),
                    # Older sidecars have no scale; 1.0 is the identity.
                    scale=float(s.get("scale", 1.0)),
                    reason=str(s.get("reason", REASON_MANUAL)),
                    reason_values=tuple(s.get("reason_values", ())),
                    note_indices=tuple(s.get("note_indices", ())),
                )
                for s in entry.get("segments", [])
            ]
            tracks.append(SegmentedTrack(track=index, name=entry.get("name", ""),
                                         notes=notes, segments=segments))
        pedals: dict[int, list[PedalSpan]] = {}
        for key, values in (data.get("pedals") or {}).items():
            pedals[int(key)] = [PedalSpan.from_dict(item) for item in values]
        raw_bpm = data.get("bpm")
        try:
            bpm = None if raw_bpm is None else float(raw_bpm)
        except (TypeError, ValueError):
            bpm = None
        if bpm is not None and not 1.0 <= bpm <= 1000.0:
            bpm = None  # a nonsense value reads as "unset", not as a broken file
        pans: dict[int, int] = {}
        for key, value in (data.get("pans") or {}).items():
            try:
                channel, pan = int(key), int(value)
            except (TypeError, ValueError):
                continue
            if 0 <= channel <= 15 and 0 <= pan <= 127:
                pans[channel] = pan
        raw_master = data.get("master")
        master = dict(raw_master) if isinstance(raw_master, dict) else None
        project = cls(source=Path(source), ticks_per_beat=tpb, tracks=tracks,
                      pedals=pedals, bpm=bpm, pans=pans, master=master)
        project.warnings.extend(project.verify_notes())
        return project

    def save(self, path: Optional[Path | str] = None) -> Path:
        target = Path(path) if path is not None else sidecar_path(self.source)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(
        cls,
        path: Path | str,
        notes_by_track: Optional[list[list[Note]]] = None,
    ) -> "SegmentProject":
        sidecar = Path(path)
        midi = sidecar_path_to_midi(sidecar)
        data = json.loads(sidecar.read_text(encoding="utf-8"))
        # The sidecar usually sits next to its .mid; if it was moved away on its
        # own, fall back to the name it recorded *next to the sidecar*.  A bare
        # name with no suffix (`proj.json`) used to resolve to itself, which is
        # never a MIDI file.
        candidates = [midi]
        recorded = str(data.get("source") or "").strip()
        if recorded:
            recorded_path = sidecar.parent / recorded
            if recorded_path != midi:
                candidates.append(recorded_path)
        source = next((c for c in candidates if c.is_file()), midi)
        if notes_by_track is None:
            # A missing MIDI file is a warning, not a crash: the sidecar still
            # describes the user's edits and can be shown to them.
            notes_by_track = read_notes(source) if source.is_file() else []
        return cls.from_dict(data, source, notes_by_track)


def sidecar_path(midi_path: Path | str) -> Path:
    """``song.mid`` -> ``song.mid.qmidi.json`` (never overwrites the .mid)."""
    p = Path(midi_path)
    return p.with_name(p.name + SIDECAR_SUFFIX)


def sidecar_path_to_midi(sidecar: Path | str) -> Path:
    p = Path(sidecar)
    name = p.name
    if name.endswith(SIDECAR_SUFFIX):
        return p.with_name(name[: -len(SIDECAR_SUFFIX)])
    return p


def load_project(
    midi_path: Path | str,
    *,
    ticks_per_beat: int = 480,
    options: Optional[AutoSegmentOptions] = None,
) -> SegmentProject:
    """Auto-segment, but prefer an existing sidecar if one is already on disk."""
    midi = Path(midi_path)
    sidecar = sidecar_path(midi)
    if sidecar.is_file():
        return SegmentProject.load(sidecar)
    tracks = build_tracks(midi, ticks_per_beat=ticks_per_beat, options=options)
    return SegmentProject(source=midi, ticks_per_beat=ticks_per_beat, tracks=tracks)