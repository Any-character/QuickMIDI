"""Editor state that must work without a window.

Everything in this module is Qt-free on purpose (AGENTS.md §4.5): selection,
mute/solo, undo, and "what should play when the user hits space" are the parts
that are easy to get wrong and cheap to test.  The widgets in the sibling
modules only paint this state and forward clicks into it.

The one rule that shapes the whole class: **the selection is the loop.**  A
user listening to a segment over and over is the core workflow this program
exists for, so selecting a segment also arms it as the playback region, unless
they have dragged an explicit region (then the drag wins until they select
again).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Optional

import mido

from ..i18n import tr
from ..offline import TickClock, resolve_track_filter
from ..pedal import PedalSpan, add_span, normalise, read_pedal, remove_at, span_at
from ..segments import (
    VEL_MAX,
    VEL_MIN,
    SegmentProject,
    SegmentedTrack,
    VelocitySegment,
    build_tracks,
    clamp_velocity,
    sidecar_path,
)

#: How many bytes of edit history to keep.  Snapshots are tiny (a few kB) and
#: a user adjusting a hundred segments wants all hundred undos.
UNDO_DEPTH = 200

#: Velocity granularity of one drag click in the velocity lane.  Twelve clicks
#: span ±36 -- coarse enough that a piano passage moves in audible steps
#: (3-6-9, as the user asked for) and fine enough never to overshoot 127.
VEL_DRAG_STEP = 3


@dataclass(frozen=True)
class TrackStats:
    """One row of the track panel."""

    track: int
    name: str
    notes: int
    segments: int
    velocity_low: int
    velocity_high: int


class Session:
    """A MIDI file, its segments, and everything the window remembers about it."""

    def __init__(
        self,
        project: SegmentProject,
        clock: TickClock,
        *,
        beats_per_bar: int = 4,
        active_track: int = 0,
    ) -> None:
        self.project = project
        #: The file's own tempo map, kept so ``set_bpm(None)`` can restore it.
        self._base_clock = clock
        self.clock = self._clock_for(project.bpm, clock)
        self.beats_per_bar = max(1, int(beats_per_bar))
        self.active_track = active_track
        #: track index -> selected segment index.  Kept per track so switching
        #: tracks back and forth does not lose the user's place.
        self.selection: dict[int, int] = {}
        self.muted: set[int] = set()
        self.solo: set[int] = set()
        self.cursor_tick = 0
        #: An explicitly dragged playback region; ``None`` means "follow the
        #: selection".  Cleared the moment the user selects a segment.
        self.manual_loop: Optional[tuple[int, int]] = None
        #: Loop playback off by default -- the transport behaves like a DAW:
        #: drop the cursor anywhere, play to the end, stop there.  The loop
        #: switch in the toolbar turns the selected segment into a repeat.
        self.loop_enabled = False
        #: ``(segment index, base delta, base target)`` while the velocity lane
        #: is being dragged; ``None`` when no gesture is in flight.
        self._velocity_drag: Optional[tuple[int, int, Optional[int]]] = None
        self.dirty = False
        #: Bumped on every mutation so painted caches can invalidate cheaply.
        self.version = 0
        self._undo: list[dict] = []
        self._redo: list[dict] = []
        #: The pedal the *file* already has, loaded once.  The project keeps only
        #: the user's edits; showing the file's spans as the starting point is
        #: what makes the first drawn span extend the existing pedal instead of
        #: replacing the whole track with it.
        self._file_pedals = read_pedal(project.source)
        self._select_first_segment()

    # -- construction ------------------------------------------------------ #
    @classmethod
    def open(cls, midi_path: Path | str) -> "Session":
        """Load a MIDI file, preferring an existing sidecar over auto-segmenting."""
        midi = Path(midi_path)
        if not midi.is_file():
            raise FileNotFoundError(f"MIDI not found: {midi}")
        sidecar = sidecar_path(midi)
        if sidecar.is_file():
            project = SegmentProject.load(sidecar)
        else:
            tpb = mido.MidiFile(str(midi)).ticks_per_beat
            tracks = build_tracks(midi, ticks_per_beat=tpb)
            project = SegmentProject(source=midi, ticks_per_beat=tpb, tracks=tracks)
        _apply_track_names(project, midi)
        return cls(project, TickClock.from_midi(midi),
                   beats_per_bar=_beats_per_bar(midi))

    # -- read-only views --------------------------------------------------- #
    @property
    def track_count(self) -> int:
        return len(self.project.tracks)

    def active(self) -> SegmentedTrack:
        """The track the selection/cursor act on."""
        for track in self.project.tracks:
            if track.track == self.active_track:
                return track
        raise IndexError(f"no track {self.active_track}")

    def selected_index(self, track: Optional[int] = None) -> Optional[int]:
        index = self.selection.get(self.active_track if track is None else track)
        if index is None:
            return None
        target = self.active() if track is None else self.project.track(track)
        return index if 0 <= index < len(target.segments) else None

    def selected_segment(self) -> Optional[VelocitySegment]:
        index = self.selected_index()
        return None if index is None else self.active().segments[index]

    def visible_tracks(self) -> frozenset[int]:
        """The tracks a render should include, given mute/solo."""
        return resolve_track_filter(self.track_count, self.muted, self.solo)

    def total_end_tick(self) -> int:
        return max((t.end_tick for t in self.project.tracks), default=0)

    #: Playhead snap, as parts per beat (4 = sixteenth notes).  The window copies
    #: it from the settings; 0 means the cursor can land anywhere.
    snap_division: int = 4

    # -- tempo -------------------------------------------------------------- #
    @staticmethod
    def _clock_for(bpm: Optional[float], base: TickClock) -> TickClock:
        """The clock to use for ``bpm`` -- the file's own map when it is unset."""
        if bpm is None:
            return base
        return base.scaled(base.bpm / float(bpm))

    @property
    def bpm(self) -> float:
        """Current tempo in BPM (the file's own when the user has not changed it)."""
        return self.clock.bpm

    def set_bpm(self, bpm: Optional[float]) -> bool:
        """Change the playback tempo; ``None`` restores the file's own.

        The whole tempo map is scaled, so a ritardando stays one.  Every tick in
        the project (segments, cursor, loop) is unaffected: ticks are musical
        positions, only their duration in seconds changes.
        """
        if bpm is not None:
            bpm = float(bpm)
            if not 20.0 <= bpm <= 300.0:
                raise ValueError(f"bpm must be in [20, 300], got {bpm!r}")
            # Typing the file's own tempo back should clear the override, not
            # freeze a rounded copy of it into the sidecar.
            if abs(bpm - self._base_clock.bpm) < 1e-9:
                bpm = None
        if self.project.bpm == bpm:
            return False
        self._push_undo()
        self.project.bpm = bpm
        self.clock = self._clock_for(bpm, self._base_clock)
        self._touch()
        return True

    def snap_tick(self, tick: float) -> int:
        """Round ``tick`` to the playhead grid.

        Only *placing* the cursor snaps.  The playhead that follows audio must
        not, or it would visibly stutter from one grid line to the next while
        playing -- the one place a snapped playhead reads as a bug.
        """
        value = max(0, int(tick))
        division = int(self.snap_division)
        if division <= 0:
            return value
        step = max(1, self.project.ticks_per_beat // division)
        # ``+ 0.5`` rather than ``round``: Python rounds halves to even, which
        # made a click exactly between two grid lines land on the *earlier* one
        # half the time -- unexplainable from the outside.
        return int(value / step + 0.5) * step

    def selected_velocity_range(self) -> Optional[tuple[int, int]]:
        """Lowest and highest velocity the selected segment produces right now.

        Shown in the velocity lane's gutter, so the user can read the number
        instead of guessing a bar height.
        """
        segment = self.selected_segment()
        if segment is None:
            return None
        try:
            lookup = self.active().note_by_index()
        except IndexError:
            return None
        values = [segment.velocity_for(lookup[i])
                  for i in segment.note_indices if i in lookup]
        if not values:
            return None
        return min(values), max(values)

    def stats(self) -> list[TrackStats]:
        """Per-track numbers for the panel, including the *current* velocities.

        Resolved (not original) values on purpose: after a nudge the row must
        show what the track now sounds like, otherwise the panel keeps
        advertising velocities the user just changed.
        """
        out = []
        for track in self.project.tracks:
            lookup = track.note_by_index()
            values = [
                segment.velocity_for(lookup[index])
                for segment in track.segments
                for index in segment.note_indices
                if index in lookup
            ]
            out.append(TrackStats(
                track=track.track,
                name=track.name or tr("轨道 {track}").format(track=track.track),
                notes=len(track.notes),
                segments=len(track.segments),
                velocity_low=min(values, default=0),
                velocity_high=max(values, default=0),
            ))
        return out

    # -- selection --------------------------------------------------------- #
    def select(self, track: int, index: Optional[int], *, move_cursor: bool = False) -> None:
        """Select segment ``index`` of ``track``; it becomes the loop region."""
        self.active_track = int(track)
        if index is None:
            self.selection.pop(self.active_track, None)
            return
        segments = self.project.track(self.active_track).segments
        if not segments:
            return
        index = max(0, min(int(index), len(segments) - 1))
        self.selection[self.active_track] = index
        self.manual_loop = None
        if move_cursor:
            self.cursor_tick = segments[index].start_tick

    def select_at_tick(self, tick: int) -> Optional[int]:
        """Select whatever segment of the active track covers ``tick``."""
        index = self.active().index_of_tick(max(0, int(tick)))
        if index is not None:
            self.select(self.active_track, index)
        return index

    def step_selection(self, delta: int) -> Optional[int]:
        """Move the selection by ``delta`` segments, clamped at both ends."""
        segments = self.active().segments
        if not segments:
            return None
        index = self.selected_index()
        if index is None:
            index = 0
        else:
            index = max(0, min(index + int(delta), len(segments) - 1))
        self.select(self.active_track, index, move_cursor=True)
        return index

    def _select_first_segment(self) -> None:
        for track in self.project.tracks:
            if track.segments:
                self.active_track = track.track
                self.selection[track.track] = 0
                return

    # -- playback region --------------------------------------------------- #
    def loop_ticks(self) -> Optional[tuple[int, int]]:
        """The region the play button will use, or ``None`` if empty."""
        if self.manual_loop is not None:
            start, end = self.manual_loop
            if end > start:
                return (start, end)
        segment = self.selected_segment()
        if segment is None:
            return None
        return (segment.start_tick, segment.end_tick)

    def loop_seconds(self) -> Optional[tuple[float, float]]:
        span = self.loop_ticks()
        if span is None:
            return None
        start, end = self.clock.seconds(span[0]), self.clock.seconds(span[1])
        if end <= start:  # a sub-block segment can round to zero length
            end = start + 0.05
        return (start, end)

    def playback_span_ticks(self) -> tuple[int, int]:
        """What the play button covers right now.

        Loop on: the selected segment (or a dragged region).  Loop off: from the
        cursor to the end of the file -- "drop the cursor anywhere, play, stop
        at the end" is the transport model the user asked for, and it is the
        opposite of the old "every play is a loop" behaviour.
        """
        if self.loop_enabled:
            span = self.loop_ticks()
            if span is not None:
                return span
        start = max(0, int(self.cursor_tick))
        return start, max(self.total_end_tick(), start + 1)

    def playback_span_seconds(self) -> tuple[float, float]:
        start_tick, end_tick = self.playback_span_ticks()
        start, end = self.clock.seconds(start_tick), self.clock.seconds(end_tick)
        if end <= start:
            end = start + 0.05
        return (start, end)

    def set_manual_loop(self, start_tick: int, end_tick: int) -> None:
        start, end = int(start_tick), int(end_tick)
        if end < start:
            start, end = end, start
        if end - start < 1:
            end = start + 1
        self.manual_loop = (max(0, start), max(0, end))

    # -- ruler / timecode --------------------------------------------------- #
    def bar_beat_label(self, tick: int) -> str:
        """``第 34 小节 · 2 拍`` -- the unit a musician navigates by."""
        tpb = max(1, self.project.ticks_per_beat)
        bar_ticks = tpb * self.beats_per_bar
        position = max(0, int(tick))
        return tr("第 {bar} 小节 · {beat} 拍").format(
            bar=position // bar_ticks + 1,
            beat=position % bar_ticks // tpb + 1)

    @staticmethod
    def timecode(seconds: float) -> str:
        """``1:23.4`` -- minutes, then seconds to a tenth."""
        total = max(0.0, float(seconds))
        minutes = int(total // 60)
        return f"{minutes}:{total - minutes * 60:04.1f}"

    def position_label(self, seconds: float) -> str:
        """The status-bar readout: bar/beat plus elapsed/total."""
        total = self.clock.seconds(self.total_end_tick())
        return (f"{self.bar_beat_label(self.clock.tick(seconds))}  ·  "
                f"{self.timecode(seconds)} / {self.timecode(total)}")

    # -- edits (all undoable) ---------------------------------------------- #
    def set_delta(self, delta: int) -> bool:
        return self._edit("apply_delta", int(delta))

    def set_target(self, velocity: int) -> bool:
        return self._edit("set_value", clamp_velocity(velocity))

    def clear_adjustment(self) -> bool:
        return self._edit("clear_adjustment")

    def merge_with_next(self) -> bool:
        return self._edit("merge_adjacent")

    def split_at_cursor(self) -> bool:
        index = self.selected_index()
        if index is None:
            return False
        self._push_undo()
        try:
            self.active().split_at_tick(self.cursor_tick)
        except Exception:
            self._undo.pop()
            raise
        self._touch()
        return True

    def apply_to_all_segments(self, delta: int) -> bool:
        """Nudge every segment of the active track -- the "whole part is too loud" case."""
        segments = self.active().segments
        if not segments:
            return False
        self._push_undo()
        for i in range(len(segments)):
            self.active().apply_delta(i, int(delta))
        self._touch()
        return True

    def _edit(self, method: str, *args) -> bool:
        index = self.selected_index()
        if index is None:
            return False
        self._push_undo()
        getattr(self.active(), method)(index, *args)
        self._touch()
        return True

    # -- tracks ------------------------------------------------------------- #
    def rename_track(self, track: int, name: str) -> bool:
        """Rename one track.  Blank restores the name stored in the MIDI file.

        The name is part of the sidecar, so a rename survives reopening; a blank
        name is not stored as blank (that would leave an unnamed row), it falls
        back to ``track_name`` from the file.
        """
        cleaned = (name or "").strip()
        try:
            target = self.project.track(track)
        except (IndexError, KeyError):
            return False
        if target.name == cleaned:
            return False
        self._push_undo()
        target.name = cleaned
        if not cleaned:
            _apply_track_names(self.project, self.project.source)
        self._touch()
        return True

    # -- resegmentation ----------------------------------------------------- #
    def resegment(self) -> bool:
        """Re-run the auto cut with the current rules, dropping adjustments.

        Adjustments are dropped on purpose: a stored delta belongs to the *old*
        note grouping, and carrying it across a re-cut would silently move
        somebody's work onto different notes.  ``_push_undo`` puts it back.
        """
        tracks = build_tracks(self.project.source,
                              ticks_per_beat=self.project.ticks_per_beat)
        if not any(track.segments for track in tracks):
            return False
        renamed = {track.track: track.name for track in self.project.tracks if track.name}
        self._push_undo()
        self.project.tracks = tracks
        _apply_track_names(self.project, self.project.source)
        # A name the user typed must not be thrown away by re-cutting: the cut
        # is about segments, the name is about the track.
        for track in self.project.tracks:
            if renamed.get(track.track):
                track.name = renamed[track.track]
        self.selection.clear()
        self._select_first_segment()
        self._touch()
        return True

    # -- pedal / pan / mastering -------------------------------------------- #
    def auto_pedal(self) -> int:
        """Re-take the sustain pedal from the harmony; returns tracks touched.

        Generated **per MIDI channel**, not per track: the two pianos of a
        two-piano reduction sit on one channel each but often span two tracks
        (one with the notes, one with the pedal), and generating per track would
        give the pedal its own, wrong, idea of the harmony.

        The spans land in the track that already carries the pedal for that
        channel, so an edited file keeps its structure; a channel with no pedal
        track gets them on its first note track.  Undoable in one step.
        """
        from ..pedal import read_pedal
        from ..pedal_auto import AutoPedalOptions, generate_spans

        by_channel: dict[int, list] = {}
        tracks_of: dict[int, list[int]] = {}
        for seg_track in self.project.tracks:
            for note in seg_track.notes:
                by_channel.setdefault(int(note.channel), []).append(note)
            if seg_track.notes:
                channel = int(seg_track.notes[0].channel)
                tracks_of.setdefault(channel, []).append(int(seg_track.track))
        if not by_channel:
            return 0

        # Which track the *file* puts its CC64 on: that is where the new spans
        # have to go, or the original pedal survives next to them and the two
        # streams fight (see qmidi.pedal -- replace, never append).
        try:
            file_pedals = set(read_pedal(self.project.source))
        except Exception:
            file_pedals = set()

        plan: dict[int, list] = {}
        clear: set[int] = set()
        for channel, notes in by_channel.items():
            candidates = tracks_of.get(channel, [])
            carriers = [t for t in candidates if t in file_pedals]
            edited = [t for t in candidates if t in self.project.pedals]
            target = carriers[0] if carriers else (edited[0] if edited else
                                                   (candidates[0] if candidates else None))
            if target is None:
                continue
            plan[target] = generate_spans(
                notes,
                ticks_per_beat=self.project.ticks_per_beat,
                end_tick=max(int(n.end_tick) for n in notes),
                options=AutoPedalOptions(),
            )
            # One pedal per channel: any other track of this channel that had
            # pedal edits loses them, so the export cannot write two CC64 sets.
            clear.update(t for t in candidates if t != target)

        if not plan:
            return 0
        self._push_undo()
        for track in clear:
            self.project.pedals.pop(track, None)
        for track, spans in plan.items():
            self.project.set_pedal_spans(track, spans)
        self._touch()
        return len(plan)

    def set_channel_pans(self, pans: dict[int, Optional[int]]) -> bool:
        """Set (or clear with ``None``) the stereo position of several channels.

        Undoable like every other edit: the other two buttons on this row are,
        and a pan you cannot take back is one you will not dare try.
        """
        if all(self.project.pan(channel) == value for channel, value in pans.items()):
            return False
        self._push_undo()
        for channel, value in pans.items():
            self.project.set_pan(channel, value)
        self._touch()
        return True

    def set_master(self, settings: Optional[dict]) -> bool:
        """Turn the MP3 mastering chain on/off (or change its settings)."""
        if (self.project.master or None) == (settings or None):
            return False
        self._push_undo()
        self.project.set_master(settings)
        self._touch()
        return True

    def channels(self) -> list[int]:
        """MIDI channels that hold notes, in ascending order."""
        return sorted({int(note.channel) for t in self.project.tracks for note in t.notes})

    # -- velocity lane (one gesture = one undo step) ------------------------ #
    def begin_velocity_drag(self, reference: int = 0) -> bool:
        """Snapshot before a lane drag; ``reference`` is the grabbed note's velocity.

        The reference is the note's *raw MIDI* velocity: the drag is absolute
        ("put this note at 50"), so the scale is always derived from the original
        value and repeated drags cannot compound into drift.
        """
        index = self.selected_index()
        if index is None:
            self._velocity_drag = None
            return False
        segment = self.selected_segment()
        self._push_undo()
        self._velocity_drag = (index, segment.scale, segment.delta, segment.target,
                               max(0, int(reference)))
        return True

    def drag_velocity(self, steps: int) -> bool:
        """Move the grabbed note by ``steps`` velocity units, scaling the segment.

        ``steps`` is absolute for the gesture -- pixels since mouse-down, not
        since the last move; a relative delta per mouse-move would multiply one
        drag by however many events the OS happens to send.

        Every other note follows *proportionally*: a segment holding 100 and 90,
        dragged to 50, becomes 50 and 45 rather than 50 and 40.  Keeping the
        internal balance is the reason the segment exists.
        """
        drag = self._velocity_drag
        if drag is None:
            return False
        index, base_scale, base_delta, base_target, reference = drag
        track = self.active()
        segment = track.segments[index]
        shift = int(steps)
        if base_target is not None:
            # A pinned segment already gives every note the same value, so
            # proportional and additive are the same operation here.
            track.segments[index] = replace(
                segment, target=clamp_velocity(base_target + shift))
        elif reference <= 0:
            # Nothing under the cursor: fall back to an additive drag rather than
            # silently doing nothing.
            track.segments[index] = replace(segment, delta=base_delta + shift)
        else:
            shown = reference * base_scale + base_delta
            target = max(float(VEL_MIN), min(float(VEL_MAX), shown + shift))
            track.segments[index] = replace(
                segment, scale=target / reference, delta=0)
        self._touch()
        return True

    def end_velocity_drag(self) -> bool:
        """Finish a lane drag; one that moved nothing leaves no undo step."""
        drag = self._velocity_drag
        self._velocity_drag = None
        if drag is None:
            return False
        _, base_scale, base_delta, base_target, _ = drag
        segment = self.selected_segment()
        moved = segment is not None and (
            segment.delta != base_delta
            or segment.target != base_target
            or abs(segment.scale - base_scale) > 1e-9)
        if not moved:
            if self._undo:
                self._undo.pop()
                self._redo.clear()
            return False
        return True

    # -- pedal ------------------------------------------------------------- #
    def pedal_spans(self, track: Optional[int] = None) -> list[PedalSpan]:
        """What the pedal lane shows.

        Order matters: the project resolves *channel-shared* pedal first (a
        transcription that split one piano over two tracks holds the sibling
        through the carrier's spans -- showing the file's track here would read
        as an empty lane on a track that actually sustains), and only when
        neither the track nor its channel mate has edits do we fall back to the
        pedal the file itself carries.
        """
        index = self.active_track if track is None else track
        if index in self.project.pedals:
            # An explicit edit -- including an empty list after a clear -- is the
            # user's statement and must not be overwritten by the file's pedal.
            return list(self.project.pedals[index])
        spans = self.project.pedal_spans(index)
        if spans:
            return spans
        return list(self._file_pedals.get(index, ()))

    def _pedal_target(self) -> int:
        """The track a pedal edit must land on: own edits, else the channel carrier.

        Writing to the active track blindly would put a second CC64 stream on a
        channel that already has one; the two streams would open and close at
        different moments and the pedal would stutter.
        """
        source = self.project.pedal_source(self.active_track)
        return self.active_track if source is None else source

    def pedal_edited(self, track: Optional[int] = None) -> bool:
        index = self.active_track if track is None else track
        return index in self.project.pedals

    def add_pedal(self, start_tick: int, end_tick: int) -> bool:
        """Draw a pedal span on the active track (overlaps are unioned)."""
        spans = add_span(self.pedal_spans(), start_tick, end_tick)
        if spans == self.pedal_spans():
            return False
        self._push_undo()
        self.project.set_pedal_spans(self._pedal_target(), spans)
        self._touch()
        return True

    def toggle_pedal_at(self, tick: int) -> bool:
        """Real damper behaviour: press down and hold, press again to lift.

        A span that reaches the last note means "still held" -- what a real
        pedal does after the final note, and how the lane shows an open press.
        The old drag-a-rectangle gesture is gone: rectangles made the user think
        in boxes, while a pedal is a foot going down and coming back up.
        """
        tick = max(0, int(tick))
        spans = list(self.pedal_spans())
        hit = span_at(spans, tick)
        if hit is None:
            spans = add_span(spans, tick, max(tick + 1, self.total_end_tick()))
        else:
            spans = [span for span in spans if span != hit]
            if tick > hit.start_tick:
                spans.append(PedalSpan(hit.start_tick, tick, hit.value))
            spans = normalise(spans)
        if spans == self.pedal_spans():
            return False
        self._push_undo()
        self.project.set_pedal_spans(self._pedal_target(), spans)
        self._touch()
        return True

    def erase_pedal(self, tick: int) -> bool:
        spans, removed = remove_at(self.pedal_spans(), tick)
        if not removed:
            return False
        self._push_undo()
        self.project.set_pedal_spans(self._pedal_target(), spans)
        self._touch()
        return True

    def clear_pedal(self) -> bool:
        """Erase every pedal span on the active track (the file's own CC64 too)."""
        if not self.pedal_spans():
            return False
        self._push_undo()
        self.project.set_pedal_spans(self._pedal_target(), [])
        self._touch()
        return True

    # -- undo -------------------------------------------------------------- #
    def can_undo(self) -> bool:
        return bool(self._undo)

    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        self._redo.append(self._snapshot())
        self._restore(self._undo.pop())
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        self._undo.append(self._snapshot())
        self._restore(self._redo.pop())
        return True

    def _snapshot(self) -> dict:
        return self.project.to_dict()

    def _restore(self, data: dict) -> None:
        # Rebuild from the snapshot but keep the notes already parsed: re-reading
        # the file on every undo would be slow and would fail if the user moved it.
        notes = [list(t.notes) for t in self.project.tracks]
        warnings = list(self.project.warnings)
        self.project = SegmentProject.from_dict(data, self.project.source,
                                                notes_by_track=notes)
        self.project.warnings = warnings
        # A snapshot carries the tempo too; the clock is derived from it, so an
        # undo that restores the project without rebuilding the clock leaves the
        # piece playing at the tempo it was just undoing.
        self.clock = self._clock_for(self.project.bpm, self._base_clock)
        self._touch()

    def _push_undo(self) -> None:
        self._undo.append(self._snapshot())
        if len(self._undo) > UNDO_DEPTH:
            del self._undo[0]
        self._redo.clear()

    # -- misc -------------------------------------------------------------- #
    def toggle_mute(self, track: int) -> None:
        self.muted.symmetric_difference_update({int(track)})

    def toggle_solo(self, track: int) -> None:
        self.solo.symmetric_difference_update({int(track)})

    def _touch(self) -> None:
        self.version += 1
        self.dirty = True

    def save(self, path: Path | str | None = None) -> Path:
        written = self.project.save(path)
        self.dirty = False
        return written

    def resolved_velocity_map(self) -> dict[tuple[int, int], int]:
        """``(track, note index) -> velocity after segment adjustments``.

        The piano roll colours every note by this; computing it once per
        version beats doing it per repaint (4065 notes in the sample file).
        """
        out: dict[tuple[int, int], int] = {}
        for track in self.project.tracks:
            lookup = track.note_by_index()
            for segment in track.segments:
                for index in segment.note_indices:
                    note = lookup.get(index)
                    if note is not None:
                        out[(track.track, index)] = segment.velocity_for(note)
        return out


def _apply_track_names(project: SegmentProject, midi_path: Path | str) -> None:
    """Fill in ``track_name`` from the file.

    MuseScore exports name every track, and without this the panel shows four
    identical "轨道 0..3" rows -- the exact thing a user needs to tell apart
    (the sample file has two pianos, and only two of its tracks carry the
    pedal).
    """
    try:
        midi = mido.MidiFile(str(midi_path))
    except Exception:
        return
    names = []
    for track in midi.tracks:
        name = next((msg.name for msg in track
                     if msg.is_meta and msg.type == "track_name"), "")
        names.append(name.strip())
    for segmented in project.tracks:
        if not segmented.name and 0 <= segmented.track < len(names):
            segmented.name = names[segmented.track]


def _beats_per_bar(midi_path: Path | str) -> int:
    """Numerator of the first time signature, or 4 when the file has none."""
    try:
        midi = mido.MidiFile(str(midi_path))
    except Exception:
        return 4
    for track in midi.tracks:
        for msg in track:
            if msg.type == "time_signature" and msg.numerator > 0:
                return int(msg.numerator)
    return 4
