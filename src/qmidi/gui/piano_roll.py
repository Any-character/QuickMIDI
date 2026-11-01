"""The piano roll: notes, segment boundaries, playhead.

It is a *view*: it paints the session and translates mouse positions into
ticks, then emits signals.  It never mutates the project, which is why the
interesting behaviour lives in :mod:`qmidi.gui.model` and can be tested without
a window.

Reading the picture
-------------------
* note colour  = velocity **after** the segment adjustments (so a nudge is
  visible immediately, before you play it)
* vertical line = segment boundary on the active track; the selected segment
  has a tinted band from top to bottom
* dimmed notes  = a muted (or non-soloed) track -- context you can see but not
  hear
* red line      = playhead; clicking the ruler drops it there.  There is no
  drag-to-select any more: the loop is the selected segment, and the toolbar
  switch decides whether it repeats

Three strips, top to bottom: notes, velocity bars, pedal.  The bars are an
editor (drag any of them to move the whole selected segment); the pedal strip is
a switch (click to press down, click again to lift).
"""

from __future__ import annotations

import bisect
from typing import Optional

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetricsF,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QPolygon,
)
from PySide6.QtWidgets import QWidget

from ..i18n import tr
from ..keymap import SCOPE_ROLL
from . import lanes, theme
from .model import VEL_DRAG_STEP

RULER_H = 22
#: Left column: note names, and the velocity readout.  Wide enough for a full
#: "101-103" range at 9 pt -- at 44 px the leading digit was being clipped, which
#: reads as a wrong number rather than a narrow column.
GUTTER_W = 56
#: Both lanes are always visible: a pedal you cannot see is a pedal you forget
#: to check, and the velocity bars are the editor this program exists for.
VEL_LANE_H = lanes.VEL_LANE_H
PEDAL_LANE_H = lanes.PEDAL_LANE_H
PITCH_MIN, PITCH_MAX = 21, 108

_MIN_ROW_H = 5
_MAX_ROW_H = 26
_MIN_PPT = 0.0002
_MAX_PPT = 0.4


class PianoRoll(QWidget):
    """Notes, velocity bars and the pedal -- the whole editor canvas.

    It is a *view*: it paints the session, translates mouse positions into
    ticks, and emits intents.  It never mutates the project, which is why the
    interesting behaviour lives in :mod:`qmidi.gui.model` and can be tested
    without a window.
    """

    segmentPicked = Signal(int, int)  # track, segment index
    cursorMoved = Signal(int)  # tick
    nudgeRequested = Signal(int)  # velocity delta
    stepRequested = Signal(int)  # previous/next segment
    clearRequested = Signal()
    pedalToggled = Signal(int)  # tick: press down, or lift if already held
    pedalErased = Signal(int)  # tick inside the span to remove
    velocityDragStarted = Signal(int)  # raw velocity of the grabbed note
    velocityDragged = Signal(int)  # velocity units from mouse-down, absolute
    velocityDragFinished = Signal()
    viewChanged = Signal()  # scrolled or zoomed: the overview bar follows

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setMinimumSize(480, 280)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self._session = None
        self._notes: list[tuple[int, int, int, int, int, int]] = []
        #: Same entries grouped by track; the velocity lane only ever wants one.
        self._notes_by_track: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        #: Start ticks of each track's notes, same order, for hover lookups.
        self._track_starts: dict[int, list[int]] = {}
        #: Velocity under the mouse in the lane; None means "show the segment range".
        self._hover_velocity: Optional[int] = None
        #: The segment's velocity range, formatted once per repaint of the cache.
        self._velocity_text = ""
        self._ppt = 0.05
        #: Zoom reference: the fit-everything scale, so the toolbar can show
        #: "100%" without inventing a unit nobody understands.
        self._base_ppt = 0.05
        self._left_tick = 0.0
        self._row_h = 9
        self._top_pitch = PITCH_MAX
        self._playhead_tick = 0.0
        self._loop: Optional[tuple[int, int]] = None
        self._fitted = False
        self._pan_last: Optional[QPoint] = None
        self._vel_drag_y: Optional[float] = None
        self._keymap = None
        #: False (default): draw only the active track's notes.  Four tracks at
        #: once is noise when only one of them can be edited.
        self._show_all_tracks = False
        #: Bitmap of every layer except the playhead (see ``paintEvent``).
        self._cache: Optional[QPixmap] = None

    # -- data -------------------------------------------------------------- #
    def set_session(self, session, velocity_map: dict) -> None:
        first = self._session is not session
        self._session = session
        self.set_velocity_map(velocity_map)
        if first or not self._fitted:
            self.zoom_fit()

    def set_show_all_tracks(self, show: bool) -> None:
        """Draw every track's notes, or only the active one's."""
        show = bool(show)
        if show == self._show_all_tracks:
            return
        self._show_all_tracks = show
        if self._session is not None:
            self.set_velocity_map(self._session.resolved_velocity_map())
        self._invalidate()

    def set_velocity_map(self, velocity_map: dict) -> None:
        if self._session is None:
            self._notes = []
            return
        only = None if self._show_all_tracks else self._session.active_track
        notes = []
        for track in self._session.project.tracks:
            if only is not None and track.track != only:
                continue
            for note in track.notes:
                velocity = velocity_map.get((track.track, note.index), note.velocity)
                notes.append((track.track, note.index, note.start_tick,
                              note.end_tick, note.pitch, velocity))
        notes.sort(key=lambda item: item[2])
        self._notes = notes
        by_track: dict[int, list[tuple[int, int, int, int, int, int]]] = {}
        for entry in notes:
            by_track.setdefault(entry[0], []).append(entry)
        self._notes_by_track = by_track
        self._track_starts = {key: [entry[2] for entry in rows]
                              for key, rows in by_track.items()}
        self._invalidate()

    def set_loop(self, span: Optional[tuple[int, int]]) -> None:
        self._loop = span
        self._invalidate()

    def set_playhead_tick(self, tick: float, *, follow: bool = True) -> None:
        self._playhead_tick = float(tick)
        if follow:
            self._follow()
        self.update()

    @property
    def playhead_tick(self) -> float:
        return self._playhead_tick

    # -- view -------------------------------------------------------------- #
    def zoom_fit(self) -> None:
        if self._session is None:
            return
        area = self._note_area()
        total = max(1, self._session.total_end_tick())
        self._ppt = max(_MIN_PPT, min(_MAX_PPT, area.width() / total))
        self._base_ppt = self._ppt
        self._left_tick = 0.0
        self._fit_pitch()
        self._fitted = True
        self.viewChanged.emit()
        self._invalidate()

    @property
    def zoom_percent(self) -> int:
        """Horizontal zoom relative to "fit everything" -- what the toolbar shows."""
        return max(1, int(round(self._ppt / max(1e-9, self._base_ppt) * 100)))

    @property
    def view_left_tick(self) -> float:
        return self._left_tick

    @property
    def visible_ticks(self) -> float:
        return self._note_area().width() / max(1e-9, self._ppt)

    def zoom_time(self, factor: float) -> None:
        """Zoom the time axis, anchored on the playhead when it is on screen.

        Anchoring on the playhead is the DAW behaviour and it matters here: zoom
        around the view centre and the thing you are editing drifts out of sight
        after two clicks, which reads as "the view jumped somewhere".
        """
        area = self._note_area()
        x = self.x_of_tick(self._playhead_tick)
        if area.left() <= x <= area.right():
            anchor = self._playhead_tick
            offset = x - area.left()
        else:
            anchor = self._left_tick + area.width() * 0.5 / self._ppt
            offset = area.width() * 0.5
        self._ppt = max(_MIN_PPT, min(_MAX_PPT, self._ppt * factor))
        self._left_tick = anchor - offset / self._ppt
        self._clamp_view()
        self.viewChanged.emit()
        self._invalidate()

    def scroll_to_tick(self, tick: float) -> None:
        """Bring ``tick`` on screen with a little lead-in (never off the left)."""
        area = self._note_area()
        self._left_tick = max(0.0, float(tick) - area.width() * 0.1 / self._ppt)
        self._clamp_view()
        self.viewChanged.emit()
        self._invalidate()

    def zoom_pitch(self, factor: float) -> None:
        """Zoom the note rows (Ctrl+Shift+wheel): taller or shorter keys."""
        self._row_h = max(_MIN_ROW_H, min(_MAX_ROW_H, self._row_h * factor))
        self._clamp_view()
        self.viewChanged.emit()
        self._invalidate()

    def set_view_left(self, tick: float) -> None:
        """Scroll so ``tick`` is at the left edge (the overview bar drags this)."""
        self._left_tick = max(0.0, float(tick))
        self._clamp_view()
        self.viewChanged.emit()
        self._invalidate()

    def fit_pitch(self) -> None:
        """Re-centre the rows on the active track, leaving the time axis alone.

        Needed the moment only one track is drawn: switching from a bass part to
        a melody two octaves up would otherwise land on empty rows and look like
        the track has no notes.
        """
        if self._session is None:
            return
        self._fit_pitch()
        self._invalidate()

    def _fit_pitch(self) -> None:
        """Centre the active track's notes vertically (they rarely span all 88)."""
        if self._session is None:
            return
        try:
            track = self._session.active()
        except IndexError:
            return
        pitches = [n.pitch for n in track.notes]
        if not pitches:
            self._top_pitch = PITCH_MAX
            return
        low, high = min(pitches), max(pitches)
        area = self._note_area()
        span = max(12, high - low + 4)
        self._row_h = max(_MIN_ROW_H, min(_MAX_ROW_H, area.height() / span))
        middle = (low + high) / 2
        self._top_pitch = min(PITCH_MAX, max(PITCH_MIN, int(middle + area.height() / (2 * self._row_h))))
        self._clamp_view()

    def _clamp_view(self) -> None:
        area = self._note_area()
        visible = area.width() / self._ppt
        total = max(1, self._session.total_end_tick()) if self._session else 1
        max_left = max(0.0, total - visible * 0.15)
        self._left_tick = max(0.0, min(self._left_tick, max_left))
        rows = area.height() / self._row_h
        if rows >= PITCH_MAX - PITCH_MIN:
            self._top_pitch = float(PITCH_MAX)
            return
        low = PITCH_MIN + rows - 2
        self._top_pitch = max(low, min(float(PITCH_MAX + 2), self._top_pitch))

    def _note_area(self) -> QRect:
        used = RULER_H + VEL_LANE_H + PEDAL_LANE_H
        return QRect(GUTTER_W, RULER_H, max(1, self.width() - GUTTER_W),
                     max(1, self.height() - used))

    def _velocity_area(self) -> QRect:
        top = max(RULER_H, self.height() - PEDAL_LANE_H - VEL_LANE_H)
        return QRect(GUTTER_W, top, max(1, self.width() - GUTTER_W), VEL_LANE_H)

    def _pedal_area(self) -> QRect:
        return QRect(GUTTER_W, max(RULER_H, self.height() - PEDAL_LANE_H),
                     max(1, self.width() - GUTTER_W), PEDAL_LANE_H)

    def tick_at_x(self, x: float) -> int:
        return max(0, int(round(self._left_tick + (x - GUTTER_W) / self._ppt)))

    def x_of_tick(self, tick: float) -> float:
        return GUTTER_W + (tick - self._left_tick) * self._ppt

    def _follow(self) -> None:
        """Keep the playhead on screen while it runs (jump a page at the edge)."""
        area = self._note_area()
        x = self.x_of_tick(self._playhead_tick)
        if x < area.left() or x > area.right() - 8:
            self._left_tick = max(0.0, self._playhead_tick - area.width() * 0.15 / self._ppt)
            self._clamp_view()
            self._cache = None  # a page turn moves everything, playhead included
            self.viewChanged.emit()  # the overview window box follows the page turn

    # -- painting ---------------------------------------------------------- #
    def paintEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        """Blit the cached layers, then draw the one thing that moves.

        Measured 2026-10-25 (4000-note file, 4 tracks): repainting every layer
        costs ~200 ms per frame, which starved the audio pump -- 4.3 fps and 86%
        of wall time spent inside paint.  Nothing but the playhead changes during
        playback, so everything else is rendered once into a bitmap.
        """
        if self._session is None:
            painter = QPainter(self)
            painter.fillRect(self.rect(), theme.BG)
            painter.setPen(theme.TEXT_DIM)
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                             tr("打开一个 MIDI 文件开始（Ctrl+O）"))
            return
        if self._cache is None:
            self._cache = self._render_static()
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self._cache)
        self._paint_velocity_value(painter)
        self._paint_playhead(painter)

    def _render_static(self) -> QPixmap:
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(QSize(int(self.width() * ratio), int(self.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(theme.BG)
        painter = QPainter(pixmap)
        self._paint_static(painter)
        painter.end()
        return pixmap

    def _paint_static(self, painter: QPainter) -> None:
        area = self._note_area()
        top_pitch = int(self._top_pitch)
        tick_end = self._left_tick + area.width() / self._ppt

        self._paint_rows(painter, area, top_pitch)
        self._paint_grid(painter, area, tick_end)
        painter.setClipRect(area)
        # The selection band is a *background*: painting it after the notes
        # would veil them with the tint and make the velocity colours unreadable.
        self._paint_selection_band(painter, area)
        self._paint_notes(painter, area, top_pitch, tick_end)
        self._paint_boundaries(painter, area)
        painter.setClipping(False)
        self._paint_velocity_lane(painter, tick_end)
        self._paint_pedal_lane(painter)
        # playhead lives in _paint_playhead: it must not be baked into the cache
        self._paint_gutter(painter, area, top_pitch)
        self._paint_ruler(painter, area, tick_end)
        # Computed once per rebuild: the dynamic layer only formats a string.
        self._velocity_text = self._format_velocity(
            self._session.selected_velocity_range() if self._session else None)

    # -- velocity readout --------------------------------------------------- #
    @staticmethod
    def _format_velocity(span: Optional[tuple[int, int]]) -> str:
        if span is None:
            return ""
        low, high = span
        return str(low) if low == high else f"{low}-{high}"

    def _paint_velocity_value(self, painter: QPainter) -> None:
        """The number in the velocity gutter: hovered note, else the segment range.

        Shrinks to fit instead of clipping: a clipped "101-103" shows as
        "01-103", which reads as a different velocity rather than a narrow
        column.  The gutter is sized for the common case; this covers the rest.
        """
        text = str(self._hover_velocity) if self._hover_velocity is not None else self._velocity_text
        if not text:
            return
        lane = self._velocity_area()
        rect = QRect(2, lane.top() + lane.height() // 2 - 2,
                     lane.left() - 8, lane.height() // 2)
        font = QFont(painter.font())
        font.setPointSizeF(9.0)
        font.setBold(True)
        while font.pointSizeF() > 6.0:
            if QFontMetricsF(font).horizontalAdvance(text) <= rect.width():
                break
            font.setPointSizeF(font.pointSizeF() - 0.5)
        painter.setFont(font)
        painter.setPen(theme.TEXT if self._hover_velocity is not None else theme.TEXT_DIM)
        painter.drawText(rect, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignTop,
                         text)

    def _hit_note_at(self, x: float) -> Optional[tuple]:
        """The lane's note entry nearest ``x``, or None when none is close."""
        if self._session is None:
            return None
        try:
            track_no = self._session.active().track
        except IndexError:
            return None
        notes = self._notes_by_track.get(track_no)
        starts = self._track_starts.get(track_no)
        if not notes or not starts:
            return None
        index = bisect.bisect_left(starts, self.tick_at_x(x))
        best: Optional[int] = None
        best_dx = 4.0
        for candidate in (index - 1, index):
            if 0 <= candidate < len(starts):
                dx = abs(self.x_of_tick(starts[candidate]) - x)
                if dx <= best_dx:
                    best_dx = dx
                    best = candidate
        return None if best is None else notes[best]

    def _velocity_at(self, x: float) -> Optional[int]:
        """Velocity shown for the bar nearest ``x`` (after adjustments)."""
        note = self._hit_note_at(x)
        return None if note is None else int(note[5])

    def _original_velocity_at(self, x: float) -> int:
        """Raw MIDI velocity of the bar under ``x``: the anchor for a proportional drag."""
        note = self._hit_note_at(x)
        if note is None or self._session is None:
            return 0
        try:
            lookup = self._session.active().note_by_index()
        except IndexError:
            return 0
        original = lookup.get(note[1])
        return 0 if original is None else int(original.velocity)

    def _invalidate(self) -> None:
        """Drop the cached layers; call it whenever the picture really changes."""
        self._cache = None
        self.update()

    def _paint_rows(self, painter: QPainter, area: QRect, top_pitch: int) -> None:
        row_h = self._row_h
        first = top_pitch
        last = top_pitch - int(area.height() / row_h) - 1
        for pitch in range(min(PITCH_MAX, first), max(PITCH_MIN, last) - 1, -1):
            y = area.top() + (top_pitch - pitch) * row_h
            if theme.is_black_key(pitch):
                painter.fillRect(QRect(area.left(), int(y), area.width(), int(row_h) + 1),
                                 theme.BLACK_KEY_ROW)
            if pitch % 12 == 0:
                painter.setPen(QPen(theme.GRID_BEAT, 1))
                painter.drawLine(area.left(), int(y + row_h), area.right(), int(y + row_h))

    def _paint_grid(self, painter: QPainter, area: QRect, tick_end: float) -> None:
        tpb = self._session.project.ticks_per_beat
        bar = max(1, tpb * self._session.beats_per_bar)
        step = bar
        # Thin the grid out when a bar is narrower than 6 px, or the paint loop
        # turns into thousands of invisible lines.
        while step * self._ppt < 6:
            step *= 2
        # Beat lines below ~8 px apart read as noise, not as a grid: at that
        # zoom the bar lines already carry the pulse.
        draw_beats = tpb * self._ppt >= 8
        start = int(self._left_tick // step) * step
        tick = start
        while tick <= tick_end:
            x = int(self.x_of_tick(tick))
            is_bar = tick % bar == 0
            painter.setPen(QPen(theme.GRID_BAR if is_bar else theme.GRID, 1))
            painter.drawLine(x, area.top(), x, area.bottom())
            if not is_bar and draw_beats:
                for b in range(1, self._session.beats_per_bar):
                    bx = int(self.x_of_tick(tick + b * tpb))
                    painter.setPen(QPen(theme.GRID, 1))
                    painter.drawLine(bx, area.top(), bx, area.bottom())
            tick += step

    def _paint_notes(self, painter: QPainter, area: QRect, top_pitch: int,
                     tick_end: float) -> None:
        row_h = self._row_h
        visible = self._session.visible_tracks()
        selected = self._session.selected_segment()
        selection_keys = set()
        if selected is not None:
            selection_keys = {(selected.track, i) for i in selected.note_indices}
        for track_no, note_index, start, end, pitch, velocity in self._notes:
            if end < self._left_tick:
                continue
            if start > tick_end:
                break
            x = int(self.x_of_tick(start))
            w = max(2, int((end - start) * self._ppt) - 1)
            y = int(area.top() + (top_pitch - pitch) * row_h)
            if y < area.top() - row_h or y > area.bottom():
                continue
            active = track_no == self._session.active_track
            color = theme.velocity_color(velocity)
            if track_no not in visible:
                color = theme.dim(color, 60)
            elif not active:
                color = theme.dim(color, 130)
            rect = QRect(x, y, w, max(2, int(row_h) - 1))
            painter.fillRect(rect, color)
            if (track_no, note_index) in selection_keys:
                painter.setPen(QPen(QColor(255, 255, 255, 150), 1))
                painter.drawRect(rect)

    def _paint_selection_band(self, painter: QPainter, area: QRect) -> None:
        selected = self._session.selected_segment()
        if selected is None:
            return
        x0 = self.x_of_tick(selected.start_tick)
        x1 = self.x_of_tick(selected.end_tick)
        painter.fillRect(QRect(int(x0), area.top(), max(1, int(x1 - x0)),
                               area.height()), theme.SELECT_FILL)

    def _paint_boundaries(self, painter: QPainter, area: QRect) -> None:
        try:
            track = self._session.active()
        except IndexError:
            return
        selected = self._session.selected_segment()
        for segment in track.segments:
            chosen = selected is not None and segment.start_tick == selected.start_tick
            x = int(self.x_of_tick(segment.start_tick))
            painter.setPen(QPen(theme.SELECT if chosen else theme.dim(theme.SELECT, 110),
                                2 if chosen else 1,
                                Qt.PenStyle.SolidLine if chosen else Qt.PenStyle.DashLine))
            painter.drawLine(x, area.top(), x, area.bottom())
        if selected is not None:
            # Both edges of the selection, so its end is as visible as its start.
            x = int(self.x_of_tick(selected.end_tick))
            painter.setPen(QPen(theme.SELECT, 2))
            painter.drawLine(x, area.top(), x, area.bottom())

    def _paint_playhead(self, painter: QPainter) -> None:
        """The only layer redrawn every frame, so it stays a handful of lines."""
        x = int(self.x_of_tick(self._playhead_tick))
        painter.setPen(QPen(theme.PLAYHEAD, 1))
        for area in (self._note_area(), self._velocity_area(), self._pedal_area()):
            if area.left() <= x <= area.right():
                painter.drawLine(x, area.top(), x, area.bottom())
        if x >= GUTTER_W:
            painter.drawLine(x, 0, x, RULER_H)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(theme.PLAYHEAD)
            painter.drawPolygon(QPolygon([QPoint(x - 4, 0), QPoint(x + 4, 0), QPoint(x, 6)]))
            painter.setBrush(Qt.BrushStyle.NoBrush)

    def _paint_velocity_lane(self, painter: QPainter, tick_end: float) -> None:
        """Bars for the active track only: four tracks of bars would be noise."""
        if self._session is None:
            return
        try:
            track_no = self._session.active().track
        except IndexError:
            return
        notes = self._notes_by_track.get(track_no, ())
        selected = self._session.selected_segment()
        chosen = (frozenset() if selected is None
                  else frozenset((selected.track, i) for i in selected.note_indices))
        lanes.draw_velocity_lane(painter, self._velocity_area(), self.width(),
                                 notes, self.x_of_tick, self._left_tick, tick_end,
                                 chosen)

    def _paint_pedal_lane(self, painter: QPainter) -> None:
        """The pedal strip: click to press down, click again to lift."""
        if self._session is None:
            return
        lanes.draw_pedal_lane(painter, self._pedal_area(), self.width(),
                              self._session.pedal_spans(), self.x_of_tick)

    def _paint_gutter(self, painter: QPainter, area: QRect, top_pitch: int) -> None:
        painter.fillRect(QRect(0, RULER_H, GUTTER_W, area.height()), theme.PANEL)
        painter.setPen(theme.TEXT_DIM)
        font = QFont(painter.font())
        font.setPointSizeF(7.5)
        painter.setFont(font)
        row_h = self._row_h
        for pitch in range(PITCH_MAX, PITCH_MIN - 1, -1):
            if pitch % 12:
                continue
            y = area.top() + (top_pitch - pitch) * row_h
            if area.top() <= y <= area.bottom():
                painter.drawText(QRect(2, int(y) - 6, GUTTER_W - 4, 12),
                                 Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                                 f"C{pitch // 12 - 1}")

    def _paint_ruler(self, painter: QPainter, area: QRect, tick_end: float) -> None:
        painter.fillRect(QRect(GUTTER_W, 0, area.width(), RULER_H), theme.PANEL_ALT)
        if self._loop is not None:
            x0 = self.x_of_tick(self._loop[0])
            x1 = self.x_of_tick(self._loop[1])
            painter.fillRect(QRect(int(x0), 0, max(1, int(x1 - x0)), RULER_H),
                             theme.dim(theme.SELECT, 70))
        tpb = self._session.project.ticks_per_beat
        bar = max(1, tpb * self._session.beats_per_bar)
        step = bar
        while step * self._ppt < 48:
            step *= 2
        tick = int(self._left_tick // step) * step
        painter.setPen(theme.TEXT_DIM)
        while tick <= tick_end:
            x = int(self.x_of_tick(tick))
            painter.drawLine(x, RULER_H - 6, x, RULER_H)
            painter.drawText(x + 3, RULER_H - 7, f"{tick // tpb + 1}")
            tick += step
        # the cursor marker is drawn every frame by _paint_playhead

    # -- interaction ------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:  # noqa: N802
        if self._session is None:
            return
        pos = event.position()
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan_last = pos.toPoint()
            return
        if pos.x() <= GUTTER_W:
            return
        tick = self.tick_at_x(pos.x())

        # pedal strip: left is the damper switch, right deletes the span
        if pos.y() >= self._pedal_area().top():
            if event.button() == Qt.MouseButton.RightButton:
                self.pedalErased.emit(tick)
            elif event.button() == Qt.MouseButton.LeftButton:
                self.setFocus()
                self.pedalToggled.emit(tick)
            return

        if event.button() != Qt.MouseButton.LeftButton:
            return

        # velocity lane: grab the bar there, select its segment, start a drag
        if pos.y() >= self._velocity_area().top():
            self.setFocus()
            index = self._session.select_at_tick(tick)
            if index is not None:
                self.segmentPicked.emit(self._session.active_track, index)
            self._vel_drag_y = pos.y()
            # The grabbed bar becomes the anchor: dragging *it* to 50 scales the
            # rest of the segment around it.
            self.velocityDragStarted.emit(self._original_velocity_at(pos.x()))
            self._invalidate()
            return

        # ruler: drop the playhead where it was clicked.  Deliberately no
        # drag-to-select -- the loop region is the selected segment now.
        if pos.y() < RULER_H:
            self.cursorMoved.emit(tick)
            return

        self.cursorMoved.emit(tick)
        index = self._session.select_at_tick(tick)
        if index is not None:
            self.segmentPicked.emit(self._session.active_track, index)
        self._invalidate()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._pan_last is not None:
            dx = pos.x() - self._pan_last.x()
            dy = pos.y() - self._pan_last.y()
            self._left_tick -= dx / self._ppt
            self._top_pitch += dy / self._row_h
            self._clamp_view()
            self._pan_last = pos.toPoint()
            self.viewChanged.emit()
            self.update()
            return
        if self._vel_drag_y is not None:
            # Absolute distance from mouse-down, so one drag is one delta no
            # matter how many move events the OS delivers.
            self.velocityDragged.emit(lanes.steps_from_drag(pos.y() - self._vel_drag_y))
            return

        lane = self._velocity_area()
        velocity = (self._velocity_at(pos.x())
                    if lane.top() <= pos.y() <= lane.bottom() and pos.x() > GUTTER_W
                    else None)
        if velocity != self._hover_velocity:
            self._hover_velocity = velocity
            self.update()

    def leaveEvent(self, event) -> None:  # noqa: N802
        if self._hover_velocity is not None:
            self._hover_velocity = None
            self.update()
        super().leaveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        if self._vel_drag_y is not None:
            self._vel_drag_y = None
            self.velocityDragFinished.emit()
        self._pan_last = None

    def wheelEvent(self, event) -> None:  # noqa: N802
        if self._session is None:
            return
        delta = event.angleDelta().y()
        mods = event.modifiers()
        control = bool(mods & Qt.KeyboardModifier.ControlModifier)
        shift = bool(mods & Qt.KeyboardModifier.ShiftModifier)
        if control and shift:
            self.zoom_pitch(1.15 if delta > 0 else 1 / 1.15)
            return
        if control:
            anchor = self.tick_at_x(event.position().x())
            factor = 1.2 if delta > 0 else 1 / 1.2
            self._ppt = max(_MIN_PPT, min(_MAX_PPT, self._ppt * factor))
            self._left_tick = anchor - (event.position().x() - GUTTER_W) / self._ppt
        elif shift:
            self._left_tick -= delta / self._ppt * 0.6
        else:
            self._top_pitch += delta / 120 * 2
        self._clamp_view()
        self.viewChanged.emit()
        self._invalidate()

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._cache = None  # the bitmap is sized to the old widget
        if not self._fitted and self._session is not None:
            self.zoom_fit()
        super().resizeEvent(event)

    def set_keymap(self, keymap) -> None:
        """Resolve the editing keys through ``keymap`` instead of hard-coding them."""
        self._keymap = keymap

    def keyPressEvent(self, event) -> None:  # noqa: N802
        """Editing keys live here, not on the window.

        A window-wide ↑/↓ or a bare "0" would steal the keystroke from the
        velocity spin box the moment the user clicks into it.  These bindings are
        therefore scoped to the roll (``SCOPE_ROLL``) and resolved through the
        same key map the settings dialog edits -- a rebind that only worked in the
        dialog would be worse than no dialog.
        """
        keymap = self._keymap
        if keymap is not None:
            action = keymap.action_for(
                QKeySequence(event.keyCombination()).toString(), scope=SCOPE_ROLL)
            # 3 per click, 9 per shift-click: the same granularity as one notch of
            # a velocity bar, so the two ways of nudging cannot disagree.
            if action == "nudge_up":
                self.nudgeRequested.emit(VEL_DRAG_STEP)
                return
            if action == "nudge_down":
                self.nudgeRequested.emit(-VEL_DRAG_STEP)
                return
            if action == "nudge_up_big":
                self.nudgeRequested.emit(VEL_DRAG_STEP * 3)
                return
            if action == "nudge_down_big":
                self.nudgeRequested.emit(-VEL_DRAG_STEP * 3)
                return
            if action == "prev_segment":
                self.stepRequested.emit(-1)
                return
            if action == "next_segment":
                self.stepRequested.emit(1)
                return
            if action == "clear_segment":
                self.clearRequested.emit()
                return
        super().keyPressEvent(event)
