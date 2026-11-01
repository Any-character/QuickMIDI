"""The two lanes under the note canvas: velocity bars and the sustain pedal.

Geometry only -- no session, no widget, no state.  ``piano_roll`` owns the
tick->x mapping and passes it in; the point of the split is that the largest
widget in the project does not also become the largest painting file.

Reading the velocity lane: every visible note of the active track is one bar.
Bar height *is* the velocity, so the outline of a passage is visible without
reading a single number -- which is exactly what the DAW screenshot showed.
Dragging any bar moves the whole segment (the project's model has one delta per
segment, not one per note), so the bars keep their relative shape and the
passage rises or falls as a block.
"""

from __future__ import annotations

from typing import Sequence

from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPen

from ..i18n import tr
from . import theme

#: Heights of the two lanes.  The velocity lane is taller because it is a real
#: editor (bar height is the value); the pedal lane only says down or up.
VEL_LANE_H = 64
PEDAL_LANE_H = 38

#: Vertical mouse travel per velocity unit.  Dragging is continuous (one unit at
#: a time) so a proportional drag can land exactly on 50; the keyboard keeps its
#: ±3 steps.  1.5 px per unit makes a 50-unit swing a 75 px movement.
PX_PER_STEP = 1.5

#: Bar width.  Narrow on purpose: a chord stacks several bars within a few
#: pixels and a dense passage has one note every 3-4 px when zoomed out, so a
#: wider bar turns the lane into a solid wall instead of a velocity profile.
BAR_W = 3

#: Guides drawn behind the bars (velocity 32 / 64 / 96).
_GUIDES = (32, 64, 96)


def steps_from_drag(dy: float) -> int:
    """Velocity units for a vertical drag; up (negative dy) is louder.

    One unit, not three: with ±3 steps a proportional drag could never reach a
    round target like 50, which is exactly what the user asked for.
    """
    return int(round(-dy / PX_PER_STEP))


def velocity_to_y(velocity: int, lane: QRect) -> int:
    """Top of a bar for ``velocity``: loud at the top of the lane."""
    top = lane.top() + 5
    height = max(1, lane.height() - 10)
    value = max(1, min(127, int(velocity)))
    return int(round(top + height * (1.0 - value / 127.0)))


def draw_velocity_lane(
    painter: QPainter,
    lane: QRect,
    full_width: int,
    notes: Sequence[tuple],
    x_of,
    left_tick: float,
    right_tick: float,
    selected_keys: frozenset[tuple[int, int]],
) -> None:
    """Bars for ``notes`` (``(track, index, start, end, pitch, velocity)``)."""
    painter.fillRect(QRect(0, lane.top(), full_width, lane.height()), theme.PANEL)
    painter.setPen(QPen(theme.BORDER, 1))
    painter.drawLine(lane.left(), lane.top(), lane.right(), lane.top())

    for guide in _GUIDES:
        y = velocity_to_y(guide, lane)
        painter.setPen(QPen(theme.GRID, 1, Qt.PenStyle.DotLine))
        painter.drawLine(lane.left(), y, lane.right(), y)

    # One bar per pixel column.  A dense passage puts 3-4 notes in the same
    # column, and drawing all of them is both slower and visually identical to
    # drawing the tallest one.  Measured: this is the difference between ~200 ms
    # and ~15 ms to rebuild the whole widget.
    columns: dict[int, tuple[int, bool]] = {}
    for track_no, index, start, end, pitch, velocity in notes:
        if start > right_tick or end < left_tick:
            continue
        x = int(x_of(start))
        if x < lane.left() - BAR_W or x > lane.right():
            continue
        chosen = (track_no, index) in selected_keys
        previous = columns.get(x)
        if previous is None:
            columns[x] = (velocity, chosen)
        else:
            columns[x] = (max(previous[0], velocity), chosen or previous[1])

    painter.setClipRect(lane)
    for x, (velocity, chosen) in columns.items():
        top = velocity_to_y(velocity, lane)
        rect = QRect(x, top, BAR_W, max(2, lane.bottom() - top))
        painter.fillRect(rect, theme.velocity_color(velocity))
        if chosen:
            painter.setPen(QPen(QColor(255, 255, 255, 190), 1))
            painter.drawRect(rect.adjusted(0, 0, -1, -1))
    painter.setClipping(False)

    # The label sits in the top half of the gutter; the *number* is drawn by the
    # widget every frame (it follows the mouse) and would be stale if baked in.
    font = QFont(painter.font())
    font.setPointSizeF(7.5)
    painter.setFont(font)
    painter.setPen(theme.TEXT_DIM)
    painter.drawText(QRect(2, lane.top() + 2, lane.left() - 6, lane.height() // 2 - 2),
                     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignBottom,
                     tr("力度"))


def draw_pedal_lane(
    painter: QPainter,
    lane: QRect,
    full_width: int,
    spans: Sequence,
    x_of,
) -> None:
    """One block per pedal span; a block reaching the right edge is still held.

    The playhead is deliberately not drawn here: it is the one thing that moves
    every frame, and it lives in the uncached layer of ``piano_roll``.
    """
    painter.fillRect(QRect(0, lane.top(), full_width, lane.height()), theme.PANEL)
    painter.setPen(QPen(theme.BORDER, 1))
    painter.drawLine(lane.left(), lane.top(), lane.right(), lane.top())

    font = QFont(painter.font())
    font.setPointSizeF(7.5)
    painter.setFont(font)
    painter.setPen(theme.TEXT_DIM)
    painter.drawText(QRect(2, lane.top(), lane.left() - 6, lane.height()),
                     Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                     tr("踏板"))

    top = lane.top() + 7
    height = max(4, lane.height() - 14)
    # Baseline where the (empty) lane sits when the pedal is up.  Without it an
    # unpedalled track looks like a broken strip rather than a lifted pedal.
    painter.setPen(QPen(theme.GRID, 1))
    painter.drawLine(lane.left(), lane.bottom() - 6, lane.right(), lane.bottom() - 6)
    painter.setClipRect(lane)
    for span in spans:
        x0 = x_of(span.start_tick)
        x1 = x_of(span.end_tick)
        if x1 < lane.left() or x0 > lane.right():
            continue
        left = int(max(lane.left(), x0))
        right = int(min(lane.right(), x1))
        painter.fillRect(QRect(left, top, max(2, right - left), height), theme.PEDAL)
    painter.setClipping(False)
