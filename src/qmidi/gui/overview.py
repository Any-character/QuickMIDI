"""Whole-file overview strip: where am I, and how much is left.

The piano roll shows one screenful; this strip shows the piece.  It is the
progress bar the user asked for, with the one behaviour that makes it useful
rather than decorative: clicking or dragging it moves the playhead, so getting
to bar 80 is one gesture instead of a scroll hunt.

Two things are drawn because two questions are being answered: a density
histogram ("where is there music at all") and a window box ("what can I see
right now").  The playhead is the third, in the same red as the roll.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen, QPixmap
from PySide6.QtWidgets import QWidget

from ..i18n import tr
from . import theme

HEIGHT = 28
#: Same left margin as the piano roll, so tick 0 lands on the same pixel.
GUTTER_W = 44


class OverviewBar(QWidget):
    """Density + view window + playhead; click or drag to seek."""

    seekRequested = Signal(int)  # tick: move the playhead here
    viewDragged = Signal(float)  # new left edge (tick): scroll the roll

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFixedHeight(HEIGHT)
        self.setToolTip(tr("全曲总览：点击或拖动跳转播放位置"))
        self._session = None
        self._starts = np.zeros(0, dtype=np.int64)
        self._total = 1
        self._view_left = 0.0
        self._view_ticks = 1.0
        self._playhead = 0.0
        self._dragging: Optional[str] = None  # "view" (scroll) or "seek" (playhead)
        self._press_x = 0.0
        self._press_left = 0.0
        #: The density histogram, which only changes with the file or the size.
        self._density: Optional[QPixmap] = None

    # -- data -------------------------------------------------------------- #
    def set_session(self, session) -> None:
        self._session = session
        if session is None:
            self._starts = np.zeros(0, dtype=np.int64)
            self._total = 1
        else:
            self._starts = np.asarray(
                [note.start_tick for track in session.project.tracks for note in track.notes],
                dtype=np.int64,
            )
            self._total = max(1, session.total_end_tick())
        self._density = None
        self.update()

    def set_view(self, left_tick: float, visible_ticks: float) -> None:
        self._view_left = float(left_tick)
        self._view_ticks = max(1.0, float(visible_ticks))
        self.update()

    def set_playhead(self, tick: float) -> None:
        self._playhead = float(tick)
        self.update()

    # -- geometry ---------------------------------------------------------- #
    def _plot(self) -> QRect:
        return QRect(GUTTER_W, 4, max(1, self.width() - GUTTER_W - 4), HEIGHT - 9)

    def tick_at(self, x: float) -> int:
        plot = self._plot()
        ratio = (x - plot.left()) / max(1, plot.width())
        return int(round(max(0.0, min(1.0, ratio)) * self._total))

    def _view_rect(self) -> QRect:
        plot = self._plot()
        x0 = plot.left() + plot.width() * self._view_left / self._total
        x1 = plot.left() + plot.width() * (self._view_left + self._view_ticks) / self._total
        return QRect(int(x0), plot.top(), max(2, int(x1 - x0)), plot.height())

    # -- interaction ------------------------------------------------------- #
    def mousePressEvent(self, event) -> None:  # noqa: N802
        """Grab the window box to scroll; click outside it to move the playhead.

        Two different questions, two different gestures: "what am I looking at"
        (the box) and "where does it play from" (the playhead).  Binding both to
        one drag made zoomed-in navigation impossible.
        """
        if self._session is None or event.button() != Qt.MouseButton.LeftButton:
            return
        pos = event.position()
        if self._view_rect().contains(pos.toPoint()):
            self._dragging = "view"
            self._press_x = pos.x()
            self._press_left = self._view_left
        else:
            self._dragging = "seek"
            self.seekRequested.emit(self.tick_at(pos.x()))

    def mouseMoveEvent(self, event) -> None:  # noqa: N802
        pos = event.position()
        if self._dragging == "view":
            span = max(1, self._plot().width())
            delta = (pos.x() - self._press_x) / span * self._total
            left = max(0.0, min(float(self._total - self._view_ticks),
                                self._press_left + delta))
            self.viewDragged.emit(left)
            return
        if self._dragging == "seek":
            self.seekRequested.emit(self.tick_at(pos.x()))
            return
        inside = self._view_rect().contains(pos.toPoint())
        self.setCursor(Qt.CursorShape.SizeHorCursor if inside
                       else Qt.CursorShape.ArrowCursor)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        self._dragging = None

    # -- painting ---------------------------------------------------------- #
    def paintEvent(self, event) -> None:  # noqa: N802
        painter = QPainter(self)
        painter.fillRect(self.rect(), theme.PANEL)
        painter.setPen(QPen(theme.BORDER, 1))
        painter.drawLine(0, 0, self.width(), 0)

        font = QFont(painter.font())
        font.setPointSizeF(7.5)
        painter.setFont(font)
        painter.setPen(theme.TEXT_DIM)
        painter.drawText(QRect(2, 0, GUTTER_W - 6, HEIGHT),
                         Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                         tr("总览"))

        if self._session is None or self._starts.size == 0:
            return

        if self._density is None:
            self._density = self._render_density()
        painter.drawPixmap(0, 0, self._density)

        plot = self._plot()
        window = self._view_rect()
        painter.fillRect(window, QColor(79, 163, 255, 30))
        painter.setPen(QPen(theme.ACCENT, 1))
        painter.drawRect(window.adjusted(0, 0, -1, -1))

        x = int(plot.left() + plot.width() * self._playhead / self._total)
        painter.setPen(QPen(theme.PLAYHEAD, 2))
        painter.drawLine(x, plot.top() - 2, x, plot.bottom())

    def _render_density(self) -> QPixmap:
        """Bars for every pixel column, drawn once per file (not once per frame)."""
        ratio = self.devicePixelRatioF()
        pixmap = QPixmap(QSize(int(self.width() * ratio), int(self.height() * ratio)))
        pixmap.setDevicePixelRatio(ratio)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        plot = self._plot()
        bins = max(1, plot.width())
        hist, _ = np.histogram(np.clip(self._starts, 0, self._total),
                               bins=bins, range=(0, self._total))
        peak = max(1, int(hist.max()))
        base = plot.bottom()
        for index, count in enumerate(hist):
            if count == 0:
                continue
            height = max(1, int(round(count / peak * plot.height())))
            painter.fillRect(QRect(plot.left() + index, base - height, 1, height),
                             theme.TEXT_DIM)
        painter.end()
        return pixmap

    def resizeEvent(self, event) -> None:  # noqa: N802
        self._density = None  # bins are per pixel column
        super().resizeEvent(event)
