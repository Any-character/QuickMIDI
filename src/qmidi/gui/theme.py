"""Colours for the editor.

A dark UI on purpose: the window is judged next to a DAW, and a bright canvas
makes low-velocity notes (which is most of what gets edited here) hard to see.
The velocity ramp runs cool -> warm so a glance at the roll shows the dynamic
shape of a passage without reading numbers.
"""

from __future__ import annotations

from PySide6.QtGui import QColor

BG = QColor("#15171c")
PANEL = QColor("#1c1f26")
PANEL_ALT = QColor("#22262e")
BORDER = QColor("#2e333d")
GRID = QColor("#232731")
GRID_BAR = QColor("#333947")
GRID_BEAT = QColor("#2a2f3a")
TEXT = QColor("#c9d1dc")
TEXT_DIM = QColor("#79828f")
ACCENT = QColor("#4fa3ff")
SELECT = QColor("#ffb454")
SELECT_FILL = QColor(255, 180, 84, 26)
PLAYHEAD = QColor("#ff5f6d")
LOOP_FILL = QColor(255, 180, 84, 18)
#: Sustain-pedal spans.  Teal, deliberately far from the velocity ramp (which
#: runs blue -> green -> orange) so a pedal block never reads as a very loud note.
PEDAL = QColor("#2f8f86")
BLACK_KEY_ROW = QColor("#191c22")

#: Velocity ramp stops: (velocity, colour).  Linear in velocity, not in
#: perceptual space -- the point is to tell 60 from 80 at a glance, not to be
#: photometrically even.
_RAMP = (
    (1, QColor("#3a5f8a")),
    (45, QColor("#3f7f96")),
    (70, QColor("#4c9375")),
    (92, QColor("#8fa14c")),
    (110, QColor("#d19a4a")),
    (127, QColor("#d96a4a")),
)

_BLACK_KEYS = frozenset({1, 3, 6, 8, 10})


def velocity_color(velocity: int) -> QColor:
    """Map 1..127 to the ramp."""
    value = max(1, min(127, int(velocity)))
    for (v0, c0), (v1, c1) in zip(_RAMP, _RAMP[1:]):
        if value <= v1:
            t = 0.0 if v1 == v0 else (value - v0) / (v1 - v0)
            return QColor(
                round(c0.red() + (c1.red() - c0.red()) * t),
                round(c0.green() + (c1.green() - c0.green()) * t),
                round(c0.blue() + (c1.blue() - c0.blue()) * t),
            )
    return _RAMP[-1][1]


def is_black_key(pitch: int) -> bool:
    return pitch % 12 in _BLACK_KEYS


def dim(color: QColor, alpha: int) -> QColor:
    out = QColor(color)
    out.setAlpha(alpha)
    return out
