"""Geometry of the two lanes -- pure maths, no widget, no session.

The painting itself needs a QPainter, but the two numbers that decide what the
user actually gets (how far a drag moved the velocity, and where a bar sits) are
plain functions, so they are tested like any other rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

pytest.importorskip("PySide6")

from PySide6.QtCore import QRect  # noqa: E402

from qmidi.gui import lanes  # noqa: E402


def test_a_drag_moves_in_single_velocity_units():
    """One unit, not three: a proportional drag must be able to land on exactly 50."""
    step = lanes.PX_PER_STEP
    assert lanes.steps_from_drag(0) == 0
    assert lanes.steps_from_drag(-step) == 1
    assert lanes.steps_from_drag(-2 * step) == 2
    assert lanes.steps_from_drag(2 * step) == -2
    assert lanes.steps_from_drag(step / 4) == 0  # a stray pixel must not nudge


def test_velocity_maps_into_the_lane_with_127_at_the_top():
    lane = QRect(0, 100, 400, 64)
    top = lanes.velocity_to_y(127, lane)
    bottom = lanes.velocity_to_y(1, lane)
    assert top < bottom
    assert lane.top() <= top
    assert bottom <= lane.bottom()
    assert lanes.velocity_to_y(64, lane) == pytest.approx((top + bottom) / 2, abs=2)


def test_out_of_range_velocities_clamp_instead_of_drawing_outside_the_lane():
    lane = QRect(0, 100, 400, 64)
    assert lanes.velocity_to_y(0, lane) == lanes.velocity_to_y(1, lane)
    assert lanes.velocity_to_y(999, lane) == lanes.velocity_to_y(127, lane)


def test_the_lane_leaves_room_for_the_full_velocity_range():
    """127 units must not collapse into a few rows, or a bar height means nothing."""
    lane = QRect(0, 0, 400, lanes.VEL_LANE_H)
    assert lanes.velocity_to_y(1, lane) - lanes.velocity_to_y(127, lane) >= 40
    # bars are narrow so a dense passage shows a profile, not a solid wall
    assert lanes.BAR_W <= 3
