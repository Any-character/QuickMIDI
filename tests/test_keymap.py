"""Shortcut table: defaults, conflicts, normalisation, persistence.

Everything here is Qt-free on purpose -- the settings dialog is a thin skin over
this module, so if these rules are right the dialog has almost nothing left to
get wrong.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi import keymap  # noqa: E402
from qmidi.keymap import (  # noqa: E402
    SCOPE_ROLL,
    SCOPE_WINDOW,
    KeyMap,
    KeyMapError,
    normalise,
    pretty,
)


# --------------------------------------------------------------------------- #
# the shipped defaults
# --------------------------------------------------------------------------- #
def test_default_actions_and_keys_are_unique():
    actions = [item.action for item in keymap.DEFAULTS]
    assert len(actions) == len(set(actions)), "two actions share an id"
    # Button-only actions ("重新切段", "声像", "母带") ship with no key; several
    # empty defaults are not a conflict, only two identical real keys are.
    keys = [normalise(item.default) for item in keymap.DEFAULTS if item.default]
    assert len(keys) == len(set(keys)), "two actions ship with the same key"


def test_every_default_resolves_back_to_its_own_action():
    table = KeyMap()
    for item in keymap.DEFAULTS:
        if not normalise(item.default):
            continue  # an action may ship with no key at all (button-only)
        assert table.action_for(item.default, scope=item.scope) == item.action


def test_the_roll_scope_only_answers_roll_bindings():
    table = KeyMap()
    assert table.action_for("Up", scope=SCOPE_ROLL) == "nudge_up"
    assert table.action_for("Up", scope=SCOPE_WINDOW) is None
    assert table.action_for("Space", scope=SCOPE_ROLL) is None


# --------------------------------------------------------------------------- #
# normalisation
# --------------------------------------------------------------------------- #
def test_normalise_fixes_modifier_order_and_letter_case():
    assert normalise("shift+ctrl+z") == "Ctrl+Shift+Z"
    assert normalise("  Ctrl + O ") == "Ctrl+O"
    assert normalise("ctrl+shift+Z") == "Ctrl+Shift+Z"
    assert normalise("") == ""


def test_normalise_spells_special_keys_the_way_qt_does():
    assert normalise("space") == "Space"
    assert normalise("escape") == "Esc"
    assert normalise("delete") == "Del"
    assert normalise("f4") == "F4"
    assert normalise("BracketLeft") == "BracketLeft"


def test_pretty_shows_arrows_and_empties():
    assert pretty("Shift+Up") == "Shift+↑"
    assert pretty("Down") == "↓"
    assert pretty("") == "未设置"
    assert pretty("Space") == "空格"


# --------------------------------------------------------------------------- #
# rebinding
# --------------------------------------------------------------------------- #
def test_rebinding_takes_effect_in_both_directions():
    table = KeyMap()
    table.bind("nudge_up", "W")
    assert table.sequence("nudge_up") == "W"
    assert table.action_for("W", scope=SCOPE_ROLL) == "nudge_up"
    assert table.action_for("Up", scope=SCOPE_ROLL) is None


def test_a_key_already_in_use_is_refused_with_a_readable_message():
    table = KeyMap()
    with pytest.raises(KeyMapError) as info:
        table.bind("nudge_up", "Space")
    assert "播放" in str(info.value)
    assert table.sequence("nudge_up") == "Up"


def test_a_binding_can_be_cleared_and_restored():
    table = KeyMap()
    table.bind("clear_segment", "")
    assert table.sequence("clear_segment") == ""
    assert table.action_for("0", scope=SCOPE_ROLL) is None
    table.reset("clear_segment")
    assert table.sequence("clear_segment") == "0"


def test_reset_without_an_action_returns_every_default():
    table = KeyMap({"nudge_up": "W", "loop": "Ctrl+L"})
    assert not table.is_default()
    table.reset()
    assert table.is_default()


def test_unknown_actions_are_rejected():
    with pytest.raises(KeyMapError):
        KeyMap().bind("no-such-action", "Q")


# --------------------------------------------------------------------------- #
# persistence
# --------------------------------------------------------------------------- #
def test_bindings_survive_a_save_and_load_round_trip(tmp_path: Path):
    path = tmp_path / "keymap.json"
    table = KeyMap()
    table.bind("nudge_up", "W")
    table.bind("loop", "")
    table.save(path)

    restored = KeyMap.load(path)
    assert restored.sequence("nudge_up") == "W"
    assert restored.sequence("loop") == ""
    assert restored.sequence("play") == "Space"


def test_a_missing_file_means_defaults(tmp_path: Path):
    assert KeyMap.load(tmp_path / "nope.json").is_default()


def test_a_broken_file_does_not_stop_the_app_from_opening(tmp_path: Path):
    path = tmp_path / "keymap.json"
    path.write_text("{ this is not json", encoding="utf-8")
    assert KeyMap.load(path).is_default()


def test_unknown_keys_in_the_file_are_ignored_forward_compatibly(tmp_path: Path):
    path = tmp_path / "keymap.json"
    path.write_text(json.dumps({"future_action": "Ctrl+Q", "loop": "Ctrl+L"}),
                    encoding="utf-8")
    table = KeyMap.load(path)
    assert table.sequence("loop") == "Ctrl+L"
    assert "future_action" not in table.to_dict()


def test_a_hand_edited_duplicate_does_not_leave_two_actions_on_one_key():
    table = KeyMap({"nudge_up": "Space", "loop": "Space"})
    taken = table.action_for("Space")
    assert taken in ("play", "nudge_up", "loop")
    owners = [action for action in table.to_dict()
              if table.sequence(action) == "Space"]
    assert len(owners) == 1


def test_the_config_path_is_overridable_for_tests(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("QMIDI_CONFIG_DIR", str(tmp_path))
    assert keymap.keymap_path() == tmp_path / "keymap.json"


# --------------------------------------------------------------------------- #
# the hint line
# --------------------------------------------------------------------------- #
def test_the_hint_line_follows_the_bindings():
    table = KeyMap()
    assert "↑ 力度+3" in table.hint_text()
    table.bind("nudge_up", "W")
    hint = table.hint_text()
    assert "W 力度+3" in hint
    assert "↑ 力度+3" not in hint
