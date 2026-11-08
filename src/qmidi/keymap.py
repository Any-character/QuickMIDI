"""Rebindable shortcuts: the table, the conflicts, and the file on disk.

The point of a module rather than strings scattered in the window: a settings
screen is only honest if the *same* table drives the QActions, their tooltips
and the hint line above the segment panel.  Three copies of "↑↓ ±3" is exactly
how a rebind ends up working in one place and lying in the other two.

No Qt here (AGENTS.md §4.5).  Sequences stay plain strings and the conflict rule
compares normalised text; the dialog converts to ``QKeySequence`` at the edge.

Two scopes exist because not every shortcut may steal a keystroke:

``window``
    A QAction on the main window.  Fires wherever focus is (a toolbar button or
    the piano roll alike), which is what you want for Ctrl+Z.
``roll``
    Lives in the piano roll's own key handler.  A bare ``0`` or ``[`` bound as a
    window shortcut would eat those keys while the user is typing a velocity into
    the spin box, so these only fire when the roll has focus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from .i18n import tr
from .settings import config_dir

SCOPE_WINDOW = "window"
SCOPE_ROLL = "roll"

GROUP_FILE = "文件"
GROUP_PLAY = "播放"
GROUP_EDIT = "编辑"
GROUP_VIEW = "视图"


@dataclass(frozen=True)
class Binding:
    """One rebindable action."""

    action: str
    label: str
    default: str
    group: str
    scope: str = SCOPE_WINDOW
    #: Toolbar text.  Empty means "not on the toolbar"; the dialog always shows
    #: ``label``, which is allowed to be longer and clearer.
    button: str = ""

    @property
    def title(self) -> str:
        return self.button or self.label


DEFAULTS: tuple[Binding, ...] = (
    Binding("open", "打开 MIDI 文件", "Ctrl+O", GROUP_FILE, button="打开"),
    Binding("save", "保存工程（力度段）", "Ctrl+S", GROUP_FILE, button="保存工程"),
    Binding("export_midi", "导出 MIDI", "Ctrl+E", GROUP_FILE, button="导出 MIDI"),
    Binding("export_mp3", "导出音频", "Ctrl+Shift+E", GROUP_FILE, button="导出音频"),
    Binding("resegment", "重新自动切段", "", GROUP_FILE, button="重切"),
    Binding("auto_pedal", "按和声自动重做踏板", "P", GROUP_EDIT, button="自动踏板"),
    Binding("stereo_split", "双钢琴分声像", "", GROUP_EDIT, button="声像"),
    Binding("master", "母带处理（混响/压缩/响度）", "", GROUP_EDIT, button="母带"),
    Binding("play", "播放 / 停止", "Space", GROUP_PLAY, button="▶ 播放"),
    Binding("loop", "循环当前段", "L", GROUP_PLAY, button="↺ 循环"),
    Binding("home", "回到开头", "Home", GROUP_PLAY),
    Binding("nudge_up", "力度 +3", "Up", GROUP_EDIT, SCOPE_ROLL),
    Binding("nudge_down", "力度 −3", "Down", GROUP_EDIT, SCOPE_ROLL),
    Binding("nudge_up_big", "力度 +9", "Shift+Up", GROUP_EDIT, SCOPE_ROLL),
    Binding("nudge_down_big", "力度 −9", "Shift+Down", GROUP_EDIT, SCOPE_ROLL),
    Binding("prev_segment", "上一段", "[", GROUP_EDIT, SCOPE_ROLL),
    Binding("next_segment", "下一段", "]", GROUP_EDIT, SCOPE_ROLL),
    Binding("clear_segment", "清除本段调整", "0", GROUP_EDIT, SCOPE_ROLL),
    Binding("undo", "撤销", "Ctrl+Z", GROUP_EDIT, button="撤销"),
    Binding("redo", "重做", "Ctrl+Shift+Z", GROUP_EDIT, button="重做"),
    Binding("zoom_in", "放大时间轴", "Ctrl+=", GROUP_VIEW, button="＋"),
    Binding("zoom_out", "缩小时间轴", "Ctrl+-", GROUP_VIEW, button="−"),
    Binding("fit", "全曲适配", "F", GROUP_VIEW),
    Binding("settings", "设置（音源 / 显示 / 快捷键）", "Ctrl+,", GROUP_VIEW, button="设置"),
)

_BINDINGS: dict[str, Binding] = {item.action: item for item in DEFAULTS}

_MODIFIER_ORDER = ("Ctrl", "Alt", "Shift", "Meta")
_MODIFIER_ALIASES = {
    "ctrl": "Ctrl", "control": "Ctrl",
    "alt": "Alt", "option": "Alt",
    "shift": "Shift",
    "meta": "Meta", "win": "Meta", "cmd": "Meta",
}

#: Human names for the keys that have no printable character, used in the hint
#: line so "Shift+Up" reads as "Shift+↑".
_KEY_GLYPHS = {
    "Up": "↑", "Down": "↓", "Left": "←", "Right": "→",
    "Space": "空格", "Return": "回车", "Enter": "回车",
    "Backspace": "退格", "Delete": "Del", "Esc": "Esc", "Escape": "Esc",
}


class KeyMapError(ValueError):
    """A binding was rejected (unknown action, or the key is already taken)."""


def action_ids() -> tuple[str, ...]:
    return tuple(item.action for item in DEFAULTS)


#: Lower-case spellings that mean a specific Qt key name.
_KEY_NAMES = {
    "up": "Up", "down": "Down", "left": "Left", "right": "Right",
    "space": "Space", "return": "Return", "enter": "Return",
    "escape": "Esc", "esc": "Esc", "backspace": "Backspace",
    "delete": "Del", "del": "Del", "home": "Home", "end": "End",
    "insert": "Ins", "ins": "Ins", "tab": "Tab",
}


def _normalise_key(key: str) -> str:
    """One key name, spelled the way Qt writes it."""
    if len(key) == 1:
        return key.upper()
    known = _KEY_NAMES.get(key.lower())
    if known is not None:
        return known
    if key[0].lower() == "f" and key[1:].isdigit():
        return "F" + key[1:]
    return key[:1].upper() + key[1:]


def normalise(sequence: str) -> str:
    """Canonical text for comparison: fixed modifier order, canonical key name.

    Qt's ``toString`` already produces this shape, so this mostly guards
    hand-edited config files: a file saying ``shift+ctrl+z`` must not create a
    phantom second binding that the conflict check then misses.
    """
    text = (sequence or "").strip()
    if not text:
        return ""
    modifiers: list[str] = []
    keys: list[str] = []
    for part in (piece.strip() for piece in text.split("+")):
        if not part:
            continue
        alias = _MODIFIER_ALIASES.get(part.lower())
        if alias is not None:
            if alias not in modifiers:
                modifiers.append(alias)
        else:
            keys.append(_normalise_key(part))
    ordered = [name for name in _MODIFIER_ORDER if name in modifiers]
    return "+".join(ordered + keys)


def pretty(sequence: str) -> str:
    """Display form: ``Shift+Up`` -> ``Shift+↑``, empty -> ``未设置``."""
    text = normalise(sequence)
    if not text:
        return tr("未设置")
    parts = text.split("+")
    parts[-1] = tr(_KEY_GLYPHS.get(parts[-1], parts[-1]))
    return "+".join(parts)


def keymap_path() -> Path:
    """Where the user's own bindings live, next to the other settings."""
    return config_dir() / "keymap.json"


class KeyMap:
    """Action id -> key sequence, with the defaults as the floor."""

    def __init__(self, values: Optional[dict[str, str]] = None) -> None:
        self._values: dict[str, str] = {}
        for item in DEFAULTS:
            self._values[item.action] = normalise(item.default)
        if values:
            for action, sequence in values.items():
                if action in _BINDINGS and isinstance(sequence, str):
                    self._values[action] = normalise(sequence)
        self._resolve_duplicates()

    # -- reading ----------------------------------------------------------- #
    def sequence(self, action: str) -> str:
        return self._values.get(action, "")

    def binding(self, action: str) -> Binding:
        # Translated on *read*, not at import: the module-level table holds the
        # Chinese originals, so a language picked up after import (or in a test)
        # still produces the right language here.
        base = _BINDINGS[action]
        return replace(
            base,
            label=tr(base.label),
            button=tr(base.button) if base.button else "",
            group=tr(base.group),
        )

    def action_for(self, sequence: str, *, scope: Optional[str] = None) -> Optional[str]:
        """The action ``sequence`` is bound to, honouring ``scope``."""
        target = normalise(sequence)
        if not target:
            return None
        for action, bound in self._values.items():
            if bound != target:
                continue
            if scope is None or _BINDINGS[action].scope == scope:
                return action
        return None

    def owner(self, sequence: str, *, ignore: Optional[str] = None) -> Optional[str]:
        target = normalise(sequence)
        if not target:
            return None
        for action, bound in self._values.items():
            if action != ignore and bound == target:
                return action
        return None

    def is_default(self) -> bool:
        return all(self._values[item.action] == normalise(item.default)
                   for item in DEFAULTS)

    # -- writing ------------------------------------------------------------ #
    def bind(self, action: str, sequence: str) -> None:
        """Rebind ``action``; raises when the key is already taken."""
        if action not in _BINDINGS:
            raise KeyMapError(tr("未知的快捷键动作：{action}").format(action=action))
        text = normalise(sequence)
        if text:
            clash = self.owner(text, ignore=action)
            if clash is not None:
                raise KeyMapError(
                    tr("{key} 已经分配给「{label}」").format(
                        key=pretty(text), label=tr(_BINDINGS[clash].label))
                )
        self._values[action] = text

    def reset(self, action: Optional[str] = None) -> None:
        targets = [item.action for item in DEFAULTS] if action is None else [action]
        for name in targets:
            self._values[name] = normalise(_BINDINGS[name].default)

    def copy(self) -> "KeyMap":
        clone = KeyMap()
        clone._values = dict(self._values)
        return clone

    # -- persistence -------------------------------------------------------- #
    def to_dict(self) -> dict[str, str]:
        return dict(self._values)

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else keymap_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "KeyMap":
        """Read the user's bindings; a broken or missing file means defaults."""
        target = Path(path) if path is not None else keymap_path()
        if not target.is_file():
            return cls()
        try:
            data = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls()
        return cls(data if isinstance(data, dict) else None)

    # -- display ------------------------------------------------------------ #
    def hint_text(self) -> str:
        """The one line above the segment panel, driven by the live bindings."""
        up = pretty(self.sequence("nudge_up"))
        up_big = pretty(self.sequence("nudge_up_big"))
        prev_key = pretty(self.sequence("prev_segment"))
        next_key = pretty(self.sequence("next_segment"))
        return tr("{up} 力度+3 · {up_big} 力度+9 · "
                  "{prev} {next} 换段 · "
                  "拖力度条整段升降 · 踏板点一下踩、再点一下松").format(
            up=up, up_big=up_big, prev=prev_key, next=next_key)

    def _resolve_duplicates(self) -> None:
        """A hand-edited file may bind one key twice; the first action keeps it."""
        seen: set[str] = set()
        for action in self._values:
            bound = self._values[action]
            if not bound:
                continue
            if bound in seen:
                self._values[action] = ""
            else:
                seen.add(bound)
