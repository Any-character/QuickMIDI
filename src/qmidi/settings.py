"""User settings that are not shortcuts: the instrument and the display mode.

No Qt here (AGENTS.md §4.5), and no exceptions on read: a settings file that
fails to parse must never stop the app from opening.  Every load falls back to
the defaults, and a broken file is simply overwritten on the next save.

Why the instrument path lives here rather than in the environment: ``SFZ_PATH``
is a shell variable, and the user runs this by double-clicking a ``.bat``.  An
env var only he can set is not a feature.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .i18n import tr


def config_dir() -> Path:
    """Per-user configuration folder (``%APPDATA%\\QuickMIDI`` on Windows)."""
    override = os.environ.get("QMIDI_CONFIG_DIR")
    if override:
        return Path(override)
    if sys.platform.startswith("win"):
        root = Path(os.environ.get("APPDATA") or Path.home())
    else:
        root = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config")
    return root / "QuickMIDI"


def settings_path() -> Path:
    return config_dir() / "settings.json"


@dataclass(frozen=True)
class Settings:
    """Everything the user can change that is not a key binding."""

    #: Instrument (.sfz file or a folder of them).  Empty means "whatever the
    #: engine finds": the SFZ_PATH variable, else the repo's junction.
    sfz_path: str = ""
    #: Show every track's notes in the roll instead of only the active one.
    #: The default is the quieter view: four tracks of notes at once is noise
    #: when you can only edit one of them.
    show_all_tracks: bool = False
    #: Playhead snap, as "parts per beat": 4 = sixteenth notes, 8 = thirty-seconds,
    #: 2 = eighths, 0 = free placement.  A grid is what makes a click land on a
    #: note instead of 37 ticks before it, and the user can still turn it off.
    snap_division: int = 4
    #: sfizz's sample interpolation quality for the MP3 export: 10 is the
    #: maximum (the preview runs at 6).  Lower is much faster -- measured 0.88x
    #: real time at 10 vs 3.26x at 6 on a 20 s full mix -- and slightly softer
    #: in the highs.  Exposed because a 4x export difference is a decision,
    #: not a detail.
    export_quality: int = 10
    #: UI language, "zh" or "en".  Applied once, before the GUI is imported;
    #: changing it in the settings dialog asks for a restart.
    language: str = "zh"

    def to_dict(self) -> dict:
        return {
            "sfz_path": self.sfz_path,
            "show_all_tracks": self.show_all_tracks,
            "snap_division": self.snap_division,
            "export_quality": self.export_quality,
            "language": self.language,
        }

    @classmethod
    def from_dict(cls, data: object) -> "Settings":
        if not isinstance(data, dict):
            return cls()
        raw_path = data.get("sfz_path", "")
        raw_snap = data.get("snap_division", 4)
        try:
            snap = int(raw_snap)
        except (TypeError, ValueError):
            snap = 4
        raw_language = data.get("language", "zh")
        raw_quality = data.get("export_quality", 10)
        try:
            quality = int(raw_quality)
        except (TypeError, ValueError):
            quality = 10
        return cls(
            sfz_path=raw_path if isinstance(raw_path, str) else "",
            show_all_tracks=bool(data.get("show_all_tracks", False)),
            snap_division=snap if snap >= 0 else 4,
            export_quality=quality if 0 <= quality <= 10 else 10,
            language=raw_language if raw_language in ("zh", "en") else "zh",
        )

    def save(self, path: Path | str | None = None) -> Path:
        target = Path(path) if path is not None else settings_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    @classmethod
    def load(cls, path: Path | str | None = None) -> "Settings":
        target = Path(path) if path is not None else settings_path()
        if not target.is_file():
            return cls()
        try:
            return cls.from_dict(json.loads(target.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return cls()

    @property
    def instrument_label(self) -> str:
        """What to show in the dialog: the file name, or the default wording."""
        if not self.sfz_path:
            return tr("默认音源（Concert Grand）")
        return Path(self.sfz_path).name or self.sfz_path
