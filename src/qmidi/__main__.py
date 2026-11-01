"""``python -m qmidi`` / ``python -m qmidi song.mid``."""

from __future__ import annotations

import sys

from .gui.app import main

if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
