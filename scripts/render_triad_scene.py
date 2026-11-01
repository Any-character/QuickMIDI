"""Step 2a: re-render the step-1 triad+pedal scene through the adapter.

Same timeline as scripts/verify_sfizz_render.py, but every sfz action now
goes through qmidi.engine.SfzEngine -- no pysfizz import anywhere below.

  0.00  CC64 = 127   (sustain DOWN)     1.30  note_off C4/E4/G4 (held by pedal)
  0.10  note_on  C4                    2.00  CC64 = 0  (pedal UP)
  0.20  note_on  E4                    3.00  end
  0.30  note_on  G4

Headless, no Qt. Usage:  python scripts/render_triad_scene.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from qmidi.engine import SfzEngine  # noqa: E402

DURATION_S = 3.0
VEL = 80
TRIAD = (60, 64, 67)  # C4 E4 G4

# (time_s, kind, a, b) -- identical to the step-1 scene.
EVENTS = [
    (0.00, "cc", 64, 127),
    (0.10, "on", 60, VEL),
    (0.20, "on", 64, VEL),
    (0.30, "on", 67, VEL),
    (1.30, "off", 60, 0),
    (1.30, "off", 64, 0),
    (1.30, "off", 67, 0),
    (2.00, "cc", 64, 0),
]

KIND = {"cc": "cc", "on": "note_on", "off": "note_off"}


def main() -> int:
    eng = SfzEngine(sample_rate=48000, block_size=1024)
    sfz = eng.load_sfz()  # SFZ_PATH env, else repo-relative default
    print(f"sfz main file : {sfz.name}")
    print(f"playable keys : {len(eng.playable_keys)}")

    sr, bs = eng.sample_rate, eng.block_size
    total_blocks = int(np.ceil(DURATION_S * sr / bs))
    out = np.empty((2, total_blocks * bs), dtype=np.float32)

    pending = sorted(EVENTS, key=lambda e: e[0])
    idx = 0
    for b in range(total_blocks):
        t_next = (b + 1) * bs / sr
        # Fire everything scheduled at or before this block's end.
        while idx < len(pending) and pending[idx][0] < t_next:
            _, kind, a, c = pending[idx]
            getattr(eng, KIND[kind])(a, c)
            idx += 1
        out[:, b * bs : (b + 1) * bs] = eng.render_block()

    eng.all_sound_off()

    n = int(DURATION_S * sr)
    audio = np.ascontiguousarray(out[:, :n])
    mono = audio.mean(axis=0)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(mono**2)))

    out_dir = ROOT / "renders"
    out_dir.mkdir(exist_ok=True)
    wav = out_dir / "engine_concert_grand_triad.wav"
    sf.write(str(wav), audio.T, sr, subtype="FLOAT")

    print(f"wav           : {wav}")
    print(f"dur_s         : {n / sr:.3f}")
    print(f"peak_abs      : {peak:.6f}")
    print(f"rms           : {rms:.6f}")
    print(f"dBFS peak     : {20 * np.log10(peak + 1e-12):.2f}")
    print(f"dBFS rms      : {20 * np.log10(rms + 1e-12):.2f}")
    print(f"silent?       : {'YES (BAD)' if peak < 1e-6 else 'no (good)'}")

    win = sr // 10
    env = [float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2))) for i in range(n // win)]
    print("envelope(100ms rms):")
    for i, e in enumerate(env):
        print(f"  {i * 0.1:4.1f}s {e:.5f} {'#' * int(e / max(env) * 40)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())