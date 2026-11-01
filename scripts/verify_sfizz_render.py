"""ONE-OFF prototype validation: does pysfizz render a real Concert Grand scene?

Not business code. Safe to delete once the prototype renderer exists.

Timeline (48 kHz, 3.0 s):
  0.00  CC64 = 127   (sustain pedal DOWN)
  0.10  note_on  C4 vel 80
  0.20  note_on  E4 vel 80
  0.30  note_on  G4 vel 80
  1.30  note_off C4/E4/G4   (pedal still down -> notes should sustain)
  2.00  CC64 = 0     (pedal UP -> release tail)
  3.00  end
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import soundfile as sf
from pysfizz.synth import Synth

ROOT = Path(__file__).resolve().parent.parent

# The .sfz_links entry is a directory symlink/junction -- resolve it, never
# hardcode the target.
LINK = ROOT / ".sfz_links" / "Concert_Grand__Reverb"

DURATION_S = 3.0
VEL = 80
# Middle-register C major triad.
TRIAD = (60, 64, 67)  # C4 E4 G4

# (time_s, kind, a, b)
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

SAMPLE_RATE = 48000  # pysfizz engine default; asserted below
BLOCK_SIZE = 1024


def find_main_sfz(directory: Path) -> Path:
    """Canonical main file: the one whose stem ENDS in 'Concert Grand'.

    All 4 candidates here are byte-distinct variants:
      Concert Grand / (Realistic) / (2 seconds release) / (2.5 seconds release)
    The unparenthesised one is the base preset; the others are named variants
    with baked-in tweaks, so they must NOT be picked by size (all 4 are 5626 B).
    """
    candidates = sorted(directory.glob("*.sfz"))
    if not candidates:
        raise SystemExit(f"no .sfz under {directory}")
    base = [c for c in candidates if c.stem.strip().lower().endswith("concert grand")]
    if base:
        return base[0]
    return max(candidates, key=lambda p: p.stat().st_size)


def main() -> int:
    if not LINK.exists():
        raise SystemExit(f"missing sfz link: {LINK}")
    sfz = find_main_sfz(LINK)
    print(f"sfz main file : {sfz.name}")
    print(f"sfz resolved  : {sfz.resolve()}")

    # Public wrapper for loading (this part is stable API).
    synth = Synth(sample_rate=SAMPLE_RATE, block_size=BLOCK_SIZE)
    if not synth.load_sfz_file(str(sfz)):
        raise SystemExit("load_sfz_file() returned False")
    print(f"regions       : {synth._synth.get_num_regions()}")
    print(f"playable keys : {len(synth.playable_keys)}")

    # NOTE: 0.1.3 exposes no public note_on/note_off/cc -- they live on the
    # private _sfizz handle. This is the crux of the API-sufficiency question.
    eng = synth._synth
    assert eng.get_sample_rate() == SAMPLE_RATE
    assert eng.get_block_size() == BLOCK_SIZE

    total_blocks = int(np.ceil(DURATION_S * SAMPLE_RATE / BLOCK_SIZE))
    left = np.empty((2, total_blocks * BLOCK_SIZE), dtype=np.float32)

    pending = sorted(EVENTS, key=lambda e: e[0])
    idx = 0

    for b in range(total_blocks):
        t_block = b * BLOCK_SIZE / SAMPLE_RATE
        t_next = (b + 1) * BLOCK_SIZE / SAMPLE_RATE
        # Fire every event whose timestamp falls inside this block.
        while idx < len(pending) and pending[idx][0] < t_next:
            _, kind, a, c = pending[idx]
            if kind == "cc":
                eng.cc(0, a, c)
            elif kind == "on":
                eng.note_on(0, a, c)
            else:
                eng.note_off(0, a, c)
            idx += 1
        l, r = eng.render_block()
        left[0, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE] = l
        left[1, b * BLOCK_SIZE : (b + 1) * BLOCK_SIZE] = r

    n = int(DURATION_S * SAMPLE_RATE)
    audio = np.ascontiguousarray(left[:, :n])

    mono = audio.mean(axis=0)
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(mono**2)))

    out_dir = ROOT / "renders"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / "verify_concert_grand_triad.wav"
    sf.write(str(out), audio.T, SAMPLE_RATE, subtype="FLOAT")

    print(f"wav           : {out}")
    print(f"dur_s         : {n / SAMPLE_RATE:.3f}")
    print(f"peak_abs      : {peak:.6f}")
    print(f"rms           : {rms:.6f}")
    print(f"dBFS peak     : {20 * np.log10(peak + 1e-12):.2f}")
    print(f"dBFS rms      : {20 * np.log10(rms + 1e-12):.2f}")
    print(f"int16 clip?   : {'YES' if peak > 1.0 else 'no'}")
    print(f"silent?       : {'YES (BAD)' if peak < 1e-6 else 'no (good)'}")

    # Coarse envelope so we can see note-on, sustain-under-pedal, then release.
    win = SAMPLE_RATE // 10
    env = [float(np.sqrt(np.mean(mono[i * win : (i + 1) * win] ** 2))) for i in range(n // win)]
    print("envelope(100ms rms):")
    for i, e in enumerate(env):
        bar = "#" * int(e / max(env) * 40)
        print(f"  {i * 0.1:4.1f}s {e:.5f} {bar}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
