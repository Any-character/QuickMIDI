"""Tests for qmidi.mastering -- the MP3 mastering chain.

Loudness is checked against a value that can be derived by hand (a stereo sine
at -20 dBFS peak is about -23 LUFS), because "sounds fine" is not a test.  The
tolerances are wide enough to accept the magnitude-only K-weighting, and tight
enough to catch the two bugs this file already had: an IR that doubled the
filter (every level 4 dB high) and a smoothing helper that silently shrank the
signal.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from qmidi import mastering  # noqa: E402
from qmidi.mastering import (  # noqa: E402
    MasterSettings,
    compress,
    limit,
    loudness_lufs,
    make_ir,
    master,
    reverb,
)

SR = 48000


def _sine(freq=1000.0, amplitude=0.3, seconds=1.0, channels=2):
    t = np.arange(int(SR * seconds)) / SR
    mono = np.sin(2.0 * np.pi * freq * t) * amplitude
    return np.stack([mono] * channels).astype(np.float32)


# --------------------------------------------------------------------------- #
# loudness
# --------------------------------------------------------------------------- #
def test_loudness_of_a_known_sine_matches_the_bs1770_formula():
    """Stereo sine, peak -10.46 dBFS -> RMS -13.47 -> sum +3.01 -> -0.691."""

    audio = _sine(1000.0, 0.3, 2.0)
    expected = 20 * np.log10(0.3) - 3.01 + 3.01 - 0.691
    assert loudness_lufs(audio, SR) == pytest.approx(expected, abs=1.2)


def test_silence_has_no_loudness():
    assert loudness_lufs(np.zeros((2, SR), dtype=np.float32), SR) == float("-inf")


def test_loudness_ignores_low_frequency_rumble_more_than_midrange():
    """K-weighting is a high-pass: 40 Hz must read quieter than 1 kHz."""
    low = loudness_lufs(_sine(40.0, 0.3, 2.0), SR)
    mid = loudness_lufs(_sine(1000.0, 0.3, 2.0), SR)
    assert low < mid - 6.0


# --------------------------------------------------------------------------- #
# chain
# --------------------------------------------------------------------------- #
def test_disabled_settings_return_the_input_untouched():
    audio = _sine()
    out, report = master(audio, SR, MasterSettings(enabled=False))
    assert np.array_equal(out, audio)
    assert report.gain_db == 0.0


def test_mastering_hits_the_requested_loudness():
    audio = _sine(220.0, 0.25, 3.0)
    out, report = master(audio, SR, MasterSettings(enabled=True, target_lufs=-14.0))
    assert report.lufs_after == pytest.approx(-14.0, abs=0.6)
    assert np.all(np.isfinite(out))


def test_mastering_never_exceeds_the_ceiling():
    audio = _sine(220.0, 0.9, 2.0)
    out, _ = master(audio, SR, MasterSettings(enabled=True, target_lufs=-9.0,
                                              ceiling_db=-1.0))
    ceiling = 10 ** (-1.0 / 20.0)
    assert float(np.max(np.abs(out))) <= ceiling + 1e-6


def test_the_output_keeps_its_dynamics_below_the_threshold():
    """A quiet passage must not be squashed: gain stays ~1 under the threshold."""
    quiet = _sine(440.0, 0.01, 1.0)
    out = compress(quiet, SR, threshold_db=-18.0, ratio=4.0)
    assert np.sqrt(np.mean(out ** 2)) == pytest.approx(
        np.sqrt(np.mean(quiet ** 2)), rel=0.02)


def test_compression_reduces_a_signal_above_the_threshold():
    loud = _sine(440.0, 0.7, 1.0)
    out = compress(loud, SR, threshold_db=-18.0, ratio=4.0)
    assert np.sqrt(np.mean(out ** 2)) < np.sqrt(np.mean(loud ** 2)) * 0.9


def test_limiter_holds_the_ceiling_on_a_transient():
    audio = np.zeros((2, SR), dtype=np.float32)
    audio[:, 1000:1010] = 3.0
    out = limit(audio, SR, ceiling_db=-1.0)
    assert float(np.max(np.abs(out))) <= 10 ** (-1.0 / 20.0) + 1e-6


def test_the_limiter_does_not_shrink_a_signal_already_below_the_ceiling():
    audio = _sine(880.0, 0.2, 1.0)
    out = limit(audio, SR, ceiling_db=-1.0)
    assert np.allclose(out, audio, atol=0.02)


# --------------------------------------------------------------------------- #
# reverb
# --------------------------------------------------------------------------- #
def test_impulse_response_is_energy_normalised():
    ir = make_ir(SR, 1.5)
    energy = np.sqrt(np.sum(ir.astype(np.float64) ** 2, axis=1))
    assert energy == pytest.approx(np.ones(2), abs=1e-3)


def test_wet_zero_is_a_no_op():
    audio = _sine()
    assert np.array_equal(reverb(audio, SR, wet=0.0), audio)


def test_reverb_adds_a_tail_after_the_dry_signal_ends():
    audio = np.zeros((2, SR), dtype=np.float32)
    audio[:, 1000:1200] = 0.8  # a 4 ms click
    wet = reverb(audio, SR, wet=0.5, seconds=1.0)
    assert np.max(np.abs(wet[:, -SR // 2 :])) > 1e-3


def test_fft_convolve_matches_a_direct_convolution_on_a_short_signal():
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((2, 400)).astype(np.float32)
    ir = rng.standard_normal((2, 37)).astype(np.float32)
    fast = mastering.fft_convolve(audio, ir, block=64)
    slow = np.stack([np.convolve(audio[c], ir[c])[:400] for c in range(2)])
    assert np.allclose(fast, slow, atol=1e-4)
