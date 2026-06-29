"""Objective quality checks for pitch_shift_audio (timbre / dynamics preservation)."""

import numpy as np
import pytest
import librosa

from pitch_core import calculate_semitones, cents_difference, detect_pitch
from pitch_shift_tool import pitch_shift_audio


def _string_like_tone(sr: int, f0: float, seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    y = np.zeros_like(t)
    for h, amp in [(1, 1.0), (2, 0.45), (3, 0.25), (4, 0.12)]:
        y += amp * np.sin(2 * np.pi * f0 * h * t)
    env = np.exp(-0.8 * t) * (1.0 - np.exp(-12.0 * t))
    y *= env
    return (y / (np.max(np.abs(y)) + 1e-10)).astype(np.float32)


def _rms_envelope_corr(a: np.ndarray, b: np.ndarray, hop: int = 512) -> float:
    ro = librosa.feature.rms(y=a, hop_length=hop)[0]
    rs = librosa.feature.rms(y=b, hop_length=hop)[0]
    n = min(len(ro), len(rs))
    if n < 3 or np.std(ro[:n]) < 1e-9:
        return 1.0
    return float(np.corrcoef(ro[:n], rs[:n])[0, 1])


@pytest.fixture
def sr():
    return 44100


@pytest.mark.parametrize("cents", [15.0, 35.0, 50.0])
def test_pitch_shift_hits_target_within_auto_retune_range(sr, cents):
    f0 = 220.0
    target = f0 * (2.0 ** (cents / 1200.0))
    st = calculate_semitones(f0, target)
    orig = _string_like_tone(sr, f0)
    shifted = pitch_shift_audio(orig, sr, st)
    detected = detect_pitch(shifted, sr, expected_note="A3")
    assert cents_difference(detected, target) < 8.0


@pytest.mark.parametrize("cents", [20.0, 50.0])
def test_pitch_shift_preserves_loudness_and_envelope(sr, cents):
    f0 = 174.61  # F3
    st = cents / 100.0
    orig = _string_like_tone(sr, f0)
    shifted = pitch_shift_audio(orig, sr, st)
    rms_ratio = np.sqrt(np.mean(shifted ** 2)) / (np.sqrt(np.mean(orig ** 2)) + 1e-10)
    assert 0.98 <= rms_ratio <= 1.02
    assert _rms_envelope_corr(orig, shifted) >= 0.97


def test_pitch_shift_zero_is_noop(sr):
    orig = _string_like_tone(sr, 440.0)
    out = pitch_shift_audio(orig, sr, 0.0)
    np.testing.assert_array_equal(out, orig)
