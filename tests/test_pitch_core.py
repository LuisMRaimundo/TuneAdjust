"""Tests for pitch_core."""

import numpy as np
import pytest

import pitch_core as pc


def _tone(sr: int, freq: float, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def test_parse_note_from_filename_rightmost():
    assert pc.parse_note_from_filename("Violin_A4_sustain_C5") == ("C", 5)
    assert pc.parse_note_from_filename("A4_only") == ("A", 4)


def test_parse_orchidea_velocity_suffix_filenames():
    """Orchidea-style names: note before -ff/-mf/-pp velocity + round-robin (-2c)."""
    assert pc.parse_note_from_filename("OrchSol_Cr.Baixo_F#2-ff-2c-N") == ("F#", 2)
    assert pc.parse_note_from_filename("OrchSol_Cr.Baixo_F#2-mf-2c-N") == ("F#", 2)
    assert pc.parse_note_from_filename("OrchSol_Cr.Baixo_F#2-pp-2c-N") == ("F#", 2)
    assert pc.parse_note_from_filename("OrchSol_Cr.Baixo_F#1-ff-4c-N") == ("F#", 1)
    assert pc.parse_note_from_filename("OrchSol_Cr.Baixo_G#2-ff-2c-N") == ("G#", 2)


def test_parse_note_from_filename_first_strategy():
    assert pc.parse_note_from_filename("A4_only", strategy="first") == ("A", 4)


def test_note_to_frequency_a4():
    assert pc.note_to_frequency("A4") == pytest.approx(440.0, rel=1e-3)


def test_are_enharmonic():
    assert pc.are_enharmonic("A#4", "Bb4")
    assert not pc.are_enharmonic("A4", "B4")


def test_cents_difference():
    assert pc.cents_difference(440.0, 440.0) == 0.0
    assert pc.cents_difference(466.16, 440.0) == pytest.approx(100.0, abs=1.0)


def test_cross_check_expected_table():
    r = pc.cross_check_expected_table(440.0, "A4", tolerance_cents=10)
    assert r["validated"] and r["is_valid"]
    r2 = pc.cross_check_expected_table(466.0, "A4", tolerance_cents=10)
    assert r2["validated"] and not r2["is_valid"]


def test_detect_frequency_pure_tone(sr):
    y = _tone(sr, 440.0)
    f = pc.detect_frequency(y, sr)
    assert 420 < f < 460


def test_detect_frequency_robust_pure_tone(sr):
    y = _tone(sr, 261.63, seconds=0.4)
    f = pc.detect_frequency_robust(y, sr)
    assert 240 < f < 285


def test_evaluate_tune_match_ok():
    cents, ok, mis, status = pc.evaluate_tune_match(440.0, "A4", "A4", 440.0, 20.0)
    assert status == "OK" and ok and not mis
    assert cents == pytest.approx(0.0, abs=0.1)


def test_evaluate_tune_match_mislabeled():
    cents, ok, mis, status = pc.evaluate_tune_match(440.0, "A4", "B4", pc.note_to_frequency("B4"), 20.0)
    assert status == "MISLABELED" and mis


def test_evaluate_tune_match_no_detection():
    cents, ok, mis, status = pc.evaluate_tune_match(0.0, "Unknown", "A4", 440.0, 20.0)
    assert status == "NO_DETECTION"


def test_detect_frequency_fast(sr):
    y = _tone(sr, 220.0)
    f = pc.detect_frequency(y, sr, fast=True, apply_harmonic_check=False)
    assert 200 < f < 240


def test_harmonic_correction_octave_high(sr):
    """When the detector returns 2× f0, resolve_fundamental_octave should halve to ~220 Hz."""
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.8 * np.sin(2 * np.pi * 220.0 * t) + 0.2 * np.sin(2 * np.pi * 440.0 * t)
    corrected = pc.resolve_fundamental_octave(y, sr, 440.0, fmin=65.0, fmax=2000.0)
    assert 190 < corrected < 250


def test_align_frequency_to_expected_octave():
    c5_hz = pc.note_to_frequency("C5")
    aligned = pc.align_frequency_to_expected_octave(c5_hz, "C4")
    assert pc.frequency_to_note(aligned) in ("C4", "B3", "C#4")
    assert pc.cents_difference(aligned, pc.note_to_frequency("C4")) < 30


def test_align_skips_when_already_close():
    c4_hz = pc.note_to_frequency("C4")
    aligned = pc.align_frequency_to_expected_octave(c4_hz * 1.01, "C4")
    assert pc.cents_difference(aligned, c4_hz) < 30


def test_pitch_smoother_octave_stable():
    s = pc.PitchSmoother(window=5)
    c4 = pc.note_to_frequency("C4")
    c5 = pc.note_to_frequency("C5")
    out = s.update(c4)
    for _ in range(4):
        out = s.update(c5)
    assert pc.cents_difference(out, c4) < 100


def test_detect_pitch_octave_with_expected(sr):
  # Fundamental C4; detector might lock high — filename should pull to C4
    f0 = pc.note_to_frequency("C4")
    t = np.arange(int(sr * 1.5)) / sr
    y = 0.5 * np.sin(2 * np.pi * f0 * t) + 0.45 * np.sin(2 * np.pi * 2 * f0 * t)
    hz = pc.detect_pitch(y, sr, expected_note="C4")
    assert pc.cents_difference(hz, f0) < 50


def test_evaluate_tune_match_octave_not_mislabeled():
    c5 = pc.note_to_frequency("C5")
    cents, ok, mis, status = pc.evaluate_tune_match(c5, "C5", "C4", pc.note_to_frequency("C4"), 20.0)
    assert status == "OCTAVE_ERROR"
    assert not mis


def test_check_instrument_range_warning():
    msg = pc.check_instrument_range(80.0, "flute")
    assert msg is not None and "flute" in msg.lower()


def test_check_instrument_range_ok():
    assert pc.check_instrument_range(440.0, "flute") is None


def test_generate_chromatic_scale():
    scale = pc.generate_chromatic_scale("C4", "E4")
    names = [f"{n}{o}" for n, o in scale]
    assert names == ["C4", "C#4", "D4", "D#4", "E4"]


def test_calculate_semitones():
    assert pc.calculate_semitones(425.0, 440.0) == pytest.approx(0.60, abs=0.05)


def test_pitch_search_bounds_low_bass():
    fmin, fmax = pc.pitch_search_bounds("A#1")
    assert fmin < 45.0
    assert fmax > 80.0


def test_detect_pitch_low_a_sharp1(sr):
    hz_target = pc.note_to_frequency("A#1")
    y = _tone(sr, hz_target, seconds=2.0)
    detected = pc.detect_pitch(y, sr, expected_note="A#1")
    assert pc.cents_difference(detected, hz_target) < 80


def test_list_instruments():
    inst = pc.list_instruments()
    assert "violin" in inst and "tuba" in inst and "double_bass" in inst
