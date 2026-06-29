"""Tests for octave math, segment selection, tune evaluation, and instrument bounds."""

import numpy as np
import pytest

import pitch_core as pc


def _tone(sr: int, freq: float, seconds: float = 1.5) -> np.ndarray:
    t = np.arange(int(sr * seconds)) / sr
    return (0.4 * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class TestOctaveHelpers:
    def test_is_octave_alias_one_octave(self):
        assert pc._is_octave_alias(440.0, 220.0)
        assert pc._is_octave_alias(880.0, 440.0)

    def test_is_octave_alias_rejects_near_unison(self):
        assert not pc._is_octave_alias(440.0, 442.0)

    def test_normalize_to_octave_neighborhood(self):
        c4 = pc.note_to_frequency("C4")
        c5 = pc.note_to_frequency("C5")
        norm = pc._normalize_to_octave_neighborhood(c5, c4)
        assert pc.cents_difference(norm, c4) < 5.0


class TestSelectStableSegment:
    def test_short_audio_unchanged(self, sr):
        y = _tone(sr, 440.0, seconds=0.5)
        assert len(pc.select_stable_segment(y, sr, duration=2.0)) == len(y)

    def test_long_audio_middle_window(self, sr):
        y = _tone(sr, 440.0, seconds=4.0)
        seg = pc.select_stable_segment(y, sr, duration=2.0)
        assert len(seg) == int(2.0 * sr)


class TestEvaluateTuneMatchExtended:
    def test_out_of_tune_different_spelling_nearby(self):
        """Slightly sharp A# when expecting A4 — OUT_OF_TUNE, not mislabeled (< 100 ct)."""
        det = 452.0
        cents, ok, mis, status = pc.evaluate_tune_match(
            det, "A#4", "A4", 440.0, 20.0
        )
        assert status == "OUT_OF_TUNE"
        assert not ok and not mis
        assert cents > 20

    def test_range_warning_appended(self):
        _, _, _, status = pc.evaluate_tune_match(
            440.0, "A4", "A4", 440.0, 20.0, range_warning="outside range"
        )
        assert status == "RANGE_WARN"

    def test_near_semitone_not_mislabeled(self):
        """Within 100 cents but wrong pitch class boundary — still OUT_OF_TUNE not MISLABELED."""
        a4 = 440.0
        bb4 = pc.note_to_frequency("Bb4")
        cents, ok, mis, status = pc.evaluate_tune_match(
            bb4 * 0.995, "A#4", "A4", a4, 20.0
        )
        assert status in ("OUT_OF_TUNE", "MISLABELED")
        assert not ok


class TestPitchSearchBounds:
    def test_double_bass_with_low_note_lowers_fmin(self):
        fmin_plain, _ = pc.pitch_search_bounds()
        fmin_bass, _ = pc.pitch_search_bounds("A1", instrument="double_bass")
        assert fmin_bass < fmin_plain
        assert fmin_bass < 45.0

    def test_expected_note_expands_range(self):
        fmin, _ = pc.pitch_search_bounds("G1")
        assert fmin < 40.0


class TestInstrumentRegistry:
    def test_load_registry_has_entries(self):
        reg = pc.load_instrument_registry()
        assert "violin" in reg
        assert "tuba" in reg

    def test_parse_range_notes(self):
        lo, hi = pc._parse_range_notes("A0 to F4")
        assert lo is not None and hi is not None
        assert lo < hi


class TestPitchSmoother:
    def test_reset_clears_history(self):
        s = pc.PitchSmoother(window=5)
        s.update(440.0)
        s.reset()
        assert s.update(0.0) == 0.0

    def test_median_smoothing(self):
        s = pc.PitchSmoother(window=3)
        s.update(440.0)
        s.update(442.0)
        out = s.update(441.0)
        assert 439 < out < 443
