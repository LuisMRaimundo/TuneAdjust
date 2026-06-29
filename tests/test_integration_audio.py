"""Integration tests: analyze → plan → apply on synthetic WAV samples."""

import numpy as np
import pytest
import soundfile as sf

from auto_correct import analyze_file, apply_auto_corrections, plan_corrections
import pitch_core as pc


def _write_tone_wav(path, freq: float, sr: int = 22050, seconds: float = 2.0, detune_cents: float = 0.0):
    """Write a monophonic sine WAV; optional detune in cents."""
    f = freq * (2.0 ** (detune_cents / 1200.0))
    t = np.arange(int(sr * seconds)) / sr
    y = (0.45 * np.sin(2 * np.pi * f * t)).astype(np.float32)
    sf.write(str(path), y, sr, format="WAV")


@pytest.fixture
def sr():
    return 22050


class TestAnalyzeFile:
    def test_in_tune_sample_reports_ok(self, tmp_path, sr):
        fp = tmp_path / "Violin_A4_1.wav"
        _write_tone_wav(fp, pc.note_to_frequency("A4"), sr=sr)
        result = analyze_file(fp, tolerance=20.0)
        assert result.status == "OK"
        assert result.expected_note == "A4"
        assert result.cents_off < 20

    def test_detuned_sample_within_retune_range(self, tmp_path, sr):
        fp = tmp_path / "Violin_A4_1.wav"
        _write_tone_wav(fp, pc.note_to_frequency("A4"), sr=sr, detune_cents=-35.0)
        result = analyze_file(fp, tolerance=20.0)
        assert result.cents_off > 20
        assert result.cents_off <= pc.AUTO_RETUNE_MAX_CENTS
        planned = plan_corrections(
            result.status,
            result.expected_note,
            result.detected_note,
            result.detected_freq,
            result.expected_freq,
            result.cents_off,
        )
        assert "retune" in planned

    def test_mislabeled_filename(self, tmp_path, sr):
        fp = tmp_path / "Violin_B4_1.wav"
        _write_tone_wav(fp, pc.note_to_frequency("A4"), sr=sr)
        result = analyze_file(fp, tolerance=20.0)
        assert result.status.split("+")[0] == "MISLABELED"


class TestApplyAutoCorrections:
    def test_retune_in_place_improves_tuning(self, tmp_path, sr):
        fp = tmp_path / "Cello_A3_1.wav"
        target = pc.note_to_frequency("A3")
        _write_tone_wav(fp, target, sr=sr, detune_cents=30.0)
        before = analyze_file(fp, tolerance=20.0)
        assert before.cents_off > 20

        planned = plan_corrections(
            before.status,
            before.expected_note,
            before.detected_note,
            before.detected_freq,
            before.expected_freq,
            before.cents_off,
        )
        assert "retune" in planned

        new_path, actions = apply_auto_corrections(
            fp,
            before.status,
            before.expected_note,
            before.detected_note,
            before.detected_freq,
            before.expected_freq,
            before.cents_off,
        )
        assert new_path.exists()
        assert any(a.action == "retune" and a.success for a in actions)

        after = analyze_file(new_path, tolerance=20.0)
        assert after.cents_off < before.cents_off
        assert after.status == "OK"

    def test_rename_mislabeled_file(self, tmp_path, sr):
        fp = tmp_path / "Horn_F4_1.wav"
        _write_tone_wav(fp, pc.note_to_frequency("G4"), sr=sr)
        before = analyze_file(fp, tolerance=20.0)
        assert before.status.split("+")[0] == "MISLABELED"

        new_path, actions = apply_auto_corrections(
            fp,
            before.status,
            before.expected_note,
            before.detected_note,
            before.detected_freq,
            before.expected_freq,
            before.cents_off,
        )
        assert any(a.action == "rename" and a.success for a in actions)
        assert "G4" in new_path.stem
        assert not fp.exists()
