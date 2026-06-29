"""Extended tests for pitch_shift_tool helpers and format handling."""

import numpy as np
import pytest
import soundfile as sf

from pitch_shift_tool import (
    _adaptive_n_fft,
    _restore_global_rms,
    soundfile_format_kwargs,
    save_audio_preserving_format,
    pitch_shift_audio,
)


class TestAdaptiveNFFT:
    @pytest.mark.parametrize(
        "semitones,expected",
        [(0.05, 4096), (0.5, 2048), (2.0, 2048)],
    )
    def test_fft_size_tiers(self, semitones, expected):
        assert _adaptive_n_fft(semitones) == expected


class TestRestoreGlobalRMS:
    def test_restores_loudness(self):
        orig = np.random.randn(8000).astype(np.float32) * 0.3
        quiet = orig * 0.7
        restored = _restore_global_rms(orig, quiet)
        r_orig = np.sqrt(np.mean(orig ** 2))
        r_rest = np.sqrt(np.mean(restored ** 2))
        assert abs(r_rest / r_orig - 1.0) < 0.01

    def test_silent_shift_unchanged(self):
        orig = np.zeros(1000, dtype=np.float32)
        out = _restore_global_rms(orig, orig.copy())
        np.testing.assert_array_equal(out, orig)


class TestSoundfileFormatKwargs:
    @pytest.mark.parametrize(
        "ext,expected_key",
        [
            (".flac", "FLAC"),
            (".ogg", "OGG"),
            (".aifc", "AIFF"),
        ],
    )
    def test_format_mapping(self, ext, expected_key):
        kw = soundfile_format_kwargs(ext)
        assert kw.get("format") == expected_key


class TestSaveAudioPreservingFormat:
    def test_save_wav_roundtrip(self, tmp_path):
        sr = 22050
        y = np.sin(2 * np.pi * 440 * np.arange(sr) / sr).astype(np.float32) * 0.5
        out = tmp_path / "A4.wav"
        assert save_audio_preserving_format(y, sr, out, ".wav")
        loaded, sr_back = sf.read(str(out))
        assert sr_back == sr
        assert len(loaded) == sr

    def test_save_aiff(self, tmp_path):
        sr = 22050
        y = np.zeros(sr, dtype=np.float32)
        out = tmp_path / "A4.aif"
        assert save_audio_preserving_format(y, sr, out, ".aif")
        assert out.exists()


class TestPitchShiftNegativeSemitones:
    def test_downward_shift(self):
        sr = 22050
        t = np.arange(sr) / sr
        y = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        out = pitch_shift_audio(y, sr, -0.25)
        assert out.shape == y.shape
        assert not np.allclose(out, y)
