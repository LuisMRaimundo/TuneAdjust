"""Extended tests for pitch_shift_tool helpers and format handling."""

import numpy as np
import pytest
import soundfile as sf

import subprocess
import sys
from pathlib import Path

from pitch_shift_tool import (
    _adaptive_n_fft,
    _restore_global_rms,
    downmix_mono,
    load_audio_native,
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


class TestStereoAndCLI:
    def test_pitch_shift_preserves_stereo_shape(self):
        sr = 22050
        t = np.arange(sr) / sr
        left = (0.4 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        right = (0.25 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
        stereo = np.stack([left, right], axis=1)
        out = pitch_shift_audio(stereo, sr, 0.25)
        assert out.shape == stereo.shape
        assert np.isfinite(out).all()

    def test_load_audio_native_keeps_channels(self, tmp_path):
        sr = 22050
        t = np.arange(sr) / sr
        stereo = np.stack([0.3 * np.sin(2 * np.pi * 440 * t)] * 2, axis=1).astype(np.float32)
        path = tmp_path / "stereo_A4.wav"
        sf.write(str(path), stereo, sr, format="WAV")
        audio, sr_back = load_audio_native(path)
        assert sr_back == sr
        assert audio.ndim == 2 and audio.shape[1] == 2
        mix = downmix_mono(audio)
        assert mix.ndim == 1 and len(mix) == audio.shape[0]

    def test_cli_requires_input_without_show_table(self):
        script = Path(__file__).resolve().parents[1] / "pitch_shift_tool.py"
        proc = subprocess.run(
            [sys.executable, str(script)],
            capture_output=True,
            text=True,
        )
        assert proc.returncode != 0
        combined = proc.stderr + proc.stdout
        assert "TypeError" not in combined
        assert "required" in combined.lower() or "usage" in combined.lower()
