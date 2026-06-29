"""Extended tests for auto_correct planning, renaming, and display helpers."""

from pathlib import Path

import pytest

from auto_correct import (
    AUTO_RETUNE_MAX_CENTS,
    display_note_token,
    should_rename_to_detected,
    plan_corrections,
    build_renamed_path,
    resolve_unique_rename_path,
    list_audio_files,
)
import pitch_core as pc


class TestDisplayNoteToken:
    def test_preserves_flat_spelling(self):
        assert display_note_token("Bb4") == "Bb4"

    def test_normalizes_sharp(self):
        assert display_note_token("A#4") == "A#4"

    def test_invalid_raises(self):
        with pytest.raises(ValueError):
            display_note_token("not_a_note")


class TestShouldRenameToDetected:
    def test_ok_never_renames(self):
        assert not should_rename_to_detected("OK", "A4", "A4", 0.0)

    def test_no_detection_never_renames(self):
        assert not should_rename_to_detected("NO_DETECTION", "A4", "Unknown", float("inf"))

    def test_mislabeled_renames(self):
        assert should_rename_to_detected("MISLABELED", "B4", "A4", 200.0)

    def test_out_of_tune_large_drift_different_note(self):
        assert should_rename_to_detected("OUT_OF_TUNE", "F3", "E3", 68.0)

    def test_out_of_tune_within_retune_range_same_note(self):
        assert not should_rename_to_detected("OUT_OF_TUNE", "A4", "A4", 35.0)

    def test_range_warn_suffix_still_renames_mislabeled(self):
        assert should_rename_to_detected("MISLABELED+RANGE", "B4", "A4", 200.0)


class TestPlanCorrectionsEdgeCases:
    def test_no_note_in_filename_empty(self):
        assert plan_corrections("NO_NOTE_IN_FILENAME", None, "A4", 440.0, 0.0, float("inf")) == []

    def test_enharmonic_same_note_retune(self):
        actions = plan_corrections(
            "OUT_OF_TUNE", "A#4", "Bb4", 438.0, pc.note_to_frequency("A#4"), 35.0
        )
        assert actions == ["retune"]

    def test_octave_error_no_retune_when_cents_huge(self):
        actions = plan_corrections(
            "OCTAVE_ERROR", "C4", "C5", 523.25, pc.note_to_frequency("C4"), 1200.0
        )
        assert actions == ["rename"]
        assert "retune" not in actions


class TestBuildRenamedPath:
    def test_no_note_in_stem_appends_token(self):
        p = Path("sample.wav")
        out = build_renamed_path(p, "G4")
        assert out.name == "sample_G4.wav"

    def test_preserves_bb_spelling(self):
        p = Path("Violin_A4.wav")
        out = build_renamed_path(p, "Bb4")
        assert out.name == "Violin_Bb4.wav"


class TestListAudioFiles:
    def test_finds_common_extensions(self, tmp_path):
        (tmp_path / "A4.wav").write_bytes(b"x")
        (tmp_path / "B4.aif").write_bytes(b"y")
        (tmp_path / "readme.txt").write_bytes(b"z")
        names = {p.name for p in list_audio_files(tmp_path)}
        assert names == {"A4.wav", "B4.aif"}

    def test_ignores_subfolders(self, tmp_path):
        sub = tmp_path / "nested"
        sub.mkdir()
        (sub / "C4.wav").write_bytes(b"x")
        (tmp_path / "D4.wav").write_bytes(b"y")
        assert len(list_audio_files(tmp_path)) == 1
