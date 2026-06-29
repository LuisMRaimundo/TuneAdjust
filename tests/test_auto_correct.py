"""Tests for auto_correct planning and path rules."""

from pathlib import Path

import pytest

from auto_correct import AUTO_RETUNE_MAX_CENTS, build_renamed_path, plan_corrections
import pitch_core as pc


def test_auto_retune_max_is_half_semitone():
    assert AUTO_RETUNE_MAX_CENTS == 50.0


def test_plan_retune_out_of_tune_within_half_step():
    actions = plan_corrections(
        "OUT_OF_TUNE", "A4", "A4", 435.0, 440.0, 19.6
    )
    assert actions == ["retune"]


def test_plan_no_retune_beyond_half_step():
    actions = plan_corrections(
        "OUT_OF_TUNE", "A4", "A4", 415.0, 440.0, 100.0
    )
    assert "retune" not in actions
    assert "rename" not in actions


def test_plan_rename_out_of_tune_beyond_half_step_different_note():
    """F3 file playing ~E3: rename to closest detected note."""
    actions = plan_corrections(
        "OUT_OF_TUNE", "F3", "E3", 164.81, pc.note_to_frequency("F3"), 68.2
    )
    assert actions == ["rename"]


def test_plan_no_rename_out_of_tune_same_note_beyond_half_step():
    actions = plan_corrections(
        "OUT_OF_TUNE", "A4", "A4", 415.0, 440.0, 100.0
    )
    assert actions == []


def test_plan_rename_mislabeled():
    actions = plan_corrections(
        "MISLABELED", "B4", "A4", 440.0, pc.note_to_frequency("B4"), 200.0
    )
    assert actions == ["rename"]


def test_plan_rename_then_retune_small_drift():
    actions = plan_corrections(
        "MISLABELED", "A#4", "A4", 438.0, pc.note_to_frequency("A#4"), 35.0
    )
    assert actions == ["rename", "retune"]


def test_plan_rename_octave_error():
    actions = plan_corrections(
        "OCTAVE_ERROR", "C4", "C5", 523.25, pc.note_to_frequency("C4"), 1200.0
    )
    assert actions[0] == "rename"


def test_plan_ok_empty():
    assert plan_corrections("OK", "A4", "A4", 440.0, 440.0, 0.0) == []


def test_build_renamed_path_replaces_note_token():
    p = Path("Violin_A4_sustain.wav")
    out = build_renamed_path(p, "Bb4")
    assert out.name == "Violin_Bb4_sustain.wav"


def test_resolve_unique_rename_path_collision(tmp_path):
    from auto_correct import resolve_unique_rename_path
    folder = tmp_path
    (folder / "A2.aif").write_bytes(b"x")
    a1 = folder / "A1.aif"
    a1.write_bytes(b"y")
    out = resolve_unique_rename_path(a1, "A2")
    assert out.name == "A2_2.aif"


def test_list_batch_folders(tmp_path):
    from auto_correct import list_batch_folders, list_audio_files

    parent = tmp_path / "parent"
    sub_a = parent / "set_a"
    sub_b = parent / "set_b"
    sub_empty = parent / "empty"
    sub_a.mkdir(parents=True)
    sub_b.mkdir(parents=True)
    sub_empty.mkdir(parents=True)
    (sub_a / "A4.wav").write_bytes(b"x")
    (sub_b / "B4.wav").write_bytes(b"y")

    folders = list_batch_folders(parent, include_parent=False, recursive=False)
    assert [f.name for f in folders] == ["set_a", "set_b"]
    assert len(list_audio_files(sub_a)) == 1


def test_list_batch_folders_recursive(tmp_path):
    from auto_correct import batch_folder_label, list_batch_folders

    parent = tmp_path / "parent"
    nested = parent / "layer_a" / "layer_b"
    shallow = parent / "top"
    nested.mkdir(parents=True)
    shallow.mkdir(parents=True)
    (nested / "C4.wav").write_bytes(b"x")
    (shallow / "D4.wav").write_bytes(b"y")

    shallow_only = list_batch_folders(parent, include_parent=False, recursive=False)
    assert [f.name for f in shallow_only] == ["top"]

    all_levels = list_batch_folders(parent, include_parent=False, recursive=True)
    labels = [batch_folder_label(f, parent) for f in all_levels]
    assert "top" in labels
    assert "layer_a/layer_b" in labels
    assert len(all_levels) == 2


def test_build_renamed_path_rightmost_note():
    p = Path("take_A4_C5.wav")
    out = build_renamed_path(p, "D5")
    assert out.name == "take_A4_D5.wav"
