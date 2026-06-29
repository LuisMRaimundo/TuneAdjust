"""Unit tests for note parsing, MIDI mapping, and frequency table (pitch_core)."""

import pytest

import pitch_core as pc


class TestParseNote:
    def test_sharp_and_flat_spellings(self):
        assert pc.parse_note("F#2") == ("F#", 2)
        assert pc.parse_note("Bb4") == ("A#", 4)
        assert pc.parse_note("Db3") == ("C#", 3)

    def test_unicode_accidentals(self):
        assert pc.parse_note("G♯4") == ("G#", 4)
        assert pc.parse_note("A♭3") == ("G#", 3)

    def test_enharmonic_aliases(self):
        assert pc.parse_note("E#4") == ("F", 4)
        assert pc.parse_note("Cb4") == ("B", 4)  # octave from token, not respelled down
        assert pc.parse_note("B#3") == ("C", 3)

    def test_rejects_invalid_octave(self):
        assert pc.parse_note("A9") is None
        assert pc.parse_note("C-1") is None

    def test_rejects_velocity_false_positive(self):
        """Filenames like F-2 from velocity suffixes must not parse as F octave -2."""
        assert pc.parse_note("F-2") is None

    def test_rejects_garbage(self):
        assert pc.parse_note("") is None
        assert pc.parse_note("velocity_127") is None
        assert pc.parse_note(None) is None


class TestParseAllNotes:
    def test_collects_multiple_tokens(self):
        found = pc.parse_all_notes("take_A4_C5_final")
        assert len(found) == 2
        assert found[0][:2] == ("A", 4)
        assert found[1][:2] == ("C", 5)

    def test_skips_invalid_octaves_in_filename(self):
        found = pc.parse_all_notes("sample_F-2-ff-2c")
        assert found == []


class TestMidiRoundtrip:
    @pytest.mark.parametrize(
        "note,octave",
        [("C", 4), ("F#", 2), ("A#", 1), ("B", 0), ("G", 8)],
    )
    def test_midi_roundtrip(self, note, octave):
        midi = pc.note_to_midi(note, octave)
        back = pc.midi_to_note(midi)
        assert back == (note, octave)


class TestFrequencyTable:
    def test_a0_and_c8_boundaries(self):
        assert pc.note_to_frequency("A0") == pytest.approx(27.5, rel=1e-3)
        assert pc.note_to_frequency("C8") == pytest.approx(4186.0, rel=1e-2)

    def test_flat_entries_match_sharp(self):
        assert pc.note_to_frequency("Bb4") == pc.note_to_frequency("A#4")


class TestFrequencyToNote:
    def test_known_pitch(self):
        assert pc.frequency_to_note(440.0) in ("A4", "A#3", "Ab4")

    def test_invalid_returns_unknown(self):
        assert pc.frequency_to_note(0.0) == "Unknown"
        assert pc.frequency_to_note(-10.0) == "Unknown"


class TestSamePitchClass:
    def test_octave_variants(self):
        assert pc.same_pitch_class("C4", "C5")
        assert not pc.same_pitch_class("C4", "D4")
