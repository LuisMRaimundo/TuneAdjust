"""Live Tuner helpers: signed cents and parse path (no Tk mainloop)."""

import pitch_core as pc
import live_tuner as lt


def test_live_tuner_uses_signed_cents_and_parse_note():
    assert lt.signed_cents_difference is pc.signed_cents_difference
    assert lt.parse_note("Bb4") == ("A#", 4)
    flat = pc.signed_cents_difference(430.0, 440.0)
    assert flat < 0
