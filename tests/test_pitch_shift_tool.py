"""Tests for pitch_shift_tool format handling."""





def test_soundfile_format_kwargs_aif():

    from pitch_shift_tool import soundfile_format_kwargs



    assert soundfile_format_kwargs(".aif") == {"format": "AIFF"}

    assert soundfile_format_kwargs(".AIFF") == {"format": "AIFF"}

    assert soundfile_format_kwargs(".wav") == {"format": "WAV"}

