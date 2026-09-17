# Changelog

## Unreleased

### Fixed

- Same-note drift (for example A4 ±35 cents) is classified as `OUT_OF_TUNE`, not `OCTAVE_ERROR`. The previous test treated any same pitch class as an octave error and could **rename** a merely sharp or flat file.
- Digital silence is `NO_DETECTION` (`max|x| < 1e-8`). Zero buffers no longer report a spurious high $f_0$.
- Live Tuner uses **signed** cents for the needle and text (negative = flat). Missing `parse_note` import on the no-target path is fixed.
- `pitch_shift_tool.py` without an input file exits with argparse usage instead of `TypeError`.
- Retune preserves stereo (and other multi-channel) layouts: detect from an equal-weight downmix, apply the same shift per channel.
- CLI analysis of unreadable files returns `ERROR` and continues the folder instead of aborting.
- Analyzer worker no longer reads Tk `BooleanVar`/`StringVar` from the background thread (options are snapshotted when **Analyze Folder** is pressed).
- Windows CLI logs use ASCII `->` so cp1252 consoles do not crash on rename/retune messages.

### Documentation

- Added `docs/TuneAdjust_math_formula.md`.
- README and technical manual: signed vs unsigned cents; filename target vs measured $f_0$; stereo; silence; test count 121.

### Not published

- Local unpublished pin `numpy>=1.23.0,<1.27` remains on archival branches only. Current librosa 0.11 + numba 0.67 accepted NumPy 2.2.6 in the review environment.
