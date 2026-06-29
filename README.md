# Tune Detection B (v2.3)

Pitch analysis and correction for **monophonic sample libraries** with note names in filenames (e.g. `Violin_A4_1.2s.wav`).

**Full technical reference:** [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md)

## Quick start

```bash
pip install -r requirements.txt
```

**Windows:** double-click **`START-Tune-Detection.bat`** — opens the GUI directly.  
Other tools: `START-Tune-Detection.bat menu` (CLI / batch CLI / Live Tuner)

| Task | Command |
|------|---------|
| Batch QC + auto-fix (GUI) | `python note_frequency_analyzer.py` |
| Auto-fix one folder (CLI) | `python auto_correct.py "C:\folder"` |
| Auto-fix all subfolders (CLI) | `python auto_correct.py "C:\parent" --batch` (nested by default; add `--shallow` for one level only) |
| Live tuning meter | `python live_tuner.py` |
| Manual pitch shift | `python pitch_shift_gui.py file.wav` |

Optional dependencies:

- **ffmpeg** on PATH — MP3/M4A export from pitch tools  
- **pyrubberband** (+ Rubber Band library) — best timbre preservation for auto-retune (`pip install pyrubberband`)

## Project layout

| File | Purpose |
|------|---------|
| `START-Tune-Detection.bat` | Windows launcher (GUI, menu, CLI batch) |
| `note_frequency_analyzer.py` | Main GUI — analyze, auto-fix, batch subfolders, export report |
| `auto_correct.py` | CLI — retune, rename, verify (single folder or `--batch`) |
| `pitch_core.py` | Shared detection, note math, instrument ranges |
| `pitch_shift_tool.py` | Core pitch-shift + format-preserving save (used by auto-fix) |
| `pitch_shift_gui.py` | Manual pitch-shift GUI (opened from analyzer) |
| `live_tuner.py` | Playback + real-time tuning meter |
| `instrument_registry.json` | Instrument range warnings (`double_bass`, `violin`, …) |
| `requirements.txt` | Python dependencies |
| `tests/` | Unit tests (`test_pitch_core`, `test_auto_correct`, `test_pitch_shift_tool`, `test_pitch_shift_quality`) |

## Auto-fix rules (GUI default ON, or CLI)

1. **Retune** — same note as filename, off by **≤ 50 cents** (½ semitone)  
2. **Rename** — mislabeled, octave-wrong, or **> 50 cents** off with a different closest note; uses `_2`, `_3`, … if the target name already exists  
3. **Verify** — full folder re-analyzed; remaining issues reported  

CLI dry-run: `python auto_correct.py "C:\folder" --dry-run`  
Low instruments: `--instrument double_bass`

## GUI options

- **Process all subfolders** — sequential batch through child folders  
- **Include nested subfolders** — when batch is on, search **all depth levels** (default ON)  
- **Fix octave errors** — filename-guided harmonic correction (recommended for bass/tuba)  
- **Instrument** — adaptive low-frequency search + range warnings  
- Writes `tuning_analysis_report.txt` + `tuning_analysis_results.csv` per analyzed folder  

## Filename parsing

- Note token: `A4`, `F#2`, `Bb3`, etc. (rightmost match in filename wins)  
- **Octaves 0–8 only** — rejects false matches like `F-2` from velocity suffixes (`-ff-2c`, `-mf-4c`)  
- Orchidea-style names work: `OrchSol_Cr.Baixo_F#2-ff-2c-N.wav` → **F#2**

## Pitch shifting (auto-retune)

For shifts **≤ 50 cents** (auto-fix limit):

1. **Rubber Band** (`pyrubberband`) when installed — best timbre preservation  
2. **librosa** phase vocoder fallback — adaptive FFT, `soxr_hq` resampling  
3. **RMS restore** — keeps loudness and envelope shape (~0.998 correlation in tests)

## Detection (`pitch_core`)

1. **pYIN** (voiced, confidence-weighted)  
2. **YIN** fallback  
3. **Autocorrelation** fallback  
4. **Harmonic template check** — reduces octave errors  
5. Adaptive search bounds for low notes (A1, A#1, …)  
6. Middle **2 s** stable segment (or full file if shorter)

Live tuner uses fast autocorrelation only.

## Tests

```bash
python -m pytest tests/ -q
```

## Math

- Cents: `1200 * log2(f_detected / f_expected)`  
- Semitones: `12 * log2(f_target / f_current)`  
- Equal temperament, A4 = 440 Hz  

## Version

2.3 — June 2026 — recursive batch, Orchidea filename fix, Rubber Band retune, quality tests.  
See [TECHNICAL_MANUAL.md](TECHNICAL_MANUAL.md) for full reference.
