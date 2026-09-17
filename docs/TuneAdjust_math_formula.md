# TuneAdjust — mathematical reference

**Status:** production documentation of the implementation on `fix/tuneadjust-validation-and-documentation`.  
**Language:** English. StackEdit-compatible Markdown + LaTeX (`$...$` inline, `$$...$$` display). No custom macros.

This document extracts **project-defined** mathematics from first-party Python. External-library internals are **not** reproduced; only the call site, arguments, and surrounding project math are recorded.

**This document does not certify scientific validity.** Synthetic tests and documentation are not a substitute for a labelled research-corpus study.

---

## Implementation baseline

| Item | Value |
|------|--------|
| GitHub pin at alignment | `be56130bbb8b781d98bf19b07a074942f96de864` |
| Working-tree provenance | Review branch built on that pin; unpublished local NumPy `<1.27` pin **not** included |
| Isolated environment | Python 3.10.11; `librosa 0.11.0`, `numpy 2.2.6`, `numba 0.67.0`, `soundfile 0.14.0`, `scipy 1.15.3`, `pytest 9.1.1` |
| Tuning reference | 12-TET, $A_4 = 440\,\mathrm{Hz}$ |
| Auto-retune limit | $50$ cents ($1/2$ semitone) |

Hashes below describe the Python sources at documentation time (after the validation fixes). Recompute if those files change.

| File | SHA-256 | Lines | Coverage |
|------|---------|------:|----------|
| `pitch_core.py` | `69bc41d28fe31bfb49fb0b5b91df75035f879d512e3835f33f875067b511574f` | 806 | All project math |
| `pitch_shift_tool.py` | `0676012ecfe419aed8ed689063a3847027bfcbbaf915b9456cf2bb0f4bc017b8` | 497 | Shift / RMS / load |
| `auto_correct.py` | `8657873e7ec85dffa5fcfb375fb2e40b84cbba01162c600ce84ff120e57ef886` | 700 | Planning thresholds |
| `live_tuner.py` | `6da119721312fd134c1f0d83d978c0e34ef17e787e5aed0f1e5d2a436e03c5cb` | 625 | Signed-cents display |
| `pitch_shift_gui.py` | `581d1e6eb0cc75c2ac6219b733de81849846986226a9103074be04c82a281e8e` | 700 | Uses core/shift math |
| `note_frequency_analyzer.py` | `67c32c1f4e7573ad34e4dd54b8d5e9e5b4c7c969eedf45937218a4bc547e8a48` | 2076 | Orchestration only |

Tests (`tests/*.py`) contain no independent production mathematics. `instrument_registry.json` is lookup data, not a formula. `START-Tune-Detection.bat` has no math.

---

## What is used for a correction

| Quantity | Source | Role |
|----------|--------|------|
| Target note / $f_\mathrm{target}$ | Filename token via `parse_note_from_filename` → `NOTE_FREQUENCY_MAP` | Intended pitch of the sample |
| Measured $f_0$ | Audio via pYIN / YIN / autocorrelation, then octave resolve | Estimate of what was recorded |
| Filename-guided octave | `detect_pitch(..., expected_note=...)` if Fix octave is ON | Prior for search bounds and octave choice — **not** a claim that audio already matches the label |
| Correction | `calculate_semitones(f_measured, f_target)` | Signed shift actually applied |
| Verification | Re-detect after write; compare to **filename** target | Must not be confused with repeating the label |

`cents_difference` is **unsigned** (QC magnitude). `signed_cents_difference` is **signed** (Live Tuner needle). Auto-retune uses `calculate_semitones`, which is signed.

---

## Project formulae

### M-001 — Equal-temperament frequency table (production)

**Source:** `pitch_core.py`, `NOTE_FREQUENCY_MAP`, lines 34–53 (plus flat aliases 54–59).  
**Function:** `note_to_frequency` lines 132–137.

```python
def note_to_frequency(note: str) -> float:
    parsed = parse_note(note)
    if not parsed:
        return 0.0
    key = f"{parsed[0]}{parsed[1]}"
    return NOTE_FREQUENCY_MAP.get(key, 0.0)
```

**LaTeX:** table lookup $f(n) = T[n]$ where $T$ is a rounded 12-TET table with $T(\mathrm{A4})=440$. Not recomputed as $440\cdot 2^{(m-69)/12}$ at runtime.

**Symbols:** $n$ note token; $f$ Hz; domain C0–B8 plus flat spellings. Invalid parse → $0$.

**Layman:** Each named note has a stored frequency. A4 is 440 Hz.

**Specialist:** Values are rounded (e.g. C4 $=261.63$). Targets therefore differ slightly from exact $440\cdot 2^{k/12}$. Tests use `approx` / relative tolerances.

**Downstream:** filename expected frequency; retune target; Live Tuner target. Tests: `test_note_to_frequency_a4`, `test_a0_and_c8_boundaries`.

---

### M-002 — MIDI from note name (production)

**Source:** `pitch_core.py` `note_to_midi` lines 123–124.

```python
def note_to_midi(note_name: str, octave: int) -> int:
    return (octave + 1) * 12 + NOTE_TO_SEMITONE.get(note_name, 0)
```

$$m = 12(o+1) + s(n)$$

**Symbols:** $m$ MIDI number; $o$ octave (0–8); $s(n)$ pitch-class index (C $=0$). Scientific pitch: C4 $\mapsto 60$.

**Layman:** Turns “A4” into the MIDI number 69.

**Specialist:** Unknown `note_name` uses $s=0$ (C). Callers should pass a parsed chromatic name.

**Downstream:** enharmonic / pitch-class tests; instrument range. Tests: `test_midi_roundtrip`.

---

### M-003 — Note from MIDI (production)

**Source:** `pitch_core.py` `midi_to_note` lines 127–129.

```python
def midi_to_note(midi: int) -> Tuple[str, int]:
    octave = (midi // 12) - 1
    return CHROMATIC_NOTES[midi % 12], octave
```

$$o=\lfloor m/12\rfloor-1,\quad n=C[m \bmod 12]$$

**Downstream:** chromatic scale generation; range labels.

---

### M-004 — Signed cents (production)

**Source:** `pitch_core.py` `signed_cents_difference` lines 166–173.

```python
def signed_cents_difference(freq_actual: float, freq_reference: float) -> float:
    if freq_actual <= 0 or freq_reference <= 0:
        return float("inf")
    return float(1200.0 * np.log2(freq_actual / freq_reference))
```

$$c_\mathrm{signed}=1200\log_2\frac{f_\mathrm{actual}}{f_\mathrm{ref}}$$

**Symbols:** $c$ cents; $f$ Hz; $c>0$ sharp; $c<0$ flat. Non-positive $f$ → $+\infty$.

**Layman:** How far the pitch is from the reference, including sharp vs flat.

**Specialist:** Standard logarithmic cents. Used by Live Tuner display/needle only. Classification still uses M-005.

**Tests:** `test_signed_cents_difference_direction`, `test_live_tuner_uses_signed_cents_and_parse_note`.

---

### M-005 — Absolute cents (production)

**Source:** `pitch_core.py` `cents_difference` lines 160–163.

```python
def cents_difference(freq1: float, freq2: float) -> float:
    signed = signed_cents_difference(freq1, freq2)
    return abs(signed) if signed != float("inf") else signed
```

$$c=|1200\log_2(f_1/f_2)|$$

**Layman:** How many cents off, ignoring direction.

**Specialist:** QC / `evaluate_tune_match` / reports. Do not treat a report of “+38 cents” from this function as sharp — it is unsigned.

**Tests:** `test_cents_difference`.

---

### M-006 — Semitone shift (production)

**Source:** `pitch_core.py` `calculate_semitones` lines 342–345.

```python
def calculate_semitones(current_freq: float, target_freq: float) -> float:
    if current_freq <= 0 or target_freq <= 0:
        return 0.0
    return 12.0 * np.log2(target_freq / current_freq)
```

$$\Delta = 12\log_2\frac{f_\mathrm{target}}{f_\mathrm{current}}$$

**Symbols:** $\Delta$ semitones (signed). Non-positive inputs → $0$ (no-op), not an error.

**Layman:** How much to raise or lower the recording to hit the target.

**Specialist:** This is the quantity passed to the phase vocoder / Rubber Band. Sign is opposite to signed cents of current vs target: if current is flat, $\Delta>0$.

**Tests:** `test_calculate_semitones`; integration retune.

---

### M-007 — Digital silence gate (production)

**Source:** `pitch_core.py` `_is_digital_silence` lines 176–179; `SILENCE_PEAK_EPS` line 19.

```python
def _is_digital_silence(audio: np.ndarray, eps: float = SILENCE_PEAK_EPS) -> bool:
    if audio is None or getattr(audio, "size", 0) == 0:
        return True
    return float(np.max(np.abs(audio))) < eps
```

$$\mathrm{silence}\iff N=0 \lor \max_i |x_i| < 10^{-8}$$

**Layman:** All-zero (or empty) audio is not given a fake pitch.

**Specialist:** Without this gate, YIN/autocorrelation on zeros can return $\approx f_s/P_\min$ (observed $4410\,\mathrm{Hz}$ at $f_s=22050$). Applied at raw detect and autocorr. Quiet but non-zero samples are unchanged.

**Tests:** `test_detect_frequency_digital_silence`, `test_digital_silence_named_note_is_no_detection`.

---

### M-008 — Same pitch class (production)

**Source:** `pitch_core.py` `same_pitch_class` lines 182–187.

```python
    return note_to_midi(pa[0], pa[1]) % 12 == note_to_midi(pb[0], pb[1]) % 12
```

$$\mathrm{samePC}(a,b)\iff m(a)\equiv m(b)\pmod{12}$$

True for C4 vs C5 **and** C4 vs C4.

---

### M-009 — Octave-error classification (production)

**Source:** `pitch_core.py` `evaluate_tune_match` lines 295–306.

```python
        if cents_off <= effective_tolerance:
            is_in_tune = True
        elif (
            same_pitch_class(expected_note, detected_note)
            and not are_enharmonic(expected_note, detected_note)
        ):
            octave_only = True
```

**LaTeX:** `OCTAVE_ERROR` iff same pitch class **and** different MIDI (not enharmonic), after failing the in-tune test.

**Layman:** “Octave error” means the letter is right but the octave is wrong. A slightly sharp A4 is **out of tune**, not an octave error.

**Specialist:** Prior to this review, `same_pitch_class` alone labelled A4±35 ct as `OCTAVE_ERROR`, which planned a **rename**. That contradicted the technical manual. Effective in-tune band is $\tau\times 1.1$ (M-010).

**Tests:** `test_evaluate_tune_match_same_note_drift_is_out_of_tune`, `test_evaluate_tune_match_octave_not_mislabeled`.

---

### M-010 — Effective in-tune tolerance (production / heuristic)

**Source:** `pitch_core.py` line 293.

```python
    effective_tolerance = tolerance * 1.1
```

$$\tau_\mathrm{eff}=1.1\,\tau$$

Default GUI/CLI $\tau=20$ cents → $\tau_\mathrm{eff}=22$. Deliberate slack, not a detector resolution claim.

---

### M-011 — Octave-alias test (production)

**Source:** `pitch_core.py` `_is_octave_alias` lines 190–196.

```python
    log2r = abs(np.log2(f1 / f2))
    n = round(log2r)
    return n >= 1 and abs(log2r - n) < 0.09
```

$$\bigl||\log_2(f_1/f_2)| - n\bigr| < 0.09,\quad n=\mathrm{round}(|\log_2(f_1/f_2)|)\ge 1$$

$0.09$ in $\log_2$ is about $108$ cents from an integer octave. Heuristic for harmonic/octave ties.

---

### M-012 — Normalize to octave neighbourhood (production)

**Source:** `pitch_core.py` `_normalize_to_octave_neighborhood` lines 199–211.

$$f' = f\cdot 2^{k^\star},\quad k^\star=\arg\min_{k\in\{-2,\ldots,2\}} \bigl|\log_2(f\cdot 2^k / f_\mathrm{ref})\bigr|$$

Used by `PitchSmoother` so live $f_0$ does not flip C4/C5.

---

### M-013 — Filename octave align (production / heuristic)

**Source:** `pitch_core.py` `align_frequency_to_expected_octave` lines 214–244.

Applies $f\cdot 2^{k}$ for $k\in[-K,K]\setminus\{0\}$ only if current error $\ge 400$ cents, candidate in $[f_\min,f_\max]$, and improvement exceeds $60$ cents. **Does not** run when already close. This is a filename prior, not an independent measurement.

**Tests:** `test_align_frequency_to_expected_octave`, `test_align_skips_when_already_close`.

---

### M-014 — Median pitch smoother (production)

**Source:** `pitch_core.py` `PitchSmoother.update` lines 257–265.

After M-012, $f_\mathrm{out}=\mathrm{median}(h)$ over a window of length $\max(3,W)$, default $W=7$. Non-positive input repeats last value or $0$.

---

### M-015 — Middle stable segment (production)

**Source:** `pitch_core.py` `select_stable_segment` lines 424–429.

$$i_0=\bigl\lfloor(N-d)/2\bigr\rfloor,\quad d=\lfloor t_\mathrm{seg}f_s\rfloor$$

Default $t_\mathrm{seg}=2\,\mathrm{s}$. If $N\le d$, the whole file is used.

---

### M-016 — Harmonic coherence score (production)

**Source:** `pitch_core.py` `_harmonic_coherence_score` lines 439–457.

$$S(f_0)=\sum_{h=1}^{H}\frac{|X|(h f_0)}{h}$$

If energy at $f_0/2$ exceeds $1.12$ times the fundamental bin, $S\leftarrow 0.45 S$. $H=8$. Bin = nearest FFT frequency. Custom algorithm using NumPy/`librosa` STFT magnitudes (library STFT is L-005).

---

### M-017 — Candidate merge (production)

**Source:** `pitch_core.py` `_merge_candidates` lines 616–626.

Pick highest (confidence, method-priority). If another candidate is within $50$ cents, replace with confidence-weighted mean:

$$f=\frac{f_b c_b + f_j c_j}{c_b+c_j}$$

Priority: pyin $3$, pyin_low $2$, yin $1$, autocorr $0$.

---

### M-018 — Autocorrelation $f_0$ (production)

**Source:** `pitch_core.py` `_detect_autocorr` lines 787–805.

Mean-remove, peak-normalize, full autocorrelation, search lag $[f_s/f_\max, f_s/f_\min]$:

$$f=f_s / P^\star,\quad P^\star=\arg\max_P R(P)$$

Reject if $f\notin[f_\min,f_\max]$, $N<1024$, or silence. Live Tuner `fast=True` uses this only, with $f_\max$ capped at $2000\,\mathrm{Hz}$.

---

### M-019 — Adaptive search bounds (production)

**Source:** `pitch_core.py` `pitch_search_bounds` lines 522–567.

Default $f_\min=65.41\,\mathrm{Hz}$ (C2), $f_\max=\mathrm{C8}$. Instrument sounding-range low $\times 0.85$ may lower the floor toward A0. Expected note:

$$f_\min=\min\bigl(65.41,\ \max(L, 0.55 f_\mathrm{exp})\bigr),\quad f_\max=\min(\mathrm{C8},\ \max(2.5 f_\mathrm{exp}, f_\min+50))$$

**Tests:** `test_pitch_search_bounds_low_bass`.

---

### M-020 — Adaptive FFT size (production)

**Source:** `pitch_shift_tool.py` `_adaptive_n_fft` lines 88–94.

$$N_\mathrm{FFT}=\begin{cases}4096 & |\Delta|<0.1\\ 2048 & \text{otherwise}\end{cases}$$

The $|\Delta|<1$ and $|\Delta|\ge 1$ branches are identical ($2048$).

---

### M-021 — Global RMS restore (production)

**Source:** `pitch_shift_tool.py` `_restore_global_rms` lines 97–103.

$$\hat{y}=y\cdot\frac{\mathrm{rms}(x)}{\mathrm{rms}(y)}\quad\text{if both rms}>10^{-10}$$

Does not prevent instantaneous clipping; post-shift peak is not hard-limited.

**Tests:** `test_restores_loudness`, `test_pitch_shift_preserves_loudness_and_envelope`.

---

### M-022 — Equal-weight downmix (production)

**Source:** `pitch_shift_tool.py` `downmix_mono` lines 81–85.

$$x_\mathrm{mono}[t]=\frac{1}{C}\sum_{c=1}^{C} x[t,c]$$

Detection only. Shift applies the same $\Delta$ per channel (M-006).

---

### M-023 — Auto-retune apply band (production)

**Source:** `auto_correct.py` lines 275–276.

```python
        if 0.5 < shift_cents <= AUTO_RETUNE_MAX_CENTS:
```

Retune in place iff unsigned error is in $(0.5, 50]$ cents and notes align (enharmonic or same pitch class). Errors $\le 0.5$ ct are left untouched.

---

### M-024 — Parse octave filter (production)

**Source:** `pitch_core.py` `VALID_OCTAVE_MIN/MAX` lines 32–33; `parse_note` lines 79–80.

Accept octave $o\in[0,8]$ only. Rejects velocity-suffix false tokens such as `F-2`.

---

## Library operations

Do not treat these as project-derived formulae.

### L-001 — `librosa.pyin`

**Call:** `pitch_core.py` `_collect_pitch_candidates` lines 586–598.  
`librosa.pyin(segment, fmin=fmin, fmax=fmax, sr=sr)` then median of frames with `voiced & (probs > 0.7)`, else voiced frames with confidence $\times 0.8$.  
**Version context:** librosa 0.11.0. Other defaults (frame length, hop) are library defaults — not set by this project.  
**Layman:** First-choice pitch tracker.  
**Specialist:** Probabilistic YIN; project uses median aggregation, not a custom Viterbi.

### L-002 — `librosa.yin`

**Call:** lines 603–607. Median of finite positive frames; assigned confidence $0.6$.

### L-003 — `librosa.hz_to_note`

**Call:** `frequency_to_note` lines 143–145. Fallback on exception: project MIDI rounding (M-002/M-003) with $69+12\log_2(f/440)$.

### L-004 — `librosa.note_to_hz`

**Call:** search-bound C8 and default C2 (`detect_frequency` octave window; `pitch_search_bounds`). Distinct from `NOTE_FREQUENCY_MAP`.

### L-005 — `librosa.stft` / `fft_frequencies`

**Call:** `resolve_fundamental_octave` lines 461–464. `n_fft=8192`, `hop_length=n_fft//4`. Mean magnitude over time. Scoring is M-016.

### L-006 — `librosa.effects.pitch_shift`

**Call:** `pitch_shift_tool.py` `_shift_with_librosa` lines 118–128.

```python
    return librosa.effects.pitch_shift(
        audio, sr=sr, n_steps=semitones,
        bins_per_octave=12, n_fft=n_fft, res_type="soxr_hq",
    )
```

Phase vocoder + soxr HQ. Duration of the array is preserved. Timbre is **not** proven by a correct $f_0$.

### L-007 — `librosa.load`

**Call:** `load_audio_native` (`sr=None`, `mono=False`); analysis paths still `mono=True`. Native sample rate. Multi-channel is transposed to $(N,C)$.

### L-008 — `librosa.effects.trim`

**Call:** `_detect_raw_frequency_robust` `top_db=50`. Optional extra segment.

### L-009 — `soundfile.write`

**Call:** `save_audio_preserving_format`. WAV/AIFF/FLAC/OGG via explicit format kwargs. Default subtype (typically PCM_16 for WAV). No peak limiter.

### L-010 — `pyrubberband.pitch_shift` (optional)

**Call:** `_shift_with_rubberband` lines 106–115. Used only if import succeeds and $|\Delta|\le 1$. Not installed in the review environment.

### L-011 — `numpy.median` / `numpy.correlate`

Used inside project algorithms (M-014, M-017, M-018). The surrounding algorithms remain project math.

---

## Omissions

- Rubber Band / librosa / pYIN / YIN **internal** update equations.
- ffmpeg encoder internals.
- Tk layout geometry.
- `NOTE_FREQUENCY_MAP` construction (static literals, not a runtime formula).
- Historical research recordings (none in this repository).
- StackEdit rendering was **not** opened in a browser.

---

## Scientific limitations and unresolved assumptions

1. Monophonic detector. Polyphonic input is undefined.
2. Filename note is a **target / prior**, not ground truth for the waveform.
3. `detect_pitch` can fold octaves toward the filename when $>400$ ct off (M-013). Verification after that still compares to the filename.
4. Table frequencies are rounded; they are not bit-exact 12-TET.
5. RMS restore $\ne$ envelope or timbre preservation.
6. Stereo: same $\Delta$ on every channel from an equal-weight mix $f_0$. Mid/side or per-channel $f_0$ is not implemented.
7. Auto-correct **overwrites** the input on retune (by design). Manual pitch-shift GUI can write `_shifted` or replace with a `_backup` copy.
8. Analyzer reports are written **into the analyzed folder**.
9. Near a 50-cent note boundary, `librosa.hz_to_note` may name the neighbour; if unsigned error $\le 50$ ct and the names differ, neither rename nor retune may run.
10. Unpublished local pin `numpy>=1.23.0,<1.27` was **not** published. Current librosa 0.11 + numba 0.67 accepted NumPy 2.2.6. Older librosa 0.10 / numba 0.59 can still break on NumPy 2.

---

## References actually consulted

- Project sources listed in the hash table (this working tree).
- `README.md` and `TECHNICAL_MANUAL.md` (reconciled against the implementation).
- librosa 0.11.0 installed metadata / call signatures used by this repo.
- Isolated-environment `pip freeze` (2026-09-17 review).
- Synthetic fixtures under `TuneAdjust_review_20260917_141141/exports/` (external).

No journal articles were required for extracting these identities.
