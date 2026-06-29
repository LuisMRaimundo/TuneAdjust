"""
Shared pitch detection, note parsing, and instrument range utilities.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import librosa
import numpy as np

A4_HZ = 440.0
A0_HZ = 27.5
DEFAULT_FMIN_HZ = 65.41  # C2 — too high for double bass / tuba low strings
AUTO_RETUNE_MAX_CENTS = 50.0  # ½ semitone — max auto pitch correction
CHROMATIC_NOTES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FLAT_EQUIVALENTS = {
    "Db": "C#", "Eb": "D#", "Gb": "F#", "Ab": "G#", "Bb": "A#",
    "Cb": "B", "Fb": "E", "E#": "F", "B#": "C",
}
NOTE_TO_SEMITONE = {
    "C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6,
    "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10,
    "B": 11, "Cb": 11, "B#": 0,
}
NOTE_PATTERN = re.compile(r"([A-G])([#♯b♭]?)(-?\d+)", re.IGNORECASE)
VALID_OCTAVE_MIN = 0
VALID_OCTAVE_MAX = 8

NOTE_FREQUENCY_MAP: Dict[str, float] = {
    "C0": 16.35, "C#0": 17.32, "D0": 18.35, "D#0": 19.45, "E0": 20.6, "F0": 21.83,
    "F#0": 23.12, "G0": 24.5, "G#0": 25.96, "A0": 27.5, "A#0": 29.14, "B0": 30.87,
    "C1": 32.7, "C#1": 34.65, "D1": 36.71, "D#1": 38.89, "E1": 41.2, "F1": 43.65,
    "F#1": 46.25, "G1": 49.0, "G#1": 51.91, "A1": 55.0, "A#1": 58.27, "B1": 61.74,
    "C2": 65.41, "C#2": 69.3, "D2": 73.42, "D#2": 77.78, "E2": 82.41, "F2": 87.31,
    "F#2": 92.5, "G2": 98.0, "G#2": 103.83, "A2": 110.0, "A#2": 116.54, "B2": 123.47,
    "C3": 130.81, "C#3": 138.59, "D3": 146.83, "D#3": 155.56, "E3": 164.81, "F3": 174.61,
    "F#3": 185.0, "G3": 196.0, "G#3": 207.65, "A3": 220.0, "A#3": 233.08, "B3": 246.94,
    "C4": 261.63, "C#4": 277.18, "D4": 293.66, "D#4": 311.13, "E4": 329.63, "F4": 349.23,
    "F#4": 369.99, "G4": 392.0, "G#4": 415.3, "A4": 440.0, "A#4": 466.16, "B4": 493.88,
    "C5": 523.25, "C#5": 554.37, "D5": 587.33, "D#5": 622.25, "E5": 659.25, "F5": 698.46,
    "F#5": 739.99, "G5": 783.99, "G#5": 830.61, "A5": 880.0, "A#5": 932.33, "B5": 987.77,
    "C6": 1046.5, "C#6": 1108.73, "D6": 1174.66, "D#6": 1244.51, "E6": 1318.51, "F6": 1396.91,
    "F#6": 1479.98, "G6": 1567.98, "G#6": 1661.22, "A6": 1760.0, "A#6": 1864.66, "B6": 1975.53,
    "C7": 2093.0, "C#7": 2217.46, "D7": 2349.32, "D#7": 2489.0, "E7": 2637.0, "F7": 2793.83,
    "F#7": 2959.96, "G7": 3135.96, "G#7": 3322.44, "A7": 3520.0, "A#7": 3729.31, "B7": 3951.0,
    "C8": 4186.0, "C#8": 4434.92, "D8": 4698.63, "D#8": 4978.0, "E8": 5274.0, "F8": 5587.65,
    "F#8": 5919.91, "G8": 6271.93, "G#8": 6644.88, "A8": 7040.0, "A#8": 7458.62, "B8": 7902.13,
}
for _oct in range(9):
    NOTE_FREQUENCY_MAP[f"Bb{_oct}"] = NOTE_FREQUENCY_MAP[f"A#{_oct}"]
    NOTE_FREQUENCY_MAP[f"Db{_oct}"] = NOTE_FREQUENCY_MAP[f"C#{_oct}"]
    NOTE_FREQUENCY_MAP[f"Eb{_oct}"] = NOTE_FREQUENCY_MAP[f"D#{_oct}"]
    NOTE_FREQUENCY_MAP[f"Gb{_oct}"] = NOTE_FREQUENCY_MAP[f"F#{_oct}"]
    NOTE_FREQUENCY_MAP[f"Ab{_oct}"] = NOTE_FREQUENCY_MAP[f"G#{_oct}"]

_REGISTRY_CACHE: Optional[Dict[str, Any]] = None


def parse_note(note_str: str) -> Optional[Tuple[str, int]]:
    if not isinstance(note_str, str):
        return None
    note_str = note_str.strip()
    match = NOTE_PATTERN.search(note_str)
    if not match:
        return None
    letter = match.group(1).upper()
    accidental = match.group(2).replace("♯", "#").replace("♭", "b")
    octave = int(match.group(3))
    note_name = letter + accidental
    if note_name in FLAT_EQUIVALENTS:
        note_name = FLAT_EQUIVALENTS[note_name]
    if note_name not in CHROMATIC_NOTES:
        return None
    if not (VALID_OCTAVE_MIN <= octave <= VALID_OCTAVE_MAX):
        return None
    return note_name, octave


def _is_valid_note_token(note_name: str, octave: int) -> bool:
    if note_name not in CHROMATIC_NOTES:
        return False
    if not (VALID_OCTAVE_MIN <= octave <= VALID_OCTAVE_MAX):
        return False
    return note_to_frequency(f"{note_name}{octave}") > 0


def parse_all_notes(text: str) -> List[Tuple[str, int, int, int]]:
    """Return (note_name, octave, start, end) for every note token in text."""
    found: List[Tuple[str, int, int, int]] = []
    for match in NOTE_PATTERN.finditer(text):
        letter = match.group(1).upper()
        accidental = match.group(2).replace("♯", "#").replace("♭", "b")
        octave = int(match.group(3))
        note_name = letter + accidental
        if note_name in FLAT_EQUIVALENTS:
            note_name = FLAT_EQUIVALENTS[note_name]
        if _is_valid_note_token(note_name, octave):
            found.append((note_name, octave, match.start(), match.end()))
    return found


def parse_note_from_filename(stem: str, strategy: str = "rightmost_longest") -> Optional[Tuple[str, int]]:
    """
    Extract the most likely note from a filename stem.
    Default: rightmost match; ties broken by longest span.
    """
    matches = parse_all_notes(stem)
    if not matches:
        return None
    if strategy == "first":
        n, o, _, _ = matches[0]
        return n, o
    best = max(matches, key=lambda m: (m[2], m[3] - m[2]))
    return best[0], best[1]


def note_to_midi(note_name: str, octave: int) -> int:
    return (octave + 1) * 12 + NOTE_TO_SEMITONE.get(note_name, 0)


def midi_to_note(midi: int) -> Tuple[str, int]:
    octave = (midi // 12) - 1
    return CHROMATIC_NOTES[midi % 12], octave


def note_to_frequency(note: str) -> float:
    parsed = parse_note(note)
    if not parsed:
        return 0.0
    key = f"{parsed[0]}{parsed[1]}"
    return NOTE_FREQUENCY_MAP.get(key, 0.0)


def frequency_to_note(freq: float) -> str:
    if freq <= 0:
        return "Unknown"
    try:
        note = librosa.hz_to_note(freq)
        return note.replace("♯", "#").replace("♭", "b")
    except Exception:
        semitones = 12.0 * np.log2(freq / A4_HZ)
        midi_note = int(round(69 + semitones))
        name, octave = midi_to_note(midi_note)
        return f"{name}{octave}"


def are_enharmonic(note1: str, note2: str) -> bool:
    p1, p2 = parse_note(note1), parse_note(note2)
    if not p1 or not p2:
        return False
    return note_to_midi(p1[0], p1[1]) == note_to_midi(p2[0], p2[1])


def cents_difference(freq1: float, freq2: float) -> float:
    if freq1 <= 0 or freq2 <= 0:
        return float("inf")
    return abs(1200.0 * np.log2(freq1 / freq2))


def same_pitch_class(note_a: str, note_b: str) -> bool:
    """True when notes differ only by octave (e.g. C4 vs C5)."""
    pa, pb = parse_note(note_a), parse_note(note_b)
    if not pa or not pb:
        return False
    return note_to_midi(pa[0], pa[1]) % 12 == note_to_midi(pb[0], pb[1]) % 12


def _is_octave_alias(f1: float, f2: float) -> bool:
    """True if f1 and f2 differ by approximately an integer number of octaves."""
    if f1 <= 0 or f2 <= 0:
        return False
    log2r = abs(np.log2(f1 / f2))
    n = round(log2r)
    return n >= 1 and abs(log2r - n) < 0.09


def _normalize_to_octave_neighborhood(freq: float, ref: float) -> float:
    """Express freq in the octave nearest to ref (stops C4/C5 flipping in trackers)."""
    if freq <= 0 or ref <= 0:
        return freq
    best = freq
    best_diff = abs(np.log2(freq / ref))
    for shift in range(-2, 3):
        candidate = freq * (2.0 ** shift)
        diff = abs(np.log2(candidate / ref))
        if diff < best_diff:
            best_diff = diff
            best = candidate
    return float(best)


def align_frequency_to_expected_octave(
    freq: float,
    expected_note: str,
    *,
    max_octave_shift: int = 2,
    fmin: float = 50.0,
    fmax: float = 8000.0,
    min_correction_cents: float = 400.0,
) -> float:
    """
    Shift by octaves toward the filename note only when clearly in the wrong octave.
    Avoids 'jumping' when the pitch is already close to expected.
    """
    expected_hz = note_to_frequency(expected_note)
    if freq <= 0 or expected_hz <= 0:
        return freq
    if cents_difference(freq, expected_hz) < min_correction_cents:
        return float(freq)
    best_f = freq
    best_cents = cents_difference(freq, expected_hz)
    for shift in range(-max_octave_shift, max_octave_shift + 1):
        if shift == 0:
            continue
        candidate = freq * (2.0 ** shift)
        if candidate < fmin or candidate > fmax:
            continue
        c = cents_difference(candidate, expected_hz)
        if c + 60.0 < best_cents:
            best_cents = c
            best_f = candidate
    return float(best_f)


class PitchSmoother:
    """Median smoother with octave consistency — for live display."""

    def __init__(self, window: int = 7):
        self.window = max(3, window)
        self._history: List[float] = []

    def reset(self) -> None:
        self._history.clear()

    def update(self, freq: float) -> float:
        if freq <= 0:
            return self._history[-1] if self._history else 0.0
        if self._history:
            freq = _normalize_to_octave_neighborhood(freq, self._history[-1])
        self._history.append(freq)
        if len(self._history) > self.window:
            self._history.pop(0)
        return float(np.median(self._history))


def evaluate_tune_match(
    detected_freq: float,
    detected_note: str,
    expected_note: Optional[str],
    expected_freq: float,
    tolerance: float,
    *,
    range_warning: Optional[str] = None,
) -> Tuple[float, bool, bool, str]:
    """
    Compare detected pitch to expected note from filename.
    Returns (cents_off, is_in_tune, is_mislabeled, status).
    Mislabeled = wrong pitch class; octave slips are OCTAVE_ERROR, not mislabeled.
    """
    if detected_freq <= 0 or detected_note in ("Unknown", ""):
        return float("inf"), False, False, "NO_DETECTION"

    if expected_freq > 0:
        cents_off = cents_difference(detected_freq, expected_freq)
    else:
        cents_off = float("inf")

    is_mislabeled = False
    is_in_tune = False
    octave_only = False
    effective_tolerance = tolerance * 1.1

    if expected_freq > 0 and expected_note:
        if cents_off <= effective_tolerance:
            is_in_tune = True
        elif same_pitch_class(expected_note, detected_note):
            octave_only = True
            is_in_tune = False
            is_mislabeled = False
        elif cents_off < 100.0:
            is_in_tune = False
            is_mislabeled = False
        else:
            is_mislabeled = True
            is_in_tune = False
    else:
        is_in_tune = False

    if is_in_tune:
        status = "OK"
    elif octave_only:
        status = "OCTAVE_ERROR"
    elif is_mislabeled:
        status = "MISLABELED"
    elif not is_in_tune and expected_freq > 0:
        status = "OUT_OF_TUNE"
    else:
        status = "NO_NOTE_IN_FILENAME"

    if range_warning:
        status = "RANGE_WARN" if status == "OK" else f"{status}+RANGE"
    return cents_off, is_in_tune, is_mislabeled, status


def generate_chromatic_scale(lowest_note: str, highest_note: str) -> List[Tuple[str, int]]:
    low, high = parse_note(lowest_note), parse_note(highest_note)
    if not low or not high:
        return []
    lo = note_to_midi(low[0], low[1])
    hi = note_to_midi(high[0], high[1])
    if lo > hi:
        lo, hi = hi, lo
    return [midi_to_note(m) for m in range(lo, hi + 1)]


def calculate_semitones(current_freq: float, target_freq: float) -> float:
    if current_freq <= 0 or target_freq <= 0:
        return 0.0
    return 12.0 * np.log2(target_freq / current_freq)


def cross_check_expected_table(freq: float, note: str, tolerance_cents: float = 50.0) -> Dict[str, Any]:
    """Compare detected frequency against the built-in equal-temperament table."""
    expected = note_to_frequency(note)
    if expected <= 0:
        return {"validated": False, "reason": "unknown_note"}
    cents = cents_difference(freq, expected)
    return {
        "validated": True,
        "expected_freq": expected,
        "detected_freq": freq,
        "cents_difference": cents,
        "is_valid": cents <= tolerance_cents,
    }


def load_instrument_registry(path: Optional[Path] = None) -> Dict[str, Any]:
    global _REGISTRY_CACHE
    if _REGISTRY_CACHE is not None:
        return _REGISTRY_CACHE
    if path is None:
        path = Path(__file__).with_name("instrument_registry.json")
    if not path.exists():
        _REGISTRY_CACHE = {}
        return _REGISTRY_CACHE
    with open(path, encoding="utf-8") as f:
        _REGISTRY_CACHE = json.load(f)
    return _REGISTRY_CACHE


def list_instruments() -> List[str]:
    return sorted(load_instrument_registry().keys())


def _parse_range_notes(range_text: str) -> Tuple[Optional[int], Optional[int]]:
    """Parse 'A0 to F4' into low/high MIDI."""
    parts = re.split(r"\s+to\s+", range_text, flags=re.IGNORECASE)
    if len(parts) != 2:
        return None, None
    low, high = parse_note(parts[0].strip()), parse_note(parts[1].strip())
    if not low or not high:
        return None, None
    return note_to_midi(low[0], low[1]), note_to_midi(high[0], high[1])


def check_instrument_range(
    detected_freq: float,
    instrument: Optional[str],
    registry: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Return warning message if detected pitch is outside instrument sounding range."""
    if not instrument or detected_freq <= 0:
        return None
    reg = registry if registry is not None else load_instrument_registry()
    entry = reg.get(instrument.lower())
    if not entry:
        return None
    standard = entry.get("standard") or next(iter(entry.values()), None)
    if not standard:
        return None
    sounding = standard.get("sounding_range") or standard.get("written_range") or standard.get("range")
    if not sounding:
        return None
    lo_midi, hi_midi = _parse_range_notes(sounding)
    if lo_midi is None or hi_midi is None:
        return None
    parsed = parse_note(frequency_to_note(detected_freq))
    if not parsed:
        return None
    detected_midi = note_to_midi(parsed[0], parsed[1])
    if detected_midi < lo_midi or detected_midi > hi_midi:
        lo_n = f"{midi_to_note(lo_midi)[0]}{midi_to_note(lo_midi)[1]}"
        hi_n = f"{midi_to_note(hi_midi)[0]}{midi_to_note(hi_midi)[1]}"
        return f"Detected pitch outside {instrument} range ({lo_n}–{hi_n})"
    return None


def select_stable_segment(audio: np.ndarray, sr: int, duration: float = 2.0) -> np.ndarray:
    target = int(duration * sr)
    if len(audio) <= target:
        return audio
    start = (len(audio) - target) // 2
    return audio[start : start + target]


def _peak_spectrum_energy(magnitude: np.ndarray, freqs: np.ndarray, f: float) -> float:
    if f <= 0:
        return 0.0
    idx = int(np.argmin(np.abs(freqs - f)))
    return float(magnitude[idx])


def _harmonic_coherence_score(
    magnitude: np.ndarray, freqs: np.ndarray, f0: float, *, n_harmonics: int = 8
) -> float:
    """Score how well f0 lines up with a harmonic series in the spectrum."""
    if f0 <= 0:
        return 0.0
    score = 0.0
    fund = _peak_spectrum_energy(magnitude, freqs, f0)
    if fund <= 0:
        return 0.0
    for h in range(1, n_harmonics + 1):
        fh = f0 * h
        if fh > freqs[-1] * 0.98:
            break
        score += _peak_spectrum_energy(magnitude, freqs, fh) / h
    if f0 / 2 >= freqs[1]:
        sub = _peak_spectrum_energy(magnitude, freqs, f0 / 2)
        if sub > fund * 1.12:
            score *= 0.45
    return score


def resolve_fundamental_octave(
    audio: np.ndarray,
    sr: int,
    freq: float,
    *,
    fmin: Optional[float] = None,
    fmax: Optional[float] = None,
    expected_note: Optional[str] = None,
    raw_hint: Optional[float] = None,
) -> float:
    """
    Choose the true fundamental among octave aliases (common with orchestral spectra).
    Stays stable when two octaves score similarly — prefers the detector's raw octave.
    """
    if freq <= 0:
        return freq
    raw_hint = raw_hint if raw_hint and raw_hint > 0 else freq
    fmin = float(fmin if fmin is not None else librosa.note_to_hz("C2"))
    fmax = float(fmax if fmax is not None else librosa.note_to_hz("C8"))

    segment = select_stable_segment(audio, sr, min(2.0, len(audio) / sr))
    n_fft = 8192
    spectrum = np.abs(librosa.stft(segment, n_fft=n_fft, hop_length=n_fft // 4))
    magnitude = np.mean(spectrum, axis=1)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)

    candidates: List[float] = []
    for shift in range(-3, 4):
        f = freq * (2.0 ** shift)
        if fmin <= f <= fmax:
            candidates.append(f)
    if not candidates:
        return float(freq)

    scored = sorted(
        [(f, _harmonic_coherence_score(magnitude, freqs, f)) for f in candidates],
        key=lambda x: -x[1],
    )
    best_f, best_s = scored[0]
    if best_s <= 0:
        best_f = raw_hint
    else:
        for f2, s2 in scored[1:4]:
            if _is_octave_alias(best_f, f2) and s2 >= best_s * 0.92:
                amb = [f for f, s in scored if _is_octave_alias(f, best_f) and s >= best_s * 0.85]
                best_f = min(amb, key=lambda f: abs(np.log2(f / raw_hint)))
                break

    if expected_note:
        expected_hz = note_to_frequency(expected_note)
        if expected_hz > 0:
            before = cents_difference(best_f, expected_hz)
            if before > 400.0:
                aligned = align_frequency_to_expected_octave(
                    best_f, expected_note, fmin=fmin, fmax=fmax
                )
                if cents_difference(aligned, expected_hz) + 80.0 < before:
                    best_f = aligned
    return float(best_f)


def pitch_search_bounds(
    expected_note: Optional[str] = None,
    instrument: Optional[str] = None,
) -> Tuple[float, float]:
    """
    Adaptive pYIN/YIN search range.
    Low notes (A1, A#1, B1, C2) need fmin below C2 or detectors lock onto harmonics.
    """
    fmax = float(librosa.note_to_hz("C8"))
    fmin = DEFAULT_FMIN_HZ
    low_limit = A0_HZ

    if instrument and instrument not in ("(none)", ""):
        reg = load_instrument_registry()
        key = instrument.lower().replace(" ", "_")
        entry = reg.get(key)
        if not entry:
            for name, data in reg.items():
                if key in name or name in key:
                    entry = data
                    break
        if entry:
            standard = entry.get("standard") or next(iter(entry.values()), None)
            if standard:
                sounding = (
                    standard.get("sounding_range")
                    or standard.get("written_range")
                    or standard.get("range")
                )
                if sounding:
                    lo_midi, _ = _parse_range_notes(sounding)
                    if lo_midi is not None:
                        n, o = midi_to_note(lo_midi)
                        lo_hz = note_to_frequency(f"{n}{o}")
                        if lo_hz > 0:
                            low_limit = min(low_limit, lo_hz * 0.85)

    if expected_note:
        exp_hz = note_to_frequency(expected_note)
        if exp_hz > 0:
            fmin = min(fmin, max(low_limit, exp_hz * 0.55))
            fmax = max(exp_hz * 2.5, fmin + 50.0)
            fmax = min(fmax, float(librosa.note_to_hz("C8")))

    fmin = max(low_limit, fmin)
    return float(fmin), float(fmax)


def correct_octave_with_harmonics(
    audio: np.ndarray, sr: int, f0: float, fmin: float, fmax: float
) -> float:
    """Backward-compatible wrapper around resolve_fundamental_octave."""
    return resolve_fundamental_octave(audio, sr, f0, fmin=fmin, fmax=fmax)


def _collect_pitch_candidates(
    segment: np.ndarray,
    sr: int,
    fmin: float,
    fmax: float,
    min_pyin_confidence: float = 0.7,
) -> List[Tuple[float, float, str]]:
    candidates: List[Tuple[float, float, str]] = []
    try:
        f0, voiced, probs = librosa.pyin(segment, fmin=fmin, fmax=fmax, sr=sr)
        mask = voiced & (probs > min_pyin_confidence)
        conf_f0 = f0[mask]
        if len(conf_f0) > 0:
            freq = float(np.median(conf_f0[~np.isnan(conf_f0)]))
            if freq > 0:
                candidates.append((freq, float(np.mean(probs[mask])), "pyin"))
        else:
            voiced_f0 = f0[voiced]
            if len(voiced_f0) > 0:
                freq = float(np.median(voiced_f0[~np.isnan(voiced_f0)]))
                if freq > 0:
                    candidates.append((freq, float(np.mean(probs[voiced])) * 0.8, "pyin_low"))
    except Exception:
        pass

    try:
        f0_yin = librosa.yin(segment, fmin=fmin, fmax=fmax, sr=sr)
        valid = f0_yin[~np.isnan(f0_yin) & (f0_yin > 0)]
        if len(valid) > 0:
            candidates.append((float(np.median(valid)), 0.6, "yin"))
    except Exception:
        pass

    ac = _detect_autocorr(segment, sr, fmin, fmax)
    if ac > 0:
        candidates.append((ac, 0.55, "autocorr"))
    return candidates


def _merge_candidates(candidates: List[Tuple[float, float, str]]) -> float:
    if not candidates:
        return 0.0
    priority = {"pyin": 3, "pyin_low": 2, "yin": 1, "autocorr": 0}
    candidates.sort(key=lambda x: (x[1], priority.get(x[2], 0)), reverse=True)
    best, conf, _ = candidates[0]
    for freq, c, _ in candidates[1:]:
        if abs(1200.0 * np.log2(freq / best)) < 50.0:
            best = (best * conf + freq * c) / (conf + c)
            break
    return float(best)


def _detect_raw_frequency(
    audio: np.ndarray,
    sr: int,
    *,
    fast: bool = False,
    min_pyin_confidence: float = 0.7,
    segment_duration: float = 2.0,
    expected_note: Optional[str] = None,
    instrument: Optional[str] = None,
) -> float:
    """pYIN / YIN / autocorr estimate without octave correction."""
    fmin, fmax = pitch_search_bounds(expected_note, instrument)

    if fast:
        return _detect_autocorr(audio, sr, fmin, min(fmax, 2000.0))

    if segment_duration <= 0 or len(audio) <= int(segment_duration * sr):
        stable = audio
    else:
        stable = select_stable_segment(audio, sr, segment_duration)

    candidates = _collect_pitch_candidates(stable, sr, fmin, fmax, min_pyin_confidence)
    return _merge_candidates(candidates)


def _detect_raw_frequency_robust(
    audio: np.ndarray,
    sr: int,
    *,
    expected_note: Optional[str] = None,
    instrument: Optional[str] = None,
) -> float:
    """Try several segments; return best raw estimate (no per-attempt octave fix)."""
    fmin, fmax = pitch_search_bounds(expected_note, instrument)
    dur = len(audio) / max(sr, 1)
    attempts: List[Tuple[np.ndarray, float, int]] = []

    if dur > 2.5:
        attempts.append((select_stable_segment(audio, sr, 2.0), 0.7, 3))
        attempts.append((select_stable_segment(audio, sr, min(4.0, dur * 0.8)), 0.5, 2))
    try:
        trimmed, _ = librosa.effects.trim(audio, top_db=50)
        if len(trimmed) > max(2048, int(0.15 * sr)):
            attempts.append((trimmed, 0.55, 1))
    except Exception:
        pass
    attempts.append((audio, 0.5, 0))
    attempts.append((audio, 0.35, -1))

    best_raw = 0.0
    best_rank = -999
    for segment, min_conf, rank in attempts:
        candidates = _collect_pitch_candidates(segment, sr, fmin, fmax, min_conf)
        merged = _merge_candidates(candidates)
        if merged > 0 and rank > best_rank:
            best_raw = merged
            best_rank = rank
    return float(best_raw)


def detect_frequency(
    audio: np.ndarray,
    sr: int,
    *,
    fast: bool = False,
    apply_harmonic_check: bool = True,
    min_pyin_confidence: float = 0.7,
    segment_duration: float = 2.0,
) -> float:
    """
    Detect fundamental frequency (pYIN → YIN → autocorrelation).
    fast=True uses autocorrelation only (for live UI).
    """
    fmin = float(librosa.note_to_hz("C2"))
    fmax = float(librosa.note_to_hz("C8"))

    raw = _detect_raw_frequency(
        audio,
        sr,
        fast=fast,
        min_pyin_confidence=min_pyin_confidence,
        segment_duration=segment_duration,
    )
    if raw <= 0:
        return 0.0
    if not apply_harmonic_check or fast:
        return float(raw)
    return resolve_fundamental_octave(
        audio, sr, raw, fmin=fmin, fmax=fmax, raw_hint=raw
    )


def detect_pitch(
    audio: np.ndarray,
    sr: int,
    *,
    expected_note: Optional[str] = None,
    instrument: Optional[str] = None,
) -> float:
    """
    Stable full-file pitch: one raw estimate, one octave resolve (no double correction).
    """
    fmin, fmax = pitch_search_bounds(expected_note, instrument)
    raw = _detect_raw_frequency(
        audio, sr, expected_note=expected_note, instrument=instrument
    )
    if raw <= 0:
        raw = _detect_raw_frequency_robust(
            audio, sr, expected_note=expected_note, instrument=instrument
        )
    if raw <= 0:
        return 0.0
    return resolve_fundamental_octave(
        audio,
        sr,
        raw,
        fmin=fmin,
        fmax=fmax,
        expected_note=expected_note,
        raw_hint=raw,
    )


def detect_frequency_robust(
    audio: np.ndarray,
    sr: int,
    *,
    apply_harmonic_check: bool = True,
    expected_note: Optional[str] = None,
    instrument: Optional[str] = None,
) -> float:
    """
    Try multiple segments and confidence thresholds when the default detector fails.
    Octave correction runs once on the best raw estimate (stable).
    """
    fmin, fmax = pitch_search_bounds(expected_note, instrument)
    raw = _detect_raw_frequency_robust(
        audio, sr, expected_note=expected_note, instrument=instrument
    )
    if raw <= 0:
        return 0.0
    if not apply_harmonic_check:
        return float(raw)
    return resolve_fundamental_octave(
        audio,
        sr,
        raw,
        fmin=fmin,
        fmax=fmax,
        expected_note=expected_note,
        raw_hint=raw,
    )


def _detect_autocorr(audio: np.ndarray, sr: int, fmin: float, fmax: float) -> float:
    if len(audio) < 1024:
        return 0.0
    norm = audio - np.mean(audio)
    norm = norm / (np.max(np.abs(norm)) + 1e-10)
    ac = np.correlate(norm, norm, mode="full")
    ac = ac[len(ac) // 2 :]
    min_p = int(sr / fmax)
    max_p = int(sr / fmin)
    if len(ac) <= max_p:
        return 0.0
    window = ac[min_p:max_p]
    if len(window) == 0:
        return 0.0
    peak = min_p + int(np.argmax(window))
    if peak <= 0:
        return 0.0
    freq = float(sr / peak)
    return freq if fmin <= freq <= fmax else 0.0

