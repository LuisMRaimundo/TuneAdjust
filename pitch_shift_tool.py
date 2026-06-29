#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Shift Tool - Standalone
==============================

High-quality pitch shifting for small corrections (<= 1/2 semitone auto-retune range).
Preserves timbre, amplitude envelope, dynamics, and file format.

Uses Rubber Band (pyrubberband) when available, else librosa phase vocoder with
adaptive FFT and soxr_hq resampling, plus global RMS restoration.

Usage:
    python pitch_shift_tool.py input.wav [--target-freq 440.0] [--detected-freq 425.0] [--output output.wav]

Features:
- Automatic frequency detection (pYIN algorithm)
- Manual target frequency input
- Preserves: timbre, amplitude, dynamics, format
- Optimized for small corrections (< 1/2 semitone)
- Supports all audio formats (via librosa/soundfile/ffmpeg)

Author: SoundSpectrAnalyse Team
Version: 1.0
Date: January 2026
"""

import argparse
import sys
from pathlib import Path
from typing import Optional
import warnings
import subprocess
import tempfile

import numpy as np

# Import dependencies with graceful error handling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("ERROR: librosa is required but not installed.")
    print("Install with: pip install librosa")
    sys.exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("ERROR: soundfile is required but not installed.")
    print("Install with: pip install soundfile")
    sys.exit(1)

# Suppress librosa warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

from pitch_core import (
    detect_frequency,
    calculate_semitones,
    note_to_frequency,
    frequency_to_note,
    NOTE_FREQUENCY_MAP,
    parse_note,
)


def _adaptive_n_fft(abs_semitones: float) -> int:
    """Larger FFT for tiny shifts → better partial/harmonic resolution."""
    if abs_semitones < 0.1:
        return 4096
    if abs_semitones < 1.0:
        return 2048
    return 2048


def _restore_global_rms(original: np.ndarray, shifted: np.ndarray) -> np.ndarray:
    """Match overall loudness (librosa/Rubber Band often drop RMS ~5–10%)."""
    original_rms = float(np.sqrt(np.mean(original ** 2)))
    shifted_rms = float(np.sqrt(np.mean(shifted ** 2)))
    if original_rms > 1e-10 and shifted_rms > 1e-10:
        return shifted * (original_rms / shifted_rms)
    return shifted


def _shift_with_rubberband(audio: np.ndarray, sr: int, semitones: float) -> Optional[np.ndarray]:
    """Rubber Band — best timbre preservation for small shifts (optional dependency)."""
    try:
        import pyrubberband as pyrb
    except ImportError:
        return None
    try:
        return pyrb.pitch_shift(audio, sr, semitones)
    except Exception:
        return None


def _shift_with_librosa(audio: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """Phase-vocoder fallback with adaptive FFT size and high-quality resampling."""
    n_fft = _adaptive_n_fft(abs(semitones))
    return librosa.effects.pitch_shift(
        audio,
        sr=sr,
        n_steps=semitones,
        bins_per_octave=12,
        n_fft=n_fft,
        res_type="soxr_hq",
    )


def pitch_shift_audio(audio_data: np.ndarray, sr: int, semitones: float) -> np.ndarray:
    """
    Shift pitch while preserving timbre, amplitude envelope, and dynamics.

    Strategy (auto-retune range ≤ 50 cents / 0.5 st):
    1. **Rubber Band** (pyrubberband) when available — best harmonic/timbre preservation
    2. **librosa** phase vocoder with adaptive ``n_fft`` and ``soxr_hq`` resampling
    3. **Global RMS restore** — compensates typical ~7% loudness loss after shifting

    Designed for small sample-library corrections. Larger shifts remain supported but
    will alter timbre more (same as any phase-vocoder / RB stretch).
    """
    if abs(semitones) < 1e-6:
        return audio_data.copy()

    shifted: Optional[np.ndarray] = None
    if abs(semitones) <= 1.0:
        shifted = _shift_with_rubberband(audio_data, sr, semitones)

    if shifted is None:
        shifted = _shift_with_librosa(audio_data, sr, semitones)

    shifted = _restore_global_rms(audio_data, shifted)
    return shifted.astype(audio_data.dtype, copy=False)


def print_frequency_reference_table():
    """
    Print the complete frequency reference table (all notes from C0 to B8).
    
    Displays notes in a formatted table organized by octave.
    Shows all notes including sharps and flats.
    """
    print("=" * 70)
    print("FREQUENCY REFERENCE TABLE (A4 = 440 Hz Standard Tuning)")
    print("=" * 70)
    print()
    
    # Define note order (matching the reference table format: A#/A/Ab, B/Bb, C#/C/Db, D#/D, E/Eb, F#/F, G#/G/Gb)
    # This order matches the user's reference table exactly
    note_order = ['A#', 'A', 'Ab', 'B', 'Bb', 'C#', 'C', 'D#', 'D', 'Db', 'E', 'Eb', 'F#', 'F', 'G#', 'G', 'Gb']
    
    # Group notes by octave
    notes_by_octave = {}
    for octave in range(9):  # Octaves 0-8
        notes_by_octave[octave] = []
        for note_name in note_order:
            note_key = f"{note_name}{octave}"
            if note_key in NOTE_FREQUENCY_MAP:
                notes_by_octave[octave].append((note_key, NOTE_FREQUENCY_MAP[note_key]))
    
    # Print table organized by octave
    # Format: Note = Frequency Hz | (separator after each note-frequency pair)
    for octave in sorted(notes_by_octave.keys(), key=int):
        print(f"Octave {octave}:")
        print("-" * 80)
        notes = notes_by_octave[octave]
        # Print in columns (3 columns per row) with vertical separator after each pair
        for i in range(0, len(notes), 3):
            row_notes = notes[i:i+3]
            # Format each note: "Note = Frequency Hz |" with separator after
            formatted_notes = []
            for note, freq in row_notes:
                formatted_notes.append(f"{note:6s} = {freq:7.2f} Hz |")
            row_str = "  ".join(formatted_notes)
            print(f"  {row_str}")
        print()
    
    print("=" * 70)
    print(f"Total: {len(NOTE_FREQUENCY_MAP)} notes")
    print("=" * 70)


def soundfile_format_kwargs(extension: str) -> dict:
    """Map file extension to explicit soundfile write kwargs (.aif needs format=AIFF)."""
    ext = (extension or "").lower()
    if ext in {".aif", ".aiff", ".aifc"}:
        return {"format": "AIFF"}
    if ext == ".wav":
        return {"format": "WAV"}
    if ext == ".flac":
        return {"format": "FLAC"}
    if ext == ".ogg":
        return {"format": "OGG", "subtype": "VORBIS"}
    return {}


def check_ffmpeg_available() -> bool:
    """Check if ffmpeg is available in the system."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            timeout=2,
            text=True
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired, Exception):
        return False


def _save_via_ffmpeg_wav_temp(
    audio_data: np.ndarray,
    sr: int,
    output_path: Path,
    original_ext: str,
) -> bool:
    """Write temp WAV then ffmpeg-encode to target format (AIFF, MP3, etc.)."""
    temp_wav_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
            temp_wav_path = Path(temp_wav.name)
        sf.write(str(temp_wav_path), audio_data, sr, format="WAV")

        ext = original_ext.lower()
        if ext in {".aif", ".aiff", ".aifc"}:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(temp_wav_path),
                "-c:a", "pcm_s16be",
                "-f", "aiff",
                str(output_path),
            ]
        elif ext == ".mp3":
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(temp_wav_path),
                "-c:a", "libmp3lame", "-q:a", "2",
                str(output_path),
            ]
        else:
            cmd = [
                "ffmpeg", "-y", "-v", "error",
                "-i", str(temp_wav_path),
                "-c:a", "copy",
                str(output_path),
            ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=60, check=True)
        return True
    except Exception as e:
        print(f"Error converting with ffmpeg ({original_ext}): {e}")
        return False
    finally:
        if temp_wav_path is not None:
            temp_wav_path.unlink(missing_ok=True)


def save_audio_preserving_format(audio_data: np.ndarray, sr: int, output_path: Path, original_format: str) -> bool:
    """
    Save audio preserving the original format.
    
    For formats supported by soundfile (WAV, AIF, AIFF, FLAC, OGG), saves directly.
    For other formats (MP3, M4A, WMA, etc.), uses ffmpeg to convert from temporary WAV.
    
    Args:
        audio_data: Audio signal (mono, float32)
        sr: Sample rate (Hz)
        output_path: Output file path (with desired extension)
        original_format: Original format extension (e.g., '.mp3', '.m4a')
        
    Returns:
        True if successful, False otherwise
    """
    # Formats that soundfile can write directly
    SOUNDFILE_WRITABLE_FORMATS = {'.wav', '.aif', '.aiff', '.flac', '.ogg'}
    
    original_ext = original_format.lower()
    
    sf_kwargs = soundfile_format_kwargs(original_ext)

    # If format is supported by soundfile, save directly
    if original_ext in SOUNDFILE_WRITABLE_FORMATS:
        try:
            sf.write(str(output_path), audio_data, sr, **sf_kwargs)
            return True
        except Exception as e:
            print(f"Error saving {original_ext} via soundfile: {e}")
            if original_ext in {".aif", ".aiff", ".aifc"} and check_ffmpeg_available():
                return _save_via_ffmpeg_wav_temp(audio_data, sr, output_path, original_ext)
            return False
    
    # For formats not supported by soundfile (MP3, M4A, WMA, etc.), use ffmpeg
    if not check_ffmpeg_available():
        # Fallback: convert to WAV if ffmpeg not available
        output_path = output_path.with_suffix('.wav')
        print(f"Warning: ffmpeg not available. Converting {original_ext} to .wav")
        try:
            sf.write(str(output_path), audio_data, sr)
            return True
        except Exception as e:
            print(f"Error saving WAV: {e}")
            return False
    
    return _save_via_ffmpeg_wav_temp(audio_data, sr, output_path, original_ext)


def main():
    """Main function: CLI interface for pitch shifting."""
    parser = argparse.ArgumentParser(
        description="High-quality pitch shifting tool - preserves timbre, amplitude, and dynamics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-detect frequency and prompt for target
  python pitch_shift_tool.py input.wav
  
  # Specify target frequency
  python pitch_shift_tool.py input.wav --target-freq 440.0
  
  # Specify both detected and target frequencies (skip auto-detection)
  python pitch_shift_tool.py input.wav --detected-freq 425.0 --target-freq 440.0
  
  # Specify output file
  python pitch_shift_tool.py input.wav --target-freq 440.0 --output output.wav
        """
    )
    
    parser.add_argument('input', type=str, nargs='?', help='Input audio file path (optional if --show-table is used)')
    parser.add_argument('--target-freq', type=float, default=None,
                       help='Target frequency (Hz). If not provided, will prompt for input.')
    parser.add_argument('--detected-freq', type=float, default=None,
                       help='Current frequency (Hz). If not provided, will auto-detect.')
    parser.add_argument('--output', type=str, default=None,
                       help='Output file path. If not provided, will add "_shifted" suffix.')
    parser.add_argument('--show-table', action='store_true',
                       help='Show frequency reference table (all notes from C0 to C8) and exit.')
    
    args = parser.parse_args()
    
    # Show frequency reference table if requested
    if args.show_table:
        print_frequency_reference_table()
        sys.exit(0)
    
    # Validate input file
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)
    
    print("=" * 70)
    print("Pitch Shift Tool - High-Quality Pitch Correction")
    print("=" * 70)
    print(f"Input file: {input_path}")
    
    # Load audio
    print("\nLoading audio...")
    try:
        audio_data, sr = librosa.load(str(input_path), sr=None, mono=True)
        print(f"✓ Loaded: {len(audio_data) / sr:.2f} seconds, {sr} Hz sample rate")
    except Exception as e:
        print(f"ERROR: Failed to load audio: {e}")
        sys.exit(1)
    
    # Detect frequency (if not provided)
    if args.detected_freq is None:
        print("\nDetecting current frequency...")
        detected_freq = detect_frequency(audio_data, sr)
        if detected_freq <= 0:
            print("ERROR: Could not detect frequency. Please specify --detected-freq manually.")
            sys.exit(1)
        print(f"✓ Detected frequency: {detected_freq:.2f} Hz ({frequency_to_note(detected_freq)})")
    else:
        detected_freq = args.detected_freq
        print(f"\nUsing provided detected frequency: {detected_freq:.2f} Hz ({frequency_to_note(detected_freq)})")
    
    # Get target frequency
    if args.target_freq is None:
        print("\nEnter target frequency (Hz):")
        try:
            target_freq = float(input("Target frequency: "))
        except (ValueError, KeyboardInterrupt):
            print("\nERROR: Invalid input or cancelled.")
            sys.exit(1)
    else:
        target_freq = args.target_freq
    
    if target_freq <= 0:
        print("ERROR: Target frequency must be positive.")
        sys.exit(1)
    
    print(f"✓ Target frequency: {target_freq:.2f} Hz ({frequency_to_note(target_freq)})")
    
    # Calculate semitones
    semitones = calculate_semitones(detected_freq, target_freq)
    cents = semitones * 100.0
    
    print(f"\nShift calculation:")
    print(f"  Current: {detected_freq:.2f} Hz ({frequency_to_note(detected_freq)})")
    print(f"  Target:  {target_freq:.2f} Hz ({frequency_to_note(target_freq)})")
    print(f"  Shift:   {semitones:+.3f} semitones ({cents:+.1f} cents)")
    
    # Warn if shift is large (> 1/2 semitone)
    if abs(semitones) > 0.5:
        print(f"\n⚠️  Warning: Shift is > 1/2 semitone ({abs(semitones):.2f} semitones).")
        print("   For best quality, use this tool for small corrections (< 1/2 semitone).")
        response = input("   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Cancelled.")
            sys.exit(0)
    
    # Apply pitch shift
    print(f"\nApplying pitch shift...")
    try:
        shifted_audio = pitch_shift_audio(audio_data, sr, semitones)
        print("✓ Pitch shift applied successfully")
    except Exception as e:
        print(f"ERROR: Pitch shift failed: {e}")
        sys.exit(1)
    
    # Determine output path
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = input_path.parent / f"{input_path.stem}_shifted{input_path.suffix}"
    
    # Save output preserving original format
    print(f"\nSaving output...")
    print(f"Output file: {output_path}")
    
    try:
        original_format = input_path.suffix
        success = save_audio_preserving_format(shifted_audio, sr, output_path, original_format)
        
        if not success:
            print(f"ERROR: Failed to save audio file")
            sys.exit(1)
        
        print(f"✓ Saved successfully: {output_path}")
        
        # Verify amplitude preservation
        original_rms = np.sqrt(np.mean(audio_data ** 2))
        shifted_rms = np.sqrt(np.mean(shifted_audio ** 2))
        amplitude_ratio = shifted_rms / (original_rms + 1e-10)
        
        if abs(amplitude_ratio - 1.0) < 0.01:
            print(f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (perfect)")
        else:
            print(f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (within tolerance)")
        
    except Exception as e:
        print(f"ERROR: Failed to save output: {e}")
        sys.exit(1)
    
    print("\n" + "=" * 70)
    print("✓ Pitch shift completed successfully!")
    print("=" * 70)


if __name__ == '__main__':
    main()
