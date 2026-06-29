"""
Automatic pitch correction and filename fixes for the note frequency analyzer.

Rules:
1. Out of tune by <= 1/2 semitone (50 cents), same note as filename -> retune in place.
2. Mislabeled, octave error, or > 1/2 semitone off -> rename to closest detected note.
   If that filename already exists, append _2, _3, ... (never overwrite).
3. Caller runs a second analysis pass to verify results.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import librosa

from pitch_core import (
    AUTO_RETUNE_MAX_CENTS,
    NOTE_PATTERN,
    cents_difference,
    calculate_semitones,
    detect_frequency,
    detect_pitch,
    parse_note,
    parse_note_from_filename,
    note_to_frequency,
    are_enharmonic,
    same_pitch_class,
)

# Re-export for tests / GUI
__all__ = [
    "AUTO_RETUNE_MAX_CENTS",
    "AutoCorrectAction",
    "build_renamed_path",
    "list_batch_folders",
    "plan_corrections",
    "rename_file_note",
    "retune_file_in_place",
    "apply_auto_corrections",
    "run_verify_pass",
    "run_batch_auto_correct",
]


@dataclass
class AutoCorrectAction:
    filepath: Path
    action: str
    detail: str
    success: bool = True
    error: Optional[str] = None
    new_path: Optional[Path] = None


def display_note_token(note_str: str) -> str:
    """Validated note string preserving user spelling (Bb vs A#)."""
    parsed = parse_note(note_str)
    if not parsed:
        raise ValueError(f"Invalid note: {note_str}")
    match = NOTE_PATTERN.search(note_str.strip())
    if match:
        letter = match.group(1).upper()
        accidental = match.group(2).replace("♯", "#").replace("♭", "b")
        return f"{letter}{accidental}{parsed[1]}"
    return f"{parsed[0]}{parsed[1]}"


def build_renamed_path(filepath: Path, new_note: str) -> Path:
    """Build target path using the same rules as the manual rename dialog."""
    token = display_note_token(new_note)
    parsed_old = parse_note_from_filename(filepath.stem)
    if parsed_old:
        old_note = f"{parsed_old[0]}{parsed_old[1]}"
        new_filename = filepath.name.replace(old_note, token)
    else:
        new_filename = f"{filepath.stem}_{token}{filepath.suffix}"
    return filepath.parent / new_filename


def resolve_unique_rename_path(filepath: Path, new_note: str) -> Path:
    """Pick a non-colliding path when the target note filename already exists."""
    primary = build_renamed_path(filepath, new_note)
    if not primary.exists() or primary.resolve() == filepath.resolve():
        return primary
    token = display_note_token(new_note)
    suffix = filepath.suffix
    parent = filepath.parent
    for n in range(2, 100):
        candidate = parent / f"{token}_{n}{suffix}"
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"No free name for {token} in {parent}")


def rename_file_note(filepath: Path, new_note: str) -> Path:
    filepath = filepath.resolve()
    new_path = resolve_unique_rename_path(filepath, new_note)
    filepath.rename(new_path)
    return new_path.resolve()


def retune_file_in_place(
    filepath: Path,
    target_freq: float,
    *,
    instrument: Optional[str] = None,
) -> float:
    """Pitch-shift audio in place toward target_freq. Returns applied semitone shift."""
    from pitch_shift_tool import pitch_shift_audio, save_audio_preserving_format

    filepath = filepath.resolve()
    audio, sr = librosa.load(str(filepath), sr=None, mono=True)
    expected = None
    parsed = parse_note_from_filename(filepath.stem)
    if parsed:
        expected = f"{parsed[0]}{parsed[1]}"
    inst = instrument if instrument not in (None, "", "(none)") else None
    detected = (
        detect_pitch(audio, sr, expected_note=expected, instrument=inst)
        if expected
        else detect_frequency(audio, sr)
    )
    if detected <= 0:
        detected = detect_frequency(audio, sr)
    if detected <= 0:
        raise RuntimeError("Could not detect frequency for auto-retune")

    semitones = calculate_semitones(detected, target_freq)
    if abs(semitones) > AUTO_RETUNE_MAX_CENTS / 100.0:
        raise RuntimeError(
            f"Required shift {semitones:+.3f} semitones exceeds auto limit "
            f"({AUTO_RETUNE_MAX_CENTS / 100.0:.1f} st)"
        )

    shifted = pitch_shift_audio(audio, sr, semitones)
    if not save_audio_preserving_format(shifted, sr, filepath, filepath.suffix):
        raise RuntimeError("Failed to save retuned audio")
    return float(semitones)


def _rename_target_differs(expected_note: Optional[str], detected_note: str) -> bool:
    """True when detected note token differs from filename note (enharmonic counts as same)."""
    if not expected_note or detected_note in ("Unknown", "") or not parse_note(detected_note):
        return False
    try:
        return display_note_token(detected_note) != display_note_token(expected_note)
    except ValueError:
        return False


def should_rename_to_detected(
    status: str,
    expected_note: Optional[str],
    detected_note: str,
    cents_off: float,
) -> bool:
    """
    Rename when mislabeled/octave slip, or when > 1/2 semitone off and closest note differs.
    """
    base = (status or "").split("+")[0]
    if base in ("ERROR", "NO_DETECTION", "NO_NOTE_IN_FILENAME", "OK"):
        return False
    if not parse_note(detected_note):
        return False
    if base in ("MISLABELED", "OCTAVE_ERROR"):
        return True
    if base == "OUT_OF_TUNE" and cents_off != float("inf") and cents_off > AUTO_RETUNE_MAX_CENTS:
        return _rename_target_differs(expected_note, detected_note)
    return False


def plan_corrections(
    status: str,
    expected_note: Optional[str],
    detected_note: str,
    detected_freq: float,
    expected_freq: float,
    cents_off: float,
) -> List[str]:
    """
    Return ordered actions: 'rename' and/or 'retune'.
    Rename is planned before retune when both apply.
    """
    actions: List[str] = []
    base = (status or "").split("+")[0]

    if base in ("ERROR", "NO_DETECTION", "NO_NOTE_IN_FILENAME", "OK"):
        return actions

    if should_rename_to_detected(status, expected_note, detected_note, cents_off):
        actions.append("rename")

    if detected_freq > 0 and expected_freq > 0 and expected_note:
        if base == "OUT_OF_TUNE" and "rename" not in actions:
            notes_align = are_enharmonic(expected_note, detected_note) or same_pitch_class(
                expected_note, detected_note
            )
            if notes_align and cents_off != float("inf") and cents_off <= AUTO_RETUNE_MAX_CENTS:
                actions.append("retune")
        elif "rename" in actions:
            # After rename, expected will match detected note name; retune if still slightly off
            if cents_off != float("inf") and cents_off <= AUTO_RETUNE_MAX_CENTS:
                actions.append("retune")

    return actions


def _notes_align_for_retune(expected_note: Optional[str], detected_note: str) -> bool:
    if not expected_note or detected_note in ("Unknown", ""):
        return False
    return are_enharmonic(expected_note, detected_note) or same_pitch_class(
        expected_note, detected_note
    )


def apply_auto_corrections(
    filepath: Path,
    status: str,
    expected_note: Optional[str],
    detected_note: str,
    detected_freq: float,
    expected_freq: float,
    cents_off: float,
    tolerance: float = 20.0,
    *,
    instrument: Optional[str] = None,
) -> Tuple[Path, List[AutoCorrectAction]]:
    """Apply auto-corrections to one file. Returns (possibly new path, action log)."""
    path = filepath.resolve()
    actions_log: List[AutoCorrectAction] = []
    current_expected = expected_note
    current_expected_freq = expected_freq
    base = (status or "").split("+")[0]

    if should_rename_to_detected(status, expected_note, detected_note, cents_off):
        try:
            token = display_note_token(detected_note)
            old_name = path.name
            new_path = rename_file_note(path, token)
            reason = base if base in ("MISLABELED", "OCTAVE_ERROR") else ">½ st off"
            actions_log.append(
                AutoCorrectAction(
                    filepath=path,
                    action="rename",
                    detail=f"{old_name} → {new_path.name} ({reason}, detected {token})",
                    new_path=new_path,
                )
            )
            path = new_path
            current_expected = token
            current_expected_freq = note_to_frequency(token)
        except (OSError, ValueError) as exc:
            actions_log.append(
                AutoCorrectAction(
                    filepath=path,
                    action="rename",
                    detail=f"Failed rename to {detected_note}",
                    success=False,
                    error=str(exc),
                )
            )

    if detected_freq > 0 and current_expected_freq > 0 and _notes_align_for_retune(
        current_expected, detected_note
    ):
        shift_cents = cents_difference(detected_freq, current_expected_freq)
        if 0.5 < shift_cents <= AUTO_RETUNE_MAX_CENTS:
            try:
                semitones = retune_file_in_place(
                    path, current_expected_freq, instrument=instrument
                )
                actions_log.append(
                    AutoCorrectAction(
                        filepath=path,
                        action="retune",
                        detail=(
                            f"{path.name}: {shift_cents:.1f} cents → "
                            f"{current_expected} ({semitones:+.3f} st)"
                        ),
                    )
                )
            except (OSError, RuntimeError, ValueError) as exc:
                actions_log.append(
                    AutoCorrectAction(
                        filepath=path,
                        action="retune",
                        detail=f"Failed retune {path.name}",
                        success=False,
                        error=str(exc),
                    )
                )

    return path, actions_log


def run_verify_pass(
    folder: Path,
    analyze_fn: Callable[[Path, float], object],
    tolerance: float,
) -> List[object]:
    """Re-analyze every audio file in folder (verification pass)."""
    folder = folder.resolve()
    audio_files = list_audio_files(folder)
    return [analyze_fn(fp, tolerance) for fp in audio_files]


def list_audio_files(folder: Path) -> List[Path]:
    folder = folder.resolve()
    audio_extensions = {".wav", ".mp3", ".flac", ".ogg", ".aif", ".aiff", ".m4a", ".wma"}
    audio_files: List[Path] = []
    for ext in audio_extensions:
        for pattern in (f"*{ext}", f"*{ext.upper()}"):
            for filepath in folder.glob(pattern):
                if filepath.parent.resolve() == folder and filepath.is_file():
                    audio_files.append(filepath)
    return sorted(set(audio_files), key=lambda p: p.name.lower())


def batch_folder_label(folder: Path, parent: Path) -> str:
    """Relative path label for batch progress / results (e.g. `set_a/ff`)."""
    folder = folder.resolve()
    parent = parent.resolve()
    if folder == parent:
        return "(root)"
    try:
        return folder.relative_to(parent).as_posix()
    except ValueError:
        return folder.name


def list_batch_folders(
    parent: Path,
    *,
    include_parent: bool = True,
    recursive: bool = False,
) -> List[Path]:
    """
    Folders under parent that contain audio files (non-recursive into each folder).

    - recursive=False: immediate children only (+ parent if include_parent)
    - recursive=True: every descendant directory at any depth (+ parent if include_parent)

    Sorted by relative path (case-insensitive).
    """
    parent = parent.resolve()
    found: List[Path] = []

    def add_if_has_audio(directory: Path) -> None:
        if list_audio_files(directory):
            found.append(directory.resolve())

    if include_parent:
        add_if_has_audio(parent)

    if recursive:
        candidates = [
            p for p in parent.rglob("*") if p.is_dir() and p.resolve() != parent
        ]
    else:
        candidates = [p for p in parent.iterdir() if p.is_dir()]

    for directory in candidates:
        add_if_has_audio(directory)

    unique: List[Path] = []
    seen: set[Path] = set()
    for folder in found:
        if folder not in seen:
            seen.add(folder)
            unique.append(folder)

    return sorted(unique, key=lambda p: batch_folder_label(p, parent).lower())


@dataclass
class FileAnalysis:
    filepath: Path
    expected_note: Optional[str]
    expected_freq: float
    detected_freq: float
    detected_note: str
    cents_off: float
    status: str


def analyze_file(
    filepath: Path,
    tolerance: float,
    *,
    fix_octave: bool = True,
    instrument: Optional[str] = None,
) -> FileAnalysis:
    """Single-file pitch analysis for CLI / batch auto-correct."""
    from pitch_core import (
        detect_frequency,
        detect_frequency_robust,
        detect_pitch,
        evaluate_tune_match,
        frequency_to_note,
    )

    stem = filepath.stem
    expected_note = None
    expected_freq = 0.0
    parsed = parse_note_from_filename(stem)
    if parsed:
        expected_note = f"{parsed[0]}{parsed[1]}"
        expected_freq = note_to_frequency(expected_note)

    audio, sr = librosa.load(str(filepath), sr=None, mono=True)
    inst = instrument if instrument not in (None, "", "(none)") else None
    if fix_octave and expected_note:
        detected_freq = detect_pitch(
            audio, sr, expected_note=expected_note, instrument=inst
        )
    else:
        detected_freq = detect_frequency(audio, sr)
        if detected_freq <= 0:
            detected_freq = detect_frequency_robust(
                audio, sr, expected_note=expected_note, instrument=inst
            )

    if detected_freq <= 0:
        return FileAnalysis(
            filepath=filepath,
            expected_note=expected_note,
            expected_freq=expected_freq,
            detected_freq=0.0,
            detected_note="Unknown",
            cents_off=float("inf"),
            status="NO_DETECTION",
        )

    detected_note = frequency_to_note(detected_freq)
    cents_off, _, _, status = evaluate_tune_match(
        detected_freq,
        detected_note,
        expected_note,
        expected_freq,
        tolerance,
    )
    return FileAnalysis(
        filepath=filepath,
        expected_note=expected_note,
        expected_freq=expected_freq,
        detected_freq=detected_freq,
        detected_note=detected_note,
        cents_off=cents_off,
        status=status,
    )


def run_folder_auto_correct(
    folder: Path,
    tolerance: float = 20.0,
    *,
    fix_octave: bool = True,
    dry_run: bool = False,
    instrument: Optional[str] = None,
) -> int:
    """
    Analyze folder → auto-fix → verify. Returns process exit code (0 = all OK).
    """
    folder = folder.resolve()
    if not folder.is_dir():
        print(f"ERROR: Not a folder: {folder}")
        return 1

    files = list_audio_files(folder)
    if not files:
        print(f"ERROR: No audio files in: {folder}")
        return 1

    print(f"Folder: {folder}")
    print(f"Files: {len(files)} | Tolerance: {tolerance} cents | Dry-run: {dry_run}")
    print("=" * 60)

    pass1: List[FileAnalysis] = []
    for i, fp in enumerate(files, 1):
        print(f"[1/3] Analyze {i}/{len(files)}: {fp.name}")
        pass1.append(
            analyze_file(fp, tolerance, fix_octave=fix_octave, instrument=instrument)
        )

    to_fix = [
        r for r in pass1
        if r.status.split("+")[0] not in ("OK", "ERROR", "NO_DETECTION", "NO_NOTE_IN_FILENAME")
    ]
    print(f"\nIssues found: {len(to_fix)} / {len(pass1)}")

    action_count = 0
    if dry_run:
        for r in to_fix:
            planned = plan_corrections(
                r.status, r.expected_note, r.detected_note,
                r.detected_freq, r.expected_freq, r.cents_off,
            )
            if planned:
                print(f"  Would {', '.join(planned)}: {r.filepath.name} ({r.status})")
    else:
        print("\n[2/3] Auto-correcting...")
        for i, r in enumerate(to_fix, 1):
            planned = plan_corrections(
                r.status, r.expected_note, r.detected_note,
                r.detected_freq, r.expected_freq, r.cents_off,
            )
            if not planned:
                continue
            print(f"  Fix {i}/{len(to_fix)}: {r.filepath.name} -> {planned}")
            _, actions = apply_auto_corrections(
                r.filepath,
                r.status,
                r.expected_note,
                r.detected_note,
                r.detected_freq,
                r.expected_freq,
                r.cents_off,
                tolerance,
                instrument=instrument,
            )
            for act in actions:
                if act.success:
                    action_count += 1
                    print(f"    OK {act.action}: {act.detail}")
                else:
                    print(f"    FAIL {act.action}: {act.detail} — {act.error}")

        print(f"\n[3/3] Verification pass...")
        pass2 = [
            analyze_file(fp, tolerance, fix_octave=fix_octave, instrument=instrument)
            for fp in list_audio_files(folder)
        ]
        ok = sum(1 for r in pass2 if r.status == "OK")
        issues = [r for r in pass2 if r.status.split("+")[0] != "OK"]
        print(f"\nResult: {ok}/{len(pass2)} OK after auto-fix ({action_count} action(s))")
        if issues:
            print("\nStill need review:")
            for r in issues:
                cents = f"{r.cents_off:.1f}" if r.cents_off != float("inf") else "N/A"
                print(f"  {r.filepath.name}: {r.status} (exp {r.expected_note}, det {r.detected_note}, {cents} ct)")
            return 2
        return 0

    return 0


def run_batch_auto_correct(
    parent: Path,
    tolerance: float = 20.0,
    *,
    fix_octave: bool = True,
    dry_run: bool = False,
    include_parent: bool = True,
    recursive: bool = True,
    instrument: Optional[str] = None,
) -> int:
    """
    Run auto-correct sequentially on each subfolder under parent.
    Returns 0 if all folders OK, 2 if any folder still has issues, 1 on fatal error.
    """
    parent = parent.resolve()
    if not parent.is_dir():
        print(f"ERROR: Not a folder: {parent}")
        return 1

    folders = list_batch_folders(
        parent, include_parent=include_parent, recursive=recursive
    )
    if not folders:
        print(f"ERROR: No audio files in subfolders of: {parent}")
        return 1

    print(f"Batch parent: {parent}")
    print(f"Subfolders to process: {len(folders)} ({'recursive' if recursive else 'immediate children only'})")
    print("=" * 60)

    worst_code = 0
    for idx, folder in enumerate(folders, 1):
        label = batch_folder_label(folder, parent)
        print(f"\n### [{idx}/{len(folders)}] {label} ###")
        code = run_folder_auto_correct(
            folder,
            tolerance=tolerance,
            fix_octave=fix_octave,
            dry_run=dry_run,
            instrument=instrument,
        )
        worst_code = max(worst_code, code)

    if worst_code == 0:
        print(f"\nBatch complete: all {len(folders)} folder(s) OK")
    else:
        print(f"\nBatch complete: some folders still need review (exit {worst_code})")
    return worst_code


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Batch auto-fix for a folder of named audio samples.\n"
            "Rules: retune if <= 1/2 semitone off; rename if mislabeled or > 1/2 st off; then verify."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  python auto_correct.py "C:\\samples\\violin"\n'
            '  python auto_correct.py "C:\\samples\\violin" --dry-run\n'
            '  python auto_correct.py "C:\\samples\\parent" --batch\n'
            "\n"
            "Windows menu: START-Tune-Detection.bat (option 2)\n"
            "GUI alternative: START-Tune-Detection.bat (option 1)"
        ),
    )
    parser.add_argument(
        "folder",
        nargs="?",
        help="Folder with audio files (note name in each filename)",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=20.0,
        help="In-tune tolerance in cents (default: 20)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned fixes only; do not modify files",
    )
    parser.add_argument(
        "--no-octave-fix",
        action="store_true",
        help="Disable harmonic octave correction during detection",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process each subfolder (with audio) under the given parent folder",
    )
    parser.add_argument(
        "--shallow",
        action="store_true",
        help="With --batch: only immediate subfolders (default is all nested levels)",
    )
    parser.add_argument(
        "--instrument",
        default=None,
        help="Instrument registry key (e.g. double_bass) for low-range detection",
    )
    args = parser.parse_args()

    if not args.folder:
        parser.print_help()
        print("\nERROR: Provide a folder path, or use the GUI:")
        print("  python note_frequency_analyzer.py")
        return 1

    folder = Path(args.folder)
    if args.batch:
        return run_batch_auto_correct(
            folder,
            tolerance=args.tolerance,
            fix_octave=not args.no_octave_fix,
            dry_run=args.dry_run,
            recursive=not args.shallow,
            instrument=args.instrument,
        )
    return run_folder_auto_correct(
        folder,
        tolerance=args.tolerance,
        fix_octave=not args.no_octave_fix,
        dry_run=args.dry_run,
        instrument=args.instrument,
    )


if __name__ == "__main__":
    raise SystemExit(main())
