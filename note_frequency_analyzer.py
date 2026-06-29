#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Note Frequency Analyzer - Practical Tool
==========================================

A unified, practical tool that:
1. Scans a folder for audio files
2. Detects REAL frequencies from audio (not just filenames)
3. Compares detected vs expected frequencies
4. Highlights out-of-tune or mislabeled notes
5. Detects repeated notes (equal or enharmonic)
6. Optional cross-check against the equal-temperament frequency table

Author: SoundSpectrAnalyse Team
Version: 2.2
Date: June 2026
"""

import sys
import threading
import queue
import csv
from datetime import datetime
from pathlib import Path
from typing import List, Set, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import warnings

import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext

# Audio processing
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False
    print("ERROR: librosa is required. Install with: pip install librosa")
    sys.exit(1)

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False
    print("ERROR: soundfile is required. Install with: pip install soundfile")
    sys.exit(1)

# Audio playback
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Suppress warnings
warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


from auto_correct import (
    apply_auto_corrections,
    batch_folder_label,
    list_audio_files,
    list_batch_folders,
    run_verify_pass,
)
from pitch_core import (
    parse_note,
    parse_note_from_filename,
    note_to_midi,
    midi_to_note,
    generate_chromatic_scale,
    note_to_frequency,
    frequency_to_note,
    are_enharmonic,
    cents_difference,
    detect_frequency,
    detect_frequency_robust,
    detect_pitch,
    evaluate_tune_match,
    cross_check_expected_table,
    list_instruments,
    check_instrument_range,
)


@dataclass
class NoteAnalysis:
    """Analysis result for a single audio file."""
    filepath: Path
    filename: str
    expected_note: Optional[str]
    expected_freq: float
    detected_freq: float
    detected_note: str
    cents_off: float
    is_in_tune: bool
    is_mislabeled: bool
    status: str
    error_msg: Optional[str] = None
    range_warning: Optional[str] = None
    table_check: Optional[Dict[str, Any]] = None
    detected_manual: bool = False
    auto_fixed: bool = False
    batch_subfolder: Optional[str] = None


# ============================================================================
# Main GUI Application
# ============================================================================

class NoteFrequencyAnalyzerGUI(tk.Tk):
    """Main GUI application for note frequency analysis."""
    
    def __init__(self):
        super().__init__()
        self.title("Note Frequency Analyzer - Practical Tool")
        self.geometry("1200x800")
        
        self.folder_path = None
        self.analysis_results: List[NoteAnalysis] = []
        self.session_log: List[str] = []
        self.processing_thread = None
        self.progress_queue = queue.Queue()
        self.abort_flag = threading.Event()
        
        # Audio playback state
        self.is_playing = False
        self.playback_thread = None
        self.current_playing_file = None
        
        self._build_ui()
        self._check_progress_queue()
    
    def _build_ui(self):
        """Build the user interface."""
        # Top frame for controls
        top_frame = ttk.Frame(self, padding="10")
        top_frame.pack(fill=tk.X)
        
        # Folder selection
        ttk.Label(top_frame, text="Folder:").grid(row=0, column=0, sticky=tk.W, padx=5)
        self.folder_var = tk.StringVar()
        self.folder_entry = ttk.Entry(top_frame, textvariable=self.folder_var, width=60)
        self.folder_entry.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(top_frame, text="Browse...", command=self.browse_folder).grid(row=0, column=2, padx=5)
        
        top_frame.columnconfigure(1, weight=1)
        
        # Tolerance input
        ttk.Label(top_frame, text="Tolerance (cents):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        self.tolerance_var = tk.StringVar(value="20")
        tolerance_entry = ttk.Entry(top_frame, textvariable=self.tolerance_var, width=10)
        tolerance_entry.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Note range for missing notes check
        ttk.Label(top_frame, text="Check Range (for missing notes):").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        range_frame = ttk.Frame(top_frame)
        range_frame.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        ttk.Label(range_frame, text="Lowest:").pack(side=tk.LEFT, padx=2)
        self.lowest_note_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.lowest_note_var, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Label(range_frame, text="Highest:").pack(side=tk.LEFT, padx=2)
        self.highest_note_var = tk.StringVar()
        ttk.Entry(range_frame, textvariable=self.highest_note_var, width=10).pack(side=tk.LEFT, padx=2)
        
        ttk.Button(range_frame, text="Auto-detect", command=self._auto_detect_range).pack(side=tk.LEFT, padx=5)
        
        # Buttons
        button_frame = ttk.Frame(top_frame)
        button_frame.grid(row=5, column=0, columnspan=3, pady=10)
        
        self.analyze_btn = ttk.Button(button_frame, text="Analyze Folder", command=self.analyze_folder)
        self.analyze_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_btn = ttk.Button(button_frame, text="Stop", command=self.stop_processing, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(button_frame, text="Export Results", command=self.export_results).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Set Detected Note", command=self._set_detected_note).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Re-analyze Selected", command=self._reanalyze_selected).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="🔄 Refresh Analysis", command=self.refresh_analysis).pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        self.play_btn = ttk.Button(button_frame, text="▶ Play Selected", command=self.play_selected, state=tk.DISABLED)
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        self.stop_play_btn = ttk.Button(button_frame, text="⏹ Stop Playback", command=self.stop_playback, state=tk.DISABLED)
        self.stop_play_btn.pack(side=tk.LEFT, padx=5)
        
        self.use_table_check_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(top_frame, text="Cross-check vs ET frequency table",
                       variable=self.use_table_check_var).grid(row=1, column=2, sticky=tk.W, padx=5, pady=5)

        self.fix_octave_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame,
            text="Fix octave errors (use filename + harmonics)",
            variable=self.fix_octave_var,
        ).grid(row=2, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.auto_fix_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame,
            text="Auto-fix (≤½ st retune; else rename to detected note) then verify",
            variable=self.auto_fix_var,
        ).grid(row=3, column=2, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.batch_subfolders_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            top_frame,
            text="Process all subfolders (sequential batch)",
            variable=self.batch_subfolders_var,
        ).grid(row=3, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        self.batch_recursive_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(
            top_frame,
            text="Include nested subfolders (all levels)",
            variable=self.batch_recursive_var,
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, padx=5, pady=2)

        ttk.Label(top_frame, text="Instrument:").grid(row=1, column=3, sticky=tk.W, padx=5)
        inst_values = ['(none)'] + list_instruments()
        self.instrument_var = tk.StringVar(value='(none)')
        self.instrument_combo = ttk.Combobox(top_frame, textvariable=self.instrument_var,
                                           values=inst_values, width=14, state='readonly')
        self.instrument_combo.grid(row=1, column=4, sticky=tk.W, padx=5, pady=5)
        
        # Progress bar
        self.progress = ttk.Progressbar(top_frame, mode='determinate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), padx=5, pady=5)
        top_frame.columnconfigure(0, weight=0)
        top_frame.columnconfigure(1, weight=1)
        top_frame.columnconfigure(2, weight=0)
        
        # Status label
        self.status_label = ttk.Label(top_frame, text="Ready")
        self.status_label.grid(row=7, column=0, columnspan=3, sticky=tk.W, padx=5)
        
        # Main area with notebook
        notebook = ttk.Notebook(self)
        notebook.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Tab 1: Results table
        results_frame = ttk.Frame(notebook)
        notebook.add(results_frame, text="Analysis Results")
        
        # Treeview for results
        columns = ('Filename', 'Expected Note', 'Expected Freq', 'Detected Freq', 
                   'Detected Note', 'Cents Off', 'Status')
        self.tree = ttk.Treeview(results_frame, columns=columns, show='headings', height=15)
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=120)
        
        # Scrollbars
        scrollbar_y = ttk.Scrollbar(results_frame, orient=tk.VERTICAL, command=self.tree.yview)
        scrollbar_x = ttk.Scrollbar(results_frame, orient=tk.HORIZONTAL, command=self.tree.xview)
        self.tree.configure(yscrollcommand=scrollbar_y.set, xscrollcommand=scrollbar_x.set)
        
        self.tree.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        scrollbar_y.grid(row=0, column=1, sticky=(tk.N, tk.S))
        scrollbar_x.grid(row=1, column=0, sticky=(tk.W, tk.E))
        results_frame.grid_rowconfigure(0, weight=1)
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Boxes for missing and repeated notes
        info_frame = ttk.Frame(results_frame)
        info_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(10, 0))
        results_frame.grid_columnconfigure(0, weight=1)
        
        # Missing notes box
        missing_frame = ttk.LabelFrame(info_frame, text="Missing Notes", padding="5")
        missing_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S), padx=(0, 5))
        
        self.missing_notes_text = scrolledtext.ScrolledText(missing_frame, wrap=tk.WORD, height=6, width=30)
        self.missing_notes_text.pack(fill=tk.BOTH, expand=True)
        self.missing_notes_text.insert(1.0, "No analysis yet. Run analysis to see missing notes.")
        self.missing_notes_text.config(state=tk.DISABLED)
        
        # Repeated notes box
        repeated_frame = ttk.LabelFrame(info_frame, text="Repeated Notes", padding="5")
        repeated_frame.grid(row=0, column=1, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.repeated_notes_text = scrolledtext.ScrolledText(repeated_frame, wrap=tk.WORD, height=6, width=30)
        self.repeated_notes_text.pack(fill=tk.BOTH, expand=True)
        self.repeated_notes_text.insert(1.0, "No analysis yet. Run analysis to see repeated notes.")
        self.repeated_notes_text.config(state=tk.DISABLED)
        
        info_frame.grid_columnconfigure(0, weight=1)
        info_frame.grid_columnconfigure(1, weight=1)
        
        # Bind double-click to open in pitch shift tool
        self.tree.bind('<Double-1>', self._on_item_double_click)
        
        # Bind right-click for context menu (play option)
        # Button-3 is right-click on Windows/Linux, Button-2 on Mac
        self.tree.bind('<Button-3>', self._on_right_click)
        self.tree.bind('<Button-2>', self._on_right_click)  # Mac compatibility
        
        # Bind spacebar to play selected
        self.tree.bind('<space>', lambda e: self.play_selected())
        
        # Store mapping from tree items to file paths
        self.tree_item_to_filepath = {}
        
        # Tab 2: Summary
        summary_frame = ttk.Frame(notebook)
        notebook.add(summary_frame, text="Summary")
        
        self.summary_text = scrolledtext.ScrolledText(summary_frame, wrap=tk.WORD, height=30)
        self.summary_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
    
    def browse_folder(self):
        """Browse for folder containing audio files."""
        folder = filedialog.askdirectory(title="Select folder with audio files")
        if folder:
            self.folder_var.set(folder)
            self.folder_path = Path(folder)
    
    def _auto_detect_range(self):
        """Auto-detect note range from analyzed results."""
        if not self.analysis_results:
            messagebox.showinfo("Info", "Please analyze files first to auto-detect range.")
            return
        
        # Get all detected notes
        detected_notes = []
        for result in self.analysis_results:
            if result.detected_note != "Unknown" and result.detected_freq > 0:
                parsed = parse_note(result.detected_note)
                if parsed:
                    detected_notes.append(parsed)
        
        if not detected_notes:
            messagebox.showinfo("Info", "No valid notes found in analysis results.")
            return
        
        # Sort by MIDI number
        detected_notes.sort(key=lambda x: note_to_midi(x[0], x[1]))
        
        # Set lowest and highest
        lowest = detected_notes[0]
        highest = detected_notes[-1]
        
        self.lowest_note_var.set(f"{lowest[0]}{lowest[1]}")
        self.highest_note_var.set(f"{highest[0]}{highest[1]}")
    
    def analyze_folder(self):
        """Start analyzing folder in separate thread."""
        folder_str = self.folder_var.get().strip()
        if not folder_str:
            messagebox.showwarning("No folder", "Please select a folder first.")
            return
        
        self.folder_path = Path(folder_str).resolve()
        if not self.folder_path.exists() or not self.folder_path.is_dir():
            messagebox.showerror("Error", f"Folder does not exist or is not a directory: {folder_str}")
            return
        
        try:
            tolerance = float(self.tolerance_var.get())
            if tolerance < 0:
                raise ValueError("Tolerance must be non-negative")
        except ValueError:
            messagebox.showerror("Error", "Invalid tolerance value. Please enter a number.")
            return
        
        # Disable analyze button, enable stop button
        self.analyze_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.abort_flag.clear()
        
        # Clear previous results
        self.analysis_results = []
        self.session_log = []
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_item_to_filepath.clear()
        
        # Start processing thread
        self.processing_thread = threading.Thread(
            target=self._process_folder_threaded,
            args=(tolerance,),
            daemon=True
        )
        self.processing_thread.start()
    
    def stop_processing(self):
        """Stop processing."""
        self.abort_flag.set()
        self.status_label.config(text="Stopping...")
    
    def _check_progress_queue(self):
        """Check for progress updates from processing thread."""
        try:
            while True:
                msg = self.progress_queue.get_nowait()
                
                if msg["type"] == "progress":
                    self.progress["value"] = msg["value"]
                    self.progress["maximum"] = msg["maximum"]
                    if "status" in msg:
                        self.status_label.config(text=msg["status"])
                
                elif msg["type"] == "complete":
                    self.progress["value"] = msg["value"]
                    self.progress["maximum"] = msg["maximum"]
                    self.status_label.config(text=msg["status"])
                    self.analyze_btn.config(state=tk.NORMAL)
                    self.stop_btn.config(state=tk.DISABLED)
                    
                    # Auto-populate lowest/highest notes if not set
                    if self.analysis_results and (not self.lowest_note_var.get().strip() or not self.highest_note_var.get().strip()):
                        self._auto_detect_range()
                    
                    # Update display
                    self._update_display()
                    self._update_summary()
                    self._update_missing_notes_box()
                    self._update_repeated_notes_box()
                    report_path = self._write_folder_analysis_report()
                    status = msg.get("status", "Complete")
                    if report_path:
                        status = f"{status} | Report: {report_path.name}"
                    self.status_label.config(text=status)
                    
                    # Enable play button if results exist
                    if self.analysis_results:
                        self.play_btn.config(state=tk.NORMAL)
                
                elif msg["type"] == "clear_results":
                    self.analysis_results = []
                    for item in self.tree.get_children():
                        self.tree.delete(item)
                    self.tree_item_to_filepath.clear()

                elif msg["type"] == "log":
                    self._log_session_change(msg["message"])

                elif msg["type"] == "result":
                    # Add result to list
                    self.analysis_results.append(msg["result"])
                    self._add_result_to_tree(msg["result"])
        except queue.Empty:
            pass
        
        # Schedule next check
        self.after(100, self._check_progress_queue)
    
    def _process_folder_threaded(self, tolerance: float):
        """Process folder (or batch of subfolders) in a separate thread."""
        try:
            if not self.folder_path or not self.folder_path.exists():
                self.progress_queue.put({
                    "type": "complete",
                    "value": 0,
                    "maximum": 1,
                    "status": "Error: Folder does not exist",
                })
                return

            parent_path = self.folder_path.resolve()
            batch_mode = bool(
                getattr(self, "batch_subfolders_var", None) and self.batch_subfolders_var.get()
            )
            if batch_mode:
                recursive = bool(
                    getattr(self, "batch_recursive_var", None)
                    and self.batch_recursive_var.get()
                )
                target_folders = list_batch_folders(
                    parent_path, include_parent=True, recursive=recursive
                )
            else:
                target_folders = [parent_path]

            if not target_folders:
                self.progress_queue.put({
                    "type": "complete",
                    "value": 0,
                    "maximum": 1,
                    "status": "No audio files found",
                })
                return

            total_files = sum(len(list_audio_files(f)) for f in target_folders)
            processed_files = 0
            total_fixed = 0
            total_ok = 0
            total_review = 0
            folder_summaries: List[str] = []

            self.progress_queue.put({
                "type": "progress",
                "value": 0,
                "maximum": max(1, total_files),
                "status": (
                    f"Batch: {len(target_folders)} folder(s), {total_files} file(s)..."
                    if batch_mode
                    else f"Found {total_files} audio files. Processing..."
                ),
            })

            for folder_idx, folder_path in enumerate(target_folders, 1):
                if self.abort_flag.is_set():
                    break

                batch_label = (
                    batch_folder_label(folder_path, parent_path) if batch_mode else None
                )

                audio_files = list_audio_files(folder_path)
                if not audio_files:
                    continue

                folder_title = batch_label or folder_path.name
                if batch_mode:
                    self.progress_queue.put({
                        "type": "log",
                        "message": f"--- Folder {folder_idx}/{len(target_folders)}: {folder_title} ---",
                    })

                ok_count, issues, fixed_count = self._process_single_folder(
                    folder_path,
                    tolerance,
                    audio_files,
                    processed_files,
                    total_files,
                    batch_label=batch_label,
                )
                processed_files += len(audio_files)
                total_fixed += fixed_count
                total_ok += ok_count
                total_review += issues
                if batch_mode:
                    folder_summaries.append(f"{folder_title}: {ok_count}/{len(audio_files)} OK")

            if batch_mode and len(target_folders) > 1:
                summary_detail = "; ".join(folder_summaries) if folder_summaries else ""
                status_msg = (
                    f"Batch complete: {total_ok}/{total_files} OK across {len(target_folders)} folder(s) "
                    f"({total_fixed} action(s); {total_review} still need review)"
                )
                if summary_detail:
                    status_msg = f"{status_msg} | {summary_detail}"
            elif getattr(self, "auto_fix_var", None) and self.auto_fix_var.get():
                status_msg = (
                    f"Complete: {total_ok}/{total_files} OK after auto-fix "
                    f"({total_fixed} action(s); {total_review} still need review)"
                )
            else:
                status_msg = f"Complete: {processed_files} files analyzed"

            self.progress_queue.put({
                "type": "complete",
                "value": total_files,
                "maximum": max(1, total_files),
                "status": status_msg,
            })

        except Exception as e:
            self.progress_queue.put({
                "type": "complete",
                "value": 0,
                "maximum": 1,
                "status": f"Error: {str(e)}",
            })

    def _process_single_folder(
        self,
        folder_path: Path,
        tolerance: float,
        audio_files: List[Path],
        files_done_before: int,
        total_files: int,
        *,
        batch_label: Optional[str] = None,
    ) -> Tuple[int, int, int]:
        """Analyze one folder; optional auto-fix + verify. Returns (ok, issues, fixed_count)."""
        total = len(audio_files)
        auto_fix = bool(getattr(self, "auto_fix_var", None) and self.auto_fix_var.get())
        pass1_results: List[NoteAnalysis] = []

        for i, filepath in enumerate(audio_files):
            if self.abort_flag.is_set():
                break

            global_idx = files_done_before + i + 1
            prefix = f"{batch_label}: " if batch_label else ""
            self.progress_queue.put({
                "type": "progress",
                "value": files_done_before + i,
                "maximum": max(1, total_files),
                "status": f"Analyzing {global_idx}/{total_files}: {prefix}{filepath.name}",
            })

            result = self._analyze_file(filepath, tolerance)
            if result:
                result.batch_subfolder = batch_label
                pass1_results.append(result)
                if not auto_fix:
                    self.progress_queue.put({"type": "result", "result": result})

        fixed_count = 0
        if auto_fix and pass1_results and not self.abort_flag.is_set():
            self.progress_queue.put({
                "type": "progress",
                "value": files_done_before,
                "maximum": max(1, total_files),
                "status": f"Auto-correcting {folder_path.name} ({len(pass1_results)} files)...",
            })
            for i, result in enumerate(pass1_results):
                if self.abort_flag.is_set():
                    break
                inst = self.instrument_var.get() if hasattr(self, "instrument_var") else "(none)"
                inst_arg = inst if inst and inst != "(none)" else None
                _, actions = apply_auto_corrections(
                    result.filepath,
                    result.status,
                    result.expected_note,
                    result.detected_note,
                    result.detected_freq,
                    result.expected_freq,
                    result.cents_off,
                    tolerance,
                    instrument=inst_arg,
                )
                for act in actions:
                    if act.success:
                        fixed_count += 1
                        self.progress_queue.put({
                            "type": "log",
                            "message": f"Auto-{act.action}: {act.detail}",
                        })
                    else:
                        self.progress_queue.put({
                            "type": "log",
                            "message": f"Auto-{act.action} FAILED: {act.detail} ({act.error})",
                        })

            if not batch_label:
                self.progress_queue.put({"type": "clear_results"})

            self.progress_queue.put({
                "type": "progress",
                "value": files_done_before,
                "maximum": max(1, total_files),
                "status": f"Verify {folder_path.name}...",
            })

            def analyze_for_verify(fp: Path, tol: float) -> Optional[NoteAnalysis]:
                r = self._analyze_file(fp, tol)
                if r:
                    r.batch_subfolder = batch_label
                return r

            verify_results = run_verify_pass(folder_path, analyze_for_verify, tolerance)
            for i, result in enumerate(verify_results):
                if result:
                    result.auto_fixed = fixed_count > 0
                    self.progress_queue.put({"type": "result", "result": result})
                if self.abort_flag.is_set():
                    break

            ok_count = sum(1 for r in verify_results if r and r.status == "OK")
            issues = sum(
                1 for r in verify_results
                if r and r.status.split("+")[0] not in ("OK",)
            )
            return ok_count, issues, fixed_count

        return len(pass1_results), 0, 0
    
    def _analyze_file(self, filepath: Path, tolerance: float) -> Optional[NoteAnalysis]:
        """Analyze a single audio file."""
        try:
            # Extract expected note from filename
            filename = filepath.stem
            expected_note = None
            expected_freq = 0.0
            
            # Try to parse note from filename
            parsed = parse_note_from_filename(filename)
            if parsed:
                note_name, octave = parsed
                expected_note = f"{note_name}{octave}"
                expected_freq = note_to_frequency(expected_note)
            
            # Load and detect frequency
            try:
                audio_data, sr = librosa.load(str(filepath), sr=None, mono=True)
                use_octave_fix = getattr(self, "fix_octave_var", None) and self.fix_octave_var.get()
                inst = self.instrument_var.get() if hasattr(self, "instrument_var") else "(none)"
                inst_arg = inst if inst and inst != "(none)" else None
                if use_octave_fix:
                    detected_freq = detect_pitch(
                        audio_data, sr, expected_note=expected_note, instrument=inst_arg
                    )
                else:
                    detected_freq = detect_frequency(audio_data, sr)
                    if detected_freq <= 0:
                        detected_freq = detect_frequency_robust(
                            audio_data, sr, expected_note=expected_note, instrument=inst_arg
                        )
            except Exception as e:
                return NoteAnalysis(
                    filepath=filepath,
                    filename=filepath.name,
                    expected_note=expected_note,
                    expected_freq=expected_freq,
                    detected_freq=0.0,
                    detected_note="Unknown",
                    cents_off=float('inf'),
                    is_in_tune=False,
                    is_mislabeled=False,
                    status="ERROR",
                    error_msg=str(e)
                )
            
            if detected_freq <= 0:
                return NoteAnalysis(
                    filepath=filepath,
                    filename=filepath.name,
                    expected_note=expected_note,
                    expected_freq=expected_freq,
                    detected_freq=0.0,
                    detected_note="Unknown",
                    cents_off=float('inf'),
                    is_in_tune=False,
                    is_mislabeled=False,
                    status="NO_DETECTION",
                    error_msg="Could not detect frequency — use Set Detected Note"
                )
            
            # Convert detected frequency to note
            detected_note = frequency_to_note(detected_freq)

            range_warning = None
            inst = self.instrument_var.get() if hasattr(self, 'instrument_var') else '(none)'
            if inst and inst != '(none)':
                range_warning = check_instrument_range(detected_freq, inst)

            table_check = None
            if getattr(self, 'use_table_check_var', None) and self.use_table_check_var.get():
                if expected_note:
                    table_check = cross_check_expected_table(detected_freq, expected_note, tolerance)

            cents_off, is_in_tune, is_mislabeled, status = evaluate_tune_match(
                detected_freq,
                detected_note,
                expected_note,
                expected_freq,
                tolerance,
                range_warning=range_warning,
            )
            
            return NoteAnalysis(
                filepath=filepath,
                filename=filepath.name,
                expected_note=expected_note,
                expected_freq=expected_freq,
                detected_freq=detected_freq,
                detected_note=detected_note,
                cents_off=cents_off,
                is_in_tune=is_in_tune,
                is_mislabeled=is_mislabeled,
                status=status,
                error_msg=None,
                range_warning=range_warning,
                table_check=table_check,
            )
        
        except Exception as e:
            return NoteAnalysis(
                filepath=filepath,
                filename=filepath.name,
                expected_note=None,
                expected_freq=0.0,
                detected_freq=0.0,
                detected_note="Unknown",
                cents_off=float('inf'),
                is_in_tune=False,
                is_mislabeled=False,
                status="ERROR",
                error_msg=str(e)
            )
    
    def _tags_for_result(self, result: NoteAnalysis) -> List[str]:
        tags: List[str] = []
        if result.detected_manual:
            tags.append("manual")
        base_status = result.status.split("+")[0]
        if base_status == "OK":
            tags.append("ok")
        elif base_status == "OUT_OF_TUNE":
            tags.append("out_of_tune")
        elif base_status == "MISLABELED":
            tags.append("mislabeled")
        elif base_status in ("ERROR",):
            tags.append("error")
        elif base_status == "NO_DETECTION":
            tags.append("no_detection")
        elif base_status == "OCTAVE_ERROR":
            tags.append("octave_error")
        return tags

    def _format_result_tree_values(self, result: NoteAnalysis) -> Tuple[str, ...]:
        expected_note_str = result.expected_note if result.expected_note else "N/A"
        expected_freq_str = f"{result.expected_freq:.2f}" if result.expected_freq > 0 else "N/A"
        detected_freq_str = f"{result.detected_freq:.2f}" if result.detected_freq > 0 else "N/A"
        if result.detected_note != "Unknown":
            detected_note_str = result.detected_note
            if result.detected_manual:
                detected_note_str = f"{detected_note_str} (manual)"
        else:
            detected_note_str = "N/A"
        cents_str = f"{result.cents_off:.1f}" if result.cents_off != float("inf") else "N/A"
        status_str = result.status
        if result.detected_manual and "MANUAL" not in status_str:
            status_str = f"{status_str} [manual]"
        display_name = result.filename
        if result.batch_subfolder:
            display_name = f"{result.batch_subfolder}\\{result.filename}"
        return (
            display_name,
            expected_note_str,
            expected_freq_str,
            detected_freq_str,
            detected_note_str,
            cents_str,
            status_str,
        )

    def _refresh_tree_row(self, item: str, result: NoteAnalysis) -> None:
        self.tree.item(item, values=self._format_result_tree_values(result), tags=self._tags_for_result(result))

    def _result_for_item(self, item: str) -> Optional[NoteAnalysis]:
        filepath = self.tree_item_to_filepath.get(item)
        if not filepath:
            return None
        return next((r for r in self.analysis_results if r.filepath == filepath), None)

    def _get_tolerance(self) -> float:
        try:
            return float(self.tolerance_var.get() or 20)
        except (TypeError, ValueError):
            return 20.0

    def _apply_detection_to_result(
        self,
        result: NoteAnalysis,
        detected_freq: float,
        detected_note: str,
        tolerance: float,
        *,
        manual: bool = False,
    ) -> None:
        result.detected_freq = detected_freq
        result.detected_note = detected_note
        result.detected_manual = manual
        result.error_msg = None

        inst = self.instrument_var.get() if hasattr(self, "instrument_var") else "(none)"
        result.range_warning = None
        if inst and inst != "(none)" and detected_freq > 0:
            result.range_warning = check_instrument_range(detected_freq, inst)

        result.table_check = None
        if getattr(self, "use_table_check_var", None) and self.use_table_check_var.get():
            if result.expected_note and detected_freq > 0:
                result.table_check = cross_check_expected_table(
                    detected_freq, result.expected_note, tolerance
                )

        cents_off, is_in_tune, is_mislabeled, status = evaluate_tune_match(
            detected_freq,
            detected_note,
            result.expected_note,
            result.expected_freq,
            tolerance,
            range_warning=result.range_warning,
        )
        result.cents_off = cents_off
        result.is_in_tune = is_in_tune
        result.is_mislabeled = is_mislabeled
        result.status = status

    def _add_result_to_tree(self, result: NoteAnalysis):
        """Add a result to the treeview (keeps scroll position so the list does not jump)."""
        yview = self.tree.yview()
        at_bottom = yview[1] >= 0.99
        item = self.tree.insert(
            "",
            tk.END,
            values=self._format_result_tree_values(result),
            tags=self._tags_for_result(result),
        )
        self.tree_item_to_filepath[item] = result.filepath
        if at_bottom:
            self.tree.see(item)
        else:
            self.tree.yview_moveto(yview[0])

    def _set_detected_note(self, item: Optional[str] = None) -> None:
        """Let the user set or correct the detected note when auto-detection fails."""
        if item is None:
            selection = self.tree.selection()
            if not selection:
                messagebox.showinfo("Select a row", "Select a file in the results table first.")
                return
            item = selection[0]

        result = self._result_for_item(item)
        if not result:
            messagebox.showwarning("Error", "Could not find analysis for the selected row.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Set Detected Note")
        dialog.geometry("440x220")
        dialog.transient(self)
        dialog.grab_set()

        ttk.Label(dialog, text=f"File: {result.filename}").pack(pady=8)
        if result.expected_note:
            ttk.Label(dialog, text=f"Expected (filename): {result.expected_note}").pack()

        hint = "Use when pitch detection missed the fundamental."
        if result.status == "NO_DETECTION" or result.detected_note == "Unknown":
            hint += " Detection failed for this file."
        ttk.Label(dialog, text=hint, wraplength=400).pack(pady=4)

        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=8)
        ttk.Label(input_frame, text="Detected note (e.g. A4, C#5):").pack(side=tk.LEFT, padx=5)
        initial = result.detected_note if result.detected_note != "Unknown" else (result.expected_note or "")
        note_var = tk.StringVar(value=initial)
        note_entry = ttk.Entry(input_frame, textvariable=note_var, width=14)
        note_entry.pack(side=tk.LEFT, padx=5)
        note_entry.focus()
        note_entry.select_range(0, tk.END)

        rename_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            dialog,
            text="Also rename file so expected note matches (filename)",
            variable=rename_var,
        ).pack(pady=4)

        def apply_note() -> None:
            note_text = note_var.get().strip()
            if not note_text:
                messagebox.showwarning("Invalid Input", "Enter a note name.")
                return
            parsed = parse_note(note_text)
            if not parsed:
                messagebox.showerror(
                    "Invalid Note",
                    f"'{note_text}' is not valid. Use A4, C#5, Bb3, etc.",
                )
                return
            canonical = f"{parsed[0]}{parsed[1]}"
            freq = note_to_frequency(canonical)
            tolerance = self._get_tolerance()

            if rename_var.get():
                filepath = result.filepath.resolve()
                parsed_old = parse_note_from_filename(filepath.stem)
                if parsed_old:
                    old_note = f"{parsed_old[0]}{parsed_old[1]}"
                    new_filename = filepath.name.replace(old_note, canonical)
                else:
                    new_filename = f"{filepath.stem}_{canonical}{filepath.suffix}"
                new_filepath = filepath.parent / new_filename
                if new_filepath.exists() and new_filepath != filepath:
                    messagebox.showerror("Error", f"File already exists: {new_filename}")
                    return
                try:
                    filepath.rename(new_filepath)
                    result.filepath = new_filepath
                    result.filename = new_filename
                    result.expected_note = canonical
                    result.expected_freq = note_to_frequency(canonical)
                    self.tree_item_to_filepath[item] = new_filepath
                except OSError as e:
                    messagebox.showerror("Error", f"Could not rename file:\n{e}")
                    return

            old_name = result.filename
            self._apply_detection_to_result(result, freq, canonical, tolerance, manual=True)
            self._refresh_tree_row(item, result)
            if rename_var.get() and result.filename != old_name:
                self._log_session_change(
                    f"Renamed + set detected note: {old_name} → {result.filename} (detected {canonical})"
                )
            else:
                self._log_session_change(
                    f"Set detected note (manual): {result.filename} → {canonical}"
                )
            self._refresh_folder_report()
            dialog.destroy()

        btn_frame = ttk.Frame(dialog)
        btn_frame.pack(pady=10)
        ttk.Button(btn_frame, text="Apply", command=apply_note).pack(side=tk.LEFT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        note_entry.bind("<Return>", lambda e: apply_note())

    def _reanalyze_selected(self) -> None:
        """Re-run pitch detection on the selected file."""
        selection = self.tree.selection()
        if not selection:
            messagebox.showinfo("Select a row", "Select a file to re-analyze.")
            return
        item = selection[0]
        result = self._result_for_item(item)
        if not result or not result.filepath.exists():
            messagebox.showwarning("Error", "File not found.")
            return

        tolerance = self._get_tolerance()
        try:
            audio_data, sr = librosa.load(str(result.filepath), sr=None, mono=True)
            expected = result.expected_note if (
                getattr(self, "fix_octave_var", None) and self.fix_octave_var.get()
            ) else None
            inst = self.instrument_var.get() if hasattr(self, "instrument_var") else "(none)"
            inst_arg = inst if inst and inst != "(none)" else None
            if expected:
                detected_freq = detect_pitch(
                    audio_data, sr, expected_note=expected, instrument=inst_arg
                )
            else:
                detected_freq = detect_frequency(audio_data, sr)
                if detected_freq <= 0:
                    detected_freq = detect_frequency_robust(
                        audio_data, sr, expected_note=None, instrument=inst_arg
                    )
        except Exception as e:
            messagebox.showerror("Error", f"Could not load audio:\n{e}")
            return

        result.detected_manual = False
        if detected_freq <= 0:
            result.detected_freq = 0.0
            result.detected_note = "Unknown"
            result.cents_off = float("inf")
            result.is_in_tune = False
            result.is_mislabeled = False
            result.status = "NO_DETECTION"
            result.error_msg = "Could not detect frequency — use Set Detected Note"
        else:
            detected_note = frequency_to_note(detected_freq)
            self._apply_detection_to_result(
                result, detected_freq, detected_note, tolerance, manual=False
            )

        self._refresh_tree_row(item, result)
        self._update_summary()
        self._update_missing_notes_box()
        self._update_repeated_notes_box()

    def _on_item_double_click(self, event):
        """Double-click Detected Note column to edit; otherwise open pitch tool."""
        col = self.tree.identify_column(event.x)
        if col == "#5":
            item = self.tree.identify_row(event.y)
            if item:
                self._set_detected_note(item)
            return
        # Get the item that was clicked
        item = self.tree.identify_row(event.y)
        if not item:
            # Fallback to selection
            item = self.tree.selection()[0] if self.tree.selection() else None
        
        if not item:
            return
        
        # Get filepath from mapping
        filepath = self.tree_item_to_filepath.get(item)
        if not filepath:
            messagebox.showwarning("Error", "Could not find file path for selected item.")
            return
        
        if not filepath.exists():
            messagebox.showwarning("File not found", f"File not found:\n{filepath}")
            return
        
        # Launch pitch shift GUI with the file
        self._launch_pitch_shift_tool(filepath)
    
    def _on_right_click(self, event):
        """Handle right-click on treeview item - show context menu."""
        # Get the item that was clicked
        item = self.tree.identify_row(event.y)
        if not item:
            # Try to get selected item if click was outside
            selection = self.tree.selection()
            if selection:
                item = selection[0]
            else:
                return
        
        # Select the item
        self.tree.selection_set(item)
        
        # Get filepath from mapping
        filepath = self.tree_item_to_filepath.get(item)
        if not filepath:
            return
        
        # Create context menu (capture filepath in closure to avoid issues)
        filepath_copy = filepath  # Make a copy for the lambda closures
        
        context_menu = tk.Menu(self, tearoff=0)
        context_menu.add_command(
            label="Set detected note…",
            command=lambda i=item: self._set_detected_note(i),
        )
        context_menu.add_command(
            label="Re-analyze pitch",
            command=self._reanalyze_selected,
        )
        context_menu.add_separator()
        context_menu.add_command(
            label="Rename filename (expected note)…",
            command=lambda f=filepath_copy: self._rename_note(item, f),
        )
        context_menu.add_command(label="Delete File", command=lambda f=filepath_copy: self._delete_file(item, f))
        context_menu.add_separator()
        context_menu.add_command(label="Play Audio", command=lambda f=filepath_copy: self._play_selected_file(f))
        context_menu.add_command(label="Open in Pitch Tuner", command=lambda f=filepath_copy: self._launch_pitch_shift_tool(f))
        
        # Show context menu at cursor position
        try:
            # Use x_root and y_root for absolute screen coordinates
            context_menu.tk_popup(event.x_root, event.y_root)
        except tk.TclError:
            # Menu was already destroyed or window closed - ignore
            pass
        except Exception:
            # Fallback: try with widget-relative coordinates
            try:
                x = event.x + self.tree.winfo_rootx()
                y = event.y + self.tree.winfo_rooty()
                context_menu.tk_popup(x, y)
            except:
                pass
    
    def _rename_note(self, item, filepath: Path):
        """Rename the note in the filename."""
        # Resolve to absolute path immediately
        filepath = filepath.resolve()
        
        if not filepath.exists():
            messagebox.showerror("Error", "File not found.")
            return
        
        # Extract current note from filename
        filename = filepath.stem
        parsed = parse_note_from_filename(filename)
        current_note = None
        if parsed:
            current_note = f"{parsed[0]}{parsed[1]}"
        
        # Create dialog to get new note name
        dialog = tk.Toplevel(self)
        dialog.title("Rename Note")
        dialog.geometry("400x150")
        dialog.transient(self)
        dialog.grab_set()
        
        # Center the dialog
        dialog.update_idletasks()
        x = (dialog.winfo_screenwidth() // 2) - (dialog.winfo_width() // 2)
        y = (dialog.winfo_screenheight() // 2) - (dialog.winfo_height() // 2)
        dialog.geometry(f"+{x}+{y}")
        
        ttk.Label(dialog, text=f"Current filename: {filepath.name}").pack(pady=10)
        
        if current_note:
            ttk.Label(dialog, text=f"Current note: {current_note}").pack()
        else:
            ttk.Label(dialog, text="No note found in filename").pack()
        
        input_frame = ttk.Frame(dialog)
        input_frame.pack(pady=10)
        
        ttk.Label(input_frame, text="New note (e.g., A4, C#5, Bb3):").pack(side=tk.LEFT, padx=5)
        new_note_var = tk.StringVar(value=current_note if current_note else "")
        new_note_entry = ttk.Entry(input_frame, textvariable=new_note_var, width=15)
        new_note_entry.pack(side=tk.LEFT, padx=5)
        new_note_entry.focus()
        
        def do_rename():
            new_note = new_note_var.get().strip()
            if not new_note:
                messagebox.showwarning("Invalid Input", "Please enter a note name.")
                return
            
            # Validate note format
            parsed_new = parse_note(new_note)
            if not parsed_new:
                messagebox.showerror("Invalid Note", f"'{new_note}' is not a valid note format.\n\nUse format like: A4, C#5, Bb3, etc.")
                return
            
            # Build new filename
            old_filename = filepath.name
            file_extension = filepath.suffix
            
            # Replace note in filename
            if current_note:
                # Replace the old note with new note
                new_filename = old_filename.replace(current_note, new_note)
            else:
                # Add note to filename (before extension)
                new_filename = f"{filename}_{new_note}{file_extension}"
            
            # Check if new filename already exists
            new_filepath = filepath.parent / new_filename
            if new_filepath.exists() and new_filepath != filepath:
                messagebox.showerror("Error", f"A file with name '{new_filename}' already exists.")
                return
            
            try:
                # Resolve new filepath to absolute
                new_filepath = new_filepath.resolve()
                
                # Verify file exists before renaming (filepath already resolved)
                if not filepath.exists():
                    messagebox.showerror("Error", f"Source file not found:\n{filepath}")
                    return
                
                # Verify we're in the original folder (safety check)
                if self.folder_path and filepath.parent != self.folder_path.resolve():
                    response = messagebox.askyesno(
                        "Warning",
                        f"File is not in the selected folder.\n\n"
                        f"File: {filepath.parent}\n"
                        f"Selected folder: {self.folder_path}\n\n"
                        f"Continue with rename anyway?"
                    )
                    if not response:
                        return
                
                # Rename the file (this actually moves/renames on disk)
                filepath.rename(new_filepath)
                
                # Verify rename succeeded
                if not new_filepath.exists():
                    raise Exception("Rename operation completed but file not found at new location")
                
                # Update analysis result
                result = next((r for r in self.analysis_results if r.filepath == filepath), None)
                if result:
                    # Update the result
                    result.filepath = new_filepath
                    result.filename = new_filename
                    
                    # Re-parse expected note
                    parsed_new_full = parse_note(new_filename)
                    if parsed_new_full:
                        note_name, octave = parsed_new_full
                        result.expected_note = f"{note_name}{octave}"
                        result.expected_freq = note_to_frequency(result.expected_note)
                        
                        # Recalculate status if needed
                        if result.detected_freq > 0 and result.expected_freq > 0:
                            cents_off = 1200 * np.log2(result.detected_freq / result.expected_freq)
                            result.cents_off = cents_off
                            
                            # Re-determine status
                            tolerance = float(self.tolerance_var.get() or 20)
                            parsed_detected = parse_note(result.detected_note)
                            if parsed_detected:
                                notes_match = are_enharmonic(result.expected_note, result.detected_note)
                                if notes_match:
                                    effective_tolerance = tolerance * 1.1
                                    result.is_in_tune = abs(cents_off) <= effective_tolerance
                                    result.is_mislabeled = False
                                else:
                                    midi_expected = note_to_midi(parsed_new_full[0], parsed_new_full[1])
                                    midi_detected = note_to_midi(parsed_detected[0], parsed_detected[1])
                                    result.is_mislabeled = (midi_expected != midi_detected)
                                    result.is_in_tune = False
                            
                            # Update status string
                            if result.is_mislabeled:
                                result.status = "MISLABELED"
                            elif not result.is_in_tune:
                                result.status = "OUT_OF_TUNE"
                            else:
                                result.status = "OK"
                    
                    # Update treeview item
                    self.tree_item_to_filepath[item] = new_filepath
                    
                    # Update treeview values
                    expected_note_str = result.expected_note if result.expected_note else "N/A"
                    expected_freq_str = f"{result.expected_freq:.2f}" if result.expected_freq > 0 else "N/A"
                    detected_freq_str = f"{result.detected_freq:.2f}" if result.detected_freq > 0 else "N/A"
                    detected_note_str = result.detected_note if result.detected_note != "Unknown" else "N/A"
                    cents_str = f"{result.cents_off:.1f}" if result.cents_off != float('inf') else "N/A"
                    
                    self.tree.item(item, values=(
                        new_filename,
                        expected_note_str,
                        expected_freq_str,
                        detected_freq_str,
                        detected_note_str,
                        cents_str,
                        result.status
                    ))
                    
                    # Update color tags
                    tags = []
                    if result.status == "OK":
                        tags.append("ok")
                    elif result.status == "OUT_OF_TUNE":
                        tags.append("out_of_tune")
                    elif result.status == "MISLABELED":
                        tags.append("mislabeled")
                    elif result.status == "ERROR":
                        tags.append("error")
                    self.tree.item(item, tags=tags)
                
                self._log_session_change(
                    f"Renamed file (expected note): {old_filename} → {new_filename}"
                )
                self._refresh_folder_report()
                
                dialog.destroy()
                messagebox.showinfo(
                    "Success", 
                    f"File renamed successfully!\n\n"
                    f"Old: {old_filename}\n"
                    f"New: {new_filename}\n\n"
                    f"Location: {new_filepath.parent}\n"
                    f"✓ File saved to disk."
                )
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to rename file:\n{str(e)}")
        
        button_frame = ttk.Frame(dialog)
        button_frame.pack(pady=10)
        
        ttk.Button(button_frame, text="Rename", command=do_rename).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Cancel", command=dialog.destroy).pack(side=tk.LEFT, padx=5)
        
        # Bind Enter key
        new_note_entry.bind('<Return>', lambda e: do_rename())
    
    def _delete_file(self, item, filepath: Path):
        """Delete a file from both interface and folder."""
        # Resolve to absolute path
        filepath = filepath.resolve()
        
        if not filepath.exists():
            messagebox.showerror("Error", "File not found.")
            return
        
        # Confirm deletion
        response = messagebox.askyesno(
            "Confirm Deletion",
            f"Are you sure you want to delete this file?\n\n"
            f"File: {filepath.name}\n"
            f"Location: {filepath.parent}\n\n"
            f"This action cannot be undone!",
            icon=messagebox.WARNING
        )
        
        if not response:
            return
        
        try:
            deleted_name = filepath.name
            filepath.unlink()
            
            self.analysis_results = [r for r in self.analysis_results if r.filepath != filepath]
            self.tree.delete(item)
            if item in self.tree_item_to_filepath:
                del self.tree_item_to_filepath[item]
            
            self._log_session_change(f"Deleted file: {deleted_name}")
            self._refresh_folder_report()
            
            messagebox.showinfo("Success", f"File deleted successfully:\n{deleted_name}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to delete file:\n{str(e)}")
    
    def _play_selected_file(self, filepath: Path):
        """Play a specific file (used by context menu)."""
        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showerror(
                "Error",
                "sounddevice not available. Install with: pip install sounddevice"
            )
            return
        
        if not filepath.exists():
            messagebox.showerror("Error", "File not found.")
            return
        
        # Stop any currently playing file
        self.stop_playback()
        
        # Load and play the file
        self._play_audio_threaded(filepath)
    
    def _launch_pitch_shift_tool(self, filepath: Path):
        """Launch pitch shift GUI tool with the specified file."""
        try:
            import subprocess
            import sys
            
            # Get the path to pitch_shift_gui.py (same directory as this script)
            script_dir = Path(__file__).parent
            pitch_shift_gui_path = script_dir / "pitch_shift_gui.py"
            
            if not pitch_shift_gui_path.exists():
                messagebox.showerror(
                    "Error",
                    f"pitch_shift_gui.py not found at:\n{pitch_shift_gui_path}\n\n"
                    "Please ensure pitch_shift_gui.py is in the same directory."
                )
                return
            
            # Launch in a new process
            subprocess.Popen(
                [sys.executable, str(pitch_shift_gui_path), str(filepath)],
                cwd=str(script_dir)
            )
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to launch pitch shift tool:\n{str(e)}")
    
    def _update_display(self):
        """Update treeview with color tags."""
        # Configure tags
        self.tree.tag_configure("ok", background="#d4edda")
        self.tree.tag_configure("out_of_tune", background="#fff3cd")
        self.tree.tag_configure("mislabeled", background="#f8d7da")
        self.tree.tag_configure("error", background="#f5c6cb")
        self.tree.tag_configure("no_detection", background="#e9ecef")
        self.tree.tag_configure("octave_error", background="#e2d9f3")
        self.tree.tag_configure("manual", background="#cfe2ff")
    
    def _log_session_change(self, message: str) -> None:
        """Record a user action (rename, manual note, delete) for the folder report."""
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.session_log.append(f"[{stamp}] {message}")

    def _compose_analysis_report(self) -> str:
        """Build full text report (summary + issues + file list + session changes)."""
        if not self.analysis_results:
            return ""

        summary_lines: List[str] = []
        summary_lines.append("NOTE FREQUENCY ANALYSIS — FINAL REPORT")
        summary_lines.append("=" * 80)
        summary_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if self.folder_path:
            summary_lines.append(f"Analyzed folder: {self.folder_path}")
        summary_lines.append(f"Tolerance: {self.tolerance_var.get()} cents")
        if hasattr(self, "instrument_var"):
            summary_lines.append(f"Instrument filter: {self.instrument_var.get()}")
        if hasattr(self, "fix_octave_var"):
            summary_lines.append(
                f"Octave correction: {'ON' if self.fix_octave_var.get() else 'OFF'}"
            )
        summary_lines.append("")

        if self.session_log:
            summary_lines.append("=" * 80)
            summary_lines.append("CHANGES MADE IN THIS SESSION (files renamed / notes corrected)")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            for entry in self.session_log:
                summary_lines.append(f"  {entry}")
            summary_lines.append("")

        summary_lines.append("=" * 80)
        summary_lines.append("ANALYSIS SUMMARY")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        total = len(self.analysis_results)
        ok_count = sum(1 for r in self.analysis_results if r.status == "OK")
        out_of_tune_count = sum(1 for r in self.analysis_results if r.status == "OUT_OF_TUNE")
        mislabeled_count = sum(1 for r in self.analysis_results if r.status == "MISLABELED")
        octave_err_count = sum(
            1 for r in self.analysis_results if r.status.split("+")[0] == "OCTAVE_ERROR"
        )
        error_count = sum(1 for r in self.analysis_results if r.status == "ERROR")
        no_det_count = sum(1 for r in self.analysis_results if r.status == "NO_DETECTION")
        manual_count = sum(1 for r in self.analysis_results if r.detected_manual)
        no_note_count = sum(1 for r in self.analysis_results if r.status == "NO_NOTE_IN_FILENAME")
        
        summary_lines.append(f"Total files analyzed: {total}")
        summary_lines.append(f"  ✓ OK (in tune): {ok_count}")
        summary_lines.append(f"  ⚠ Out of tune: {out_of_tune_count}")
        summary_lines.append(f"  ✗ Mislabeled: {mislabeled_count}")
        summary_lines.append(f"  ↕ Octave error (C4↔C5): {octave_err_count}")
        summary_lines.append(f"  ❌ Errors: {error_count}")
        summary_lines.append(f"  ? No detection: {no_det_count}")
        summary_lines.append(f"  ✎ Manual detected note: {manual_count}")
        summary_lines.append(f"  ? No note in filename: {no_note_count}")
        summary_lines.append("")
        
        # Repeated notes detection (enharmonic-aware)
        summary_lines.append("=" * 80)
        summary_lines.append("REPEATED NOTES DETECTION (Enharmonic-aware)")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        # Group by MIDI number (handles enharmonic equivalence)
        note_groups: Dict[int, List[NoteAnalysis]] = {}
        
        for result in self.analysis_results:
            if result.detected_note != "Unknown" and result.detected_freq > 0:
                parsed = parse_note(result.detected_note)
                if parsed:
                    note_name, octave = parsed
                    midi = note_to_midi(note_name, octave)
                    
                    # Use MIDI number as key (handles enharmonic equivalence)
                    if midi not in note_groups:
                        note_groups[midi] = []
                    note_groups[midi].append(result)
        
        # Find repeated notes (same MIDI = same pitch, even if enharmonic)
        repeated_notes = {k: v for k, v in note_groups.items() if len(v) > 1}
        
        if repeated_notes:
            summary_lines.append(f"Found {len(repeated_notes)} notes with multiple files (including enharmonic):")
            summary_lines.append("")
            
            for midi, results in sorted(repeated_notes.items()):
                # Get all note representations for this MIDI
                note_reprs = set()
                for r in results:
                    parsed = parse_note(r.detected_note)
                    if parsed:
                        note_reprs.add(f"{parsed[0]}{parsed[1]}")
                
                note_str = ", ".join(sorted(note_reprs))
                summary_lines.append(f"  {note_str} (MIDI {midi}) - {len(results)} files:")
                for r in results:
                    expected_str = f" (expected: {r.expected_note})" if r.expected_note else ""
                    summary_lines.append(f"    - {r.filename}{expected_str}")
            summary_lines.append("")
        else:
            summary_lines.append("✓ No repeated notes found (all notes are unique)")
            summary_lines.append("")
        
        # Missing notes detection
        summary_lines.append("=" * 80)
        summary_lines.append("MISSING NOTES DETECTION")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        
        lowest_str = self.lowest_note_var.get().strip()
        highest_str = self.highest_note_var.get().strip()
        
        # Auto-detect if not set
        if not lowest_str or not highest_str:
            # Try to auto-detect from detected notes
            detected_notes = []
            for result in self.analysis_results:
                if result.detected_note != "Unknown" and result.detected_freq > 0:
                    parsed = parse_note(result.detected_note)
                    if parsed:
                        detected_notes.append(parsed)
            
            if detected_notes:
                detected_notes.sort(key=lambda x: note_to_midi(x[0], x[1]))
                lowest = detected_notes[0]
                highest = detected_notes[-1]
                lowest_str = f"{lowest[0]}{lowest[1]}"
                highest_str = f"{highest[0]}{highest[1]}"
        
        if lowest_str and highest_str:
            # Generate expected chromatic scale
            expected_scale = self._generate_chromatic_scale(lowest_str, highest_str)
            
            if expected_scale:
                # Get all detected notes (by MIDI to handle enharmonic)
                detected_midi = set()
                for result in self.analysis_results:
                    if result.detected_note != "Unknown" and result.detected_freq > 0:
                        parsed = parse_note(result.detected_note)
                        if parsed:
                            midi = note_to_midi(parsed[0], parsed[1])
                            detected_midi.add(midi)
                
                # Convert expected scale to MIDI
                expected_midi = set()
                for note_name, octave in expected_scale:
                    midi = note_to_midi(note_name, octave)
                    expected_midi.add(midi)
                
                # Find missing notes
                missing_midi = expected_midi - detected_midi
                
                if missing_midi:
                    summary_lines.append(f"Range: {lowest_str} to {highest_str}")
                    summary_lines.append(f"Expected notes: {len(expected_scale)}")
                    summary_lines.append(f"Found notes: {len(detected_midi)}")
                    summary_lines.append(f"Missing notes: {len(missing_midi)}")
                    summary_lines.append("")
                    summary_lines.append("MISSING NOTES:")
                    summary_lines.append("-" * 80)
                    
                    # Convert missing MIDI back to notes
                    missing_notes = []
                    for midi in sorted(missing_midi):
                        note_name, octave = midi_to_note(midi)
                        missing_notes.append((note_name, octave))
                    
                    for note_name, octave in missing_notes:
                        summary_lines.append(f"  {note_name}{octave}")
                    summary_lines.append("")
                else:
                    summary_lines.append(f"Range: {lowest_str} to {highest_str}")
                    summary_lines.append(f"✓ All notes in the chromatic scale are present!")
                    summary_lines.append("")
            else:
                summary_lines.append("⚠ Could not parse note range. Please check format (e.g., C3, A#4, Bb2)")
                summary_lines.append("")
        else:
            summary_lines.append("⚠ Note range not specified. Enter lowest and highest notes to check for missing notes.")
            summary_lines.append("")
        
        # Out of tune notes
        if out_of_tune_count > 0:
            summary_lines.append("=" * 80)
            summary_lines.append("OUT OF TUNE NOTES")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            
            out_of_tune = [r for r in self.analysis_results if r.status == "OUT_OF_TUNE"]
            for r in sorted(out_of_tune, key=lambda x: abs(x.cents_off), reverse=True)[:20]:
                summary_lines.append(f"  {r.filename}:")
                summary_lines.append(f"    Expected: {r.expected_note} ({r.expected_freq:.2f} Hz)")
                summary_lines.append(f"    Detected: {r.detected_note} ({r.detected_freq:.2f} Hz)")
                summary_lines.append(f"    Difference: {r.cents_off:.1f} cents")
                summary_lines.append("")
        
        # Mislabeled notes
        if mislabeled_count > 0:
            summary_lines.append("=" * 80)
            summary_lines.append("MISLABELED NOTES")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            
            mislabeled = [r for r in self.analysis_results if r.status == "MISLABELED"]
            for r in mislabeled:
                summary_lines.append(f"  {r.filename}:")
                summary_lines.append(f"    Labeled as: {r.expected_note} ({r.expected_freq:.2f} Hz)")
                summary_lines.append(f"    Actually is: {r.detected_note} ({r.detected_freq:.2f} Hz)")
                summary_lines.append(f"    Difference: {r.cents_off:.1f} cents")
                summary_lines.append("")

        if octave_err_count > 0:
            summary_lines.append("=" * 80)
            summary_lines.append("OCTAVE ERRORS (same note name, wrong octave — e.g. C4 vs C5)")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            for r in self.analysis_results:
                if r.status.split("+")[0] == "OCTAVE_ERROR":
                    summary_lines.append(f"  {r.filename}:")
                    summary_lines.append(f"    Expected: {r.expected_note} ({r.expected_freq:.2f} Hz)")
                    summary_lines.append(f"    Detected: {r.detected_note} ({r.detected_freq:.2f} Hz)")
                    summary_lines.append(f"    Difference: {r.cents_off:.1f} cents")
                    summary_lines.append("")

        if no_det_count > 0:
            summary_lines.append("=" * 80)
            summary_lines.append("NO PITCH DETECTION")
            summary_lines.append("=" * 80)
            summary_lines.append("")
            for r in self.analysis_results:
                if r.status == "NO_DETECTION":
                    summary_lines.append(f"  {r.filename}  (expected: {r.expected_note or 'N/A'})")
            summary_lines.append("")

        summary_lines.append("=" * 80)
        summary_lines.append("COMPLETE FILE LIST")
        summary_lines.append("=" * 80)
        summary_lines.append("")
        summary_lines.append(
            f"{'Filename':<42} {'Expected':<8} {'Detected':<10} {'Cents':>8}  {'Status'}"
        )
        summary_lines.append("-" * 80)
        for r in sorted(self.analysis_results, key=lambda x: x.filename.lower()):
            exp = r.expected_note or "N/A"
            det = r.detected_note if r.detected_note != "Unknown" else "N/A"
            if r.detected_manual:
                det = f"{det}*"
            cents = f"{r.cents_off:.1f}" if r.cents_off != float("inf") else "N/A"
            summary_lines.append(
                f"{r.filename:<42} {exp:<8} {det:<10} {cents:>8}  {r.status}"
            )
        summary_lines.append("")
        summary_lines.append(
            "Output files in this folder: tuning_analysis_report.txt, tuning_analysis_results.csv"
        )
        summary_lines.append("")

        return "\n".join(summary_lines)

    def _write_folder_analysis_report(self) -> Optional[Path]:
        """Write final report + CSV into the analyzed audio folder."""
        if not self.folder_path or not self.analysis_results:
            return None
        try:
            text = self._compose_analysis_report()
            report_path = self.folder_path / "tuning_analysis_report.txt"
            report_path.write_text(text, encoding="utf-8")
            csv_path = self.folder_path / "tuning_analysis_results.csv"
            self._export_results_csv(csv_path)
            return report_path
        except OSError as exc:
            messagebox.showwarning(
                "Report not saved",
                f"Could not write report into the analyzed folder:\n{exc}",
            )
            return None

    def _export_results_csv(self, filepath: Path) -> None:
        """Write results table to CSV (used by export dialog and auto-report)."""
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "Filename",
                "Expected Note",
                "Expected Freq (Hz)",
                "Detected Freq (Hz)",
                "Detected Note",
                "Cents Off",
                "Status",
                "Is In Tune",
                "Is Mislabeled",
                "Manual Override",
                "Error Message",
            ])
            for result in self.analysis_results:
                writer.writerow([
                    result.filename,
                    result.expected_note or "",
                    f"{result.expected_freq:.2f}" if result.expected_freq > 0 else "",
                    f"{result.detected_freq:.2f}" if result.detected_freq > 0 else "",
                    result.detected_note,
                    f"{result.cents_off:.1f}" if result.cents_off != float("inf") else "",
                    result.status,
                    "Yes" if result.is_in_tune else "No",
                    "Yes" if result.is_mislabeled else "No",
                    "Yes" if result.detected_manual else "No",
                    result.error_msg or "",
                ])

    def _refresh_folder_report(self) -> None:
        """Refresh Summary tab and rewrite report files in the analyzed folder."""
        self._update_summary()
        self._update_missing_notes_box()
        self._update_repeated_notes_box()
        report_path = self._write_folder_analysis_report()
        if report_path:
            self.status_label.config(
                text=f"Report updated in folder: {report_path.name}"
            )

    def _update_summary(self):
        """Update summary tab text."""
        if not self.analysis_results:
            return
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, self._compose_analysis_report())
    
    def _generate_chromatic_scale(self, lowest_note: str, highest_note: str) -> List[Tuple[str, int]]:
        """Wrapper for generate_chromatic_scale."""
        return generate_chromatic_scale(lowest_note, highest_note)
    
    def _update_missing_notes_box(self):
        """Update the missing notes box."""
        self.missing_notes_text.config(state=tk.NORMAL)
        self.missing_notes_text.delete(1.0, tk.END)
        
        if not self.analysis_results:
            self.missing_notes_text.insert(1.0, "No analysis yet. Run analysis to see missing notes.")
            self.missing_notes_text.config(state=tk.DISABLED)
            return
        
        lowest_str = self.lowest_note_var.get().strip()
        highest_str = self.highest_note_var.get().strip()
        
        # Auto-detect if not set
        if not lowest_str or not highest_str:
            detected_notes = []
            for result in self.analysis_results:
                if result.detected_note != "Unknown" and result.detected_freq > 0:
                    parsed = parse_note(result.detected_note)
                    if parsed:
                        detected_notes.append(parsed)
            
            if detected_notes:
                detected_notes.sort(key=lambda x: note_to_midi(x[0], x[1]))
                lowest = detected_notes[0]
                highest = detected_notes[-1]
                lowest_str = f"{lowest[0]}{lowest[1]}"
                highest_str = f"{highest[0]}{highest[1]}"
        
        if lowest_str and highest_str:
            expected_scale = self._generate_chromatic_scale(lowest_str, highest_str)
            
            if expected_scale:
                # Get all detected notes (by MIDI to handle enharmonic)
                detected_midi = set()
                for result in self.analysis_results:
                    if result.detected_note != "Unknown" and result.detected_freq > 0:
                        parsed = parse_note(result.detected_note)
                        if parsed:
                            midi = note_to_midi(parsed[0], parsed[1])
                            detected_midi.add(midi)
                
                # Convert expected scale to MIDI
                expected_midi = set()
                for note_name, octave in expected_scale:
                    midi = note_to_midi(note_name, octave)
                    expected_midi.add(midi)
                
                # Find missing notes
                missing_midi = expected_midi - detected_midi
                
                if missing_midi:
                    # Convert missing MIDI back to notes
                    missing_notes = []
                    for midi in sorted(missing_midi):
                        note_name, octave = midi_to_note(midi)
                        missing_notes.append(f"{note_name}{octave}")
                    
                    self.missing_notes_text.insert(1.0, f"Range: {lowest_str} to {highest_str}\n")
                    self.missing_notes_text.insert(tk.END, f"Missing: {len(missing_notes)} notes\n\n")
                    self.missing_notes_text.insert(tk.END, "\n".join(missing_notes))
                else:
                    self.missing_notes_text.insert(1.0, f"Range: {lowest_str} to {highest_str}\n\n")
                    self.missing_notes_text.insert(tk.END, "✓ All notes in the chromatic scale are present!")
            else:
                self.missing_notes_text.insert(1.0, "⚠ Could not parse note range.\nPlease check format (e.g., C3, A#4, Bb2)")
        else:
            self.missing_notes_text.insert(1.0, "⚠ Note range not specified.\nEnter lowest and highest notes to check for missing notes.")
        
        self.missing_notes_text.config(state=tk.DISABLED)
    
    def _update_repeated_notes_box(self):
        """Update the repeated notes box."""
        self.repeated_notes_text.config(state=tk.NORMAL)
        self.repeated_notes_text.delete(1.0, tk.END)
        
        if not self.analysis_results:
            self.repeated_notes_text.insert(1.0, "No analysis yet. Run analysis to see repeated notes.")
            self.repeated_notes_text.config(state=tk.DISABLED)
            return
        
        # Group by MIDI number (handles enharmonic equivalence)
        note_groups: Dict[int, List[NoteAnalysis]] = {}
        
        for result in self.analysis_results:
            if result.detected_note != "Unknown" and result.detected_freq > 0:
                parsed = parse_note(result.detected_note)
                if parsed:
                    note_name, octave = parsed
                    midi = note_to_midi(note_name, octave)
                    
                    if midi not in note_groups:
                        note_groups[midi] = []
                    note_groups[midi].append(result)
        
        # Find repeated notes (same MIDI = same pitch, even if enharmonic)
        repeated_notes = {k: v for k, v in note_groups.items() if len(v) > 1}
        
        if repeated_notes:
            lines = []
            lines.append(f"Found {len(repeated_notes)} repeated note(s):\n")
            
            for midi, results in sorted(repeated_notes.items()):
                # Get all note representations for this MIDI
                note_reprs = set()
                for r in results:
                    parsed = parse_note(r.detected_note)
                    if parsed:
                        note_reprs.add(f"{parsed[0]}{parsed[1]}")
                
                note_str = ", ".join(sorted(note_reprs))
                lines.append(f"{note_str}: {len(results)} file(s)")
                for r in results[:3]:  # Show first 3 files
                    lines.append(f"  • {r.filename}")
                if len(results) > 3:
                    lines.append(f"  ... and {len(results) - 3} more")
                lines.append("")
            
            self.repeated_notes_text.insert(1.0, "\n".join(lines))
        else:
            self.repeated_notes_text.insert(1.0, "✓ No repeated notes found\n(all notes are unique)")
        
        self.repeated_notes_text.config(state=tk.DISABLED)
    
    def refresh_analysis(self):
        """Refresh analysis of the current folder (useful after tuning files)."""
        if not self.folder_path:
            messagebox.showwarning("No Folder", "Please select a folder first.")
            return
        
        # Clear current results
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.tree_item_to_filepath.clear()
        self.analysis_results.clear()
        self.summary_text.delete(1.0, tk.END)
        
        # Re-analyze
        self.analyze_folder()
    
    def export_results(self):
        """Export results to CSV (default: save into the analyzed folder)."""
        if not self.analysis_results:
            messagebox.showwarning("No results", "No results to export. Run analysis first.")
            return

        initial = str(self.folder_path) if self.folder_path else ""
        filepath = filedialog.asksaveasfilename(
            title="Save results CSV",
            initialdir=initial,
            initialfile="tuning_analysis_results.csv",
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filepath:
            return
        try:
            self._export_results_csv(Path(filepath))
            messagebox.showinfo("Success", f"Results exported to:\n{filepath}")
        except OSError as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def play_selected(self):
        """Play the selected audio file."""
        if not SOUNDDEVICE_AVAILABLE:
            messagebox.showerror(
                "Error",
                "sounddevice not available. Install with: pip install sounddevice"
            )
            return
        
        # Get selected item
        selection = self.tree.selection()
        if not selection:
            messagebox.showwarning("No Selection", "Please select a file from the list.")
            return
        
        item = selection[0]
        filepath = self.tree_item_to_filepath.get(item)
        
        if not filepath or not filepath.exists():
            messagebox.showerror("Error", "File not found.")
            return
        
        # Stop any currently playing file
        self.stop_playback()
        
        # Load and play the file
        self._play_audio_threaded(filepath)
    
    def stop_playback(self):
        """Stop audio playback."""
        if SOUNDDEVICE_AVAILABLE:
            try:
                sd.stop()
            except:
                pass
        
        self.current_playing_file = None
        self.stop_play_btn.config(state=tk.DISABLED)
    
    def _play_audio_threaded(self, filepath: Path):
        """Play audio file in a separate thread."""
        def play_thread():
            try:
                # Load audio file
                audio_data, sample_rate = sf.read(str(filepath))
                
                # Convert to mono if stereo
                if len(audio_data.shape) > 1:
                    audio_data = np.mean(audio_data, axis=1)
                
                # Update UI (use try-except to handle if window is closed)
                try:
                    self.after(0, lambda: self.play_btn.config(state=tk.DISABLED) if hasattr(self, 'play_btn') else None)
                    self.after(0, lambda: self.stop_play_btn.config(state=tk.NORMAL) if hasattr(self, 'stop_play_btn') else None)
                    self.after(0, lambda: setattr(self, 'current_playing_file', filepath) if hasattr(self, 'current_playing_file') else None)
                except:
                    pass
                
                # Play audio
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()  # Wait until playback is finished
                
                # Update UI when done (use try-except to handle if window is closed)
                try:
                    self.after(0, lambda: self.play_btn.config(state=tk.NORMAL) if hasattr(self, 'play_btn') else None)
                    self.after(0, lambda: self.stop_play_btn.config(state=tk.DISABLED) if hasattr(self, 'stop_play_btn') else None)
                    self.after(0, lambda: setattr(self, 'current_playing_file', None) if hasattr(self, 'current_playing_file') else None)
                except:
                    pass
                
            except Exception as e:
                # Only show error if window still exists
                try:
                    self.after(0, lambda: messagebox.showerror("Playback Error", f"Failed to play audio:\n{str(e)}") if hasattr(self, 'play_btn') else None)
                    self.after(0, lambda: self.play_btn.config(state=tk.NORMAL) if hasattr(self, 'play_btn') else None)
                    self.after(0, lambda: self.stop_play_btn.config(state=tk.DISABLED) if hasattr(self, 'stop_play_btn') else None)
                    self.after(0, lambda: setattr(self, 'current_playing_file', None) if hasattr(self, 'current_playing_file') else None)
                except:
                    pass
        
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()


def main():
    """Main function."""
    if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
        messagebox.showerror(
            "Missing Dependencies",
            "Required libraries not available.\n\n"
            "Install with:\npip install librosa soundfile"
        )
        return
    
    app = NoteFrequencyAnalyzerGUI()
    app.mainloop()


if __name__ == '__main__':
    main()
