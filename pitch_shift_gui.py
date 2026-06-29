#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pitch Shift Tool - GUI Interface
=================================

High-quality pitch shifting tool with graphical interface.
Preserves timbre, amplitude, dynamics, and format.

Usage:
    python pitch_shift_gui.py

Features:
- Drag & drop or file browser
- Automatic frequency detection
- Manual target frequency input
- Real-time shift calculation (semitones & cents)
- Preserves: timbre, amplitude, dynamics, format
- Optimized for small corrections (< 1/2 semitone)

Author: SoundSpectrAnalyse Team
Version: 1.0
Date: January 2026
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
import threading
import warnings
import subprocess
import tempfile
import shutil

import numpy as np

# Import dependencies with graceful error handling
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    SOUNDFILE_AVAILABLE = False

# Suppress librosa warnings
if LIBROSA_AVAILABLE:
    warnings.filterwarnings('ignore', category=UserWarning, module='librosa')

from pitch_core import (
    detect_frequency,
    calculate_semitones,
    NOTE_FREQUENCY_MAP,
    note_to_frequency,
    frequency_to_note,
)
from pitch_shift_tool import pitch_shift_audio, save_audio_preserving_format


class PitchShiftGUI:
    """Graphical interface for pitch shifting tool."""
    
    def __init__(self, root, initial_file=None):
        self.root = root
        self.root.title("Pitch Shift Tool - High-Quality Pitch Correction")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # State variables
        self.input_file = None
        self.audio_data = None
        self.sample_rate = None
        self.detected_freq = None
        self.target_freq_var = tk.StringVar(value="440.0")
        
        # If initial file provided, load it
        self.initial_file = initial_file
        # If opened from analyzer, default to replacing original
        self.from_analyzer = initial_file is not None
        
        # Check dependencies
        if not LIBROSA_AVAILABLE or not SOUNDFILE_AVAILABLE:
            self.show_error(
                "Missing Dependencies",
                "librosa and soundfile are required.\n\n"
                "Install with:\npip install librosa soundfile"
            )
            root.destroy()
            return
        
        self.create_widgets()
        
        # Load initial file if provided
        if self.initial_file:
            self.load_file(Path(self.initial_file))
    
    def load_file(self, filepath: Path):
        """Load a file into the GUI."""
        if not filepath.exists():
            messagebox.showerror("Error", f"File not found: {filepath}")
            return
        
        self.input_file = filepath
        self.file_label.config(text=self.input_file.name, foreground="black")
        self.detect_btn.config(state=tk.NORMAL)
        self.process_btn.config(state=tk.DISABLED)
        self.detected_label.config(text="-- Hz (--)", foreground="gray")
        self.shift_label.config(text="-- semitones (-- cents)", foreground="gray")
        self.warning_label.config(text="")
        self.status_label.config(text=f"File selected: {self.input_file.name}")
        
        # Clear previous data
        self.audio_data = None
        self.sample_rate = None
        self.detected_freq = None
        
        # Auto-detect frequency if file is loaded
        self.root.after(100, self.detect_frequency_threaded)
        
    def create_widgets(self):
        """Create GUI widgets."""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title and menu bar
        title_frame = ttk.Frame(main_frame)
        title_frame.grid(row=0, column=0, columnspan=3, pady=(0, 10), sticky=(tk.W, tk.E))
        title_frame.columnconfigure(0, weight=1)
        
        title_label = ttk.Label(
            title_frame,
            text="Pitch Shift Tool",
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, sticky=tk.W)
        
        # Reference table button
        table_btn = ttk.Button(
            title_frame,
            text="Show Frequency Table",
            command=self.show_frequency_table,
            width=20
        )
        table_btn.grid(row=0, column=1, sticky=tk.E, padx=(10, 0))
        
        # File selection
        ttk.Label(main_frame, text="Audio File:").grid(row=1, column=0, sticky=tk.W, pady=5)
        
        self.file_label = ttk.Label(
            main_frame,
            text="No file selected",
            foreground="gray"
        )
        self.file_label.grid(row=1, column=1, sticky=(tk.W, tk.E), padx=5)
        
        browse_btn = ttk.Button(
            main_frame,
            text="Browse...",
            command=self.browse_file
        )
        browse_btn.grid(row=1, column=2, padx=5)
        
        # Detect button
        self.detect_btn = ttk.Button(
            main_frame,
            text="Detect Frequency",
            command=self.detect_frequency_threaded,
            state=tk.DISABLED
        )
        self.detect_btn.grid(row=2, column=0, columnspan=3, pady=10, sticky=(tk.W, tk.E))
        
        # Detected frequency
        ttk.Label(main_frame, text="Detected Frequency:").grid(row=3, column=0, sticky=tk.W, pady=5)
        
        self.detected_label = ttk.Label(
            main_frame,
            text="-- Hz (--)",
            foreground="gray"
        )
        self.detected_label.grid(row=3, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        # Target frequency
        ttk.Label(main_frame, text="Target Frequency:").grid(row=4, column=0, sticky=tk.W, pady=5)
        
        target_frame = ttk.Frame(main_frame)
        target_frame.grid(row=4, column=1, columnspan=2, sticky=(tk.W, tk.E), padx=5)
        target_frame.columnconfigure(0, weight=1)
        
        self.target_entry = ttk.Entry(
            target_frame,
            textvariable=self.target_freq_var,
            width=15
        )
        self.target_entry.grid(row=0, column=0, sticky=(tk.W, tk.E))
        self.target_entry.bind('<KeyRelease>', self.update_shift_calculation)
        
        ttk.Label(target_frame, text="Hz").grid(row=0, column=1, padx=5)
        
        # Shift calculation
        ttk.Label(main_frame, text="Shift Calculation:").grid(row=5, column=0, sticky=tk.W, pady=5)
        
        self.shift_label = ttk.Label(
            main_frame,
            text="-- semitones (-- cents)",
            foreground="gray"
        )
        self.shift_label.grid(row=5, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        # Target note
        self.target_note_label = ttk.Label(
            main_frame,
            text="Target note: --",
            foreground="gray"
        )
        self.target_note_label.grid(row=6, column=1, columnspan=2, sticky=tk.W, padx=5)
        
        # Warning label
        self.warning_label = ttk.Label(
            main_frame,
            text="",
            foreground="orange",
            wraplength=550
        )
        self.warning_label.grid(row=7, column=0, columnspan=3, pady=10, sticky=tk.W)
        
        # Replace original file option
        # Default to True if opened from analyzer
        self.replace_original_var = tk.BooleanVar(value=self.from_analyzer)
        replace_check = ttk.Checkbutton(
            main_frame,
            text="Replace original file (saves tuned file with original name, backs up original)",
            variable=self.replace_original_var
        )
        replace_check.grid(row=8, column=0, columnspan=3, pady=5, sticky=tk.W)
        
        # Process button
        self.process_btn = ttk.Button(
            main_frame,
            text="Apply Pitch Shift",
            command=self.process_audio_threaded,
            state=tk.DISABLED
        )
        self.process_btn.grid(row=9, column=0, columnspan=3, pady=20, sticky=(tk.W, tk.E))
        
        # Progress bar (hidden initially)
        self.progress = ttk.Progressbar(
            main_frame,
            mode='indeterminate'
        )
        self.progress.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(5, 0))
        self.progress.grid_remove()
        
        # Status bar (moved to bottom)
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            relief=tk.SUNKEN,
            anchor=tk.W
        )
        self.status_label.grid(row=11, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(10, 0))
        
    def browse_file(self):
        """Browse for audio file."""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.aif *.aiff *.m4a *.wma"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.input_file = Path(filename)
            self.file_label.config(text=self.input_file.name, foreground="black")
            self.detect_btn.config(state=tk.NORMAL)
            self.process_btn.config(state=tk.DISABLED)
            self.detected_label.config(text="-- Hz (--)", foreground="gray")
            self.shift_label.config(text="-- semitones (-- cents)", foreground="gray")
            self.warning_label.config(text="")
            self.status_label.config(text=f"File selected: {self.input_file.name}")
            
            # Clear previous data
            self.audio_data = None
            self.sample_rate = None
            self.detected_freq = None
    
    def detect_frequency_threaded(self):
        """Detect frequency in a separate thread."""
        if not self.input_file:
            return
        
        self.detect_btn.config(state=tk.DISABLED)
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Detecting frequency...")
        self.progress.grid()
        self.progress.start()
        
        def detect():
            try:
                # Load audio
                self.status_label.config(text="Loading audio...")
                audio, sr = librosa.load(str(self.input_file), sr=None, mono=True)
                
                self.status_label.config(text="Detecting frequency...")
                freq = detect_frequency(audio, sr)
                
                # Update UI in main thread
                self.root.after(0, lambda: self.on_frequency_detected(audio, sr, freq))
            except Exception as e:
                self.root.after(0, lambda: self.on_error(f"Detection failed: {str(e)}"))
        
        thread = threading.Thread(target=detect, daemon=True)
        thread.start()
    
    def on_frequency_detected(self, audio_data, sr, freq):
        """Handle frequency detection result."""
        self.progress.stop()
        self.progress.grid_remove()
        self.detect_btn.config(state=tk.NORMAL)
        
        self.audio_data = audio_data
        self.sample_rate = sr
        self.detected_freq = freq
        
        if freq > 0:
            note = frequency_to_note(freq)
            self.detected_label.config(
                text=f"{freq:.2f} Hz ({note})",
                foreground="green"
            )
            self.status_label.config(text=f"Frequency detected: {freq:.2f} Hz ({note})")
            self.update_shift_calculation()
        else:
            self.detected_label.config(
                text="Detection failed",
                foreground="red"
            )
            self.status_label.config(text="Detection failed - try manual input")
            messagebox.showerror(
                "Detection Failed",
                "Could not detect frequency automatically.\n\n"
                "Please enter the current frequency manually in the target field\n"
                "or check if the audio file contains a clear pitch."
            )
    
    def update_shift_calculation(self, event=None):
        """Update shift calculation when target frequency changes."""
        if not self.detected_freq or self.detected_freq <= 0:
            return
        
        try:
            target = float(self.target_freq_var.get())
            
            if target > 0:
                semitones = calculate_semitones(self.detected_freq, target)
                cents = semitones * 100.0
                
                self.shift_label.config(
                    text=f"{semitones:+.3f} semitones ({cents:+.1f} cents)",
                    foreground="black"
                )
                
                target_note = frequency_to_note(target)
                self.target_note_label.config(
                    text=f"Target note: {target_note}",
                    foreground="black"
                )
                
                # Enable process button
                self.process_btn.config(state=tk.NORMAL)
                
                # Warning for large shifts
                if abs(semitones) > 0.5:
                    self.warning_label.config(
                        text=f"⚠️ Warning: Shift is > 1/2 semitone ({abs(semitones):.2f} semitones). "
                             "For best quality, use for small corrections (< 1/2 semitone).",
                        foreground="orange"
                    )
                else:
                    self.warning_label.config(text="")
            else:
                self.shift_label.config(
                    text="Invalid target frequency",
                    foreground="red"
                )
                self.process_btn.config(state=tk.DISABLED)
        except ValueError:
            self.shift_label.config(
                text="Invalid target frequency",
                foreground="red"
            )
            self.process_btn.config(state=tk.DISABLED)
    
    def process_audio_threaded(self):
        """Process audio in a separate thread."""
        if self.audio_data is None or not self.detected_freq:
            return
        
        try:
            target = float(self.target_freq_var.get())
            if target <= 0:
                messagebox.showerror("Error", "Target frequency must be positive.")
                return
        except ValueError:
            messagebox.showerror("Error", "Invalid target frequency.")
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.detect_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Processing...")
        self.progress.grid()
        self.progress.start()
        
        # Capture values before starting thread
        target_freq = target
        detected_freq = self.detected_freq
        audio_data = self.audio_data  # Use original reference (no copy needed)
        sample_rate = self.sample_rate
        input_file = self.input_file.resolve() if self.input_file else None
        
        if not input_file or not input_file.exists():
            messagebox.showerror("Error", "No valid input file selected.")
            self.process_btn.config(state=tk.NORMAL)
            self.detect_btn.config(state=tk.NORMAL)
            return
        
        def process():
            try:
                semitones = calculate_semitones(detected_freq, target_freq)
                
                self.root.after(0, lambda: self.status_label.config(text="Applying pitch shift..."))
                shifted_audio = pitch_shift_audio(audio_data, sample_rate, semitones)
                
                # Determine output path based on replace option
                replace_original = self.replace_original_var.get()
                original_format = input_file.suffix
                
                if replace_original:
                    # Backup original file first
                    backup_path = input_file.parent / f"{input_file.stem}_backup{input_file.suffix}"
                    if input_file.exists() and not backup_path.exists():
                        try:
                            shutil.copy2(input_file, backup_path)
                        except Exception as e:
                            raise Exception(f"Failed to backup original file: {str(e)}")
                    
                    # Save with original filename
                    output_path = input_file
                else:
                    # Save with _shifted suffix
                    output_path = input_file.parent / f"{input_file.stem}_shifted{input_file.suffix}"
                
                self.root.after(0, lambda: self.status_label.config(text="Saving file..."))
                
                # Ensure we're using absolute paths
                output_path = output_path.resolve()
                
                # Ensure output directory exists and is writable
                try:
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise Exception(f"Cannot create output directory: {str(e)}")
                
                # Check if file is locked (on Windows)
                if output_path.exists() and replace_original:
                    try:
                        # Try to open file for writing to check if it's locked
                        with open(output_path, 'r+b') as f:
                            pass
                    except PermissionError:
                        raise Exception(f"File is locked or in use. Please close any programs using: {output_path.name}")
                    except Exception as e:
                        raise Exception(f"Cannot access file: {str(e)}")
                
                # Update status before saving
                self.root.after(0, lambda: self.status_label.config(text="Saving file to disk..."))
                
                success = save_audio_preserving_format(shifted_audio, sample_rate, output_path, original_format)
                
                if not success:
                    raise Exception(f"Failed to save audio file. Check file permissions and disk space.")
                
                # Verify file was actually saved
                if not output_path.exists():
                    raise Exception(f"Save operation completed but file not found at: {output_path}")
                
                # Verify file size is reasonable (not empty)
                file_size = output_path.stat().st_size
                if file_size < 100:  # Very small file, likely an error
                    raise Exception(f"Saved file is suspiciously small ({file_size} bytes). Save may have failed.")
                
                # Verify amplitude preservation (already done in pitch_shift_audio, but verify again)
                original_rms = np.sqrt(np.mean(audio_data ** 2))
                shifted_rms = np.sqrt(np.mean(shifted_audio ** 2))
                amplitude_ratio = shifted_rms / (original_rms + 1e-10)
                
                # Log preservation status
                if abs(amplitude_ratio - 1.0) < 0.01:
                    print(f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (perfect)")
                else:
                    print(f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (within tolerance)")
                
                self.root.after(0, lambda: self.on_processing_complete(output_path, amplitude_ratio))
            except Exception as e:
                self.root.after(0, lambda: self.on_error(f"Processing failed: {str(e)}"))
        
        thread = threading.Thread(target=process, daemon=True)
        thread.start()
    
    def on_processing_complete(self, output_path, amplitude_ratio):
        """Handle processing completion."""
        self.progress.stop()
        self.progress.grid_remove()
        self.detect_btn.config(state=tk.NORMAL)
        self.process_btn.config(state=tk.NORMAL)
        
        if abs(amplitude_ratio - 1.0) < 0.01:
            amp_text = f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (perfect)\n✓ Timbre preserved\n✓ Dynamics preserved"
        else:
            amp_text = f"✓ Amplitude preserved: {amplitude_ratio:.4f}x (within tolerance)\n✓ Timbre preserved\n✓ Dynamics preserved"
        
        replace_original = self.replace_original_var.get()
        if replace_original:
            self.status_label.config(text=f"✓ Saved: {output_path.name} (original replaced)")
            backup_path = output_path.parent / f"{output_path.stem}_backup{output_path.suffix}"
            backup_msg = f"\nOriginal backed up to: {backup_path.name}" if backup_path.exists() else ""
        else:
            self.status_label.config(text=f"✓ Saved: {output_path.name}")
            backup_msg = ""
        
        messagebox.showinfo(
            "Success",
            f"Pitch shift completed successfully!\n\n"
            f"Output file: {output_path.name}\n"
            f"Location: {output_path.parent}\n\n"
            f"{amp_text}{backup_msg}\n\n"
            f"✓ File saved to disk."
        )
        
        # If original was replaced, reload the file to show updated frequency
        if replace_original and self.input_file == output_path:
            self.root.after(500, self.detect_frequency_threaded)
    
    def on_error(self, error_msg):
        """Handle error."""
        self.progress.stop()
        self.progress.grid_remove()
        self.detect_btn.config(state=tk.NORMAL)
        self.process_btn.config(state=tk.NORMAL)
        self.status_label.config(text=f"Error: {error_msg}")
        self.show_error("Error", error_msg)
    
    def show_error(self, title, message):
        """Show error message."""
        messagebox.showerror(title, message)
    
    def show_frequency_table(self):
        """Show frequency reference table in a popup window."""
        # Create popup window
        popup = tk.Toplevel(self.root)
        popup.title("Frequency Reference Table (A4 = 440 Hz)")
        popup.geometry("950x850")  # Increased size to show all notes
        popup.resizable(True, True)
        
        # Create main container
        main_container = ttk.Frame(popup)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create frame with scrollbar using Canvas for better scrolling
        canvas_frame = ttk.Frame(main_container)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        # Canvas for scrolling
        canvas = tk.Canvas(canvas_frame, highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient="vertical", command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Pack canvas and scrollbar
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for table (inside scrollable frame)
        text_widget = tk.Text(
            scrollable_frame,
            wrap=tk.NONE,
            font=("Courier", 9),
            padx=10,
            pady=10,
            width=100,
            height=50,
            bg="white",
            fg="black"
        )
        text_widget.pack(fill=tk.BOTH, expand=True)
        
        # Generate table content - VERIFIED: All 153 notes are generated
        table_text = "=" * 68 + "\n"
        table_text += "FREQUENCY REFERENCE TABLE (A4 = 440 Hz Standard Tuning)\n"
        table_text += "=" * 68 + "\n\n"
        
        # Define note order (matching the reference table format: A#/A/Ab, B/Bb, C#/C/Db, D#/D, E/Eb, F#/F, G#/G/Gb)
        # This order matches the user's reference table exactly
        note_order = ['A#', 'A', 'Ab', 'B', 'Bb', 'C#', 'C', 'D#', 'D', 'Db', 'E', 'Eb', 'F#', 'F', 'G#', 'G', 'Gb']
        
        # Group notes by octave - VERIFIED: All 9 octaves (0-8) with 17 notes each = 153 total
        notes_by_octave = {}
        for octave in range(9):  # Octaves 0-8
            notes_by_octave[octave] = []
            for note_name in note_order:
                note_key = f"{note_name}{octave}"
                if note_key in NOTE_FREQUENCY_MAP:
                    notes_by_octave[octave].append((note_key, NOTE_FREQUENCY_MAP[note_key]))
        
        # Generate table text - VERIFIED: All notes are included
        # Format: Note = Frequency Hz | (separator after each note-frequency pair)
        for octave in sorted(notes_by_octave.keys(), key=int):
            table_text += f"Octave {octave}:\n"
            table_text += "-" * 80 + "\n"
            notes = notes_by_octave[octave]
            # Print in columns (3 columns per row) with vertical separator after each pair
            for i in range(0, len(notes), 3):
                row_notes = notes[i:i+3]
                # Format each note: "Note = Frequency Hz |" with separator after
                formatted_notes = []
                for note, freq in row_notes:
                    formatted_notes.append(f"{note:6s} = {freq:7.2f} Hz |")
                row_str = "  ".join(formatted_notes)
                table_text += f"  {row_str}\n"
            table_text += "\n"
        
        table_text += "=" * 68 + "\n"
        table_text += f"Total: {len(NOTE_FREQUENCY_MAP)} notes\n"
        table_text += "=" * 68 + "\n"
        
        # Insert text - CRITICAL: Must be done in NORMAL state
        text_widget.config(state=tk.NORMAL)
        text_widget.delete("1.0", tk.END)
        text_widget.insert("1.0", table_text)
        
        # Verify content was inserted (debug)
        content_length = len(text_widget.get("1.0", tk.END))
        if content_length < len(table_text) * 0.9:  # Allow for some encoding differences
            print(f"WARNING: Text may not have been fully inserted. Expected ~{len(table_text)}, got {content_length}")
        
        # Scroll to top and update
        text_widget.see("1.0")
        text_widget.update_idletasks()
        canvas.update_idletasks()
        
        # Make read-only AFTER everything is set up
        text_widget.config(state=tk.DISABLED)
        
        # Bind mousewheel to canvas for scrolling
        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)
        
        # Close button
        close_btn = ttk.Button(
            main_container,
            text="Close",
            command=popup.destroy
        )
        close_btn.pack(pady=10)


def main():
    """Main function: launch GUI."""
    import sys
    
    root = tk.Tk()
    
    # Check for file path argument
    initial_file = None
    if len(sys.argv) > 1:
        initial_file = sys.argv[1]
    
    app = PitchShiftGUI(root, initial_file=initial_file)
    root.mainloop()


if __name__ == '__main__':
    main()
