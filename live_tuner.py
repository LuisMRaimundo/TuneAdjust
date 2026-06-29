#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Live Tuner - Real-Time Audio Tuner
==================================

A real-time tuner app that:
- Plays audio file and analyzes in real-time (internal audio, no microphone)
- Detects frequency in real-time
- Shows visual feedback with colors (green = in tune, red/yellow = out of tune)
- Displays detected note and frequency
- Shows tuning meter/needle

Author: SoundSpectrAnalyse Team
Version: 1.0
Date: January 2026
"""

import sys
import threading
import queue
import warnings
from pathlib import Path
from typing import Optional, Tuple
import time

import numpy as np
import tkinter as tk
from tkinter import ttk, messagebox, filedialog

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

# Audio playback (for internal audio)
try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False
    print("WARNING: sounddevice not available. Install with: pip install sounddevice")

warnings.filterwarnings('ignore', category=UserWarning, module='librosa')


import pitch_core as pc
from pitch_core import (
    NOTE_FREQUENCY_MAP,
    note_to_frequency,
    frequency_to_note,
    cents_difference,
    detect_frequency,
    PitchSmoother,
)

def detect_frequency_fast(audio_data, sr):
    return detect_frequency(audio_data, sr, fast=True, apply_harmonic_check=False)

# ============================================================================
# Live Tuner GUI
# ============================================================================

class LiveTunerGUI(tk.Tk):
    """Real-time tuner application with internal audio playback."""
    
    def __init__(self):
        super().__init__()
        self.title("Live Tuner - Real-Time Audio Tuner (Internal Audio)")
        self.geometry("600x750")
        
        # Audio settings
        self.sample_rate = 44100
        self.chunk_size = 4096  # Audio buffer size
        self.tolerance = 20.0  # Cents tolerance
        
        # State
        self.is_playing = False
        self.audio_file = None
        self.audio_data = None
        self.audio_sr = None
        self.audio_thread = None
        self.audio_queue = queue.Queue()
        self.current_freq = 0.0
        self.current_note = "Unknown"
        self.current_cents = 0.0
        self.playback_position = 0
        self._pitch_smoother = PitchSmoother(window=9)
        
        # Target note (optional - if set, shows tuning relative to that note)
        self.target_note = None
        self.target_freq = 0.0
        
        self._build_ui()
    
    def _build_ui(self):
        """Build the user interface."""
        # Main frame
        main_frame = ttk.Frame(self, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame,
            text="Live Tuner (Internal Audio)",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=10)
        
        # File selection
        file_frame = ttk.Frame(main_frame)
        file_frame.pack(pady=10, fill=tk.X)
        
        ttk.Label(file_frame, text="Audio File:").pack(side=tk.LEFT, padx=5)
        self.file_label = ttk.Label(
            file_frame,
            text="No file selected",
            foreground="gray"
        )
        self.file_label.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(
            file_frame,
            text="Browse...",
            command=self._browse_file
        ).pack(side=tk.LEFT, padx=5)
        
        # Playback controls
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(pady=10)
        
        self.play_btn = ttk.Button(
            control_frame,
            text="▶ Play",
            command=self._toggle_playback,
            state=tk.DISABLED
        )
        self.play_btn.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="⏹ Stop",
            command=self._stop_playback
        ).pack(side=tk.LEFT, padx=5)
        
        ttk.Button(
            control_frame,
            text="⏮ Restart",
            command=self._restart_playback
        ).pack(side=tk.LEFT, padx=5)
        
        # Tolerance setting
        tolerance_frame = ttk.Frame(main_frame)
        tolerance_frame.pack(pady=10)
        
        ttk.Label(tolerance_frame, text="Tolerance (cents):").pack(side=tk.LEFT, padx=5)
        self.tolerance_var = tk.StringVar(value="15")
        tolerance_entry = ttk.Entry(tolerance_frame, textvariable=self.tolerance_var, width=10)
        tolerance_entry.pack(side=tk.LEFT, padx=5)
        tolerance_entry.bind('<KeyRelease>', self._update_tolerance)
        
        # Target note (optional)
        target_frame = ttk.Frame(main_frame)
        target_frame.pack(pady=5)
        
        ttk.Label(target_frame, text="Target note (optional):").pack(side=tk.LEFT, padx=5)
        self.target_note_var = tk.StringVar(value="")
        target_entry = ttk.Entry(target_frame, textvariable=self.target_note_var, width=10)
        target_entry.pack(side=tk.LEFT, padx=5)
        target_entry.bind('<KeyRelease>', self._update_target_note)
        
        # Frequency display
        freq_frame = ttk.LabelFrame(main_frame, text="Detected Frequency", padding="10")
        freq_frame.pack(pady=20, fill=tk.X)
        
        self.freq_label = ttk.Label(
            freq_frame,
            text="-- Hz",
            font=("Arial", 32, "bold")
        )
        self.freq_label.pack()
        
        # Note display
        note_frame = ttk.LabelFrame(main_frame, text="Detected Note", padding="10")
        note_frame.pack(pady=10, fill=tk.X)
        
        self.note_label = ttk.Label(
            note_frame,
            text="--",
            font=("Arial", 48, "bold")
        )
        self.note_label.pack()
        
        # Cents display
        self.cents_label = ttk.Label(
            note_frame,
            text="-- cents",
            font=("Arial", 16)
        )
        self.cents_label.pack()
        
        # Tuning meter (visual indicator)
        meter_frame = ttk.LabelFrame(main_frame, text="Tuning Meter", padding="10")
        meter_frame.pack(pady=20, fill=tk.BOTH, expand=True)
        
        # Canvas for tuning meter
        self.meter_canvas = tk.Canvas(meter_frame, width=500, height=200, bg="white")
        self.meter_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Status
        self.status_label = ttk.Label(
            main_frame,
            text="Select an audio file to begin",
            font=("Arial", 12)
        )
        self.status_label.pack(pady=10)
        
        # Start processing audio queue
        self._process_audio_queue()
    
    def _browse_file(self):
        """Browse for audio file."""
        filename = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio files", "*.wav *.mp3 *.flac *.ogg *.aif *.aiff *.m4a *.wma"),
                ("All files", "*.*")
            ]
        )
        
        if filename:
            self.audio_file = Path(filename)
            self.file_label.config(text=self.audio_file.name, foreground="black")
            self._load_audio_file()
    
    def _load_audio_file(self):
        """Load audio file for playback and analysis."""
        try:
            self.status_label.config(text="Loading audio file...", foreground="black")
            self.update()
            
            # Load audio
            self.audio_data, self.audio_sr = librosa.load(
                str(self.audio_file),
                sr=None,
                mono=True
            )
            
            self.sample_rate = self.audio_sr
            self.playback_position = 0
            self.play_btn.config(state=tk.NORMAL)
            self.status_label.config(text="Ready to play", foreground="green")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load audio file:\n{e}")
            self.status_label.config(text="Error loading file", foreground="red")
    
    def _toggle_playback(self):
        """Toggle playback."""
        if not self.is_playing:
            self._start_playback()
        else:
            self._pause_playback()
    
    def _start_playback(self):
        """Start playback and analysis."""
        if self.audio_data is None:
            return
        
        self.is_playing = True
        self.play_btn.config(text="⏸ Pause")
        self.status_label.config(text="Playing and analyzing...", foreground="green")
        
        # Start playback thread
        if SOUNDDEVICE_AVAILABLE:
            self._start_sounddevice_playback()
        else:
            messagebox.showwarning(
                "Warning",
                "sounddevice not available. Install with: pip install sounddevice\n\n"
                "Audio will be analyzed but not played."
            )
            self._start_analysis_only()
    
    def _start_sounddevice_playback(self):
        """Start playback using sounddevice."""
        def playback_thread():
            try:
                position = self.playback_position
                total_samples = len(self.audio_data)
                
                while self.is_playing and position < total_samples:
                    # Get chunk
                    end_pos = min(position + self.chunk_size, total_samples)
                    chunk = self.audio_data[position:end_pos]
                    
                    if len(chunk) < self.chunk_size:
                        # Pad if needed
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')
                    
                    # Play chunk
                    sd.play(chunk, samplerate=self.sample_rate, blocking=True)
                    
                    # Queue for analysis
                    self.audio_queue.put(chunk.copy())
                    
                    position = end_pos
                    self.playback_position = position
                    
                    if position >= total_samples:
                        # End of file
                        self.is_playing = False
                        self.play_btn.config(text="▶ Play")
                        self.status_label.config(text="Playback complete", foreground="blue")
                        break
                
            except Exception as e:
                self.status_label.config(text=f"Playback error: {e}", foreground="red")
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
        
        self.audio_thread = threading.Thread(target=playback_thread, daemon=True)
        self.audio_thread.start()
    
    def _start_analysis_only(self):
        """Start analysis without playback (fallback)."""
        def analysis_thread():
            try:
                position = self.playback_position
                total_samples = len(self.audio_data)
                
                while self.is_playing and position < total_samples:
                    # Get chunk
                    end_pos = min(position + self.chunk_size, total_samples)
                    chunk = self.audio_data[position:end_pos]
                    
                    if len(chunk) < self.chunk_size:
                        chunk = np.pad(chunk, (0, self.chunk_size - len(chunk)), mode='constant')
                    
                    # Queue for analysis
                    self.audio_queue.put(chunk.copy())
                    
                    position = end_pos
                    self.playback_position = position
                    
                    # Simulate playback timing
                    time.sleep(self.chunk_size / self.sample_rate)
                    
                    if position >= total_samples:
                        self.is_playing = False
                        self.play_btn.config(text="▶ Play")
                        self.status_label.config(text="Analysis complete", foreground="blue")
                        break
                
            except Exception as e:
                self.status_label.config(text=f"Analysis error: {e}", foreground="red")
                self.is_playing = False
                self.play_btn.config(text="▶ Play")
        
        self.audio_thread = threading.Thread(target=analysis_thread, daemon=True)
        self.audio_thread.start()
    
    def _pause_playback(self):
        """Pause playback."""
        self.is_playing = False
        self.play_btn.config(text="▶ Play")
        self.status_label.config(text="Paused", foreground="orange")
    
    def _stop_playback(self):
        """Stop playback and reset."""
        self.is_playing = False
        self.playback_position = 0
        self.play_btn.config(text="▶ Play")
        self.status_label.config(text="Stopped", foreground="gray")
        
        # Clear current readings
        self.current_freq = 0.0
        self.current_note = "Unknown"
        self.current_cents = 0.0
        self._pitch_smoother.reset()
        self._update_display()
    
    def _restart_playback(self):
        """Restart playback from beginning."""
        self._stop_playback()
        if self.audio_data is not None:
            self._start_playback()
    
    def _update_tolerance(self, event=None):
        """Update tolerance value."""
        try:
            self.tolerance = float(self.tolerance_var.get())
        except ValueError:
            pass
    
    def _update_target_note(self, event=None):
        """Update target note."""
        target_str = self.target_note_var.get().strip()
        if target_str:
            self.target_note = target_str
            self.target_freq = note_to_frequency(target_str)
        else:
            self.target_note = None
            self.target_freq = 0.0
    
    
    def _process_audio_queue(self):
        """Process audio from queue and update display."""
        try:
            while True:
                audio_data = self.audio_queue.get_nowait()
                
                # Detect frequency
                raw_freq = detect_frequency_fast(audio_data, self.sample_rate)
                freq = self._pitch_smoother.update(raw_freq)
                
                if freq > 0:
                    self.current_freq = freq
                    self.current_note = frequency_to_note(freq)
                    
                    # Calculate cents if target note is set
                    if self.target_note and self.target_freq > 0:
                        self.current_cents = cents_difference(freq, self.target_freq)
                    else:
                        # Calculate cents from nearest note
                        parsed = self._parse_note(self.current_note)
                        if parsed:
                            note_name, octave = parsed
                            note_key = f"{note_name}{octave}"
                            expected_freq = note_to_frequency(note_key)
                            if expected_freq > 0:
                                self.current_cents = cents_difference(freq, expected_freq)
                            else:
                                self.current_cents = 0.0
                        else:
                            self.current_cents = 0.0
                    
                    # Update display
                    self._update_display()
        except queue.Empty:
            pass
        
        # Schedule next update
        self.after(50, self._process_audio_queue)
    
    def _parse_note(self, note_str: str) -> Optional[Tuple[str, int]]:
        """Parse note string into (note_name, octave)."""
        import re
        pattern = re.compile(r'([A-G])([#b]?)(-?\d+)', re.IGNORECASE)
        match = pattern.search(note_str)
        if match:
            letter = match.group(1).upper()
            accidental = match.group(2).replace('♯', '#').replace('♭', 'b')
            octave = int(match.group(3))
            note_name = letter + accidental
            if note_name in FLAT_EQUIVALENTS:
                note_name = FLAT_EQUIVALENTS[note_name]
            if note_name in CHROMATIC_NOTES:
                return (note_name, octave)
        return None
    
    def _update_display(self):
        """Update the display with current frequency and tuning status."""
        # Update frequency
        self.freq_label.config(text=f"{self.current_freq:.2f} Hz")
        
        # Update note
        self.note_label.config(text=self.current_note)
        
        # Update cents
        if self.current_cents != float('inf'):
            cents_text = f"{self.current_cents:+.1f} cents"
            self.cents_label.config(text=cents_text)
        else:
            self.cents_label.config(text="-- cents")
        
        # Determine tuning status and color
        if self.current_cents <= self.tolerance:
            # In tune - GREEN
            color = "#4CAF50"  # Green
            status = "In Tune"
        elif self.current_cents <= self.tolerance * 2:
            # Slightly out - YELLOW
            color = "#FFC107"  # Yellow/Orange
            status = "Slightly Out"
        else:
            # Out of tune - RED
            color = "#F44336"  # Red
            status = "Out of Tune"
        
        # Update note label color
        self.note_label.config(foreground=color)
        self.freq_label.config(foreground=color)
        
        # Update status
        self.status_label.config(text=status, foreground=color)
        
        # Update tuning meter
        self._update_tuning_meter()
    
    def _update_tuning_meter(self):
        """Update the visual tuning meter."""
        self.meter_canvas.delete("all")
        
        width = self.meter_canvas.winfo_width()
        height = self.meter_canvas.winfo_height()
        
        if width < 10 or height < 10:
            return
        
        center_x = width // 2
        center_y = height // 2
        
        # Draw background
        self.meter_canvas.create_rectangle(0, 0, width, height, fill="#f0f0f0", outline="")
        
        # Draw center line (perfect tuning)
        self.meter_canvas.create_line(
            center_x, 0, center_x, height,
            fill="black", width=2
        )
        
        # Draw tolerance zones
        tolerance_pixels = (self.tolerance / 50.0) * (width / 2)  # Scale to screen
        
        # Green zone (in tune)
        self.meter_canvas.create_rectangle(
            center_x - tolerance_pixels, 0,
            center_x + tolerance_pixels, height,
            fill="#4CAF50", outline="", stipple="gray25"
        )
        
        # Yellow zone (slightly out)
        yellow_zone = tolerance_pixels * 2
        self.meter_canvas.create_rectangle(
            center_x - yellow_zone, 0,
            center_x - tolerance_pixels, height,
            fill="#FFC107", outline="", stipple="gray25"
        )
        self.meter_canvas.create_rectangle(
            center_x + tolerance_pixels, 0,
            center_x + yellow_zone, height,
            fill="#FFC107", outline="", stipple="gray25"
        )
        
        # Draw needle (current tuning position)
        if self.current_cents != float('inf') and self.current_freq > 0:
            # Calculate needle position (-50 to +50 cents mapped to screen width)
            max_cents = 50.0
            cents_clamped = max(-max_cents, min(max_cents, self.current_cents))
            needle_pos = center_x + (cents_clamped / max_cents) * (width / 2)
            
            # Draw needle
            needle_color = "#4CAF50" if abs(cents_clamped) <= self.tolerance else "#F44336"
            self.meter_canvas.create_line(
                needle_pos, 10,
                needle_pos, height - 10,
                fill=needle_color, width=4
            )
            
            # Draw needle tip (triangle)
            tip_size = 10
            self.meter_canvas.create_polygon(
                needle_pos, 10,
                needle_pos - tip_size, 10 + tip_size * 2,
                needle_pos + tip_size, 10 + tip_size * 2,
                fill=needle_color, outline=""
            )
        
        # Draw scale markers
        for cents in [-50, -25, -10, 0, 10, 25, 50]:
            pos = center_x + (cents / 50.0) * (width / 2)
            if 0 <= pos <= width:
                marker_height = 15 if cents == 0 else 10
                self.meter_canvas.create_line(
                    pos, height - marker_height,
                    pos, height,
                    fill="black", width=1
                )
                # Label
                if cents != 0:
                    self.meter_canvas.create_text(
                        pos, height - marker_height - 5,
                        text=f"{cents:+d}",
                        font=("Arial", 8)
                    )
        
        # Draw center label
        self.meter_canvas.create_text(
            center_x, height - 20,
            text="0",
            font=("Arial", 10, "bold")
        )
    
    def on_closing(self):
        """Handle window closing."""
        self.is_playing = False
        if SOUNDDEVICE_AVAILABLE and hasattr(self, 'stream'):
            try:
                sd.stop()
            except:
                pass
        self.destroy()


def main():
    """Main function."""
    if not LIBROSA_AVAILABLE:
        messagebox.showerror(
            "Missing Dependencies",
            "librosa is required.\n\nInstall with: pip install librosa"
        )
        return
    
    if not SOUNDFILE_AVAILABLE:
        messagebox.showerror(
            "Missing Dependencies",
            "soundfile is required.\n\nInstall with: pip install soundfile"
        )
        return
    
    app = LiveTunerGUI()
    app.protocol("WM_DELETE_WINDOW", app.on_closing)
    app.mainloop()


if __name__ == '__main__':
    main()
