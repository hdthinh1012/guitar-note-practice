#!/usr/bin/env python3
"""
Real-time Guitar Note Detector
===============================
Listens to your microphone, detects the pitch using a YIN algorithm
(implemented in pure numpy), and displays which musical note is being played.

Dependencies: numpy, sounddevice
Usage:        python guitar_note_detector.py

Author: Generated with GitHub Copilot
"""

import sys
import numpy as np
import sounddevice as sd

# ─── Audio Settings ──────────────────────────────────────────────────────────
SAMPLE_RATE = 44100       # Hz
BUFFER_SIZE = 4096         # samples per frame (~93ms at 44100 Hz)
CHANNELS = 1               # mono

# ─── YIN Pitch Detection Settings ────────────────────────────────────────────
YIN_THRESHOLD = 0.15       # lower = stricter pitch detection
MIN_FREQ = 60.0            # Hz – lowest guitar note is E2 ≈ 82 Hz (some margin)
MAX_FREQ = 1200.0          # Hz – highest practical guitar note

# ─── Display Settings ────────────────────────────────────────────────────────
CENTS_TOLERANCE = 10       # cents within which we consider "in tune"
SILENCE_THRESHOLD = 0.01   # RMS below this = silence
PITCH_BUFFER_SIZE = 5      # median smoothing window

# ─── Note Names ──────────────────────────────────────────────────────────────
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Standard guitar tuning reference
GUITAR_STRINGS = {
    "6th (E2)": 82.41,
    "5th (A2)": 110.00,
    "4th (D3)": 146.83,
    "3rd (G3)": 196.00,
    "2nd (B3)": 246.94,
    "1st (E4)": 329.63,
}


# ─── Terminal Colors ─────────────────────────────────────────────────────────
RST = "\033[0m"
DIM = "\033[2m"

# ─── ASCII Music Staff Renderer (G-clef 8vb) ────────────────────────────────
#
# Diatonic note order within an octave (C-based):
#   C=0  D=1  E=2  F=3  G=4  A=5  B=6
# Treble clef lines (bottom→top): E4 G4 B4 D5 F5  (positions 0,2,4,6,8)
# Guitar notation is written one octave higher than sounding pitch.

_DIATONIC = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}


def _parse_note(note_str: str):
    """Parse 'C#4' → ('C', True, 4) or 'A2' → ('A', False, 2)."""
    if "#" in note_str:
        return note_str[0], True, int(note_str[2:])
    else:
        return note_str[0], False, int(note_str[1:])


def _diatonic_from_c0(letter: str, octave: int) -> int:
    """Absolute diatonic position from C0.  C0=0, D0=1 … B0=6, C1=7 …"""
    return octave * 7 + _DIATONIC[letter]


_TREBLE_BOTTOM = _diatonic_from_c0("E", 4)  # bottom line of treble staff


def _staff_position(letter: str, octave: int, clef_bottom: int) -> int:
    """Return the staff position (0 = bottom line). Even→line, Odd→space."""
    return _diatonic_from_c0(letter, octave) - clef_bottom


def _render_staff(note_str: str | None, clef_bottom: int, clef_label: str,
                  width: int = 37) -> list[str]:
    """Render a single 5-line staff with an optional note placed on it."""
    POS_MIN = -8
    POS_MAX = 16
    STAFF_LINES = {0, 2, 4, 6, 8}

    pos = None
    is_sharp = False
    if note_str:
        letter, is_sharp, octave = _parse_note(note_str)
        pos = _staff_position(letter, octave, clef_bottom)

    rows = []
    COL_NOTE = 20
    COL_SHARP = 17
    LEDGER_L = COL_NOTE - 2
    LEDGER_R = COL_NOTE + 2

    for p in range(POS_MAX, POS_MIN - 1, -1):
        is_line = p in STAFF_LINES
        on_note = (pos is not None and p == pos)

        if is_line:
            left = f"  {clef_label:>12s} " if p == 4 else "               "
            bar = list("─" * width)
        else:
            left = "               "
            bar = list(" " * width)

        if on_note:
            if is_line:
                bar[COL_NOTE - 1] = " "
                bar[COL_NOTE] = "●"
                bar[COL_NOTE + 1] = " "
            else:
                bar[COL_NOTE] = "●"
            if is_sharp:
                bar[COL_SHARP] = "♯"
            if p % 2 == 0 and p not in STAFF_LINES:
                for c in range(LEDGER_L, LEDGER_R + 1):
                    if 0 <= c < width and c != COL_NOTE:
                        bar[c] = "─"
        else:
            if pos is not None and p % 2 == 0 and p not in STAFF_LINES:
                need_ledger = False
                if p < 0 and pos <= p:
                    need_ledger = True
                if p > 8 and pos >= p:
                    need_ledger = True
                if need_ledger:
                    for c in range(LEDGER_L, LEDGER_R + 1):
                        if 0 <= c < width:
                            bar[c] = "─"

        rows.append(left + "".join(bar))
    return rows


def render_guitar_staff(note_str: str | None) -> list[str]:
    """Render guitar staff (G-clef 8vb). Written pitch = sounding + 1 octave."""
    if note_str:
        letter, is_sharp, octave = _parse_note(note_str)
        written_octave = octave + 1
        written_note = f"{letter}{'#' if is_sharp else ''}{written_octave}"
    else:
        written_note = None

    header = [f"  {DIM}┌─ Guitar staff (G-clef 8vb) ──────────────────────────┐{RST}"]
    staff = _render_staff(written_note, _TREBLE_BOTTOM, "𝄞₈ᵥᵦ")
    footer = [f"  {DIM}└──────────────────────────────────────────────────────┘{RST}"]
    return header + [f"  {DIM}│{RST}{r}" for r in staff] + footer


# Total lines the staff display occupies (header + 25 rows + footer)
STAFF_DISPLAY_LINES = 27
# Total display lines: 2 (note info + tuner) + 1 (blank) + 27 (staff) = 30
TOTAL_DISPLAY_LINES = 30


# ─── YIN Algorithm (Pure Numpy) ─────────────────────────────────────────────

def yin_pitch(signal: np.ndarray, sample_rate: int, threshold: float = YIN_THRESHOLD) -> float | None:
    """
    Estimate the fundamental frequency of a signal using the YIN algorithm.

    Returns the detected frequency in Hz, or None if no pitch is detected.

    Reference: De Cheveigné, A., & Kawahara, H. (2002).
    "YIN, a fundamental frequency estimator for speech and music."
    """
    # Work with float64
    signal = signal.astype(np.float64)
    n = len(signal)
    tau_max = n // 2

    # Min and max period based on frequency range
    tau_min = max(2, int(sample_rate / MAX_FREQ))
    tau_max_search = min(tau_max, int(sample_rate / MIN_FREQ))

    if tau_max_search <= tau_min:
        return None

    # Step 1 & 2: Difference function using autocorrelation trick
    # d(tau) = sum_{j=0}^{W-1} (x[j] - x[j+tau])^2
    # This can be computed efficiently using:
    # d(tau) = r(0) + r_shifted(0) - 2*r(tau)
    # where r is autocorrelation

    # Compute using the cumulative sum approach for efficiency
    w = tau_max
    x = signal[:w]

    # Difference function
    d = np.zeros(tau_max_search)
    for tau in range(1, tau_max_search):
        diff = signal[:w] - signal[tau:tau + w]
        d[tau] = np.sum(diff ** 2)

    # Step 3: Cumulative mean normalized difference function (CMND)
    d[0] = 1.0
    cumsum = 0.0
    cmnd = np.zeros(tau_max_search)
    cmnd[0] = 1.0
    for tau in range(1, tau_max_search):
        cumsum += d[tau]
        if cumsum == 0:
            cmnd[tau] = 1.0
        else:
            cmnd[tau] = d[tau] * tau / cumsum

    # Step 4: Absolute threshold – find the first tau where CMND < threshold
    tau_estimate = None
    for tau in range(tau_min, tau_max_search):
        if cmnd[tau] < threshold:
            # Find the local minimum following this point
            while tau + 1 < tau_max_search and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_estimate = tau
            break

    if tau_estimate is None:
        return None

    # Step 5: Parabolic interpolation for sub-sample accuracy
    if 0 < tau_estimate < tau_max_search - 1:
        s0 = cmnd[tau_estimate - 1]
        s1 = cmnd[tau_estimate]
        s2 = cmnd[tau_estimate + 1]
        denom = 2.0 * s1 - s2 - s0
        if denom != 0:
            tau_estimate = tau_estimate + (s2 - s0) / (2.0 * denom)

    frequency = sample_rate / tau_estimate
    return frequency


# ─── Note Mapping ────────────────────────────────────────────────────────────

def freq_to_note(freq: float) -> tuple[str, int, float]:
    """
    Convert a frequency to the nearest musical note.

    Returns: (note_name, octave, cents_off)
        - note_name: e.g. "A", "C#"
        - octave: e.g. 4 for A4=440Hz
        - cents_off: how many cents sharp (+) or flat (-) from the note
    """
    if freq <= 0:
        return ("?", 0, 0.0)

    # Number of semitones from A4 (440 Hz)
    semitones = 12.0 * np.log2(freq / 440.0)
    nearest_semitone = round(semitones)
    cents_off = (semitones - nearest_semitone) * 100.0

    # Convert to note name and octave
    # A4 is MIDI note 69, C4 is MIDI note 60
    midi_note = 69 + nearest_semitone
    note_index = midi_note % 12
    octave = (midi_note // 12) - 1

    return (NOTE_NAMES[note_index], octave, cents_off)


def find_nearest_string(freq: float) -> str | None:
    """Find the nearest standard guitar string to the detected frequency."""
    if freq <= 0:
        return None

    nearest = None
    min_cents = float("inf")

    for name, string_freq in GUITAR_STRINGS.items():
        cents = abs(1200.0 * np.log2(freq / string_freq))
        if cents < min_cents:
            min_cents = cents
            nearest = name

    if min_cents < 200:  # within ~2 semitones
        return nearest
    return None


# ─── Display ─────────────────────────────────────────────────────────────────

def build_tuner_bar(cents: float, width: int = 40) -> str:
    """
    Build an ASCII tuner bar showing how sharp/flat the note is.

    ◄ FLAT ──────────┃──────────── SHARP ►
    """
    half = width // 2
    # Clamp cents to ±50 for display
    clamped = max(-50, min(50, cents))
    pos = int((clamped / 50.0) * half)

    bar = [" "] * width
    bar[half] = "┃"  # center marker

    # Place the indicator
    indicator_pos = half + pos
    indicator_pos = max(0, min(width - 1, indicator_pos))

    if abs(cents) <= CENTS_TOLERANCE:
        bar[indicator_pos] = "●"  # in tune!
        status = "\033[92m✓ IN TUNE\033[0m"
    elif cents < 0:
        bar[indicator_pos] = "◄"
        status = f"\033[93m♭ FLAT  {cents:+.1f}¢\033[0m"
    else:
        bar[indicator_pos] = "►"
        status = f"\033[93m♯ SHARP {cents:+.1f}¢\033[0m"

    # Add scale marks
    for i in [half - half // 2, half + half // 2]:
        if bar[i] == " ":
            bar[i] = "·"

    bar_str = "".join(bar)
    return f"[{bar_str}]  {status}"


def display_detection(freq: float, note: str, octave: int, cents: float, rms: float):
    """Display the current detection result with guitar staff notation."""
    nearest_string = find_nearest_string(freq)
    string_info = f"  (nearest: {nearest_string})" if nearest_string else ""

    tuner_bar = build_tuner_bar(cents)
    note_str = f"{note}{octave}"

    # Build the display lines
    lines = [
        f"\033[2K  Note: \033[1;97m{note_str}\033[0m   "
        f"Freq: \033[96m{freq:.1f} Hz\033[0m   "
        f"Volume: {'█' * min(20, int(rms * 200))}{'░' * (20 - min(20, int(rms * 200)))}",
        f"\033[2K  {tuner_bar}{string_info}",
        f"\033[2K",
    ]

    # Add the guitar staff
    staff_lines = render_guitar_staff(note_str)
    for sl in staff_lines:
        lines.append(f"\033[2K{sl}")

    # Move cursor up and overwrite
    sys.stdout.write(f"\033[{TOTAL_DISPLAY_LINES}A")
    for line in lines:
        sys.stdout.write(f"\r{line}\n")
    sys.stdout.flush()


# ─── Main Loop ───────────────────────────────────────────────────────────────

def main():
    print("\033[2J\033[H")  # clear screen
    print("╔══════════════════════════════════════════════════════════╗")
    print("║          🎸  Guitar Note Detector  🎸                   ║")
    print("╠══════════════════════════════════════════════════════════╣")
    print("║  Play a note on your guitar near the microphone.        ║")
    print("║  The detector will show which note is being played.     ║")
    print("║                                                         ║")
    print("║  Standard tuning: E2 A2 D3 G3 B3 E4                    ║")
    print("║  Press Ctrl+C to exit.                                  ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print()

    # Pitch smoothing buffer
    pitch_buffer: list[float] = []

    # Reserve blank lines for the display area (note info + tuner + staff)
    for _ in range(TOTAL_DISPLAY_LINES):
        print()

    def audio_callback(indata: np.ndarray, frames: int, time_info, status):
        nonlocal pitch_buffer

        if status:
            pass  # ignore overflow warnings silently

        # Get mono signal
        signal = indata[:, 0]

        # Check volume (RMS)
        rms = float(np.sqrt(np.mean(signal ** 2)))
        if rms < SILENCE_THRESHOLD:
            sys.stdout.write(f"\033[{TOTAL_DISPLAY_LINES}A")
            sys.stdout.write(f"\r\033[2K  🔇 Listening... (play a note)\n")
            for _ in range(TOTAL_DISPLAY_LINES - 1):
                sys.stdout.write(f"\r\033[2K\n")
            sys.stdout.flush()
            pitch_buffer.clear()
            return

        # Detect pitch using YIN
        freq = yin_pitch(signal, SAMPLE_RATE)

        if freq is None or freq < MIN_FREQ or freq > MAX_FREQ:
            sys.stdout.write(f"\033[{TOTAL_DISPLAY_LINES}A")
            sys.stdout.write(f"\r\033[2K  🔇 No pitch detected...\n")
            for _ in range(TOTAL_DISPLAY_LINES - 1):
                sys.stdout.write(f"\r\033[2K\n")
            sys.stdout.flush()
            return

        # Add to smoothing buffer
        pitch_buffer.append(freq)
        if len(pitch_buffer) > PITCH_BUFFER_SIZE:
            pitch_buffer.pop(0)

        # Use median for stability
        smoothed_freq = float(np.median(pitch_buffer))

        # Map to note
        note, octave, cents = freq_to_note(smoothed_freq)

        # Display
        display_detection(smoothed_freq, note, octave, cents, rms)

    # Start audio stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BUFFER_SIZE,
            channels=CHANNELS,
            callback=audio_callback,
            dtype="float32",
        ):
            print("  ✅ Microphone active. Listening...\n\n")
            while True:
                sd.sleep(100)
    except KeyboardInterrupt:
        print("\n\n  👋 Goodbye! Happy playing! 🎸")
    except sd.PortAudioError as e:
        print(f"\n  ❌ Audio error: {e}")
        print("  Make sure your microphone is connected and accessible.")
        print("  On Fedora, you may need: sudo dnf install pipewire pipewire-pulseaudio")
        sys.exit(1)


if __name__ == "__main__":
    main()
