#!/usr/bin/env python3
"""
Guitar Note Practice Tool
=========================
A metronome-based practice tool that:
  1. Plays a 4-beat metronome with distinct tones per beat
  2. Shows a random target note (E2 → G4) on beat 4
  3. Listens for you to play the correct note from beat 2 of the next bar
  4. Evaluates and scores your accuracy

Dependencies: numpy, sounddevice
Usage:        python guitar_practice.py
"""

import sys
import time
import random
import threading
from collections import Counter

import numpy as np
import sounddevice as sd


# ─── Audio Settings ──────────────────────────────────────────────────────────
SAMPLE_RATE = 44100
MIC_BLOCK_SIZE = 4096       # ~93ms per block at 44100 Hz

# ─── Click / Metronome Sound Settings ────────────────────────────────────────
CLICK_DURATION = 0.06       # seconds
CLICK_VOLUME = 0.35

# Beat tones (as requested: 1st highest, 2nd second-highest, 3rd & 4th lowest)
BEAT_FREQS = {
    1: 1200,    # highest
    2: 900,     # second highest
    3: 660,     # lowest
    4: 660,     # lowest
}

# ─── Pitch Detection Settings ────────────────────────────────────────────────
YIN_THRESHOLD = 0.15
MIN_FREQ = 60.0
MAX_FREQ = 850.0
SILENCE_THRESHOLD = 0.012

# ─── Echo / Noise Removal ────────────────────────────────────────────────────
ECHO_FLUSH_DURATION = 0.5   # seconds to discard audio after detection

# ─── Note Definitions ────────────────────────────────────────────────────────
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def generate_practice_notes():
    """Generate all notes from E2 (MIDI 40) to G4 (MIDI 67)."""
    notes = []
    for midi in range(40, 68):
        note_idx = midi % 12
        octave = (midi // 12) - 1
        name = f"{NOTE_NAMES[note_idx]}{octave}"
        freq = 440.0 * (2.0 ** ((midi - 69) / 12.0))
        notes.append({"name": name, "midi": midi, "freq": freq})
    return notes


PRACTICE_NOTES = generate_practice_notes()


# ─── YIN Pitch Detection (pure numpy) ───────────────────────────────────────

def yin_pitch(signal: np.ndarray, sample_rate: int, threshold: float = YIN_THRESHOLD):
    """
    Estimate fundamental frequency using the YIN algorithm.
    Returns frequency in Hz, or None if no pitch detected.
    """
    signal = signal.astype(np.float64)
    n = len(signal)
    tau_max = n // 2
    tau_min = max(2, int(sample_rate / MAX_FREQ))
    tau_max_search = min(tau_max, int(sample_rate / MIN_FREQ))

    if tau_max_search <= tau_min:
        return None

    w = tau_max

    # Difference function
    d = np.zeros(tau_max_search)
    for tau in range(1, tau_max_search):
        diff = signal[:w] - signal[tau : tau + w]
        d[tau] = np.sum(diff**2)

    # Cumulative mean normalized difference
    d[0] = 1.0
    cumsum = 0.0
    cmnd = np.zeros(tau_max_search)
    cmnd[0] = 1.0
    for tau in range(1, tau_max_search):
        cumsum += d[tau]
        cmnd[tau] = d[tau] * tau / cumsum if cumsum != 0 else 1.0

    # Absolute threshold
    tau_estimate = None
    for tau in range(tau_min, tau_max_search):
        if cmnd[tau] < threshold:
            while tau + 1 < tau_max_search and cmnd[tau + 1] < cmnd[tau]:
                tau += 1
            tau_estimate = tau
            break

    if tau_estimate is None:
        return None

    # Parabolic interpolation
    if 0 < tau_estimate < tau_max_search - 1:
        s0 = cmnd[tau_estimate - 1]
        s1 = cmnd[tau_estimate]
        s2 = cmnd[tau_estimate + 1]
        denom = 2.0 * s1 - s2 - s0
        if denom != 0:
            tau_estimate = tau_estimate + (s2 - s0) / (2.0 * denom)

    return sample_rate / tau_estimate


def freq_to_note_name(freq: float):
    """Convert frequency → ('NoteOctave', cents_off) e.g. ('A4', -2.3)."""
    if freq is None or freq <= 0:
        return None
    semitones = 12.0 * np.log2(freq / 440.0)
    nearest = round(semitones)
    midi = 69 + nearest
    note_idx = midi % 12
    octave = (midi // 12) - 1
    cents = (semitones - nearest) * 100.0
    return f"{NOTE_NAMES[note_idx]}{octave}", cents


# ─── Click Sound Generation ─────────────────────────────────────────────────

def generate_click(freq: float, duration: float = CLICK_DURATION, volume: float = CLICK_VOLUME):
    """Generate a short percussive click sound."""
    n_samples = int(SAMPLE_RATE * duration)
    t = np.arange(n_samples) / SAMPLE_RATE

    # Exponential decay for a "tick" feel
    envelope = np.exp(-t * 45)

    # Tiny attack to avoid a pop
    attack = int(0.002 * SAMPLE_RATE)
    if attack > 0 and attack < n_samples:
        envelope[:attack] *= np.linspace(0, 1, attack)

    signal = volume * np.sin(2 * np.pi * freq * t) * envelope
    return signal.astype(np.float32)


# Pre-generate all click sounds
CLICKS = {beat: generate_click(freq) for beat, freq in BEAT_FREQS.items()}


# ─── Background Pitch Listener ──────────────────────────────────────────────

class PitchListener:
    """Runs in the background, collecting detected note names while listening."""

    def __init__(self):
        self._detections: list[str] = []
        self._lock = threading.Lock()
        self._stream = None
        self._listening = False

    def start(self):
        """Open the microphone stream."""
        self._stream = sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=MIC_BLOCK_SIZE,
            channels=1,
            callback=self._callback,
            dtype="float32",
        )
        self._stream.start()

    def stop(self):
        """Close the microphone stream."""
        if self._stream:
            self._stream.stop()
            self._stream.close()

    def _callback(self, indata, frames, time_info, status):
        if not self._listening:
            return

        signal = indata[:, 0]
        rms = float(np.sqrt(np.mean(signal**2)))
        if rms < SILENCE_THRESHOLD:
            return

        freq = yin_pitch(signal, SAMPLE_RATE)
        if freq and MIN_FREQ <= freq <= MAX_FREQ:
            result = freq_to_note_name(freq)
            if result:
                note_name, _ = result
                with self._lock:
                    self._detections.append(note_name)

    def begin_listening(self, flush_duration: float = 0.0):
        """Start collecting detections (clears previous buffer).

        If *flush_duration* > 0, the microphone stays open for that many
        seconds first so residual echo / string ring is consumed and
        discarded before real detection begins.
        """
        with self._lock:
            self._detections.clear()
        self._listening = True
        if flush_duration > 0:
            time.sleep(flush_duration)
            with self._lock:
                self._detections.clear()

    def end_listening(self):
        """Stop collecting and return the most-detected note name (or None)."""
        self._listening = False
        with self._lock:
            detections = self._detections.copy()
            self._detections.clear()

        if not detections:
            return None

        counter = Counter(detections)
        return counter.most_common(1)[0][0]


# ─── Terminal Colors ─────────────────────────────────────────────────────────
RST = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
WHITE = "\033[97m"
CLR_LINE = "\033[2K"


# ─── ASCII Music Staff Renderer ─────────────────────────────────────────────
#
# Each staff is drawn as a grid of rows. A "position" is a diatonic step
# relative to a reference.  Even positions = lines, odd positions = spaces.
#
# Treble clef lines (bottom→top): E4 G4 B4 D5 F5   (positions 0,2,4,6,8)
# Bass clef lines   (bottom→top): G2 B2 D3 F3 A3   (positions 0,2,4,6,8)
#
# For the GUITAR treble-8vb clef the written pitches are one octave higher
# than they sound, so we add 12 semitones when computing the staff position.
#
# Diatonic note order within an octave (C-based):
#   C=0  D=1  E=2  F=3  G=4  A=5  B=6

# Map note letter → diatonic index
_DIATONIC = {"C": 0, "D": 1, "E": 2, "F": 3, "G": 4, "A": 5, "B": 6}

def _parse_note(note_str: str):
    """Parse 'C#4' → ('C', True, 4) or 'A2' → ('A', False, 2)."""
    if "#" in note_str:
        letter = note_str[0]
        octave = int(note_str[2:])
        return letter, True, octave
    else:
        letter = note_str[0]
        octave = int(note_str[1:])
        return letter, False, octave


def _diatonic_from_c0(letter: str, octave: int) -> int:
    """Absolute diatonic position from C0.  C0=0, D0=1 … B0=6, C1=7 …"""
    return octave * 7 + _DIATONIC[letter]


# ── Treble clef reference: bottom line = E4 ─────────────────────────────────
_TREBLE_BOTTOM = _diatonic_from_c0("E", 4)   # staff-position 0

# ── Bass clef reference: bottom line = G2 ───────────────────────────────────
_BASS_BOTTOM = _diatonic_from_c0("G", 2)     # staff-position 0


def _staff_position(letter: str, octave: int, clef_bottom: int) -> int:
    """
    Return the staff position (0 = bottom line) for a note.
    Even → on a line, Odd → in a space.
    """
    return _diatonic_from_c0(letter, octave) - clef_bottom


def _render_staff(note_str: str | None, clef_bottom: int, clef_label: str,
                  width: int = 37) -> list[str]:
    """
    Render a single 5-line staff with an optional note placed on it.

    Returns a list of strings (one per row, top→bottom).
    """

    # Position range rendered (bottom→top in staff coords)
    POS_MIN = -8   # 4 ledger lines below bottom line
    POS_MAX = 16   # 4 ledger lines above top line

    # The 5 staff lines are at positions 0, 2, 4, 6, 8
    STAFF_LINES = {0, 2, 4, 6, 8}

    # Parse the note
    pos = None
    is_sharp = False
    if note_str:
        letter, is_sharp, octave = _parse_note(note_str)
        pos = _staff_position(letter, octave, clef_bottom)

    rows = []

    COL_NOTE = 20     # column where the note head appears
    COL_SHARP = 17    # column for the sharp sign
    LEDGER_L = COL_NOTE - 2
    LEDGER_R = COL_NOTE + 2   # inclusive

    for p in range(POS_MAX, POS_MIN - 1, -1):
        is_line = p in STAFF_LINES
        on_note = (pos is not None and p == pos)

        # Build the base row
        if is_line:
            left = f"  {clef_label:>12s} " if p == 4 else "               "
            bar = list("─" * width)
        else:
            left = "               "
            bar = list(" " * width)

        if on_note:
            # ── Place the note head ──────────────────────────────
            # On a staff line: clear a gap around the note so ● is visible
            if is_line:
                bar[COL_NOTE - 1] = " "
                bar[COL_NOTE] = "●"
                bar[COL_NOTE + 1] = " "
            else:
                bar[COL_NOTE] = "●"

            # Sharp sign
            if is_sharp:
                bar[COL_SHARP] = "♯"

            # Ledger lines through the note if outside the 5-line staff
            if p % 2 == 0 and p not in STAFF_LINES:
                for c in range(LEDGER_L, LEDGER_R + 1):
                    if 0 <= c < width and c != COL_NOTE:
                        bar[c] = "─"
        else:
            # Ledger lines at even positions between the staff and the note
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
    """
    Render the GUITAR staff (G-clef 8vb).

    Written pitch = sounding pitch + 1 octave.
    So if the note is E2 (sounding), we place it as E3 on a treble staff.
    """
    if note_str:
        letter, is_sharp, octave = _parse_note(note_str)
        written_octave = octave + 1  # guitar notation is 8va up
        written_note = f"{letter}{'#' if is_sharp else ''}{written_octave}"
    else:
        written_note = None

    header = [f"  {DIM}┌─ Guitar staff (G-clef 8vb) ──────────────────────────┐{RST}"]
    staff = _render_staff(written_note, _TREBLE_BOTTOM, "𝄞₈ᵥᵦ")
    footer = [f"  {DIM}└──────────────────────────────────────────────────────┘{RST}"]
    return header + [f"  {DIM}│{RST}{r}" for r in staff] + footer


def render_piano_staff(note_str: str | None) -> list[str]:
    """
    Render the PIANO grand staff (treble G-clef + bass F-clef).

    Notes at or above middle C (C4, MIDI 60) go on the treble staff.
    Notes below middle C go on the bass staff.
    """
    treble_note = None
    bass_note = None

    if note_str:
        letter, is_sharp, octave = _parse_note(note_str)
        midi = _note_to_midi(letter, is_sharp, octave)
        if midi >= 60:  # C4 and above → treble
            treble_note = note_str
        else:  # below C4 → bass
            bass_note = note_str

    header = [f"  {DIM}┌─ Piano staff (Grand staff) ──────────────────────────┐{RST}"]
    treble = _render_staff(treble_note, _TREBLE_BOTTOM, "𝄞")
    separator = [f"               {'─' * 37}"]
    bass = _render_staff(bass_note, _BASS_BOTTOM, "𝄢")
    footer = [f"  {DIM}└──────────────────────────────────────────────────────┘{RST}"]

    lines = header
    lines += [f"  {DIM}│{RST}{r}" for r in treble]
    lines += [f"  {DIM}│{RST}{r}" for r in separator]
    lines += [f"  {DIM}│{RST}{r}" for r in bass]
    lines += footer
    return lines


def _note_to_midi(letter: str, is_sharp: bool, octave: int) -> int:
    """Convert note components to MIDI number."""
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    return (octave + 1) * 12 + base[letter] + (1 if is_sharp else 0)


# ─── Display ─────────────────────────────────────────────────────────────────

def beat_indicator(current_beat: int) -> str:
    """● ○ ○ ○  with color per beat."""
    colors = {1: RED, 2: YELLOW, 3: DIM, 4: DIM}
    parts = []
    for b in range(1, 5):
        if b == current_beat:
            parts.append(f"{colors[b]}●{RST}")
        else:
            parts.append(f"{DIM}○{RST}")
    return "  ".join(parts)


def render(bpm, score, total, target_note, current_beat, last_result, phase):
    """Redraw the whole UI (overwrites from the top of the screen)."""
    pct = (score / total * 100) if total > 0 else 0

    lines = [
        "",
        f"  {BOLD}🎸  Guitar Note Practice{RST}",
        f"  {DIM}{'━' * 50}{RST}",
        f"  BPM: {CYAN}{bpm}{RST}  │  Score: {GREEN}{score}{RST}/{total}  │  Accuracy: {pct:.0f}%",
        f"  {DIM}{'━' * 50}{RST}",
        "",
    ]

    if phase == "intro":
        lines.append(f"  {DIM}Get ready … the first note appears on beat 4{RST}")
        lines.append("")
    elif target_note:
        lines.append(f"  🎯  Play this note:   {BOLD}>>> {YELLOW}{target_note}{RST}{BOLD} <<<{RST}")
        lines.append("")
    else:
        lines.append(f"  {DIM}Waiting …{RST}")
        lines.append("")

    # ── Music staff notation ─────────────────────────────────────────
    if target_note and phase != "intro":
        guitar_lines = render_guitar_staff(target_note)
        lines += guitar_lines
        lines.append("")
    else:
        # Reserve space so UI doesn't jump around
        # Guitar: header(1) + 25 rows + footer(1) = 27, plus gap
        for _ in range(28):
            lines.append("")

    lines.append(f"  Beat:   {beat_indicator(current_beat)}")
    lines.append("")

    if last_result:
        lines.append(f"  {last_result}")
    else:
        lines.append(f"  {DIM}(results will appear here){RST}")

    lines.append("")
    lines.append(f"  {DIM}Press Ctrl+C to exit{RST}")
    lines.append("")

    # Jump to top-left and write
    sys.stdout.write("\033[H")
    for line in lines:
        sys.stdout.write(f"{CLR_LINE}{line}\n")
    sys.stdout.flush()


# ─── Easy Mode Display ───────────────────────────────────────────────────────

def render_easy(score, total, target_note, last_result, status,
                avg_response_time=None, current_elapsed=None):
    """Redraw the easy-mode UI."""
    pct = (score / total * 100) if total > 0 else 0

    # Build the stats line
    stats = f"  Score: {GREEN}{score}{RST}/{total}  │  Accuracy: {pct:.0f}%"
    if avg_response_time is not None:
        stats += f"  │  Avg: {CYAN}{avg_response_time:.1f}s{RST}"
    if current_elapsed is not None:
        stats += f"  │  ⏱ {YELLOW}{current_elapsed:.1f}s{RST}"

    lines = [
        "",
        f"  {BOLD}🎸  Guitar Note Practice  —  Easy Mode{RST}",
        f"  {DIM}{'━' * 50}{RST}",
        stats,
        f"  {DIM}{'━' * 50}{RST}",
        "",
    ]

    if target_note:
        lines.append(f"  🎯  Play this note:   {BOLD}>>> {YELLOW}{target_note}{RST}{BOLD} <<<{RST}")
        lines.append("")
    else:
        lines.append(f"  {DIM}Waiting …{RST}")
        lines.append("")

    # ── Music staff notation ─────────────────────────────────────────
    if target_note:
        guitar_lines = render_guitar_staff(target_note)
        lines += guitar_lines
        lines.append("")
    else:
        for _ in range(28):
            lines.append("")

    # Status line (listening / detected)
    if status:
        lines.append(f"  {status}")
    else:
        lines.append(f"  {DIM}🎤 Play the note on your guitar …{RST}")
    lines.append("")

    if last_result:
        lines.append(f"  {last_result}")
    else:
        lines.append(f"  {DIM}(results will appear here){RST}")

    lines.append("")
    lines.append(f"  {DIM}Press Ctrl+C to exit{RST}")
    lines.append("")

    # Jump to top-left and write
    sys.stdout.write("\033[H")
    for line in lines:
        sys.stdout.write(f"{CLR_LINE}{line}\n")
    sys.stdout.flush()


# ─── Easy Mode ───────────────────────────────────────────────────────────────

def easy_mode():
    """Easy mode: display a note, wait for the user to play it, verify, repeat."""

    listener = PitchListener()
    listener.start()

    score = 0
    total = 0
    last_result = None
    response_times: list[float] = []   # response times for correct notes
    avg_rt = None                       # running average response time

    # Pick the first target note
    target_note = random.choice(PRACTICE_NOTES)["name"]

    sys.stdout.write("\033[2J\033[H")  # clear for main UI

    try:
        while True:
            # ── Show target and listen ────────────────────────────
            render_easy(score, total, target_note, last_result,
                        f"{CYAN}🎤 Listening … play {YELLOW}{target_note}{RST}",
                        avg_response_time=avg_rt)

            listener.begin_listening(flush_duration=ECHO_FLUSH_DURATION)
            note_start_time = time.monotonic()   # start the clock

            # ── Collect microphone samples until a stable note is detected ──
            # We accumulate detections over short windows and require
            # a minimum number of consistent readings for reliability.
            detected = None
            MIN_CONSISTENT = 4          # need this many agreeing samples
            POLL_INTERVAL = 0.12        # seconds between polls
            consistent_count = 0
            last_detected = None

            while detected is None:
                time.sleep(POLL_INTERVAL)

                # Peek at current detections without clearing
                with listener._lock:
                    detections = listener._detections.copy()

                elapsed = time.monotonic() - note_start_time

                if not detections:
                    consistent_count = 0
                    last_detected = None
                    # Redraw with live timer even when silent
                    render_easy(score, total, target_note, last_result,
                                f"{CYAN}🎤 Listening … play {YELLOW}{target_note}{RST}",
                                avg_response_time=avg_rt,
                                current_elapsed=elapsed)
                    continue

                # Take the most common note in the buffer
                counter = Counter(detections)
                top_note = counter.most_common(1)[0][0]

                if top_note == last_detected:
                    consistent_count += 1
                else:
                    last_detected = top_note
                    consistent_count = 1

                # Update status while listening
                render_easy(score, total, target_note, last_result,
                            f"{CYAN}🎤 Hearing: {WHITE}{top_note}{RST}",
                            avg_response_time=avg_rt,
                            current_elapsed=elapsed)

                if consistent_count >= MIN_CONSISTENT:
                    detected = top_note

            listener.end_listening()
            response_time = time.monotonic() - note_start_time

            # ── Evaluate ─────────────────────────────────────────
            total += 1
            if detected == target_note:
                score += 1
                response_times.append(response_time)
                avg_rt = sum(response_times) / len(response_times)
                last_result = (
                    f"{GREEN}✓  Correct!{RST}  "
                    f"Played {GREEN}{detected}{RST}  "
                    f"in {CYAN}{response_time:.1f}s{RST}"
                )
            else:
                last_result = (
                    f"{RED}✗  Wrong!{RST}   "
                    f"Played {RED}{detected}{RST}  "
                    f"Expected {YELLOW}{target_note}{RST}"
                )

            # ── Show result briefly, then next note ──────────────
            render_easy(score, total, target_note, last_result,
                        f"{DIM}Next note in a moment …{RST}",
                        avg_response_time=avg_rt)
            time.sleep(1.5)

            # Pick next target
            target_note = random.choice(PRACTICE_NOTES)["name"]

    except KeyboardInterrupt:
        listener.stop()
        _print_scoreboard(score, total)

    except sd.PortAudioError as e:
        listener.stop()
        print(f"\n  {RED}❌ Audio error: {e}{RST}")
        print(f"  Make sure your microphone is connected.")
        print(f"  On Fedora: sudo dnf install pipewire pipewire-pulseaudio")
        sys.exit(1)


# ─── Hard Mode ───────────────────────────────────────────────────────────────

def hard_mode(bpm: int):
    """Hard mode: metronome-driven practice with timed note detection."""

    beat_duration = 60.0 / bpm

    listener = PitchListener()
    listener.start()

    score = 0
    total = 0
    target_note = None
    last_result = None

    # ── Countdown ────────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")
    for i in range(3, 0, -1):
        sys.stdout.write(f"\033[H\n\n  Starting in {BOLD}{i}{RST} …\n")
        sys.stdout.flush()
        time.sleep(1)

    sys.stdout.write("\033[2J\033[H")  # clear for main UI

    # ── Practice loop ────────────────────────────────────────────────
    try:
        bar = 0
        next_beat_time = time.monotonic()

        while True:
            bar += 1
            is_intro = bar == 1

            for beat_num in range(1, 5):
                # ── Wait for precise beat time ───────────────────────
                now = time.monotonic()
                wait = next_beat_time - now
                if wait > 0:
                    time.sleep(wait)
                next_beat_time += beat_duration

                # ── PRE-click actions ────────────────────────────────

                # On beat 4: evaluate the user's answer (before the tick)
                if beat_num == 4 and not is_intro and target_note:
                    detected = listener.end_listening()
                    total += 1

                    if detected is None:
                        last_result = (
                            f"{RED}✗  No note detected!{RST}  "
                            f"Expected {YELLOW}{target_note}{RST}"
                        )
                    elif detected == target_note:
                        score += 1
                        last_result = (
                            f"{GREEN}✓  Correct!{RST}  "
                            f"Played {GREEN}{detected}{RST}"
                        )
                    else:
                        last_result = (
                            f"{RED}✗  Wrong!{RST}   "
                            f"Played {RED}{detected}{RST}  "
                            f"Expected {YELLOW}{target_note}{RST}"
                        )

                # ── Play the tick sound ──────────────────────────────
                sd.play(CLICKS[beat_num], SAMPLE_RATE)

                # ── POST-click actions ───────────────────────────────

                # On beat 2: start listening for the user's guitar
                if beat_num == 2 and not is_intro and target_note:
                    listener.begin_listening(flush_duration=ECHO_FLUSH_DURATION)

                # On beat 4: pick & display the next target note
                if beat_num == 4:
                    target_note = random.choice(PRACTICE_NOTES)["name"]

                # ── Redraw UI ────────────────────────────────────────
                phase = "intro" if is_intro else "play"
                render(bpm, score, total, target_note, beat_num, last_result, phase)

    except KeyboardInterrupt:
        listener.stop()
        _print_scoreboard(score, total)

    except sd.PortAudioError as e:
        listener.stop()
        print(f"\n  {RED}❌ Audio error: {e}{RST}")
        print(f"  Make sure your microphone is connected.")
        print(f"  On Fedora: sudo dnf install pipewire pipewire-pulseaudio")
        sys.exit(1)


# ─── Shared Scoreboard ───────────────────────────────────────────────────────

def _print_scoreboard(score: int, total: int):
    """Print the final score summary."""
    print("\n\n")
    print(f"  {BOLD}📊  Final Score{RST}")
    print(f"  {DIM}{'━' * 30}{RST}")
    if total > 0:
        pct = score / total * 100
        print(f"  Correct: {GREEN}{score}{RST} / {total}  ({pct:.0f}%)")
        if pct >= 90:
            print(f"  {GREEN}🌟 Excellent!{RST}")
        elif pct >= 70:
            print(f"  {YELLOW}👍 Good job!{RST}")
        elif pct >= 50:
            print(f"  {YELLOW}💪 Keep practicing!{RST}")
        else:
            print(f"  {RED}🎯 More practice needed – keep at it!{RST}")
    else:
        print(f"  No notes attempted.")
    print()
    print(f"  👋  Happy practicing! 🎸")
    print()


# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    # ── Welcome screen ───────────────────────────────────────────────────
    sys.stdout.write("\033[2J\033[H")  # clear
    print()
    print(f"  {BOLD}🎸  Guitar Note Practice Tool{RST}")
    print(f"  {DIM}{'━' * 50}{RST}")
    print()
    print(f"  Practice playing the correct note on your guitar!")
    print(f"  Note range: {CYAN}E2 → G4{RST}  (28 notes, standard guitar range)")
    print()
    print(f"  {BOLD}Choose a mode:{RST}")
    print()
    print(f"    {YELLOW}1{RST}  {BOLD}Hard mode{RST}  (metronome)")
    print(f"       4-beat metronome. A new note appears on beat 4.")
    print(f"       You must play the correct note on beats 2–3 of the next bar.")
    print()
    print(f"    {GREEN}2{RST}  {BOLD}Easy mode{RST}  (no timer)")
    print(f"       A note + staff is shown. Take your time to find and play it.")
    print(f"       The script waits until it hears your note, then shows the next.")
    print()

    # ── Get mode ─────────────────────────────────────────────────────────
    while True:
        raw = input(f"  Select mode [{YELLOW}1{RST}/{GREEN}2{RST}]: ").strip()
        if raw in ("1", "2"):
            mode = int(raw)
            break
        print(f"  {RED}Please enter 1 or 2.{RST}")

    if mode == 1:
        # ── Hard mode: ask for BPM ───────────────────────────────────
        print()
        print(f"  {DIM}Hard mode – metronome practice{RST}")
        print(f"  How each bar works:")
        print(f"    Beat 1  {RED}●{RST}  HIGH tick   – new bar")
        print(f"    Beat 2  {YELLOW}●{RST}  MID tick    – 🎸 play the note HERE!")
        print(f"    Beat 3  {DIM}●{RST}  LOW tick    – still listening …")
        print(f"    Beat 4  {DIM}●{RST}  LOW tick    – result shown + next note")
        print()

        while True:
            try:
                raw = input(f"  Enter tempo (BPM, 40–200) [{CYAN}80{RST}]: ").strip()
                if raw == "":
                    bpm = 80
                else:
                    bpm = int(raw)
                if 40 <= bpm <= 200:
                    break
                print(f"  {RED}Please enter a value between 40 and 200.{RST}")
            except ValueError:
                print(f"  {RED}Please enter a valid number.{RST}")

        hard_mode(bpm)

    else:
        # ── Easy mode ────────────────────────────────────────────────
        print()
        print(f"  {DIM}Easy mode – take your time!{RST}")
        print(f"  {DIM}Starting in 2 seconds …{RST}")
        time.sleep(2)
        easy_mode()


if __name__ == "__main__":
    main()
