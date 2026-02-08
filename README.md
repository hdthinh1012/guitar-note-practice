# 🎸 Guitar Note Practice

A collection of real-time guitar tools built in Python — no heavy C libraries required. Everything runs on **pure NumPy** (YIN pitch-detection algorithm) and **sounddevice** for microphone/speaker access.

---

## Tools

### 1. `guitar_note_detector.py` — Real-Time Note Detector

Listens to your microphone and continuously displays the detected note, octave, and how many cents sharp/flat you are from perfect pitch.

```
 ♪  Detected:  A4   440.0 Hz   +0.1¢
 ├──────────────────●──────────────────┤
           ♭ flat       sharp ♯
```

**Features**
- YIN pitch detection — fast and accurate, implemented entirely in NumPy
- Visual tuner bar showing cents deviation
- Configurable silence threshold, frequency range, and detection sensitivity

### 2. `guitar_practice.py` — Metronome Note Practice

A metronome-driven practice session that shows a random target note, listens for you to play it, and scores your accuracy.

```
 🎯  Play this note:   >>> E3 <<<
 ┌─ Guitar staff (G-clef 8vb) ──────────────────────┐
 │          𝄞₈ᵥᵦ ─────────────────────────────────── │
 │               ─────────────── ● ──────────────── │
 └──────────────────────────────────────────────────┘
 Beat:   ●  ○  ○  ○
```

**Features**
- 4-beat metronome with distinct tones per beat (high → low)
- Random target notes across the full guitar range **E2 → G4** (28 notes)
- ASCII **music staff** (G-clef 8vb) showing the target note with sharps and ledger lines
- Real-time pitch evaluation with scoring & accuracy percentage
- Configurable tempo (40–200 BPM)

**How each bar works**

| Beat | Sound | Action |
|------|-------|--------|
| 1 | HIGH tick | New bar starts |
| 2 | MID tick | 🎸 **Play the note now!** |
| 3 | LOW tick | Still listening … |
| 4 | LOW tick | Result shown + next note revealed |

---

## Quick Start

```bash
# Clone
git clone https://github.com/hdthinh1012/guitar-note-practice.git
cd guitar-note-practice

# Create a virtual environment & install deps
python3 -m venv .venv
source .venv/bin/activate
pip install numpy sounddevice

# Run the real-time detector
python guitar_note_detector.py

# Run the practice tool
python guitar_practice.py
```

---

## Requirements

- **Python** ≥ 3.10
- **NumPy** and **sounddevice** (installed via pip)
- A working **microphone** (USB, built-in, or audio interface)
- Linux with PipeWire / PulseAudio / ALSA (tested on Fedora 43)

> No compiled C extensions needed — pitch detection uses a pure-Python YIN implementation on top of NumPy.

---

## How It Works

Both tools use the **YIN algorithm** for fundamental-frequency estimation:

1. Compute the *difference function* over the audio buffer
2. Apply *cumulative mean normalised difference*
3. Find the first dip below a configurable threshold
4. Parabolic interpolation for sub-sample accuracy
5. Convert the resulting frequency to the nearest note name + cents offset

This gives reliable detection down to **E2 ≈ 82 Hz** with a ~93 ms analysis window (4096 samples @ 44.1 kHz).

---

## License

MIT
