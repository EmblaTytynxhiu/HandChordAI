"""HandChordAI - gesture-controlled virtual guitar (version 3).

WHAT THIS PROGRAM DOES
----------------------
You hold both hands up in front of your webcam:
    - your LEFT hand (screen-left) picks the ROOT note  (C D E F G A B)
    - your RIGHT hand (screen-right) picks the QUALITY  (Major, Minor, 7th)
      and, if its THUMB is out, makes the chord SHARP (#)   (new in v3.3)
The program recognises which fingers are raised, works out the chord,
shows how to play it on a guitar, and plays a strummed guitar sound.

HOW A FRAME IS PROCESSED (the "pipeline")
-----------------------------------------
1. Read a camera image and mirror it (so it behaves like a mirror).
2. MediaPipe finds up to 2 hands and gives 21 points (landmarks) per hand.
3. One Euro Filter smooths the points so they don't jitter.
4. We measure each finger relative to the palm (bend, reach, direction).
5. A classifier turns those measurements into "probability finger is up".
6. Probabilities are smoothed over time + hysteresis (no flickering).
7. The finger pattern is matched against the gesture templates (C, D, ...).
8. The gesture must win several frames in a row before it counts.
9. Root + quality = chord -> play the sound and draw the UI.

CONTROLS
--------
C       Start/cancel calibration (do it once - it makes tracking much better)
G       Show/hide the full gesture guide
D       Toggle diagnostics (raw numbers for every finger)
R       Reset gesture history
SPACE   Replay the current chord
M       Mute / unmute the sound                      (new in v3.1)
S       Save a screenshot into the screenshots folder (new in v3.1)
E       Start/stop an evaluation session              (new in v3.2)
Q/ESC   Quit (closing the window with the X also works now)

COMMAND-LINE OPTIONS (new in v3.1, more in v3.2)
------------------------------------------------
    python HandChordAI.py                 normal start
    python HandChordAI.py --camera 1      use the second webcam
    python HandChordAI.py --mute          start with the sound off
    python HandChordAI.py --participant P03   name used in evaluation files (v3.2)
    python HandChordAI.py --eval-reps 2   ask for every chord twice per session (v3.2)
    python HandChordAI.py --force-fallback    ignore calibration, use the built-in
                                              rules (for comparing the two) (v3.2)
    python HandChordAI.py --help          show all options

WHAT CHANGED IN v3.4  (search the code for "UPGRADE v3.4" to find each one)
---------------------------------------------------------------------------
HELD CHORDS - the chord now rings for as long as you hold the gesture.
1. THE PROBLEM: every chord was a 2-second .wav file (LOOP_DURATION) that
   winsound looped. A plucked string dies away, so you heard: strum ->
   fade to nothing -> strum again, every 2 seconds. It felt like the
   chord only lasted 2 seconds.
2. THE FIX: a new streaming audio player (class ChordPlayer, uses the
   'sounddevice' library). When a chord starts it plays the strum ONCE,
   and underneath it slowly fades in a steady "ringing" version of the
   same chord (SUSTAIN_*). That ringing part loops seamlessly with no
   restart, so the chord keeps sounding until your hands change.
3. When the gesture ends (after RELEASE_GRACE_SECONDS, like before) the
   sound fades out over RELEASE_FADE_SECONDS instead of being cut off.
   Changing chord = old chord fades out while the new one is strummed.
4. If sounddevice is missing or the sound card can't be opened, the
   program automatically falls back to the OLD winsound behaviour, so
   nothing breaks. The panel shows "(winsound fallback)" in that case.
5. play_chord() / stop_chord() keep their names and arguments, so the
   rest of the program didn't need to change. close_audio() is new and
   is called when the program quits.

WHAT CHANGED IN v3.3  (search the code for "UPGRADE v3.3" to find each one)
---------------------------------------------------------------------------
SHARP (#) CHORDS - 21 chords became 36.
1. The right (quality) hand's THUMB is no longer ignored. It is now the
   "sharp switch":  thumb tucked = natural chord, thumb out = sharp chord.
   Example: left "fist" (C) + right "index + thumb" = C# major.
2. Only C D F G A can become sharp. E# and B# are not normal guitar
   chords (E# sounds the same as F, B# the same as C), so on E and B the
   thumb is simply ignored and you get the normal E / B chord.
   -> 7 natural roots + 5 sharp roots = 12 roots x 3 qualities = 36 chords.
3. The quality hand now has 6 gestures instead of 3: MAJOR, MINOR, 7TH
   and #MAJOR, #MINOR, #7TH (the same three with the thumb out).
   The chord's internal name is just root + quality gesture, so
   "C" + "#MAJOR" = "C#MAJOR" (and its sound file is C#MAJOR.wav).
   The helper chord_name() does this (and handles the E/B exception).
4. New guitar shapes for the 15 sharp chords (FRET_SHAPES, CHORD_FINGERINGS).
   The flat name is shown too, because guitarists often say Bb, not A#.
5. Calibration has a 7th pose: "thumb + index out" (an L shape). It helps
   the B root gesture AND the new sharp major gesture. Old calibration
   files still load, but recalibrating (C) is recommended.
6. The chord table on the panel always shows all 12 roots in piano order
   (C, C#, D, D#, E, F, F#, G, G#, A, A#, B). Sharp rows are gold and say
   e.g. "C + R thumb". The rows are shorter so all 12 fit.
   (An earlier v3.3 draft only showed sharps while the thumb was out -
   that hid them, so it was replaced.)
7. The evaluation mode now asks for all 36 chords (about 3 minutes).
8. Calibration sanity check (calibration_problems). BUG FOUND: the
   calibration saved on 2026-11-22 20:21 never saw the thumbs "out", so
   it couldn't tell thumb-in from thumb-out -> C/A and D/B got mixed up.
   Now a calibration like that is refused (not saved, old one kept), and
   an already-saved bad file is ignored at startup (built-in rules used,
   with a message telling you to press C).

WHAT CHANGED IN v3.2  (search the code for "UPGRADE v3.2" to find each one)
---------------------------------------------------------------------------
EVALUATION MODE - so the thesis can report real numbers.
1. Press E: the program asks for all 21 chords in random order. For each
   chord you get EVAL_SETTLE_SECONDS to form it, then EVAL_RECORD_SECONDS
   of recording. Every frame is written to evaluation/<id>.csv, and the
   settings used are written to evaluation/<id>.json.
2. Each frame logs: target chord, what the system recognised (per frame
   AND after voting), what the fallback rules AND the calibrated model
   would say on their own, all finger features and probabilities.
   -> accuracy, confusion matrices, latency, and fallback-vs-calibrated
      comparison can all be computed afterwards from one recording.
3. analyze_evaluation.py turns the CSV files into tables and plots.
4. --participant, --eval-reps, --force-fallback command-line options.
5. HandTrack now remembers raw_probs (before smoothing) and
   frame_gesture (this frame's guess before voting), so they can be logged.

WHAT CHANGED IN v3.1  (search the code for "UPGRADE v3.1" to find each one)
---------------------------------------------------------------------------
1. FPS counter in the top-right corner of the camera view.
2. Closing the window with the X button now really quits the program.
3. One dropped camera frame no longer kills the program; it retries.
4. The blurred camera background is computed on a tiny image -> faster.
5. --camera and --mute command-line options (no more editing the code
   just to switch webcams).
6. M key = mute, S key = screenshot.
7. Automatic tests live in tests/test_logic.py
   (run them with:  python -m unittest discover tests)

GESTURES
--------
ROOT HAND (left)               QUALITY HAND (right)
C  fist, thumb tucked          Major  index up
D  index up                    Minor  index + middle up
E  index + middle up           7th    index + middle + ring up
F  index + middle + ring up
G  all four fingers up         Right THUMB OUT as well = SHARP (#)   (v3.3)
A  thumb out only              e.g. D + "index + thumb" = D#
B  thumb + index up            (E and B ignore the thumb: no E#/B#)

Calibration is saved in finger_calibration.json next to this file.
"""

# "from __future__ import annotations" lets us write modern type hints
# on older Python versions. Harmless to keep.
from __future__ import annotations

# ---- Standard library (comes with Python) ----
import argparse             # UPGRADE v3.1: reads options like --camera 1 from the command line
import csv                  # UPGRADE v3.2: write evaluation recordings as spreadsheet-style files
import json                 # read/write the calibration file
import math                 # exp, log, sin, cos, pi ...
import random               # UPGRADE v3.2: shuffle the chord order in an evaluation session
import time                 # time.monotonic() = a clock that never goes backwards
import urllib.request       # download the MediaPipe model file
import wave                 # write .wav sound files
from collections import Counter, deque  # Counter = count votes, deque = fixed-size list
from pathlib import Path    # nicer file paths than plain strings

# ---- Third-party libraries (pip install opencv-python mediapipe numpy) ----
import cv2                  # OpenCV: camera, drawing, window
import mediapipe as mp      # Google's hand-landmark detector
import numpy as np          # fast maths on arrays

# winsound only exists on Windows. If it's missing (Mac/Linux) we set it to
# None and simply skip playing sound instead of crashing.
try:
    import winsound
except ImportError:
    winsound = None

# UPGRADE v3.4: sounddevice lets us STREAM sound (send it to the speakers
# bit by bit while the program runs). That is what makes it possible to
# hold a chord for as long as the hands hold the gesture - winsound can
# only play/loop a whole file. If sounddevice isn't installed we set it to
# None and the program falls back to the old winsound looping.
# (Install it with:  python -m pip install sounddevice)
try:
    import sounddevice
except Exception:          # ImportError, or OSError if PortAudio is missing
    sounddevice = None

import threading            # UPGRADE v3.4: a Lock so the audio thread and the main loop don't clash


# ============================================================
# 1. CONFIGURATION  (change these numbers to tweak behaviour)
# ============================================================

# ---- Camera ----
CAMERA_INDEX = 0            # 0 = default webcam. Try 1 or 2 if you have several.
CAMERA_WIDTH = 960          # resolution we ASK the camera for (it may ignore us)
CAMERA_HEIGHT = 720

# ---- Window layout ----
DISPLAY_WIDTH = 1280        # whole window width in pixels
DISPLAY_HEIGHT = 720        # whole window height
CAMERA_VIEW_WIDTH = 640     # left half = camera, right half = UI panel
WINDOW_NAME = "HandChordAI - Virtual Guitar"

# ---- Sound ----
SAMPLE_RATE = 44100         # audio samples per second (CD quality)
LOOP_DURATION = 2.0         # length of one .wav = one strum every 2 seconds
STRUM_DELAY = 0.018         # 18 ms gap between strings -> sounds like a strum
# NOTE (v3.4): LOOP_DURATION is now only the length of the STRUM part.
# The chord itself lasts as long as you hold it (see the SUSTAIN_ settings).

# ---- UPGRADE v3.4: held-chord settings (streaming player, ChordPlayer) ----
SUSTAIN_LOOP_SECONDS = 4.0    # length of the seamless "ringing" loop. Every
                              # note is nudged (max 0.125 Hz, too small to hear)
                              # so it fits a whole number of waves in 4 s ->
                              # the loop's end joins its start without a click.
SUSTAIN_LEVEL = 0.30          # loudness of the held ring (strum peak is 0.80)
SUSTAIN_FADE_IN_START = 0.15  # seconds after the strum when the ring starts fading in
SUSTAIN_FADE_IN_SECONDS = 1.2 # how long the ring takes to reach full loudness
SUSTAIN_HARMONICS = (1.0, 0.45, 0.22, 0.10)  # loudness of overtones 1x,2x,3x,4x
                              # (a pure sine sounds like a beep; overtones = warmer)
SUSTAIN_WOBBLE_HZ = 0.5       # gentle volume "breathing" so the ring sounds alive
SUSTAIN_WOBBLE_DEPTH = 0.06   # +-6 % volume (0.5 Hz fits 2 times in 4 s -> seamless)
RELEASE_FADE_SECONDS = 0.20   # fade-out when the chord ends (no hard cut / click)
AUDIO_BLOCK_SIZE = 512        # samples per chunk sent to the sound card (~12 ms)

# ---- MediaPipe detector ----
MAX_HANDS = 2
DETECTION_CONFIDENCE = 0.6  # how sure MediaPipe must be that it found a hand
PRESENCE_CONFIDENCE = 0.6   # how sure it must be the hand is still there
TRACKING_CONFIDENCE = 0.6   # how sure it must be when following between frames

# ---- One Euro Filter settings (landmark smoothing) ----
# min_cutoff: LOWER = smoother when the hand is still (but laggier)
# beta:       HIGHER = less lag when the hand moves fast
# d_cutoff:   smoothing of the speed estimate itself (rarely needs changing)
# Image points are 0..1 screen units, world points are in metres, so they
# need different beta values.
IMAGE_FILTER = {"min_cutoff": 1.5, "beta": 4.0, "d_cutoff": 1.0}
WORLD_FILTER = {"min_cutoff": 2.0, "beta": 30.0, "d_cutoff": 1.0}

# ---- Finger probability smoothing ----
PROBABILITY_EMA = 0.55      # new_value = 0.55 * this_frame + 0.45 * previous
OPEN_THRESHOLD = 0.60       # a folded finger becomes "up" only above 0.60
CLOSE_THRESHOLD = 0.40      # a raised finger becomes "folded" only below 0.40
# The gap between 0.40 and 0.60 is the "hysteresis" zone: inside it the
# finger keeps its previous state, which stops flickering.

# ---- Gesture template matching ----
MIN_GESTURE_POSTERIOR = 0.55  # best gesture must be clearly better than the rest
MIN_GESTURE_FIT = 0.55        # and must fit the fingers reasonably well

# ---- Gesture voting over several frames ----
GESTURE_HISTORY_SIZE = 6      # remember the last 6 frames
REQUIRED_GESTURE_VOTES = 4    # a gesture needs 4 of those 6 frames to count

# ---- Timing ----
RELEASE_GRACE_SECONDS = 0.50  # keep the sound playing this long after gesture ends
HAND_LOST_GRACE_SECONDS = 0.70  # forget a hand after it's been gone this long

# ---- Calibration ----
CALIBRATION_SETTLE_SECONDS = 1.0   # time to get into the pose (not recorded)
CALIBRATION_CAPTURE_SECONDS = 2.0  # time spent recording each pose
CALIBRATION_MIN_SAMPLES = 12       # fewer frames than this -> repeat the pose

# ---- Files (all stored next to this .py file) ----
BASE_DIR = Path(__file__).resolve().parent
SOUND_DIR = BASE_DIR / "sounds_v3"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
CALIBRATION_PATH = BASE_DIR / "finger_calibration.json"
CALIBRATION_VERSION = 3   # bump this if the calibration format ever changes

# ---- UPGRADE v3.1: new settings ----
# Screenshots taken with the S key are saved in this folder.
# (The folder is created automatically the first time you press S.)
SCREENSHOT_DIR = BASE_DIR / "screenshots"

# Webcams sometimes fail to deliver ONE frame (USB hiccup, driver busy).
# Before v3.1 the program quit immediately when that happened. Now we
# only give up after this many failed reads IN A ROW.
CAMERA_READ_RETRIES = 30

# How smooth the FPS number is. 0.1 means: each new frame changes the
# displayed value by only 10%, so the number doesn't jump around wildly.
FPS_SMOOTHING = 0.1

# ---- UPGRADE v3.2: evaluation mode settings (E key) ----
# Recordings are saved in this folder (created automatically).
EVAL_DIR = BASE_DIR / "evaluation"
# Each chord ("trial") has two parts:
#   settle = time to change your hands to the new chord. It IS logged
#            (we need it to measure how long recognition takes), but the
#            accuracy numbers only use the record part.
#   record = time spent holding the chord. Accuracy is measured here.
EVAL_SETTLE_SECONDS = 2.0
EVAL_RECORD_SECONDS = 3.0
# How many times each chord is asked for in one session.
# UPGRADE v3.3: there are 36 chords now (was 21), so
# 1 -> 36 trials x 5 s = about 3 minutes. Change it with --eval-reps.
EVAL_REPETITIONS = 1

# Official MediaPipe hand model (downloaded once, ~7 MB).
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


# ============================================================
# 2. COLORS
# ============================================================
# IMPORTANT: OpenCV uses BGR order, not RGB!
# (255, 0, 0) is BLUE in OpenCV, (0, 0, 255) is RED.

BACKGROUND = (17, 22, 32)     # very dark blue-grey
PANEL = (25, 33, 47)          # right-side panel background
CARD = (34, 44, 60)           # boxes on the panel
ACTIVE = (37, 83, 62)         # highlighted (selected) box - dark green
TEXT = (245, 247, 252)        # almost white
MUTED = (160, 175, 195)       # grey text
DIM = (85, 98, 118)           # darker grey lines
MINT = (105, 230, 185)        # light green accent
BLUE = (235, 155, 85)         # blue (remember: BGR)
GOLD = (65, 195, 255)         # orange/gold
SUCCESS = (90, 220, 135)      # green
DANGER = (75, 90, 235)        # red
WARNING = (0, 190, 255)       # orange
BORDER = (63, 76, 96)         # box outlines

# One colour per finger, used for the skeleton, bars and icons.
FINGER_COLORS = {
    "THUMB": (70, 180, 255),     # orange
    "INDEX": SUCCESS,            # green
    "MIDDLE": BLUE,              # blue
    "RING": (220, 130, 240),     # pink
    "PINKY": (255, 120, 120),    # light purple-blue
}

# Colours for the numbered dots on the guitar fretboard diagram.
# 1 = index, 2 = middle, 3 = ring, 4 = pinky (guitar finger numbering).
FINGER_NUMBER_COLORS = {1: SUCCESS, 2: BLUE, 3: GOLD, 4: (220, 130, 240)}


# ============================================================
# 3. MUSIC DEFINITIONS
# ============================================================

ROOTS = ["C", "D", "E", "F", "G", "A", "B"]   # the 7 natural notes
QUALITIES = ["MAJOR", "MINOR", "7TH"]          # the 3 chord types

# ---- UPGRADE v3.3: sharp (#) chords ----
# ROOTS above are still the 7 LEFT-HAND gestures. A sharp chord uses the
# SAME left-hand gesture as its natural note (C# uses the C fist); the
# right thumb is what adds the "#".
#
# Which natural notes can be made sharp. E and B are missing on purpose:
# E# is the same sound as F and B# the same as C, and guitarists never
# call them that - so on E and B the right thumb is simply ignored.
SHARP_ROOTS = ["C", "D", "F", "G", "A"]

# The sign written in front of a quality gesture name to mean "sharp".
# "#MAJOR" = major WITH the right thumb out.
SHARP_MARK = "#"

# All 6 right-hand gestures: the 3 normal ones, then the 3 sharp ones.
# -> ["MAJOR", "MINOR", "7TH", "#MAJOR", "#MINOR", "#7TH"]
QUALITY_GESTURES = QUALITIES + [SHARP_MARK + q for q in QUALITIES]

# Every sharp note also has a "flat" name (the same piano key).
# Guitarists often prefer the flat name (Bb and Eb are much more common
# than A# and D#), so we show it next to the sharp name.
FLAT_NAMES = {"C#": "Db", "D#": "Eb", "F#": "Gb", "G#": "Ab", "A#": "Bb"}

# How each quality is written after the root: C, Cm, C7.
QUALITY_SUFFIX = {"MAJOR": "", "MINOR": "m", "7TH": "7"}
# Long name shown under the big chord name: "C major", "C minor"...
QUALITY_WORD = {"MAJOR": "major", "MINOR": "minor", "7TH": "dominant 7th"}

# How many semitones (piano keys) each root is above C.
ROOT_SEMITONES = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# All 12 note names in one octave, index 0 = C.
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F",
              "F#", "G", "G#", "A", "A#", "B"]

# The notes in each chord, as semitones above the root.
#   Major = root, major third (+4), fifth (+7)
#   Minor = root, minor third (+3), fifth (+7)
#   7th   = major chord + minor seventh (+10)
CHORD_INTERVALS = {"MAJOR": (0, 4, 7), "MINOR": (0, 3, 7), "7TH": (0, 4, 7, 10)}

# Standard guitar tuning as MIDI note numbers, thickest string first:
# E2=40, A2=45, D3=50, G3=55, B3=59, E4=64.
OPEN_STRING_MIDI = (40, 45, 50, 55, 59, 64)
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]

# Guitar chord shapes. 6 characters, thickest (low E) string first.
#   x = don't play this string
#   0 = play the string open (no finger)
#   1-9 = press this fret
FRET_SHAPES = {
    "C": {"MAJOR": "x32010", "MINOR": "x35543", "7TH": "x32310"},
    "D": {"MAJOR": "xx0232", "MINOR": "xx0231", "7TH": "xx0212"},
    "E": {"MAJOR": "022100", "MINOR": "022000", "7TH": "020100"},
    "F": {"MAJOR": "133211", "MINOR": "133111", "7TH": "131211"},
    "G": {"MAJOR": "320003", "MINOR": "355333", "7TH": "320001"},
    "A": {"MAJOR": "x02220", "MINOR": "x02210", "7TH": "x02020"},
    "B": {"MAJOR": "x24442", "MINOR": "x24432", "7TH": "x21202"},
    # ---- UPGRADE v3.3: the 15 sharp chords ----
    # Sharps have (almost) no easy open-string shapes, so most of these are
    # BARRE chords: the same shape as another chord, slid up the neck.
    #   C#  = the A-shape barre at fret 4 (like Bb/A#, but 3 frets higher)
    #   D#  = small 4-string shapes near the nut (same as Eb)
    #   F#  = the F shape one fret higher (fret 2 instead of 1)
    #   G#  = the F shape at fret 4
    #   A#  = the A-shape barre at fret 1 (same as Bb)
    # build_chords() checks every one of these really contains the right
    # notes, so a typo here stops the program with a clear message.
    "C#": {"MAJOR": "x46664", "MINOR": "x46654", "7TH": "x46464"},
    "D#": {"MAJOR": "xx1343", "MINOR": "xx1342", "7TH": "xx1323"},
    "F#": {"MAJOR": "244322", "MINOR": "244222", "7TH": "242322"},
    "G#": {"MAJOR": "466544", "MINOR": "466444", "7TH": "464544"},
    "A#": {"MAJOR": "x13331", "MINOR": "x13321", "7TH": "x13131"},
}

# Which finger of your FRETTING hand presses each string (same order).
#   1 = index, 2 = middle, 3 = ring, 4 = pinky
#   0 = open string, None = muted string
# When the same finger appears on several strings at the same fret,
# that's a "barre" (one finger flat across the strings).
CHORD_FINGERINGS = {
    "C": {"MAJOR": (None, 3, 2, 0, 1, 0), "MINOR": (None, 1, 3, 4, 2, 1),
          "7TH": (None, 3, 2, 4, 1, 0)},
    "D": {"MAJOR": (None, None, 0, 1, 3, 2), "MINOR": (None, None, 0, 2, 3, 1),
          "7TH": (None, None, 0, 2, 1, 3)},
    "E": {"MAJOR": (0, 2, 3, 1, 0, 0), "MINOR": (0, 2, 3, 0, 0, 0),
          "7TH": (0, 2, 0, 1, 0, 0)},
    "F": {"MAJOR": (1, 3, 4, 2, 1, 1), "MINOR": (1, 3, 4, 1, 1, 1),
          "7TH": (1, 3, 1, 2, 1, 1)},
    "G": {"MAJOR": (2, 1, 0, 0, 0, 3), "MINOR": (1, 3, 4, 1, 1, 1),
          "7TH": (3, 2, 0, 0, 0, 1)},
    "A": {"MAJOR": (None, 0, 1, 2, 3, 0), "MINOR": (None, 0, 2, 3, 1, 0),
          "7TH": (None, 0, 1, 0, 2, 0)},
    "B": {"MAJOR": (None, 1, 3, 3, 3, 1), "MINOR": (None, 1, 3, 4, 2, 1),
          "7TH": (None, 2, 1, 3, 0, 4)},
    # ---- UPGRADE v3.3: fingerings for the 15 sharp chords ----
    # Same order as FRET_SHAPES above (low E string first).
    # A "1" on two or more strings at the same fret = index-finger barre.
    "C#": {"MAJOR": (None, 1, 2, 3, 4, 1), "MINOR": (None, 1, 3, 4, 2, 1),
           "7TH": (None, 1, 3, 1, 4, 1)},
    "D#": {"MAJOR": (None, None, 1, 3, 4, 2), "MINOR": (None, None, 1, 3, 4, 2),
           "7TH": (None, None, 1, 3, 2, 4)},
    "F#": {"MAJOR": (1, 3, 4, 2, 1, 1), "MINOR": (1, 3, 4, 1, 1, 1),
           "7TH": (1, 3, 1, 2, 1, 1)},
    "G#": {"MAJOR": (1, 3, 4, 2, 1, 1), "MINOR": (1, 3, 4, 1, 1, 1),
           "7TH": (1, 3, 1, 2, 1, 1)},
    "A#": {"MAJOR": (None, 1, 2, 3, 4, 1), "MINOR": (None, 1, 3, 4, 2, 1),
           "7TH": (None, 1, 3, 1, 4, 1)},
}


# ============================================================
# 4. GESTURE TEMPLATES  (which fingers = which chord part)
# ============================================================

FINGER_NAMES = ["THUMB", "INDEX", "MIDDLE", "RING", "PINKY"]
NON_THUMB = FINGER_NAMES[1:]          # ["INDEX", "MIDDLE", "RING", "PINKY"]
ROLE_NAMES = ["ROOT", "QUALITY"]      # left hand role, right hand role

# Each pattern is (THUMB, INDEX, MIDDLE, RING, PINKY).
#   1 = finger must be UP, 0 = finger must be FOLDED.
# To add/change a gesture, just edit these tuples.
ROOT_TEMPLATES = {
    "C": (0, 0, 0, 0, 0),
    "D": (0, 1, 0, 0, 0),
    "E": (0, 1, 1, 0, 0),
    "F": (0, 1, 1, 1, 0),
    "G": (0, 1, 1, 1, 1),
    "A": (1, 0, 0, 0, 0),
    "B": (1, 1, 0, 0, 0),
}

# Text used in the big gesture guide (G key).
ROOT_DESCRIPTIONS = {
    "C": "Fist, thumb tucked",
    "D": "Index up",
    "E": "Index + middle",
    "F": "Index + middle + ring",
    "G": "All four fingers",
    "A": "Thumb out only",
    "B": "Thumb + index",
}

# Short text used in the small chord library on the right panel.
ROOT_SHORT = {"C": "fist", "D": "index", "E": "2 fingers", "F": "3 fingers",
              "G": "4 fingers", "A": "thumb", "B": "thumb+index"}

# Quality hand templates.
# UPGRADE v3.3: before v3.3 the thumb was None here ("don't care"). Now the
# thumb MEANS something: 0 = tucked = natural chord, 1 = out = sharp chord.
# So there are 6 templates: the 3 old ones with the thumb tucked, and the
# same 3 with the thumb out (their names start with "#").
QUALITY_TEMPLATES = {
    "MAJOR": (0, 1, 0, 0, 0),
    "MINOR": (0, 1, 1, 0, 0),
    "7TH": (0, 1, 1, 1, 0),
    "#MAJOR": (1, 1, 0, 0, 0),    # UPGRADE v3.3: index + thumb  (an "L")
    "#MINOR": (1, 1, 1, 0, 0),    # UPGRADE v3.3: index + middle + thumb
    "#7TH": (1, 1, 1, 1, 0),      # UPGRADE v3.3: index + middle + ring + thumb
}

QUALITY_DESCRIPTIONS = {
    "MAJOR": "Index up",
    "MINOR": "Index + middle",
    "7TH": "Index + middle + ring",
    # UPGRADE v3.3: the sharp versions.
    "#MAJOR": "Index + thumb",
    "#MINOR": "Index + middle + thumb",
    "#7TH": "Index + middle + ring + thumb",
}


# ---- UPGRADE v3.3: small helpers for the sharp gestures ----

def is_sharp_gesture(quality_gesture):
    """True if this right-hand gesture has the thumb out (= sharp).
    "#MINOR" -> True, "MINOR" -> False, None (unclear hand) -> False."""
    return bool(quality_gesture) and quality_gesture.startswith(SHARP_MARK)


def base_quality(quality_gesture):
    """Remove the "#" to get the plain chord type: "#MINOR" -> "MINOR".
    (Used to know which column of the chord table to highlight.)"""
    if quality_gesture is None:
        return None
    return quality_gesture.lstrip(SHARP_MARK)


def quality_label(quality_gesture):
    """Nice text for a right-hand gesture: "#7TH" -> "# 7th", "MINOR" -> "Minor"."""
    words = {"MAJOR": "Major", "MINOR": "Minor", "7TH": "7th"}
    sharp = "# " if is_sharp_gesture(quality_gesture) else ""
    return sharp + words[base_quality(quality_gesture)]


def chord_name(root_gesture, quality_gesture):
    """Combine the two hands' gestures into the chord's internal name.

        "C" + "MAJOR"   -> "CMAJOR"     (normal chord, like before v3.3)
        "C" + "#MAJOR"  -> "C#MAJOR"    (thumb out -> C sharp major)
        "E" + "#MINOR"  -> "EMINOR"     (no E#: the thumb is ignored on E and B)

    The result is always a key of CHORDS and the name of its .wav file."""
    if is_sharp_gesture(quality_gesture) and root_gesture not in SHARP_ROOTS:
        # E or B with the thumb out: drop the "#" and play the normal chord.
        return root_gesture + base_quality(quality_gesture)
    # Otherwise just glue them together. The "#" (if any) lands right after
    # the root letter, which is exactly how sharp chords are written.
    return root_gesture + quality_gesture


# ============================================================
# 5. MEDIAPIPE LANDMARK NUMBERS
# ============================================================
# MediaPipe gives 21 points per hand:
#   0 = wrist
#   1-4   thumb  (CMC, MCP, IP, TIP)
#   5-8   index  (MCP = big knuckle, PIP = middle joint, DIP = top joint, TIP)
#   9-12  middle
#   13-16 ring
#   17-20 pinky

WRIST = 0
FINGER_LANDMARKS = {
    "THUMB": (1, 2, 3, 4),
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}

# Palm outline lines (the fingers themselves are drawn separately).
HAND_CONNECTIONS = [
    (0, 1), (0, 5), (5, 9), (9, 13), (13, 17), (0, 17),
]


# ============================================================
# 6. BUILD AND VALIDATE THE 36 CHORDS  (UPGRADE v3.3: was 21)
# ============================================================

def midi_to_frequency(midi):
    """Convert a MIDI note number to a frequency in Hz.
    A4 (MIDI 69) = 440 Hz, and every semitone multiplies by 2^(1/12)."""
    return 440.0 * 2 ** ((midi - 69) / 12)


def build_chords():
    """Create a dictionary with everything we know about all 36 chords
    (UPGRADE v3.3: was 21 - the 15 sharp chords were added).

    It also double-checks the tables above: every note in a guitar shape
    must belong to the chord, and the root note must be in it. If you
    make a typo in FRET_SHAPES the program stops and tells you which one.
    """

    chords = {}

    # UPGRADE v3.3: build a list of (gesture_root, sharp) pairs first:
    #   the 7 natural roots  -> ("C", False), ("D", False) ... ("B", False)
    #   the 5 sharp roots    -> ("C", True),  ("D", True)  ... ("A", True)
    # gesture_root is the LEFT-hand gesture; sharp says if the right thumb is out.
    root_variants = [(r, False) for r in ROOTS] + [(r, True) for r in SHARP_ROOTS]

    for gesture_root, sharp in root_variants:
        # UPGRADE v3.3: the note the chord is built on, e.g. "C" or "C#".
        root = gesture_root + (SHARP_MARK if sharp else "")
        for quality in QUALITIES:
            # Pitch classes = note numbers 0..11 (C=0, C#=1, ... B=11).
            # UPGRADE v3.3: a sharp is one semitone (one piano key) higher.
            root_pc = (ROOT_SEMITONES[gesture_root] + (1 if sharp else 0)) % 12
            pitch_classes = [(root_pc + i) % 12 for i in CHORD_INTERVALS[quality]]

            fret = FRET_SHAPES[root][quality]
            fingers = CHORD_FINGERINGS[root][quality]

            # Every guitar shape needs exactly 6 strings.
            if len(fret) != 6 or len(fingers) != 6:
                raise ValueError(f"Invalid shape for {root}{quality}")

            # Work out the actual MIDI note each string plays.
            string_midi = []
            for string_index, value in enumerate(fret):
                if value.lower() == "x":
                    string_midi.append(None)          # muted string
                else:
                    # open string note + fret number = played note
                    string_midi.append(OPEN_STRING_MIDI[string_index] + int(value))

            # Check the shape really is this chord.
            voiced = {m % 12 for m in string_midi if m is not None}
            if not voiced <= set(pitch_classes) or root_pc not in voiced:
                raise ValueError(f"Voicing {fret} does not match {root}{quality}")

            # UPGRADE v3.3: the right-hand gesture that picks this chord,
            # e.g. "MINOR" for Am, "#MINOR" for A#m.
            quality_gesture = (SHARP_MARK if sharp else "") + quality

            # Long name, e.g. "A minor". UPGRADE v3.3: sharp chords also get
            # their flat name, e.g. "A# minor (= Bbm)".
            long_name = f"{root} {QUALITY_WORD[quality]}"
            if sharp:
                long_name += f"  (= {FLAT_NAMES[root]}{QUALITY_SUFFIX[quality]})"

            # Internal name, e.g. "AMINOR" or "A#MINOR". Display name is "Am".
            # UPGRADE v3.3: chord_name() makes the SAME name from the two
            # gestures, so the main loop can find this chord in the table.
            name = chord_name(gesture_root, quality_gesture)
            chords[name] = {
                # "root" is the LEFT-hand gesture letter ("C" for both C and C#).
                # The drawing code uses it to look up ROOT_TEMPLATES, and the
                # evaluation logs it as the target for the left hand.
                "root": gesture_root,
                "root_note": root,                   # UPGRADE v3.3: "C" or "C#"
                "sharp": sharp,                      # UPGRADE v3.3: True for C#, D# ...
                "quality": quality,                  # chord type: MAJOR/MINOR/7TH
                "quality_gesture": quality_gesture,  # UPGRADE v3.3: right-hand gesture
                "display": root + QUALITY_SUFFIX[quality],     # "Am" / "A#m"
                "long_name": long_name,                        # "A minor"
                "notes": [NOTE_NAMES[pc] for pc in pitch_classes],  # ["A","C","E"]
                "fret": fret,
                "fingers": fingers,
                # Frequency of each string (None = muted) for the sound.
                "string_frequencies": [
                    None if m is None else midi_to_frequency(m) for m in string_midi
                ],
            }

    # UPGRADE v3.3: (7 natural + 5 sharp roots) x 3 qualities = 36 (was 21).
    # Calculated instead of typed, so it stays right if the lists change.
    expected = (len(ROOTS) + len(SHARP_ROOTS)) * len(QUALITIES)
    if len(chords) != expected:
        raise ValueError(f"Expected {expected} chords, got {len(chords)}")
    return chords


# Build the chord table once when the program starts.
CHORDS = build_chords()


# ============================================================
# 7. MEDIAPIPE MODEL AND HAND DETECTOR
# ============================================================

def download_model():
    """Download the MediaPipe hand model the first time the program runs."""

    # If the file already exists and isn't tiny/broken, reuse it.
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return

    print("Downloading MediaPipe hand model...")
    try:
        urllib.request.urlretrieve(MODEL_URL, str(MODEL_PATH))
        # A tiny file means the download failed (e.g. an error page).
        if MODEL_PATH.stat().st_size <= 100_000:
            raise RuntimeError("Downloaded model file is too small.")
    except Exception as error:
        raise RuntimeError(f"Could not download the hand model: {error}") from error


def create_hand_detector():
    """Create the MediaPipe HandLandmarker in VIDEO mode.

    VIDEO mode remembers the hands between frames (tracking), which is
    faster and steadier than detecting from scratch every frame.
    """
    download_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=MAX_HANDS,
        min_hand_detection_confidence=DETECTION_CONFIDENCE,
        min_hand_presence_confidence=PRESENCE_CONFIDENCE,
        min_tracking_confidence=TRACKING_CONFIDENCE,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def detect_hands(detector, frame, timestamp_ms):
    """Run MediaPipe on one camera frame.

    Returns a list of hands, each as a tuple:
        (center_x, image_points, world_points)
    - center_x:     average x position (0 = left edge, 1 = right edge)
    - image_points: 21x3 array, x/y in 0..1 screen units (used for drawing)
    - world_points: 21x3 array in metres, centred on the hand (used for maths,
                    because it doesn't depend on how far you are from the camera)
    The list is sorted left to right, so hands[0] is the screen-left hand.
    """

    # OpenCV images are BGR, MediaPipe wants RGB.
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    # VIDEO mode needs a timestamp that always increases.
    result = detector.detect_for_video(image, timestamp_ms)

    hands = []
    for index, image_landmarks in enumerate(result.hand_landmarks or []):
        # Safety check: every hand should also have world landmarks.
        if index >= len(result.hand_world_landmarks):
            continue

        # Convert MediaPipe's point objects into NumPy arrays.
        image_points = np.array([[p.x, p.y, p.z] for p in image_landmarks])
        world_points = np.array(
            [[p.x, p.y, p.z] for p in result.hand_world_landmarks[index]]
        )

        center_x = float(image_points[:, 0].mean())
        hands.append((center_x, image_points, world_points))

    # Sort by x so the left-most hand comes first.
    hands.sort(key=lambda hand: hand[0])
    return hands


# ============================================================
# 8. ONE EURO FILTER  (landmark smoothing)
# ============================================================

class OneEuroFilter:
    """Adaptive low-pass filter for noisy real-time signals.

    Paper: Casiez, Roussel & Vogel, "1 Euro Filter", CHI 2012.

    The idea in plain words:
      - A normal smoothing filter has to choose between "smooth but laggy"
        and "fast but jittery".
      - This filter measures how fast the signal is moving.
        * Slow / still  -> smooth a lot   (kills jitter)
        * Fast movement -> smooth a little (no lag)

    It works on whole NumPy arrays, so we filter all 21x3 numbers at once.
    """

    def __init__(self, min_cutoff=1.0, beta=0.0, d_cutoff=1.0):
        self.min_cutoff = min_cutoff  # smoothing strength when still (Hz)
        self.beta = beta              # how much speed reduces smoothing
        self.d_cutoff = d_cutoff      # smoothing of the speed estimate
        self.reset()

    def reset(self):
        """Forget all history (used when a hand disappears)."""
        self.x_prev = None   # last filtered value
        self.dx_prev = None  # last filtered speed
        self.t_prev = None   # time of last value

    @staticmethod
    def _alpha(cutoff, dt):
        """Convert a cutoff frequency into a blend factor between 0 and 1.
        alpha near 1 = trust the new value, near 0 = keep the old value."""
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, x, t):
        """Filter a new value x measured at time t (seconds).
        Usage: filtered = my_filter(raw_points, now)"""
        x = np.asarray(x, dtype=np.float64)

        # First value ever: nothing to smooth against yet.
        if self.x_prev is None:
            self.x_prev = x.copy()
            self.dx_prev = np.zeros_like(x)
            self.t_prev = t
            return x

        # Time since the previous frame (at least 1 ms to avoid dividing by 0).
        dt = max(t - self.t_prev, 1e-3)

        # 1) Estimate speed and smooth it.
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1 - a_d) * self.dx_prev

        # 2) Faster movement -> higher cutoff -> less smoothing.
        #    (done per number, so a moving fingertip isn't slowed by a still wrist)
        cutoff = self.min_cutoff + self.beta * np.abs(dx_hat)
        a = 1.0 / (1.0 + (1.0 / (2 * math.pi * cutoff)) / dt)

        # 3) Blend new value with the previous filtered value.
        x_hat = a * x + (1 - a) * self.x_prev

        # Remember for next time.
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat


# ============================================================
# 9. FEATURE EXTRACTION  (turn 21 points into finger measurements)
# ============================================================

# Short names for each measurement, shown in debug mode (D key).
FEATURE_LABELS = {
    "THUMB": ["across", "d_pinky", "d_middle", "bend"],
    "FINGER": ["flex", "reach", "up", "pip"],
}


def joint_angle(a, b, c):
    """Angle at point b (in degrees) between the lines b->a and b->c.
    180 = perfectly straight joint, smaller = more bent."""
    v1, v2 = a - b, c - b
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:     # points on top of each other
        return 180.0
    cosine = np.dot(v1, v2) / (n1 * n2)
    # clip protects arccos from tiny rounding errors like 1.0000001
    return float(np.degrees(np.arccos(np.clip(cosine, -1, 1))))


def extract_features(points):
    """Turn 21 world landmarks into 4 numbers per finger.

    WHY A "PALM FRAME"?
    If we used raw x/y/z, the numbers would change whenever you rotate or
    tilt your hand. Instead we build two directions attached to the palm:
        up     = wrist -> middle-finger knuckle
        across = index knuckle -> pinky knuckle (made perpendicular to up)
    and measure fingers relative to those. Now rotating the hand doesn't
    change the numbers, only bending the fingers does.

    NON-THUMB FINGERS -> [flex, reach, up, pip]
        flex  = total bend of all 3 joints in degrees
                (about 0-40 straight, 200+ when curled into a fist)
        reach = (tip to wrist distance) / (knuckle to wrist distance)
                (about 1.9 straight, about 1.0 curled)
        up    = how far the tip sticks out past the knuckle, along "up"
                (positive = pointing up, negative = curled back)
        pip   = angle of the middle joint (180 = straight)

    THUMB -> [across, d_pinky, d_middle, bend]
        across   = thumb tip position along "across", measured from the
                   index knuckle. Positive = over the palm (tucked in),
                   negative = out to the side.
        d_pinky  = distance from thumb tip to pinky knuckle
        d_middle = distance from thumb tip to middle knuckle
        bend     = total bend of the thumb joints
    All distances are divided by palm size so big/small hands match.
    """

    p = np.asarray(points, dtype=np.float64)
    wrist = p[WRIST]
    index_mcp, middle_mcp, pinky_mcp = p[5], p[9], p[17]

    # Palm size (length) and width, used to normalise distances.
    palm_size = max(np.linalg.norm(middle_mcp - wrist), 1e-6)
    palm_width = max(np.linalg.norm(pinky_mcp - index_mcp), 1e-6)

    # Unit vector pointing "up" the palm.
    up = (middle_mcp - wrist) / palm_size

    # Unit vector pointing "across" the palm, made perpendicular to "up"
    # (Gram-Schmidt: remove the part of 'across' that points along 'up').
    across = pinky_mcp - index_mcp
    across = across - np.dot(across, up) * up
    across /= max(np.linalg.norm(across), 1e-6)

    features = {}

    # ---- Index, middle, ring, pinky ----
    for name in NON_THUMB:
        mcp_i, pip_i, dip_i, tip_i = FINGER_LANDMARKS[name]
        mcp, pip, dip, tip = p[mcp_i], p[pip_i], p[dip_i], p[tip_i]

        # Angle at each of the 3 joints (180 = straight).
        mcp_angle = joint_angle(wrist, mcp, pip)
        pip_angle = joint_angle(mcp, pip, dip)
        dip_angle = joint_angle(pip, dip, tip)

        # Total bend = how far each joint is from straight, added up.
        flex = (180 - mcp_angle) + (180 - pip_angle) + (180 - dip_angle)

        # Reach: does the tip stick out beyond the knuckle?
        reach = np.linalg.norm(tip - wrist) / max(np.linalg.norm(mcp - wrist), 1e-6)

        # Dot product with "up" = how far along the up direction the tip is.
        up_proj = float(np.dot(tip - mcp, up)) / palm_size

        features[name] = np.array([flex, reach, up_proj, pip_angle])

    # ---- Thumb ----
    cmc, t_mcp, t_ip, t_tip = (p[i] for i in FINGER_LANDMARKS["THUMB"])
    thumb_across = float(np.dot(t_tip - index_mcp, across)) / palm_width
    d_pinky = np.linalg.norm(t_tip - pinky_mcp) / palm_width
    d_middle = np.linalg.norm(t_tip - middle_mcp) / palm_width
    bend = (180 - joint_angle(cmc, t_mcp, t_ip)) + (180 - joint_angle(t_mcp, t_ip, t_tip))
    features["THUMB"] = np.array([thumb_across, d_pinky, d_middle, bend])

    return features


# ============================================================
# 10. FINGER CLASSIFIERS  (measurements -> "probability finger is up")
# ============================================================

def sigmoid(x):
    """Squash any number into 0..1. 0 -> 0.5, big positive -> ~1,
    big negative -> ~0. Clamped to avoid overflow in math.exp."""
    return 1.0 / (1.0 + math.exp(-max(-30.0, min(30.0, x))))


def fallback_probabilities(features):
    """Simple hand-made rules, used BEFORE you calibrate.

    Each rule is written as (measurement - threshold) / scale:
      - positive result = votes "finger up"
      - negative result = votes "finger folded"
      - scale controls how sharp the decision is
    The votes are averaged and squashed with sigmoid() into a probability.

    These defaults work for many people, but hands and cameras differ -
    calibrating (press C) is much more accurate.
    """

    probs = {}

    # ---- Normal fingers ----
    for name in NON_THUMB:
        flex, reach, up_proj, _ = features[name]
        logits = [
            (120.0 - flex) / 25.0,       # less bend than 120 deg -> up
            (reach - 1.45) / 0.10,       # reaches past 1.45 -> up
            (up_proj - 0.35) / 0.10,     # points up more than 0.35 -> up
        ]
        # clip so one crazy measurement can't outvote the others
        probs[name] = sigmoid(float(np.mean(np.clip(logits, -6, 6))))

    # ---- Thumb ----
    across, d_pinky, d_middle, _ = features["THUMB"]
    logits = [
        (0.05 - across) / 0.08,          # tip out to the side -> up
        (d_pinky - 1.20) / 0.10,         # far from pinky knuckle -> up
        (d_middle - 0.90) / 0.10,        # far from middle knuckle -> up
    ]
    probs["THUMB"] = sigmoid(float(np.mean(np.clip(logits, -6, 6))))
    return probs


# Smallest allowed standard deviation per feature. Without this, if you
# held perfectly still during calibration, the classifier would become
# over-confident and fail on tiny changes.
#                           [feature1, feature2, feature3, feature4]
STD_FLOORS = {
    "THUMB": np.array([0.05, 0.06, 0.06, 6.0]),
    "FINGER": np.array([8.0, 0.05, 0.05, 6.0]),
}


def train_finger_model(X, y, finger):
    """Train a Gaussian Naive Bayes model for ONE finger.

    X = list of feature vectors (one per calibration frame)
    y = list of True/False (was this finger up in that pose?)

    "Gaussian" = we assume each measurement follows a bell curve.
    For "up" frames and "folded" frames separately, we store:
        mean = the typical value (centre of the bell curve)
        std  = how spread out the values are (width of the bell curve)
    Later, a new measurement is compared to both bell curves.

    We use median/MAD (robust statistics) so a few bad frames where
    MediaPipe glitched don't ruin the calibration.
    """

    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=bool)
    floor = STD_FLOORS["THUMB" if finger == "THUMB" else "FINGER"]

    model = {}
    for label, mask in (("open", y), ("closed", ~y)):
        if mask.sum() < 3:
            raise ValueError(f"Not enough '{label}' samples for {finger}")

        # Median = middle value, ignores outliers better than the average.
        mean = np.median(X[mask], axis=0)

        # MAD = median absolute deviation. x1.4826 makes it comparable to std.
        mad = np.median(np.abs(X[mask] - mean), axis=0) * 1.4826

        # Use the biggest of MAD, normal std, and the floor (safe choice).
        std = np.maximum(np.maximum(mad, X[mask].std(axis=0)), floor)

        # .tolist() so it can be saved as JSON.
        model[label] = {"mean": mean.tolist(), "std": std.tolist()}
    return model


def gaussian_log_likelihood(x, mean, std):
    """How well does measurement x fit the bell curve (mean, std)?
    Returns a log-likelihood: higher = better fit.

    z = how many standard deviations away from the mean x is.
    Clipping z at +-4 stops one weird feature from dominating."""
    z = (x - mean) / std
    return float(np.sum(-0.5 * np.clip(z, -4, 4) ** 2 - np.log(std)))


def calibrated_probabilities(features, profile):
    """Use the trained per-finger models to get "probability finger is up".

    For each finger we compute:
        ll_open   = how well it fits the "up" bell curves
        ll_closed = how well it fits the "folded" bell curves
    The difference goes through sigmoid() -> probability between 0 and 1.
    (This is Bayes' rule with equal 50/50 prior.)
    """
    probs = {}
    for name in FINGER_NAMES:
        model = profile[name]
        x = features[name]
        ll_open = gaussian_log_likelihood(
            x, np.array(model["open"]["mean"]), np.array(model["open"]["std"]))
        ll_closed = gaussian_log_likelihood(
            x, np.array(model["closed"]["mean"]), np.array(model["closed"]["std"]))
        probs[name] = sigmoid(ll_open - ll_closed)
    return probs


def match_gesture(probs, templates):
    """Find which gesture template best matches the finger probabilities.

    For every template (e.g. "B" = thumb up, index up, rest folded) we
    multiply the chance of each finger matching:
        finger should be up     -> use p
        finger should be folded -> use 1 - p
    (We add logarithms instead of multiplying, which is the same thing
    but avoids tiny numbers.)

    Returns (name, posterior, fit):
        name      = best gesture, or None if nothing matches well enough
        posterior = how much the best gesture beats all the others (0..1)
        fit       = average per-finger agreement with the best gesture (0..1)
    """

    scores = {}   # log-likelihood of each template
    fits = {}     # average per-finger match of each template

    for name, pattern in templates.items():
        log_likelihood = 0.0
        used = 0
        for finger, wanted in zip(FINGER_NAMES, pattern):
            if wanted is None:          # "don't care" finger, skip it
                continue
            # Keep p away from exactly 0 or 1 so log() never explodes.
            p = min(max(probs[finger], 0.02), 0.98)
            log_likelihood += math.log(p if wanted else 1 - p)
            used += 1
        scores[name] = log_likelihood
        # Geometric mean of the per-finger probabilities.
        fits[name] = math.exp(log_likelihood / max(used, 1))

    # Best template = highest score.
    best = max(scores, key=scores.get)

    # Softmax: turn scores into "share of total probability" for the best one.
    peak = scores[best]
    total = sum(math.exp(s - peak) for s in scores.values())
    posterior = 1.0 / total

    # Only accept if it's clearly the winner AND fits well.
    if posterior >= MIN_GESTURE_POSTERIOR and fits[best] >= MIN_GESTURE_FIT:
        return best, posterior, fits[best]
    return None, posterior, fits[best]


# ============================================================
# 11. PER-HAND TRACKING STATE
# ============================================================

class HandTrack:
    """Everything we remember about ONE hand (ROOT or QUALITY) between frames.

    We create two of these: tracks["ROOT"] and tracks["QUALITY"].
    """

    def __init__(self, role):
        self.role = role   # "ROOT" or "QUALITY"
        # Separate smoothing filters for drawing points and maths points.
        self.image_filter = OneEuroFilter(**IMAGE_FILTER)
        self.world_filter = OneEuroFilter(**WORLD_FILTER)
        self.reset()

    def reset(self):
        """Forget everything (hand left the screen, R key, new calibration)."""
        self.image_filter.reset()
        self.world_filter.reset()
        self.image_points = None   # smoothed 21 points for drawing
        self.features = None       # latest finger measurements
        self.probs = None          # smoothed probability each finger is up
        # UPGRADE v3.2: two extra things remembered only so the evaluation
        # mode can log them (the normal program doesn't need them):
        self.raw_probs = None      # this frame's probabilities BEFORE smoothing
        self.frame_gesture = None  # this frame's best gesture BEFORE voting
        self.states = {name: False for name in FINGER_NAMES}  # final up/down
        self.trails = {name: deque(maxlen=12) for name in FINGER_NAMES}  # tip paths
        self.history = deque(maxlen=GESTURE_HISTORY_SIZE)  # recent gesture guesses
        self.gesture = None        # stable gesture, e.g. "E" or "MINOR"
        self.gesture_confidence = 0.0
        self.last_seen = -1e9      # time we last saw this hand
        self.visible = False       # is the hand on screen this frame?

    def update(self, hand, now, profile):
        """Process this hand for the current frame.

        hand    = (center_x, image_points, world_points) from detect_hands()
        now     = current time in seconds
        profile = calibration for this hand, or None to use fallback rules
        """
        _, image_points, world_points = hand

        # Step 1: smooth the landmarks.
        self.image_points = self.image_filter(image_points, now)
        world = self.world_filter(world_points, now)

        # Step 2: measure the fingers.
        self.features = extract_features(world)

        # Step 3: probability each finger is up (calibrated or fallback).
        if profile is not None:
            raw = calibrated_probabilities(self.features, profile)
        else:
            raw = fallback_probabilities(self.features)
        # UPGRADE v3.2: keep a copy for the evaluation log.
        self.raw_probs = dict(raw)
        # UPGRADE v3.2: clear last frame's guess; vote() fills it in again.
        # (So an old guess can never be logged as if it were this frame's.)
        self.frame_gesture = None

        # Step 4: smooth the probabilities over time (exponential moving average).
        if self.probs is None:
            self.probs = dict(raw)
        else:
            for name in FINGER_NAMES:
                self.probs[name] = (PROBABILITY_EMA * raw[name]
                                    + (1 - PROBABILITY_EMA) * self.probs[name])

        # Step 5: hysteresis - a finger must clearly cross a threshold to
        # change state. Between 0.40 and 0.60 it keeps its old state.
        for name in FINGER_NAMES:
            p = self.probs[name]
            if self.states[name] and p < CLOSE_THRESHOLD:
                self.states[name] = False
            elif not self.states[name] and p > OPEN_THRESHOLD:
                self.states[name] = True

        self.last_seen = now
        self.visible = True

    def vote(self, templates):
        """Match this frame's fingers to a gesture and add it to the history.
        The stable gesture only changes when enough recent frames agree."""
        name, posterior, fit = match_gesture(self.probs, templates)
        # UPGRADE v3.2: remember this frame's guess (None = "unclear")
        # so the evaluation log can compare it with the voted result.
        self.frame_gesture = name
        self.history.append(name)
        self.gesture = stable_value(self.history)
        # Confidence shown in the UI (only meaningful if this frame agrees).
        self.gesture_confidence = posterior * fit if name == self.gesture else 0.0
        return self.gesture


def stable_value(history):
    """Return the most common recent value, but only if it appears at least
    REQUIRED_GESTURE_VOTES times (and isn't None). Otherwise None."""
    if not history:
        return None
    value, votes = Counter(history).most_common(1)[0]
    return value if value is not None and votes >= REQUIRED_GESTURE_VOTES else None


# ============================================================
# 12. CALIBRATION
# ============================================================
# Calibration records you doing 6 poses with BOTH hands. Because we know
# which fingers SHOULD be up in each pose, every frame becomes a labelled
# example, and we train one small model per finger per hand.

# (short name, instruction shown on screen, finger pattern T I M R P)
CALIBRATION_POSES = [
    ("FIST", "Make fists, thumbs tucked over your fingers", (0, 0, 0, 0, 0)),
    ("INDEX", "Raise only your index fingers, thumbs tucked", (0, 1, 0, 0, 0)),
    ("TWO", "Index + middle up (peace sign), thumbs tucked", (0, 1, 1, 0, 0)),
    ("THREE", "Index + middle + ring up, thumb holds the pinky", (0, 1, 1, 1, 0)),
    ("OPEN", "Open hands, spread all five fingers", (1, 1, 1, 1, 1)),
    ("THUMB", "Thumbs out to the side, other fingers folded", (1, 0, 0, 0, 0)),
    # UPGRADE v3.3: 7th pose. "Thumb + index" is the B root gesture AND the
    # new sharp-major gesture, but no pose recorded exactly that before.
    # Recording it teaches the thumb model what "thumb out while the index
    # is up" looks like, which is the hardest case for the sharp switch.
    # (Adding a pose doesn't change the file format: old calibration files
    # still load fine, they just didn't see this pose.)
    ("L", "Thumb + index out (an L shape), others folded", (1, 1, 0, 0, 0)),
]


def create_calibration_session():
    """A dictionary holding the calibration wizard's progress.
        active      = is calibration running?
        phase       = which pose we're on (0..6 - UPGRADE v3.3: 7 poses now)
        phase_start = when this pose started
        samples     = frames recorded for the current pose
        collected   = all (pattern, frame) pairs from finished poses
        notice      = message to show on screen
    """
    return {"active": False, "phase": 0, "phase_start": 0.0,
            "samples": [], "collected": [], "notice": ""}


def start_calibration(session, now):
    """Begin (or restart) calibration from the first pose."""
    session.update({"active": True, "phase": 0, "phase_start": now,
                    "samples": [], "collected": [], "notice": ""})
    print("Calibration started:", CALIBRATION_POSES[0][1])


def cancel_calibration(session):
    """Stop calibration. The previously saved calibration is kept."""
    session.update({"active": False, "samples": [], "collected": [],
                    "notice": "Calibration cancelled."})


# ---- UPGRADE v3.3: calibration sanity check ----
# WHY? On 2026-11-22 a calibration was saved where the thumbs were never
# really seen "out" during the OPEN / THUMB poses (maybe the thumb pointed
# at the camera, or the pose was rushed). The program trained on that
# without complaining, and afterwards it kept mixing up C/A and D/B
# (the gestures that differ ONLY by the thumb).
#
# The check: for every finger, compare the typical "up" value with the
# typical "folded" value of its MAIN measurement. If they're almost the
# same, the model can't tell up from folded -> refuse the calibration.
#   thumb  -> feature 0 "across": out to the side ~ -0.8, tucked ~ +0.6
#             (a good calibration has a gap of about 1.3-1.4)
#   others -> feature 0 "flex":   straight ~ 20 deg, curled ~ 200 deg
#             (a good calibration has a gap of 110+ degrees)
# The broken file had a thumb gap of only 0.15-0.19.
MIN_THUMB_ACROSS_GAP = 0.5     # palm widths
MIN_FINGER_FLEX_GAP = 60.0     # degrees


def calibration_problems(profiles):
    """Return a list of fingers whose calibration can't tell up from folded.
    Empty list = the calibration looks fine.

    profiles = {"ROOT": {finger: model}, "QUALITY": {...}} as made by
    train_profiles() or loaded from finger_calibration.json."""
    hand_names = {"ROOT": "left", "QUALITY": "right"}
    problems = []
    for role in ROLE_NAMES:
        for finger in FINGER_NAMES:
            model = profiles[role][finger]
            # closed minus open on feature 0. Positive and big = good:
            # a tucked thumb sits further "across" the palm, and a folded
            # finger is far more bent, than when they're up.
            gap = model["closed"]["mean"][0] - model["open"]["mean"][0]
            needed = MIN_THUMB_ACROSS_GAP if finger == "THUMB" else MIN_FINGER_FLEX_GAP
            if gap < needed:
                problems.append(f"{hand_names[role]} {finger}")
    return problems


def train_profiles(collected):
    """Train the models for both hands from all recorded frames.

    collected = list of (pose_pattern, {"ROOT": features, "QUALITY": features})
    Returns {"ROOT": {finger: model}, "QUALITY": {finger: model}}.
    """
    profiles = {}
    for role in ROLE_NAMES:
        profiles[role] = {}
        for finger_index, finger in enumerate(FINGER_NAMES):
            # Measurements of this finger in every recorded frame.
            X = [sample[role][finger] for _, sample in collected]
            # Was this finger supposed to be up in that frame's pose?
            y = [pattern[finger_index] == 1 for pattern, _ in collected]
            profiles[role][finger] = train_finger_model(X, y, finger)

    # UPGRADE v3.3: refuse a calibration that can't tell up from folded.
    # Raising ValueError is enough: update_calibration() already catches it,
    # shows "Calibration failed: ..." on screen and does NOT save the file
    # (so your previous calibration is kept).
    problems = calibration_problems(profiles)
    if problems:
        raise ValueError(f"{', '.join(problems)} not seen up AND folded - redo (C)")
    return profiles


def update_calibration(session, tracks, now):
    """Called every frame while calibrating.

    Records samples for the current pose, moves to the next pose when
    time is up, and trains + saves the models after the last pose.
    Returns the new profiles when finished, otherwise None.
    """

    if not session["active"]:
        return None

    elapsed = now - session["phase_start"]

    # We need BOTH hands visible to record a frame.
    both_visible = all(tracks[r].visible and tracks[r].features is not None
                       for r in ROLE_NAMES)

    # Record only after the "settle" time (gives you time to get into the pose).
    if both_visible and elapsed >= CALIBRATION_SETTLE_SECONDS:
        session["samples"].append(
            {role: {f: tracks[role].features[f].copy() for f in FINGER_NAMES}
             for role in ROLE_NAMES})

    # Still recording this pose? Then wait.
    if elapsed < CALIBRATION_SETTLE_SECONDS + CALIBRATION_CAPTURE_SECONDS:
        return None

    # Too few good frames (hands weren't visible) -> repeat this pose.
    if len(session["samples"]) < CALIBRATION_MIN_SAMPLES:
        session["phase_start"] = now
        session["samples"] = []
        session["notice"] = "Show BOTH hands clearly - repeating this pose."
        print(session["notice"])
        return None

    # Pose finished: label every recorded frame with this pose's pattern.
    pattern = CALIBRATION_POSES[session["phase"]][2]
    session["collected"].extend((pattern, s) for s in session["samples"])
    session["samples"] = []
    session["phase"] += 1
    session["phase_start"] = now
    session["notice"] = ""

    # More poses to go?
    if session["phase"] < len(CALIBRATION_POSES):
        print("Next pose:", CALIBRATION_POSES[session["phase"]][1])
        return None

    # All poses done -> train the models.
    try:
        profiles = train_profiles(session["collected"])
    except ValueError as error:
        session["active"] = False
        session["notice"] = f"Calibration failed: {error}"
        print(session["notice"])
        return None

    session["active"] = False
    session["notice"] = "Calibration complete and saved."
    save_calibration(profiles)
    print("Calibration complete. Saved to:", CALIBRATION_PATH)
    return profiles


def save_calibration(profiles):
    """Write the trained models to finger_calibration.json."""
    try:
        CALIBRATION_PATH.write_text(
            json.dumps({"version": CALIBRATION_VERSION, "roles": profiles}, indent=1),
            encoding="utf-8")
        return True
    except OSError as error:
        print(f"Could not save calibration: {error}")
        return False


def load_calibration():
    """Load finger_calibration.json if it exists and is valid.
    Returns the profiles dictionary, or None (-> fallback rules are used)."""
    if not CALIBRATION_PATH.exists():
        return None
    try:
        document = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))

        # Files from older versions of this program use a different format.
        if document.get("version") != CALIBRATION_VERSION:
            print("Old calibration format found - please recalibrate (press C).")
            return None

        # Check every hand / finger / label has 4 means and 4 stds.
        roles = document["roles"]
        for role in ROLE_NAMES:
            for finger in FINGER_NAMES:
                for label in ("open", "closed"):
                    entry = roles[role][finger][label]
                    if len(entry["mean"]) != 4 or len(entry["std"]) != 4:
                        raise ValueError(f"Bad entry {role}/{finger}/{label}")

        # UPGRADE v3.3: a file saved BEFORE the sanity check existed may be
        # broken (like the one from 2026-11-22). Don't use it - the built-in
        # rules are better than a calibration that has the thumb wrong.
        problems = calibration_problems(roles)
        if problems:
            raise ValueError(f"it can't tell up from folded for: {', '.join(problems)}. "
                             "Using built-in rules - please recalibrate (press C)")
        return roles
    except (OSError, ValueError, KeyError, TypeError) as error:
        # json.JSONDecodeError is a subclass of ValueError, so it's caught too.
        print("Saved calibration could not be loaded:", error)
        return None


# ============================================================
# 12b. EVALUATION MODE  (UPGRADE v3.2 - this whole section is new)
# ============================================================
# WHY?
#   A thesis needs numbers: "how often is the right chord recognised?",
#   "which gestures get confused?", "how long does it take?".
#   The evaluation mode answers these by telling you which chord to make
#   (so we KNOW the right answer = "ground truth") and logging what the
#   system recognised on every single frame.
#
# HOW A SESSION WORKS
#   1. Press E. All chords (36 since v3.3) are shuffled into a random order.
#   2. For each chord ("trial"): the target is shown on the panel.
#        - first EVAL_SETTLE_SECONDS: change your hands to the new chord
#        - then  EVAL_RECORD_SECONDS: hold it still
#   3. Every frame becomes one row in evaluation/<session id>.csv.
#   4. When all trials are done (or you press E again) the file is closed.
#   5. Run  python analyze_evaluation.py  to get the results.
#
# The recognition itself is NOT changed by this mode - it runs exactly
# like normal use. The mode only watches and writes things down.

# Which hand-related values go into each CSV row. Kept in one place so the
# header and the rows always match. "root_" / "quality_" are added in front.
EVAL_HAND_FIELDS = [
    "visible",      # 1 = hand seen this frame, 0 = not seen
    "frame",        # the pipeline's guess THIS frame (before voting)
    "stable",       # the pipeline's guess AFTER voting (what the user sees)
    "fallback",     # what the built-in rules alone would say this frame
    "calibrated",   # what the calibrated model alone would say this frame
]
# How the gesture columns are filled in:
#   a name like "E" or "MINOR" = that gesture
#   "NONE" = the classifier saw the hand but wasn't sure enough ("unclear")
#   ""  (empty) = not available (hand not visible, or no calibration exists)


def evaluation_columns():
    """The list of column names for the evaluation CSV, in order."""
    columns = [
        "participant",      # who did the session (--participant)
        "session",          # session id = start date/time
        "live_classifier",  # "calibrated" or "fallback": what the live pipeline used
        "trial",            # trial number 0, 1, 2, ...
        "target",           # the chord the user was asked for, e.g. "EMINOR"
        "target_root",      # e.g. "E"
        "target_quality",   # e.g. "MINOR"
        "phase",            # "settle" or "record"
        "t_trial",          # seconds since this trial started
        "t_session",        # seconds since the session started
        "fps",              # frames per second at that moment
        "active_chord",     # chord shown on screen (may be left over from earlier)
    ]
    for role in ROLE_NAMES:
        prefix = role.lower() + "_"           # "root_" or "quality_"
        columns += [prefix + field for field in EVAL_HAND_FIELDS]
        # Smoothed probability that each finger is up (what the pipeline uses).
        columns += [f"{prefix}p_{finger}" for finger in FINGER_NAMES]
        # The 4 raw measurements of every finger. Logging them means you can
        # later try OTHER classifiers on exactly the same data, offline.
        columns += [f"{prefix}{finger}_f{i}" for finger in FINGER_NAMES for i in range(4)]
    return columns


def make_trial_order(repetitions, rng):
    """Shuffle all chords (36 since UPGRADE v3.3) into the order the user is asked for them.

    repetitions = how many times each chord appears.
    rng         = a random.Random object (seeded, so the order can be
                  recreated later - the seed is saved in the .json file).

    The same chord never comes twice in a row. That matters for the
    latency measurement: if the chord didn't change, it would already be
    "recognised" at time 0 and the measured time would be meaningless.
    """
    chords = list(CHORDS)          # all internal names, e.g. "EMINOR", "C#MAJOR"
    order = []
    for _ in range(repetitions):
        block = chords[:]          # [:] = a copy, so shuffling doesn't touch 'chords'
        rng.shuffle(block)
        # Inside one block every chord appears once, so a repeat can only
        # happen where two blocks meet. If so, swap the first two of the new block.
        if order and block[0] == order[-1]:
            block[0], block[1] = block[1], block[0]
        order += block
    return order


def create_evaluation_session():
    """A dictionary holding the evaluation session's progress.
    Starts inactive; start_evaluation() fills it in."""
    return {"active": False, "trials": [], "index": 0, "trial_start": 0.0,
            "session_start": 0.0, "file": None, "writer": None, "rows": 0,
            "meta": None, "meta_path": None, "notice": ""}


def gesture_text(name):
    """Turn a gesture result into CSV text: None -> "NONE", "E" -> "E"."""
    return "NONE" if name is None else name


def start_evaluation(session, now, participant, live_classifier, repetitions, camera_index):
    """Begin a new evaluation session: pick the chord order, open the CSV
    file and write a .json file describing all the settings used.

    Saving the settings is important for a thesis: if you change, say,
    REQUIRED_GESTURE_VOTES later, you can still tell which recordings
    were made with which settings."""

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    session_id = time.strftime("%Y%m%d_%H%M%S")      # e.g. 20261122_143005
    base = EVAL_DIR / f"{participant}_{session_id}"  # e.g. evaluation/P01_20261122_143005

    # A random seed, saved so the exact same chord order can be recreated.
    seed = random.randrange(1_000_000)
    trials = make_trial_order(repetitions, random.Random(seed))

    # newline="" is what Python's csv module wants (avoids blank lines on Windows).
    csv_file = open(base.with_suffix(".csv"), "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_file, fieldnames=evaluation_columns())
    writer.writeheader()

    # Everything needed to describe this session in the thesis.
    meta = {
        "participant": participant,
        "session": session_id,
        "live_classifier": live_classifier,
        "camera_index": camera_index,
        "repetitions": repetitions,
        "seed": seed,
        "trials": trials,
        "settings": {
            "EVAL_SETTLE_SECONDS": EVAL_SETTLE_SECONDS,
            "EVAL_RECORD_SECONDS": EVAL_RECORD_SECONDS,
            "IMAGE_FILTER": IMAGE_FILTER,
            "WORLD_FILTER": WORLD_FILTER,
            "PROBABILITY_EMA": PROBABILITY_EMA,
            "OPEN_THRESHOLD": OPEN_THRESHOLD,
            "CLOSE_THRESHOLD": CLOSE_THRESHOLD,
            "MIN_GESTURE_POSTERIOR": MIN_GESTURE_POSTERIOR,
            "MIN_GESTURE_FIT": MIN_GESTURE_FIT,
            "GESTURE_HISTORY_SIZE": GESTURE_HISTORY_SIZE,
            "REQUIRED_GESTURE_VOTES": REQUIRED_GESTURE_VOTES,
            "DETECTION_CONFIDENCE": DETECTION_CONFIDENCE,
            "PRESENCE_CONFIDENCE": PRESENCE_CONFIDENCE,
            "TRACKING_CONFIDENCE": TRACKING_CONFIDENCE,
        },
        "completed": False,      # set to True at the end if all trials were done
        "trials_done": 0,
    }

    session.update({"active": True, "trials": trials, "index": 0,
                    "trial_start": now, "session_start": now,
                    "file": csv_file, "writer": writer, "rows": 0,
                    "meta": meta, "meta_path": base.with_suffix(".json"),
                    "notice": ""})
    write_evaluation_meta(session)
    print(f"Evaluation started ({len(trials)} trials). Recording to {base}.csv")


def write_evaluation_meta(session):
    """Save the session's .json description (called at the start and the end)."""
    try:
        session["meta_path"].write_text(json.dumps(session["meta"], indent=1),
                                        encoding="utf-8")
    except OSError as error:
        print("Could not save evaluation info:", error)


def finish_evaluation(session, completed):
    """Close the CSV file and record whether the session was finished.
    completed = True if every trial was done, False if E was pressed early.
    An unfinished recording is still saved - the analysis can use it."""
    if session["file"] is not None:
        session["file"].close()
    session["meta"]["completed"] = completed
    session["meta"]["trials_done"] = session["index"] if not completed else len(session["trials"])
    write_evaluation_meta(session)
    session.update({"active": False, "file": None, "writer": None})
    session["notice"] = ("Evaluation complete - saved." if completed
                         else "Evaluation stopped early - partial recording saved.")
    print(session["notice"], f"({session['rows']} frames)")


def evaluation_row(session, state, now):
    """Build one CSV row describing what happened in this frame."""
    trial_number = session["index"]
    target = session["trials"][trial_number]
    t_trial = now - session["trial_start"]
    meta = session["meta"]

    row = {
        "participant": meta["participant"],
        "session": meta["session"],
        "live_classifier": meta["live_classifier"],
        "trial": trial_number,
        "target": target,
        # The GESTURE each hand should make, so the analysis can compare it
        # directly with what was recognised (root_stable, quality_stable ...).
        "target_root": CHORDS[target]["root"],
        # UPGRADE v3.3: log the right-hand GESTURE ("#MINOR" for A#m), not
        # just the chord type ("MINOR"), otherwise every correctly made
        # sharp chord would be counted as a quality-hand error.
        "target_quality": CHORDS[target]["quality_gesture"],
        "phase": "settle" if t_trial < EVAL_SETTLE_SECONDS else "record",
        "t_trial": round(t_trial, 4),
        "t_session": round(now - session["session_start"], 4),
        "fps": round(state["fps"], 1),
        "active_chord": state["active_chord"] or "",
    }

    profiles = state["profiles"]
    for role in ROLE_NAMES:
        track = state["tracks"][role]
        prefix = role.lower() + "_"
        templates = ROOT_TEMPLATES if role == "ROOT" else QUALITY_TEMPLATES
        seen = track.visible and track.features is not None

        row[prefix + "visible"] = int(seen)
        # The stable (voted) gesture is what the user sees, so log it even
        # when the hand is briefly gone (it stays on screen for a moment).
        row[prefix + "stable"] = gesture_text(track.gesture)

        if not seen:
            # No hand -> no measurements. Leave those columns empty.
            for field in ("frame", "fallback", "calibrated"):
                row[prefix + field] = ""
            continue

        row[prefix + "frame"] = gesture_text(track.frame_gesture)

        # Run BOTH classifiers on this frame's measurements, no matter which
        # one the live pipeline uses. Then one recording lets you compare
        # them on exactly the same hand poses. (Raw = no smoothing/voting.)
        row[prefix + "fallback"] = gesture_text(
            match_gesture(fallback_probabilities(track.features), templates)[0])
        if profiles is not None:
            row[prefix + "calibrated"] = gesture_text(
                match_gesture(calibrated_probabilities(track.features, profiles[role]),
                              templates)[0])
        else:
            row[prefix + "calibrated"] = ""   # no calibration exists

        for finger in FINGER_NAMES:
            row[f"{prefix}p_{finger}"] = round(track.probs[finger], 4)
            for i, value in enumerate(track.features[finger]):
                row[f"{prefix}{finger}_f{i}"] = round(float(value), 5)
    return row


def update_evaluation(session, state, now):
    """Called every frame while an evaluation is running.
    Moves on to the next trial when time is up, and logs the frame."""
    if not session["active"]:
        return

    # Time for the next trial?
    if now - session["trial_start"] >= EVAL_SETTLE_SECONDS + EVAL_RECORD_SECONDS:
        session["index"] += 1
        session["trial_start"] = now
        if session["index"] >= len(session["trials"]):
            finish_evaluation(session, completed=True)
            return
        # NOTE: the hand tracks are deliberately NOT reset here. Changing
        # from the previous chord to the new one is exactly what we want
        # to time, just like in real playing.

    session["writer"].writerow(evaluation_row(session, state, now))
    session["rows"] += 1


# ============================================================
# 13. AUDIO - KARPLUS-STRONG GUITAR SYNTHESIS
# ============================================================

def pluck_string(frequency, sample_count, rng, decay=0.996):
    """Make the sound of one plucked string (Karplus-Strong algorithm, 1983).

    How it works:
      1. Fill a short buffer with random noise. The buffer length sets the
         pitch: length = sample_rate / frequency.
      2. Repeatedly play the buffer, each time replacing every sample with
         the average of two neighbours (times 'decay').
      3. The averaging removes high frequencies a bit more each round, so
         the noise turns into a clear note that fades out - just like a
         real guitar string.

    Speed trick: one whole period only depends on the PREVIOUS period, so
    we compute a full period at once with NumPy instead of a slow loop
    over every single sample.
    """

    # Buffer length (the -0.5 compensates for the delay of the averaging).
    period = max(2, int(round(SAMPLE_RATE / frequency - 0.5)))
    output = np.zeros(sample_count)

    # Initial "pluck" = random noise, lightly smoothed for a warmer tone.
    burst = rng.uniform(-1.0, 1.0, period)
    burst = np.convolve(burst, [0.5, 0.5], mode="same")
    output[:min(period, sample_count)] = burst[:sample_count]

    # Each new period = decay * average of two samples one period earlier.
    for start in range(period, sample_count, period):
        end = min(start + period, sample_count)
        idx = np.arange(start, end)
        output[start:end] = decay * 0.5 * (output[idx - period] + output[idx - period - 1])

    return output


def generate_chord(chord, filename):
    """Create a strummed chord .wav file from the chord's guitar shape."""

    sample_count = int(SAMPLE_RATE * LOOP_DURATION)
    audio = np.zeros(sample_count)

    # Random generator with a fixed seed per chord -> same sound every time.
    rng = np.random.default_rng(sum(ord(c) * (i + 1) for i, c in enumerate(chord["fret"])))

    # Only the strings that are actually played (skip muted "x" strings).
    played = [f for f in chord["string_frequencies"] if f is not None]

    for order, frequency in enumerate(played):
        # Each string starts a little later -> strum from low to high.
        offset = int(order * STRUM_DELAY * SAMPLE_RATE)
        string = pluck_string(frequency, sample_count - offset, rng)
        # Higher strings slightly quieter, like a real downstroke.
        audio[offset:] += string * (1.0 - 0.04 * order)

    # Normalise so the loudest point is 80% of max volume (no clipping).
    peak = np.max(np.abs(audio))
    if peak > 0:
        audio = audio / peak * 0.8

    # Short fade at the very end so the loop doesn't click.
    fade = int(SAMPLE_RATE * 0.03)
    audio[-fade:] *= np.linspace(1.0, 0.0, fade)

    # Convert from -1..1 floats to 16-bit integers (standard WAV format).
    pcm = np.clip(audio * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(filename), "wb") as wav_file:
        wav_file.setnchannels(1)      # mono
        wav_file.setsampwidth(2)      # 2 bytes = 16 bit
        wav_file.setframerate(SAMPLE_RATE)
        wav_file.writeframes(pcm.tobytes())


def create_chord_files():
    """Generate any of the chord .wav files that don't exist yet.
    UPGRADE v3.3: the 15 new sharp chords (C#MAJOR.wav ...) are created
    automatically the first time v3.3 runs; the old 21 files are reused."""
    SOUND_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {len(CHORDS)} guitar chord sounds...")
    for name, chord in CHORDS.items():
        path = SOUND_DIR / f"{name}.wav"
        if not path.exists():
            generate_chord(chord, path)
    print("Chord sounds ready.")


# ------------------------------------------------------------
# UPGRADE v3.4: HELD CHORDS - the streaming player
# ------------------------------------------------------------
# Before v3.4 a chord was a 2 s .wav looped by winsound: strum, die away,
# strum again every 2 s. Now the ChordPlayer below builds the sound live,
# chunk by chunk (AUDIO_BLOCK_SIZE samples at a time):
#
#     sound(t) = strum(t)                 the .wav, played ONCE
#              + fade_in(t) * ring(t)     a steady chord that loops forever
#
# - strum(t): the same Karplus-Strong strum as before (read from the .wav).
# - ring(t):  the chord's notes as gentle sine waves + overtones. Each note
#             is nudged so a whole number of waves fits in
#             SUSTAIN_LOOP_SECONDS, so when the loop restarts it continues
#             exactly where it started -> no click, no re-strum.
# - fade_in:  0 -> 1, so the ring takes over smoothly while the strum dies.
#
# The chord keeps ringing until stop_chord() is called (= the hands stopped
# making the chord for RELEASE_GRACE_SECONDS). Then it fades out over
# RELEASE_FADE_SECONDS instead of stopping with a click.
#
# Each sounding chord is a "voice". Normally there is one; while you change
# chord there are briefly two (old one fading out, new one strumming).

def build_sustain_loop(chord):
    """UPGRADE v3.4: make the seamless "ringing" loop for one chord.
    Returns a float32 array of SUSTAIN_LOOP_SECONDS whose end joins its
    start perfectly (so it can repeat forever without clicking)."""

    sample_count = int(SAMPLE_RATE * SUSTAIN_LOOP_SECONDS)
    t = np.arange(sample_count) / SAMPLE_RATE      # time of every sample, in seconds
    ring = np.zeros(sample_count)

    # Same fixed-seed idea as generate_chord(): same chord = same sound.
    rng = np.random.default_rng(sum(ord(c) * (i + 3) for i, c in enumerate(chord["fret"])))

    played = [f for f in chord["string_frequencies"] if f is not None]
    for order, frequency in enumerate(played):
        for harmonic, loudness in enumerate(SUSTAIN_HARMONICS, start=1):
            # THE SEAMLESS-LOOP TRICK: round the frequency so it does a whole
            # number of waves in the loop. E.g. with a 4 s loop, 82.41 Hz ->
            # 82.50 Hz (330 waves in 4 s). The change is at most 0.125 Hz,
            # which nobody can hear, but now sample [end] flows into [0].
            cycles = max(1, round(frequency * harmonic * SUSTAIN_LOOP_SECONDS))
            exact = cycles / SUSTAIN_LOOP_SECONDS
            if exact >= SAMPLE_RATE / 2:   # above what the sound card can play
                continue
            phase = rng.uniform(0, 2 * math.pi)   # random start -> less "buzzy"
            # Higher strings a bit quieter, like in the strum.
            ring += loudness * (1.0 - 0.04 * order) * np.sin(2 * math.pi * exact * t + phase)

    # Gentle volume "breathing" (0.5 Hz fits 2 whole times in 4 s -> still seamless).
    ring *= 1.0 + SUSTAIN_WOBBLE_DEPTH * np.sin(2 * math.pi * SUSTAIN_WOBBLE_HZ * t)

    peak = np.max(np.abs(ring))
    if peak > 0:
        ring = ring / peak * SUSTAIN_LEVEL
    return ring.astype(np.float32)


def load_strum(chord_name):
    """UPGRADE v3.4: read a chord's strum .wav into a float32 array (-1..1).
    Creates the file first if it's missing."""
    path = SOUND_DIR / f"{chord_name}.wav"
    if not path.exists():
        SOUND_DIR.mkdir(parents=True, exist_ok=True)
        generate_chord(CHORDS[chord_name], path)
    with wave.open(str(path), "rb") as wav_file:
        frames = wav_file.readframes(wav_file.getnframes())
    return np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0


class ChordPlayer:
    """UPGRADE v3.4: plays chords for as long as they are held.

    How it's used (the rest of the program only calls play_chord() and
    stop_chord() further down, which call these):
        player.play("EMINOR")   strum E minor, then keep it ringing
        player.release_all()    fade out whatever is sounding

    The sound card asks for the next chunk of sound ~86 times per second,
    on its OWN thread (in parallel with the camera loop). It calls
    _callback(), which calls render(). The Lock makes sure the main loop
    never changes the voice list halfway through a render."""

    def __init__(self):
        self.voices = []                 # chords currently sounding (dicts)
        self.cache = {}                  # chord name -> (strum, ring) arrays
        self.lock = threading.Lock()
        self.stream = None               # the sounddevice output stream
        # How much the volume drops per sample while fading out.
        self.release_step = 1.0 / max(1, int(SAMPLE_RATE * RELEASE_FADE_SECONDS))

    # ---- opening / closing the sound card connection ----
    def start(self):
        """Open the speakers. Raises an error if that's impossible
        (init_audio() catches it and falls back to winsound)."""
        self.stream = sounddevice.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype="float32",
            blocksize=AUDIO_BLOCK_SIZE, callback=self._callback)
        self.stream.start()

    def close(self):
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass
            self.stream = None

    # ---- controlling which chords sound (called from the main loop) ----
    def sounds_for(self, chord_name):
        """Get (strum, ring) for a chord, building them the first time.
        Done in the MAIN thread, never in the audio thread, because building
        takes a few milliseconds and the audio thread must never be late."""
        if chord_name not in self.cache:
            self.cache[chord_name] = (load_strum(chord_name),
                                      build_sustain_loop(CHORDS[chord_name]))
        return self.cache[chord_name]

    def play(self, chord_name):
        """Start a new chord (strum + endless ring). Any chord that is still
        sounding is faded out, so changing chord sounds smooth."""
        strum, ring = self.sounds_for(chord_name)
        with self.lock:
            for voice in self.voices:
                voice["releasing"] = True
            self.voices.append({"strum": strum, "ring": ring, "pos": 0,
                                "gain": 1.0, "releasing": False})

    def release_all(self):
        """Fade out everything (over RELEASE_FADE_SECONDS)."""
        with self.lock:
            for voice in self.voices:
                voice["releasing"] = True

    def is_sounding(self):
        with self.lock:
            return bool(self.voices)

    # ---- making the actual sound (runs on the audio thread) ----
    def render(self, frame_count):
        """Mix all voices into the next 'frame_count' samples.
        Kept separate from _callback() so the tests can call it directly."""
        out = np.zeros(frame_count, dtype=np.float32)
        with self.lock:
            for voice in self.voices:
                # Sample numbers inside this voice for this chunk.
                idx = voice["pos"] + np.arange(frame_count)

                # 1. The strum - only while the .wav still has samples left.
                strum = voice["strum"]
                signal = np.zeros(frame_count, dtype=np.float32)
                inside = idx < len(strum)
                signal[inside] = strum[idx[inside]]

                # 2. The ring - "% len(ring)" makes it repeat forever.
                fade = np.clip((idx / SAMPLE_RATE - SUSTAIN_FADE_IN_START)
                               / SUSTAIN_FADE_IN_SECONDS, 0.0, 1.0)
                signal += voice["ring"][idx % len(voice["ring"])] * fade

                # 3. Fade-out: lower the volume a tiny bit every sample.
                if voice["releasing"]:
                    gains = voice["gain"] - self.release_step * np.arange(1, frame_count + 1)
                    gains = np.clip(gains, 0.0, 1.0)
                    signal *= gains
                    voice["gain"] = float(gains[-1])

                out += signal
                voice["pos"] += frame_count

            # Voices that have faded to silence are finished -> remove them.
            self.voices = [v for v in self.voices if v["gain"] > 0.0]

        # Two chords overlapping for a moment could go above 1.0 -> clip.
        return np.clip(out, -1.0, 1.0)

    def _callback(self, outdata, frames, time_info, status):
        """Called by sounddevice whenever the speakers need more sound."""
        outdata[:, 0] = self.render(frames)


# UPGRADE v3.4: the one player the whole program shares. Stays None if
# sounddevice is unavailable -> then the old winsound code is used.
audio_player = None


def init_audio():
    """UPGRADE v3.4: try to start the streaming player. Returns True if it
    works. If not (library missing, no sound card...) we print why, and
    play_chord()/stop_chord() automatically use winsound like in v3.3."""
    global audio_player
    if sounddevice is None:
        print("sounddevice not installed - chords will re-strum every "
              f"{LOOP_DURATION:.0f} s (winsound). Install it with: "
              "python -m pip install sounddevice")
        return False
    try:
        player = ChordPlayer()
        player.start()
    except Exception as error:
        print("Could not open the sound card with sounddevice:", error)
        print(f"Falling back to winsound (chords re-strum every {LOOP_DURATION:.0f} s).")
        return False
    audio_player = player
    return True


def close_audio():
    """UPGRADE v3.4: close the sound card connection when quitting."""
    global audio_player
    if audio_player is not None:
        audio_player.close()
        audio_player = None


def stop_chord():
    """Stop whatever sound is playing.
    UPGRADE v3.4: with the streaming player this is a short fade-out
    (RELEASE_FADE_SECONDS) instead of a hard cut. Without it, the old
    winsound stop is used (does nothing without winsound)."""
    if audio_player is not None:
        audio_player.release_all()
        return
    if winsound is None:
        return
    try:
        winsound.PlaySound(None, winsound.SND_PURGE)
    except Exception:
        pass


def play_chord(chord_name, muted=False):
    """Start a chord in the background.

    UPGRADE v3.4: with the streaming player the chord is strummed once and
    then RINGS UNTIL stop_chord() is called - i.e. for as long as the hands
    hold it. Only if that player isn't available do we use the old way:
    SND_ASYNC = don't freeze the program while playing,
    SND_LOOP  = repeat the 2 s .wav until stop_chord() is called.

    UPGRADE v3.1: new 'muted' parameter. When muted is True we simply
    don't start any sound. The rest of the program still thinks the chord
    is "playing" (silently), so un-muting can restart it straight away."""
    if muted or chord_name not in CHORDS:
        return
    if audio_player is not None:            # UPGRADE v3.4: held-chord player
        try:
            audio_player.play(chord_name)
        except Exception as error:
            print("Audio playback error:", error)
        return
    if winsound is None:
        return
    path = SOUND_DIR / f"{chord_name}.wav"
    if not path.exists():
        print(f"Missing audio file: {path}")
        return
    try:
        winsound.PlaySound(str(path), winsound.SND_FILENAME
                           | winsound.SND_ASYNC | winsound.SND_LOOP)
    except Exception as error:
        print("Audio playback error:", error)


# ============================================================
# 14. DRAWING HELPERS
# ============================================================

def draw_text(image, text, position, scale=0.6, color=TEXT, thickness=1):
    """Shortcut for cv2.putText with smooth (anti-aliased) letters.
    position = (x, y) of the BOTTOM-left corner of the text."""
    cv2.putText(image, str(text), (int(position[0]), int(position[1])),
                cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def text_width(text, scale, thickness=1):
    """How many pixels wide a piece of text will be (handy for centring)."""
    return cv2.getTextSize(str(text), cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0]


def draw_mini_hand(image, x, y, pattern, mirrored, size=1.0, highlight=False):
    """Draw a small hand icon: tall coloured bar = finger up,
    short stub = finger folded, outlined stub = "doesn't matter".

    (x, y)    = bottom-left corner of the icon
    pattern   = (T, I, M, R, P) with 1 / 0 / None
    mirrored  = True for the root (screen-left) hand. In the mirrored
                camera view your left hand has its thumb on the RIGHT,
                so the icon matches what you see.
    size      = scale factor
    highlight = draw the palm in gold (used for the currently detected gesture)
    Returns the icon's width in pixels.
    """

    # How tall each finger bar is when raised (thumb shortest).
    finger_heights = {"THUMB": 11, "INDEX": 20, "MIDDLE": 23, "RING": 21, "PINKY": 16}

    order = list(FINGER_NAMES)
    if mirrored:
        order.reverse()   # pinky ... thumb

    bar_w = int(5 * size)
    gap = int(3 * size)
    palm_h = int(7 * size)
    total_w = 5 * bar_w + 4 * gap

    # The palm block at the bottom.
    palm_color = GOLD if highlight else MUTED
    cv2.rectangle(image, (x, y - palm_h), (x + total_w, y), palm_color, -1)

    for slot, finger in enumerate(order):
        wanted = pattern[FINGER_NAMES.index(finger)]
        bx = x + slot * (bar_w + gap)        # left edge of this finger's bar
        bottom = y - palm_h                   # fingers start on top of the palm
        if finger == "THUMB":
            bottom = y - 2  # thumb starts lower, at the side of the palm

        if wanted is None:
            # "Don't care": thin outline only.
            cv2.rectangle(image, (bx, bottom - int(6 * size)), (bx + bar_w, bottom),
                          DIM, 1)
        elif wanted:
            # Raised: tall bar in the finger's colour.
            h = int(finger_heights[finger] * size)
            cv2.rectangle(image, (bx, bottom - h), (bx + bar_w, bottom),
                          FINGER_COLORS[finger], -1)
        else:
            # Folded: short stub in the palm colour.
            cv2.rectangle(image, (bx, bottom - int(4 * size)), (bx + bar_w, bottom),
                          palm_color, -1)

    return total_w


def draw_hand_landmarks(frame, track, label):
    """Draw one hand's skeleton on the camera image.
    Each finger is drawn in its own colour when raised, red when folded,
    with a short trail behind the fingertip."""

    points = track.image_points
    h, w = frame.shape[:2]

    # Convert 0..1 coordinates to pixel coordinates.
    pixel = [(int(p[0] * w), int(p[1] * h)) for p in points]

    # Palm outline in grey.
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pixel[a], pixel[b], (105, 120, 140), 2, cv2.LINE_AA)

    for finger, indices in FINGER_LANDMARKS.items():
        is_open = track.states[finger]
        color = FINGER_COLORS[finger] if is_open else DANGER

        # The thumb's chain starts at the wrist; other fingers at their knuckle.
        chain = (WRIST,) + indices if finger == "THUMB" else indices

        # Draw the bones: zip(chain, chain[1:]) gives consecutive pairs.
        for a, b in zip(chain, chain[1:]):
            cv2.line(frame, pixel[a], pixel[b], color, 4, cv2.LINE_AA)

        # Fingertip trail (last 12 positions).
        tip = pixel[indices[-1]]
        track.trails[finger].append(tip)
        if len(track.trails[finger]) >= 2:
            cv2.polylines(frame, [np.array(track.trails[finger], np.int32)],
                          False, color, 1, cv2.LINE_AA)

        # Fingertip dot with a white outline.
        cv2.circle(frame, tip, 7, color, -1, cv2.LINE_AA)
        cv2.circle(frame, tip, 9, TEXT, 1, cv2.LINE_AA)

    # Label under the wrist, e.g. "ROOT: E".
    wrist = pixel[WRIST]
    text = label
    if track.gesture is not None:
        text += f": {track.gesture}"
    draw_text(frame, text, (max(5, wrist[0] - 40), min(h - 10, wrist[1] + 28)),
              0.6, GOLD, 2)


# ============================================================
# 15. FRETBOARD DIAGRAM
# ============================================================

def draw_fretboard(panel, chord):
    """Draw a guitar chord diagram: 6 vertical strings, 4 frets,
    numbered finger dots, barre bars, X for muted and O for open strings."""

    fret_shape = chord["fret"]
    fingers = chord["fingers"]

    # All fret numbers used (ignoring x and 0).
    fretted = [int(v) for v in fret_shape if v.isdigit() and v != "0"]

    # Normally the diagram starts at fret 1. If the chord goes higher than
    # fret 4 (e.g. Cm at frets 3-5), start the diagram at the lowest fret.
    first_fret = 1
    if fretted and max(fretted) > 4:
        first_fret = min(fretted)

    # Diagram position inside the panel.
    x_left, x_right, top, fret_spacing = 330, 590, 158, 24
    spacing = (x_right - x_left) / 5      # gap between strings

    # ---- Strings (vertical lines) + names underneath ----
    for i, name in enumerate(STRING_NAMES):
        x = int(x_left + i * spacing)
        draw_text(panel, name, (x - 4, top + 4 * fret_spacing + 13), 0.33, GOLD)
        cv2.line(panel, (x, top), (x, top + 4 * fret_spacing), (185, 195, 210), 1)

    # ---- Frets (horizontal lines). Top line is thick = the nut. ----
    for row in range(5):
        y = top + row * fret_spacing
        thick = 3 if row == 0 and first_fret == 1 else 1
        cv2.line(panel, (x_left, y), (x_right, y), (185, 195, 210), thick)

    # Show "3fr" etc. if the diagram doesn't start at fret 1.
    if first_fret > 1:
        draw_text(panel, f"{first_fret}fr", (x_left - 32, top + 17), 0.36, TEXT)

    # ---- Barres ----
    # Group strings by (fret, finger). If one finger presses 2+ strings on
    # the same fret, draw a thick bar from the lowest to the highest string.
    groups = {}
    for i, (v, finger) in enumerate(zip(fret_shape, fingers)):
        if v.isdigit() and v != "0" and finger not in (None, 0):
            groups.setdefault((int(v), finger), []).append(i)
    for (fret, finger), strings in groups.items():
        if len(strings) >= 2:
            y = int(top + (fret - first_fret + 0.5) * fret_spacing)
            cv2.line(panel, (int(x_left + min(strings) * spacing), y),
                     (int(x_left + max(strings) * spacing), y),
                     FINGER_NUMBER_COLORS[finger], 8, cv2.LINE_AA)

    # ---- Dots, X and O for every string ----
    for i, (v, finger) in enumerate(zip(fret_shape, fingers)):
        x = int(x_left + i * spacing)

        if v.lower() == "x":                     # muted string
            draw_text(panel, "X", (x - 5, top - 8), 0.40, DANGER, 2)
            continue

        fret = int(v)
        if fret == 0:                            # open string
            cv2.circle(panel, (x, top - 12), 5, MUTED, 1, cv2.LINE_AA)
            continue

        rel = fret - first_fret                  # row in the diagram (0-3)
        if not 0 <= rel <= 3:
            continue

        # Dot in the middle of the fret space, with the finger number inside.
        y = int(top + (rel + 0.5) * fret_spacing)
        cv2.circle(panel, (x, y), 9, FINGER_NUMBER_COLORS.get(finger, TEXT), -1,
                   cv2.LINE_AA)
        if finger:
            draw_text(panel, finger, (x - 4, y + 4), 0.32, BACKGROUND, 1)

    draw_text(panel, "1 Index 2 Middle 3 Ring 4 Pinky", (330, 285), 0.30, MUTED)


# ============================================================
# 16. PANELS  (right-side user interface)
# ============================================================

def draw_probability_bars(panel, x, y, track, mirrored):
    """Five vertical bars showing how sure we are each finger is up.
    The two thin lines on each bar are the open/close thresholds.
    Bar order matches the hand as you see it on screen."""

    order = list(FINGER_NAMES)
    if mirrored:
        order.reverse()

    bar_w, gap, height = 22, 8, 40
    for slot, finger in enumerate(order):
        bx = x + slot * (bar_w + gap)

        # Empty bar background.
        cv2.rectangle(panel, (bx, y), (bx + bar_w, y + height), BORDER, -1)

        # Filled part = probability.
        if track.probs is not None and track.visible:
            p = track.probs[finger]
            fill = int(height * p)
            color = FINGER_COLORS[finger] if track.states[finger] else DANGER
            cv2.rectangle(panel, (bx, y + height - fill), (bx + bar_w, y + height),
                          color, -1)

        # Threshold marker lines (0.60 and 0.40).
        for t in (OPEN_THRESHOLD, CLOSE_THRESHOLD):
            ty = y + height - int(height * t)
            cv2.line(panel, (bx, ty), (bx + bar_w, ty), DIM, 1)

        # Finger letter under the bar (T I M R P).
        draw_text(panel, finger[0], (bx + 6, y + height + 14), 0.36, MUTED)


def draw_hand_status(panel, x, title, track, color, mirrored, describe):
    """One hand's status block: title, probability bars, detected gesture.
    describe = function turning a gesture name into display text."""
    draw_text(panel, title, (x, 312), 0.42, color, 2)
    draw_probability_bars(panel, x, 322, track, mirrored)

    tx = x + 160   # text to the right of the bars
    if not track.visible:
        draw_text(panel, "not visible", (tx, 345), 0.42, MUTED)
        return
    if track.gesture is None:
        draw_text(panel, "unclear...", (tx, 345), 0.42, GOLD)
        draw_text(panel, "hold still", (tx, 366), 0.34, MUTED)
        return
    draw_text(panel, describe(track.gesture), (tx, 345), 0.6, MINT, 2)
    draw_text(panel, f"{track.gesture_confidence * 100:.0f}% sure", (tx, 366), 0.34, MUTED)


def draw_calibration_card(panel, session, now):
    """Card shown during calibration: instruction, pose icons, progress bars."""
    cv2.rectangle(panel, (18, 100), (622, 290), CARD, -1)
    cv2.rectangle(panel, (18, 100), (622, 290), BORDER, 1)
    draw_text(panel, "PERSONAL HAND CALIBRATION", (35, 130), 0.56, MINT, 2)

    # Current pose (min() avoids going past the last pose).
    phase = min(session["phase"], len(CALIBRATION_POSES) - 1)
    name, instruction, pattern = CALIBRATION_POSES[phase]
    draw_text(panel, f"{phase + 1}/{len(CALIBRATION_POSES)}  {instruction}",
              (35, 168), 0.42, TEXT)
    draw_text(panel, "Both hands, palms toward the camera. Hold still.",
              (35, 192), 0.38, MUTED)

    # Icons showing the pose for each hand.
    draw_text(panel, "left", (430, 128), 0.32, MUTED)
    draw_mini_hand(panel, 430, 160, pattern, mirrored=True, size=1.0, highlight=True)
    draw_text(panel, "right", (510, 128), 0.32, MUTED)
    draw_mini_hand(panel, 510, 160, pattern, mirrored=False, size=1.0, highlight=True)

    # Big bar = progress of this pose (after the settle time).
    elapsed = now - session["phase_start"]
    progress = np.clip((elapsed - CALIBRATION_SETTLE_SECONDS)
                       / CALIBRATION_CAPTURE_SECONDS, 0, 1)
    cv2.rectangle(panel, (35, 210), (600, 228), BORDER, -1)
    cv2.rectangle(panel, (35, 210), (35 + int(565 * progress), 228), SUCCESS, -1)

    # Thin bar = overall progress through all the poses (7 since v3.3).
    overall = (phase + progress) / len(CALIBRATION_POSES)
    cv2.rectangle(panel, (35, 236), (600, 240), BORDER, -1)
    cv2.rectangle(panel, (35, 236), (35 + int(565 * overall), 240), MINT, -1)

    draw_text(panel, f"Samples: {len(session['samples'])}", (35, 262), 0.36, TEXT)
    if session["notice"]:
        draw_text(panel, session["notice"], (35, 282), 0.36, WARNING)
    else:
        draw_text(panel, "Press C to cancel", (35, 282), 0.34, MUTED)


def draw_evaluation_card(panel, session, active_chord, now):
    """UPGRADE v3.2: card shown during an evaluation session.
    Shows which chord to make, how to make it, how much time is left,
    and what the system currently recognises (green = correct)."""
    cv2.rectangle(panel, (18, 100), (622, 290), CARD, -1)
    cv2.rectangle(panel, (18, 100), (622, 290), BORDER, 1)

    meta = session["meta"]
    index = min(session["index"], len(session["trials"]) - 1)
    target = CHORDS[session["trials"][index]]
    draw_text(panel, f"EVALUATION  {meta['participant']}  "
              f"trial {index + 1}/{len(session['trials'])}", (35, 127), 0.42, MINT, 2)

    # The target chord, big, plus the two hand icons showing how to make it.
    draw_text(panel, "Make:", (35, 160), 0.42, MUTED)
    draw_text(panel, target["display"], (35, 215), 2.0, TEXT, 4)
    draw_mini_hand(panel, 190, 210, ROOT_TEMPLATES[target["root"]], mirrored=True,
                   size=1.2, highlight=True)
    draw_text(panel, "+", (248, 200), 0.5, MUTED)
    # UPGRADE v3.3: use the right-hand GESTURE (e.g. "#MINOR"), so the icon
    # shows the thumb out when a sharp chord is asked for.
    draw_mini_hand(panel, 266, 210, QUALITY_TEMPLATES[target["quality_gesture"]],
                   mirrored=False, size=1.2, highlight=True)

    # What the system recognises right now (only the voted, stable result).
    draw_text(panel, "Recognised:", (360, 160), 0.42, MUTED)
    if active_chord:
        correct = active_chord == session["trials"][index]
        draw_text(panel, CHORDS[active_chord]["display"], (360, 215), 1.6,
                  SUCCESS if correct else DANGER, 3)
    else:
        draw_text(panel, "-", (360, 215), 1.6, MUTED, 3)

    # Phase text + progress bar for this trial.
    elapsed = now - session["trial_start"]
    if elapsed < EVAL_SETTLE_SECONDS:
        draw_text(panel, "Change to this chord...", (35, 250), 0.42, GOLD)
    else:
        draw_text(panel, "Recording - hold it", (35, 250), 0.42, SUCCESS)
    progress = np.clip(elapsed / (EVAL_SETTLE_SECONDS + EVAL_RECORD_SECONDS), 0, 1)
    cv2.rectangle(panel, (35, 260), (600, 270), BORDER, -1)
    cv2.rectangle(panel, (35, 260), (35 + int(565 * progress), 270),
                  GOLD if elapsed < EVAL_SETTLE_SECONDS else SUCCESS, -1)
    # Small mark where recording starts, so you can see the two phases.
    settle_x = 35 + int(565 * EVAL_SETTLE_SECONDS / (EVAL_SETTLE_SECONDS + EVAL_RECORD_SECONDS))
    cv2.line(panel, (settle_x, 256), (settle_x, 274), TEXT, 1)

    draw_text(panel, "Press E to stop (the recording so far is kept)", (35, 284), 0.33, MUTED)


def draw_chord_card(panel, active_chord, notice):
    """Card showing the current chord: name, notes, how-to icons, fretboard."""
    cv2.rectangle(panel, (18, 100), (622, 290), CARD, -1)
    cv2.rectangle(panel, (18, 100), (622, 290), BORDER, 1)
    draw_text(panel, "CURRENT CHORD", (35, 127), 0.40, MUTED)

    # No chord yet -> show help text instead.
    if not active_chord:
        draw_text(panel, "Waiting for gestures...", (35, 180), 0.6, MUTED)
        draw_text(panel, "Press G for the gesture guide", (35, 215), 0.42, GOLD)
        draw_text(panel, "Press C to calibrate your hands", (35, 240), 0.42, GOLD)
        if notice:
            draw_text(panel, notice, (35, 275), 0.36, SUCCESS)
        return

    chord = CHORDS[active_chord]
    draw_text(panel, chord["display"], (35, 195), 2.0, MINT, 4)          # "Em"
    draw_text(panel, chord["long_name"], (35, 225), 0.5, TEXT)           # "E minor"
    draw_text(panel, "Notes: " + " ".join(chord["notes"]), (35, 250), 0.40, MUTED)
    draw_text(panel, "Shape: " + chord["fret"].upper(), (35, 272), 0.40, MUTED)

    # "Left hand icon + right hand icon" = how to make this chord.
    draw_mini_hand(panel, 190, 190, ROOT_TEMPLATES[chord["root"]], mirrored=True)
    draw_text(panel, "+", (238, 182), 0.5, MUTED)
    # UPGRADE v3.3: "quality_gesture" (not "quality") so sharp chords show
    # the right-hand icon WITH the thumb out.
    draw_mini_hand(panel, 254, 190, QUALITY_TEMPLATES[chord["quality_gesture"]],
                   mirrored=False)

    draw_fretboard(panel, chord)


def draw_library(panel, active_chord, root_gesture, quality_gesture):
    """The chord table, which doubles as a gesture guide:
    each row shows the left-hand gesture, each column the right-hand one.
    The currently detected row/column/chord are highlighted.

    UPGRADE v3.3 (second version): ALL 12 roots are always shown, in piano
    order: C, C#, D, D#, E, F, F#, G, G#, A, A#, B  (= 36 chords).
    (The first v3.3 version only showed the sharps while your thumb was
    out - that hid them, so it was replaced.)
    To make 12 rows fit in the same space, the rows are a bit shorter
    (19 px instead of 30) and the icons/text a bit smaller.
    A sharp row uses the SAME left-hand gesture as the natural row above
    it, so instead of repeating the icon it says e.g. "C + R thumb"."""

    # UPGRADE v3.3: the 12 rows as (left-hand gesture, is_sharp) pairs.
    # For each natural root we add its row, then (if it has one) its sharp:
    #   ("C", False), ("C", True), ("D", False), ("D", True), ("E", False) ...
    rows = []
    for root in ROOTS:
        rows.append((root, False))
        if root in SHARP_ROOTS:
            rows.append((root, True))

    # UPGRADE v3.3: which row is "live" right now, e.g. ("C", True) for C#.
    # The sharp only counts on roots that have one (E/B ignore the thumb),
    # exactly like chord_name() does in the main loop.
    current_row = None
    if root_gesture is not None:
        current_row = (root_gesture,
                       is_sharp_gesture(quality_gesture) and root_gesture in SHARP_ROOTS)

    draw_text(panel, "GESTURE GUIDE", (22, 398), 0.50, TEXT, 2)
    draw_text(panel, "rows = left hand, columns = right | right thumb out = #",
              (190, 398), 0.32, MUTED)

    # Layout numbers (pixels inside the panel).
    # UPGRADE v3.3: smaller rows/header and the table starts a bit higher,
    # so 12 rows end at about y=683 - just above the footer text at y=700.
    left, root_w, gap = 18, 150, 6
    top, row_h, row_gap, header_h = 404, 19, 2, 26
    cell_w = (622 - left - root_w - 3 * gap) // 3

    # ---- Column headers: quality gesture icons ----
    for col, quality in enumerate(QUALITIES):
        x1 = left + root_w + gap + col * (cell_w + gap)
        # UPGRADE v3.3: base_quality() removes the "#", so the Minor column
        # lights up for both "MINOR" and "#MINOR".
        is_current = quality == base_quality(quality_gesture)
        cv2.rectangle(panel, (x1, top), (x1 + cell_w, top + header_h),
                      ACTIVE if is_current else CARD, -1)
        # Icon = the natural gesture (thumb tucked). The hint text above the
        # table says that adding the thumb makes it sharp.
        draw_mini_hand(panel, x1 + 8, top + header_h - 3,
                       QUALITY_TEMPLATES[quality], mirrored=False, size=0.7,
                       highlight=is_current)
        draw_text(panel, quality.title() if quality != "7TH" else "7th",
                  (x1 + 48, top + 18), 0.40, TEXT)

    draw_text(panel, "ROOT", (left + 8, top + 18), 0.40, BLUE)

    # ---- One row per root note (12 rows) ----
    for row, (root, sharp) in enumerate(rows):
        y1 = top + header_h + 3 + row * (row_h + row_gap)
        y2 = y1 + row_h
        is_current = (root, sharp) == current_row
        label = root + (SHARP_MARK if sharp else "")     # "C" or "C#"

        # Left cell: note name + how to make it.
        cv2.rectangle(panel, (left, y1), (left + root_w, y2),
                      ACTIVE if is_current else CARD, -1)
        # Sharp rows are written in GOLD so they stand out from the naturals.
        draw_text(panel, label, (left + 6, y1 + 15), 0.45, GOLD if sharp else MINT, 1)
        # UPGRADE v3.3: EVERY row (natural and sharp) now has the same layout:
        #   [note]  [left-hand icon]  [right-thumb icon if sharp]  [description]
        # (Before, sharp rows only had the text "C + R thumb" and no picture.)

        # 1) Left-hand icon. A sharp row uses the SAME left-hand gesture as
        #    its natural note (C# = the C fist), so it's the same icon.
        draw_mini_hand(panel, left + 36, y2 - 2, ROOT_TEMPLATES[root],
                       mirrored=True, size=0.55, highlight=is_current)

        # 2) Sharp rows only: a small RIGHT-hand icon with just the thumb up
        #    (the other fingers are drawn as outlines = "whatever the column
        #    says"). Slightly bigger than the left icon, and with a grey palm
        #    (highlight=False) so the orange thumb stands out clearly.
        if sharp:
            thumb_only = (1, None, None, None, None)   # (T, I, M, R, P)
            draw_mini_hand(panel, left + 54, y2 - 2, thumb_only,
                           mirrored=False, size=0.7, highlight=False)

        # 3) Short description, at the same x for every row so they line up.
        #    Gold on sharp rows to match the gold "#" note name.
        draw_text(panel, ROOT_SHORT[root], (left + 84, y1 + 14), 0.32,
                  GOLD if sharp else MUTED)

        # Three chord cells: name + guitar shape.
        for col, quality in enumerate(QUALITIES):
            x1 = left + root_w + gap + col * (cell_w + gap)
            # UPGRADE v3.3: chord_name() gives the same name the main loop
            # uses, e.g. ("C", "#MINOR") -> "C#MINOR".
            name = chord_name(root, (SHARP_MARK if sharp else "") + quality)
            color = ACTIVE if name == active_chord else CARD
            cv2.rectangle(panel, (x1, y1), (x1 + cell_w, y2), color, -1)
            cv2.rectangle(panel, (x1, y1), (x1 + cell_w, y2), BORDER, 1)
            draw_text(panel, CHORDS[name]["display"], (x1 + 8, y1 + 15), 0.42,
                      MINT if name == active_chord else TEXT, 1)
            draw_text(panel, CHORDS[name]["fret"].upper(), (x1 + 66, y1 + 14), 0.30, MUTED)


def draw_panel(display, state, now):
    """Draw the whole right half of the window."""

    # display[:, 640:] is a VIEW into the big image - drawing on 'panel'
    # draws directly on the right half of 'display'.
    panel = display[:, CAMERA_VIEW_WIDTH:]
    panel[:] = PANEL

    # ---- Header ----
    draw_text(panel, "VIRTUAL GUITAR", (22, 43), 0.90, TEXT, 2)
    draw_text(panel, "TWO-HAND CHORD CONTROLLER", (24, 70), 0.40, MUTED)

    calibrated = state["profiles"] is not None
    if state["force_fallback"]:
        # UPGRADE v3.2: --force-fallback was used, so the calibration is
        # ignored even if it exists. Say so, so you never forget which
        # mode a recording was made in.
        draw_text(panel, "FALLBACK RULES (forced)", (400, 40), 0.38, WARNING, 1)
    else:
        draw_text(panel, "CALIBRATED" if calibrated else "NOT CALIBRATED - PRESS C",
                  (360 if not calibrated else 500, 40), 0.38,
                  SUCCESS if calibrated else GOLD, 1)
    # UPGRADE v3.1: show "MUTED" (orange) when M was pressed / --mute was used.
    if state["muted"]:
        draw_text(panel, "MUTED (M)", (500, 70), 0.40, WARNING, 2)
    elif state["playing_chord"]:
        draw_text(panel, "PLAYING", (500, 70), 0.40, SUCCESS, 2)
    else:
        draw_text(panel, "SOUND OFF", (500, 70), 0.38, MUTED)
    # UPGRADE v3.4: say which audio path is in use. Nothing shown = the new
    # held-chord player works. Otherwise warn, so you know why the chord
    # re-strums every 2 s (fallback) or why there's no sound at all.
    if audio_player is None:
        draw_text(panel, "(winsound fallback)" if winsound is not None else "(no audio)",
                  (500, 84), 0.30, WARNING)

    cv2.line(panel, (20, 90), (620, 90), BORDER, 1)

    # ---- Big card: calibration wizard OR current chord ----
    session = state["calibration"]
    evaluation = state["evaluation"]   # UPGRADE v3.2
    if session["active"]:
        draw_calibration_card(panel, session, now)
    elif evaluation["active"]:
        # UPGRADE v3.2: evaluation card replaces the chord card.
        draw_evaluation_card(panel, evaluation, state["active_chord"], now)
    else:
        # UPGRADE v3.2: show the evaluation's "saved" message too, if newer.
        draw_chord_card(panel, state["active_chord"],
                        evaluation["notice"] or session["notice"])

    # ---- Live status of both hands ----
    tracks = state["tracks"]
    draw_hand_status(panel, 22, "LEFT HAND: ROOT", tracks["ROOT"], BLUE, True,
                     lambda g: g)
    # UPGRADE v3.3: quality_label() also knows the sharp gestures ("# Minor").
    draw_hand_status(panel, 330, "RIGHT HAND: QUALITY", tracks["QUALITY"], GOLD, False,
                     quality_label)

    # ---- Chord table / gesture guide ----
    draw_library(panel, state["active_chord"], tracks["ROOT"].gesture,
                 tracks["QUALITY"].gesture)

    # ---- Footer ----
    # UPGRADE v3.1: added M (mute) and S (screenshot) to the key list.
    # UPGRADE v3.2: added E (evaluation); "SPACE" shortened to "SPC" to fit.
    draw_text(panel, "C Calib | E Eval | G Guide | D Debug | R Reset | SPC Replay | "
              "M Mute | S Shot | Q Quit", (18, 700), 0.33, MUTED)


def draw_guide_overlay(view):
    """Full-screen gesture guide drawn over the camera view (G key)."""

    # Darken the camera image: blend 85% dark colour + 15% camera.
    overlay = view.copy()
    cv2.rectangle(overlay, (0, 0), (view.shape[1], view.shape[0]), BACKGROUND, -1)
    cv2.addWeighted(overlay, 0.85, view, 0.15, 0, view)

    draw_text(view, "GESTURE GUIDE", (24, 40), 0.8, MINT, 2)
    draw_text(view, "Coloured bar = finger up, short bar = finger folded",
              (24, 66), 0.40, MUTED)

    # ---- Left column: root gestures ----
    draw_text(view, "LEFT HAND (screen-left) = ROOT NOTE", (24, 105), 0.5, BLUE, 2)
    for i, root in enumerate(ROOTS):
        y = 150 + i * 46
        draw_text(view, root, (34, y), 0.9, MINT, 2)
        draw_mini_hand(view, 74, y + 6, ROOT_TEMPLATES[root], mirrored=True, size=1.4)
        draw_text(view, ROOT_DESCRIPTIONS[root], (160, y - 4), 0.5, TEXT)

    # ---- Right column: quality gestures ----
    draw_text(view, "RIGHT HAND = QUALITY", (360, 105), 0.5, GOLD, 2)
    for i, quality in enumerate(QUALITIES):
        y = 150 + i * 46
        draw_mini_hand(view, 366, y + 6, QUALITY_TEMPLATES[quality], mirrored=False,
                       size=1.4)
        label = {"MAJOR": "Major", "MINOR": "Minor", "7TH": "7th"}[quality]
        draw_text(view, label, (452, y - 8), 0.55, TEXT, 2)
        draw_text(view, QUALITY_DESCRIPTIONS[quality], (452, y + 10), 0.38, MUTED)
    # UPGRADE v3.3: the right thumb used to "not matter". Now it is the
    # sharp switch, so explain it with an icon (index + thumb = # Major).
    draw_text(view, "Keep the thumb TUCKED for these.", (366, 296), 0.38, MUTED)
    draw_text(view, "SHARP (#): add the THUMB", (366, 330), 0.5, GOLD, 2)
    draw_mini_hand(view, 366, 380, QUALITY_TEMPLATES["#MAJOR"], mirrored=False, size=1.4)
    draw_text(view, "e.g. index + thumb", (452, 356), 0.40, TEXT)
    draw_text(view, "= # Major", (452, 376), 0.40, TEXT)
    draw_text(view, "E and B have no #: thumb ignored", (366, 404), 0.36, MUTED)

    draw_text(view, "Example: left 'index + middle' (E)", (366, 436), 0.40, TEXT)
    draw_text(view, "+ right 'index + middle' (Minor) = Em", (366, 456), 0.40, TEXT)
    draw_text(view, "Left fist (C) + right index + thumb = C#", (366, 478), 0.38, GOLD)

    # ---- Tips ----
    tips = [
        "TIPS FOR RELIABLE TRACKING",
        # UPGRADE v3.3: the number of poses is counted, not typed (now 7).
        f"- Press C once and do the {len(CALIBRATION_POSES)} calibration poses.",
        "- Palms facing the camera, hands 40-70 cm away.",
        "- Keep the hands apart; don't let them overlap.",
        "- Light from the front, not a bright window behind you.",
        "- Hold a gesture ~0.2 s; it locks in when the bars settle.",
    ]
    for i, line in enumerate(tips):
        # First line is a heading (gold, bold), the rest normal text.
        draw_text(view, line, (24, 500 + i * 26), 0.5 if i == 0 else 0.45,
                  GOLD if i == 0 else TEXT, 2 if i == 0 else 1)
    draw_text(view, "Press G to close", (24, 700), 0.45, MUTED)


def draw_debug(view, tracks):
    """Show the raw feature numbers and probability for every finger (D key).
    Useful to see WHY a finger is detected wrong."""
    y = 100
    for role in ROLE_NAMES:
        track = tracks[role]
        if track.features is None or not track.visible:
            continue
        draw_text(view, f"{role} HAND", (10, y), 0.40, GOLD, 1)
        y += 18
        for finger in FINGER_NAMES:
            labels = FEATURE_LABELS["THUMB" if finger == "THUMB" else "FINGER"]
            # Small numbers with 2 decimals, angles as whole numbers.
            values = " ".join(f"{l}:{v:.2f}" if abs(v) < 10 else f"{l}:{v:.0f}"
                              for l, v in zip(labels, track.features[finger]))
            p = track.probs[finger]
            color = SUCCESS if track.states[finger] else MUTED
            draw_text(view, f"{finger[:3]} p={p:.2f} {values}", (10, y), 0.36, color)
            y += 16
        y += 6


def prepare_camera_view(frame):
    """Fit the camera image into the left half without stretching it.
    Empty space around it is filled with a blurred copy of the image."""
    h, w = frame.shape[:2]
    scale = min(CAMERA_VIEW_WIDTH / w, DISPLAY_HEIGHT / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(frame, (nw, nh))

    # Blurred background stretched to fill the whole left half.
    #
    # UPGRADE v3.1 (speed): before, we blurred the FULL 640x720 image with
    # a huge 51x51 blur EVERY frame - that's a lot of work for something
    # that's supposed to look fuzzy anyway!
    # Trick: shrink the image to 1/8 size (80x90), blur that with a small
    # 7x7 blur, then stretch it back up. Stretching a tiny image makes it
    # blurry for free, so the result looks the same but costs ~50x less.
    small = cv2.resize(frame, (CAMERA_VIEW_WIDTH // 8, DISPLAY_HEIGHT // 8),
                       interpolation=cv2.INTER_AREA)   # INTER_AREA = best for shrinking
    small = cv2.GaussianBlur(small, (7, 7), 0)
    canvas = cv2.resize(small, (CAMERA_VIEW_WIDTH, DISPLAY_HEIGHT),
                        interpolation=cv2.INTER_LINEAR)  # LINEAR = smooth stretching

    # Paste the sharp image in the centre.
    x, y = (CAMERA_VIEW_WIDTH - nw) // 2, (DISPLAY_HEIGHT - nh) // 2
    canvas[y:y + nh, x:x + nw] = resized
    return canvas


def compose_display(frame, state, now):
    """Build the full 1280x720 window image: camera left, panel right."""
    display = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
    display[:, :CAMERA_VIEW_WIDTH] = prepare_camera_view(frame)
    view = display[:, :CAMERA_VIEW_WIDTH]   # drawing on 'view' draws on 'display'

    if state["show_guide"]:
        draw_guide_overlay(view)
    else:
        draw_text(view, "HAND TRACKING", (20, 35), 0.60, TEXT, 2)
        draw_text(view, "Coloured = raised | Red = folded | G = guide",
                  (20, 60), 0.40, MUTED)
        session = state["calibration"]
        if session["active"]:
            pose = CALIBRATION_POSES[min(session["phase"], len(CALIBRATION_POSES) - 1)]
            draw_text(view, f"CALIBRATION {session['phase'] + 1}/"
                      f"{len(CALIBRATION_POSES)}: {pose[0]}", (20, 88), 0.5, GOLD, 2)
        elif state["evaluation"]["active"]:
            # UPGRADE v3.2: a red "REC" label so it's obvious frames are being saved.
            evaluation = state["evaluation"]
            target = CHORDS[evaluation["trials"][min(evaluation["index"],
                                                     len(evaluation["trials"]) - 1)]]
            draw_text(view, f"REC  EVALUATION {evaluation['index'] + 1}/"
                      f"{len(evaluation['trials'])}: {target['display']}",
                      (20, 88), 0.5, DANGER, 2)
        elif state["debug"]:
            draw_debug(view, state["tracks"])

        # UPGRADE v3.1: FPS counter (frames per second) in the top-right.
        # The README always promised this but the code never drew it.
        # Green = smooth (25+), orange = OK (15-25), red = slow (<15).
        fps = state["fps"]
        fps_color = SUCCESS if fps >= 25 else WARNING if fps >= 15 else DANGER
        fps_text = f"{fps:.0f} FPS"
        draw_text(view, fps_text, (CAMERA_VIEW_WIDTH - 20 - text_width(fps_text, 0.5, 2), 35),
                  0.5, fps_color, 2)

    draw_panel(display, state, now)
    return display


# ============================================================
# 17. MAIN APPLICATION
# ============================================================

def open_camera(camera_index=CAMERA_INDEX):
    """Open the webcam. DirectShow (CAP_DSHOW) starts faster on Windows;
    if it fails we try OpenCV's default backend.

    UPGRADE v3.1: camera_index is now a parameter, so --camera 1 on the
    command line can choose a different webcam without editing the code."""
    camera = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(camera_index)
    if not camera.isOpened():
        raise RuntimeError(f"Could not open camera {camera_index}. "
                           "Try another one, e.g.: python HandChordAI.py --camera 1")
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)
    camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)   # small buffer = less delay
    return camera


def assign_roles(hands):
    """Decide which detected hand is ROOT and which is QUALITY.
    Two hands: left-most = ROOT, right-most = QUALITY.
    One hand:  it gets the role of the screen side it's on.
    Returns e.g. {"ROOT": hand, "QUALITY": hand}."""
    if len(hands) >= 2:
        return {"ROOT": hands[0], "QUALITY": hands[-1]}
    if len(hands) == 1:
        return {"ROOT" if hands[0][0] < 0.5 else "QUALITY": hands[0]}
    return {}


def save_screenshot(image):
    """UPGRADE v3.1: save the current window image as a PNG (S key).

    Files are named by date and time, e.g. screenshots/handchord_20261120_213015.png
    so a new screenshot never overwrites an old one.
    Returns the path on success, or None if saving failed."""
    SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)   # create folder if needed
    # time.strftime turns the current time into text: %Y=year %m=month %d=day,
    # %H=hour %M=minute %S=second.
    path = SCREENSHOT_DIR / f"handchord_{time.strftime('%Y%m%d_%H%M%S')}.png"
    # cv2.imwrite returns True/False instead of raising an error.
    return path if cv2.imwrite(str(path), image) else None


def window_was_closed():
    """UPGRADE v3.1: True if the user clicked the window's X button.

    Without this check, clicking X just hides the window and the next
    cv2.imshow() call opens it again - so the program never ends.
    WND_PROP_VISIBLE is 1 while the window is open and 0 after it closes."""
    try:
        return cv2.getWindowProperty(WINDOW_NAME, cv2.WND_PROP_VISIBLE) < 1
    except cv2.error:
        # Some OpenCV versions raise an error once the window is gone.
        return True


def parse_arguments():
    """UPGRADE v3.1: read options typed after the script name.

    Examples:
        python HandChordAI.py --camera 1
        python HandChordAI.py --mute
    argparse also creates --help automatically from the help= texts."""
    parser = argparse.ArgumentParser(
        description="HandChordAI - play guitar chords with hand gestures.")
    parser.add_argument("--camera", type=int, default=CAMERA_INDEX,
                        help=f"webcam number (default {CAMERA_INDEX}; try 1 or 2 "
                             "if you have more than one camera)")
    parser.add_argument("--mute", action="store_true",
                        help="start with the sound turned off (press M to unmute)")
    # ---- UPGRADE v3.2: evaluation options ----
    parser.add_argument("--participant", default="P00",
                        help="participant code written into evaluation files "
                             "(e.g. P01). Use codes, not real names, for privacy.")
    parser.add_argument("--eval-reps", type=int, default=EVAL_REPETITIONS,
                        help=f"how many times each chord is asked for in one "
                             f"evaluation session (default {EVAL_REPETITIONS})")
    parser.add_argument("--force-fallback", action="store_true",
                        help="ignore the saved calibration and use the built-in "
                             "rules (to compare the two in an evaluation)")
    arguments = parser.parse_args()
    # Protect against e.g. --eval-reps 0, which would make an empty session.
    if arguments.eval_reps < 1:
        parser.error("--eval-reps must be 1 or more")
    return arguments


def run_application(camera_index=CAMERA_INDEX, start_muted=False,
                    participant="P00", eval_reps=EVAL_REPETITIONS, force_fallback=False):
    """Set everything up, then run the frame loop until you quit.

    UPGRADE v3.1: the parameters come from the command line
    (see parse_arguments). Calling run_application() with no arguments
    still works exactly like before.
    UPGRADE v3.2: added participant, eval_reps and force_fallback."""

    print("=" * 48)
    print("HANDCHORD AI - VIRTUAL GUITAR v3.4")   # UPGRADE v3.4: version number
    print("=" * 48)

    # Make the .wav files before the camera starts (only the first time is slow).
    create_chord_files()

    # UPGRADE v3.4: start the held-chord player. If it can't start,
    # init_audio() prints why and the old winsound looping is used.
    if init_audio():
        print("Audio: chords ring for as long as you hold them.")
    elif winsound is None:
        print("WARNING: no audio available (no sounddevice, no winsound); "
              "playback is disabled.")

    # All the program's changing state lives in this one dictionary,
    # so it's easy to pass to the drawing functions.
    state = {
        "profiles": load_calibration(),     # None until you calibrate
        "tracks": {role: HandTrack(role) for role in ROLE_NAMES},
        "calibration": create_calibration_session(),
        "active_chord": None,               # chord shown on screen, e.g. "EMINOR"
        "playing_chord": None,              # chord currently making sound
        "show_guide": False,                # G key
        "debug": False,                     # D key
        "muted": start_muted,               # UPGRADE v3.1: M key / --mute
        "fps": 0.0,                         # UPGRADE v3.1: smoothed frames per second
        "evaluation": create_evaluation_session(),  # UPGRADE v3.2: E key
        "force_fallback": force_fallback,   # UPGRADE v3.2: --force-fallback
    }
    print("Personal calibration loaded." if state["profiles"]
          else "No calibration yet - press C in the window.")
    if force_fallback:
        print("--force-fallback: the calibration is ignored; built-in rules are used.")

    camera = open_camera(camera_index)
    detector = create_hand_detector()
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    tracks = state["tracks"]
    last_timestamp = 0             # MediaPipe needs increasing timestamps
    last_valid_chord_time = 0.0    # when we last saw a full valid chord
    failed_reads = 0               # UPGRADE v3.1: camera failures in a row
    previous_frame_time = None     # UPGRADE v3.1: used to measure FPS

    print("\nG = guide | C = calibrate | D = debug | R = reset | "
          "M = mute | S = screenshot | Q = quit")

    # try/finally makes sure the camera is released even if something crashes.
    try:
        while True:
            # ------------------------------------------------
            # 1. Read and mirror the camera frame
            # ------------------------------------------------
            ok, frame = camera.read()
            if not ok or frame is None:
                # UPGRADE v3.1: don't quit on the first bad frame.
                # Count failures in a row; only give up after many.
                failed_reads += 1
                if failed_reads >= CAMERA_READ_RETRIES:
                    print("Camera stopped sending frames - closing.")
                    break
                # waitKey(10) pauses 10 ms (gives the camera time to
                # recover) and keeps the window responsive meanwhile.
                cv2.waitKey(10)
                continue   # 'continue' = skip the rest and try again
            failed_reads = 0   # got a good frame -> reset the counter

            frame = cv2.flip(frame, 1)   # 1 = flip horizontally (mirror)
            now = time.monotonic()

            # UPGRADE v3.1: measure FPS.
            # 1 / (seconds since the last frame) = frames per second.
            # Then blend it into the old value so the number is steady.
            if previous_frame_time is not None:
                frame_seconds = max(now - previous_frame_time, 1e-6)  # never divide by 0
                instant_fps = 1.0 / frame_seconds
                if state["fps"] == 0.0:
                    state["fps"] = instant_fps          # first measurement
                else:
                    state["fps"] += FPS_SMOOTHING * (instant_fps - state["fps"])
            previous_frame_time = now

            # Timestamp in ms, forced to always go up by at least 1.
            timestamp = max(int(now * 1000), last_timestamp + 1)
            last_timestamp = timestamp

            # ------------------------------------------------
            # 2. Detect hands and give them roles
            # ------------------------------------------------
            hands = detect_hands(detector, frame, timestamp)
            roles = assign_roles(hands)

            # ------------------------------------------------
            # 3. Update both hand tracks
            # ------------------------------------------------
            for role in ROLE_NAMES:
                track = tracks[role]
                if role in roles:
                    # Use this hand's calibration if we have one.
                    # UPGRADE v3.2: ...unless --force-fallback was given.
                    use_calibration = state["profiles"] and not state["force_fallback"]
                    profile = state["profiles"][role] if use_calibration else None
                    track.update(roles[role], now, profile)
                else:
                    # Hand not visible this frame.
                    track.visible = False
                    # Gone for a while? Forget its history.
                    if now - track.last_seen > HAND_LOST_GRACE_SECONDS:
                        track.reset()

            session = state["calibration"]

            # ------------------------------------------------
            # 4a. Calibration mode
            # ------------------------------------------------
            if session["active"]:
                profiles = update_calibration(session, tracks, now)
                if profiles is not None:
                    # Finished! Use the new models from now on.
                    state["profiles"] = profiles
                    for track in tracks.values():
                        track.reset()

            # ------------------------------------------------
            # 4b. Both hands visible -> recognise the chord
            # ------------------------------------------------
            elif tracks["ROOT"].visible and tracks["QUALITY"].visible:
                root = tracks["ROOT"].vote(ROOT_TEMPLATES)
                quality = tracks["QUALITY"].vote(QUALITY_TEMPLATES)

                if root is not None and quality is not None:
                    # UPGRADE v3.3: chord_name() instead of root + quality.
                    # "E" + "MINOR" -> "EMINOR" like before, "C" + "#MAJOR" ->
                    # "C#MAJOR", and "E" + "#MINOR" -> "EMINOR" (no E#).
                    chord = chord_name(root, quality)
                    state["active_chord"] = chord
                    last_valid_chord_time = now

                    # Only restart the sound if the chord CHANGED.
                    if state["playing_chord"] != chord:
                        stop_chord()
                        play_chord(chord, state["muted"])   # v3.1: respects mute
                        state["playing_chord"] = chord

            # ------------------------------------------------
            # 4c. Only one hand -> still update its gesture for the UI
            # ------------------------------------------------
            else:
                for role in ROLE_NAMES:
                    if tracks[role].visible:
                        tracks[role].vote(
                            ROOT_TEMPLATES if role == "ROOT" else QUALITY_TEMPLATES)

            # ------------------------------------------------
            # 5. Stop the sound if no valid chord for a while
            # ------------------------------------------------
            if (state["playing_chord"] is not None
                    and now - last_valid_chord_time > RELEASE_GRACE_SECONDS):
                stop_chord()
                state["playing_chord"] = None

            # ------------------------------------------------
            # 5b. UPGRADE v3.2: log this frame if an evaluation is running.
            #     Done AFTER recognition, so the row contains this frame's
            #     final results.
            # ------------------------------------------------
            update_evaluation(state["evaluation"], state, now)

            # ------------------------------------------------
            # 6. Draw everything and show the window
            # ------------------------------------------------
            for role in ROLE_NAMES:
                if tracks[role].visible:
                    draw_hand_landmarks(frame, tracks[role], role)

            # UPGRADE v3.1: keep the finished image in a variable so the
            # S key can save exactly what you see on screen.
            display = compose_display(frame, state, now)
            cv2.imshow(WINDOW_NAME, display)

            # ------------------------------------------------
            # 7. Keyboard
            # ------------------------------------------------
            # waitKey(1) waits 1 ms for a key (also lets the window refresh).
            key = cv2.waitKey(1) & 0xFF
            # Convert to a lowercase letter so C and c both work.
            char = chr(key).lower() if key < 128 else ""

            evaluation = state["evaluation"]   # UPGRADE v3.2: short name for below

            if char == "c":
                if session["active"]:
                    cancel_calibration(session)
                    print("Calibration cancelled.")
                elif evaluation["active"]:
                    # UPGRADE v3.2: calibrating in the middle of an evaluation
                    # would change the classifier halfway through the data.
                    print("Stop the evaluation first (press E).")
                else:
                    # Silence everything and start the wizard.
                    stop_chord()
                    state["playing_chord"] = None
                    state["active_chord"] = None
                    state["show_guide"] = False
                    start_calibration(session, time.monotonic())

            elif char == "e":
                # UPGRADE v3.2: E = start / stop an evaluation session.
                if evaluation["active"]:
                    finish_evaluation(evaluation, completed=False)
                elif session["active"]:
                    print("Finish or cancel the calibration first (press C).")
                else:
                    # Which classifier the live pipeline uses - saved with the
                    # data so the analysis can compare the two conditions.
                    live = ("calibrated" if state["profiles"] and not state["force_fallback"]
                            else "fallback")
                    state["show_guide"] = False   # the guide would hide the camera view
                    evaluation["notice"] = ""
                    start_evaluation(evaluation, time.monotonic(), participant, live,
                                     eval_reps, camera_index)

            elif char == "g":
                state["show_guide"] = not state["show_guide"]

            elif char == "d":
                state["debug"] = not state["debug"]

            elif char == "r":
                for track in tracks.values():
                    track.reset()
                state["active_chord"] = None
                state["playing_chord"] = None
                last_valid_chord_time = 0.0
                stop_chord()
                print("Gesture and audio state reset.")

            elif key == 32 and state["active_chord"] and not session["active"]:
                # 32 = SPACE: replay the current chord from the start.
                stop_chord()
                play_chord(state["active_chord"], state["muted"])  # v3.1: respects mute
                state["playing_chord"] = state["active_chord"]
                last_valid_chord_time = now

            elif char == "m":
                # UPGRADE v3.1: M = mute / unmute.
                state["muted"] = not state["muted"]
                if state["muted"]:
                    stop_chord()   # silence right now
                    print("Sound muted.")
                else:
                    print("Sound on.")
                    # If a chord is being held right now, start hearing it
                    # immediately instead of waiting for the next chord change.
                    if state["playing_chord"]:
                        play_chord(state["playing_chord"])

            elif char == "s":
                # UPGRADE v3.1: S = save a screenshot of the whole window.
                path = save_screenshot(display)
                print(f"Screenshot saved: {path}" if path else "Screenshot failed.")

            elif char == "q" or key == 27:
                # 27 = ESC
                break

            # UPGRADE v3.1: quit when the window's X button was clicked.
            if window_was_closed():
                break

    finally:
        # Always clean up: stop sound, free the camera, close the window.
        print("Closing Virtual Guitar...")
        # UPGRADE v3.2: if you quit during an evaluation, close the file
        # properly so the frames recorded so far are not lost.
        if state["evaluation"]["active"]:
            finish_evaluation(state["evaluation"], completed=False)
        stop_chord()
        close_audio()   # UPGRADE v3.4: close the sound card stream
        camera.release()
        detector.close()
        cv2.destroyAllWindows()


# ============================================================
# 18. PROGRAM ENTRY POINT
# ============================================================
# This only runs when you start this file directly (python HandChordAI.py),
# not when another file imports it.
if __name__ == "__main__":
    try:
        # UPGRADE v3.1: read --camera / --mute and pass them on.
        arguments = parse_arguments()
        # UPGRADE v3.2: also pass on the evaluation options.
        run_application(camera_index=arguments.camera, start_muted=arguments.mute,
                        participant=arguments.participant,
                        eval_reps=arguments.eval_reps,
                        force_fallback=arguments.force_fallback)
    except KeyboardInterrupt:
        # Ctrl+C in the terminal = normal quit.
        print("\nApplication interrupted.")
        stop_chord()
        close_audio()   # UPGRADE v3.4
        cv2.destroyAllWindows()