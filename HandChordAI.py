"""
AI Virtual Guitar
A two-hand gesture-controlled virtual guitar.

Requirements:
    python -m pip install mediapipe opencv-python numpy

Run:
    python HandChordAI.py

Controls:
    Q      Quit
    D      Toggle tracking diagnostics
    R      Reset gesture recognition and stop audio
    SPACE  Replay the currently selected chord

The screen-left hand selects the chord root.
The screen-right hand selects the chord quality.
Audio sustains while the chord gesture is held.
"""

from __future__ import annotations

import math
import time
import urllib.request
import wave
import winsound

from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# 1. CONFIGURATION
# ============================================================

PROJECT_NAME = "AI VIRTUAL GUITAR"

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

DISPLAY_WIDTH = 1280
CAMERA_VIEW_HEIGHT = 560
DASHBOARD_HEIGHT = 340

DISPLAY_HEIGHT = CAMERA_VIEW_HEIGHT + DASHBOARD_HEIGHT

SAMPLE_RATE = 44100
LOOP_DURATION = 2.0

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.45
PRESENCE_CONFIDENCE = 0.45
TRACKING_CONFIDENCE = 0.45

EXTENSION_THRESHOLD = 0.58
BENT_THRESHOLD = 0.46

HISTORY_SIZE = 9
REQUIRED_VOTES = 6

RELEASE_GRACE_SECONDS = 0.45

BASE_DIR = Path(__file__).resolve().parent

SOUND_DIR = BASE_DIR / "sounds"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

SOUND_DIR.mkdir(exist_ok=True)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

WINDOW_NAME = "AI Virtual Guitar - Studio"


# OpenCV uses BGR instead of RGB.

COLORS = {
    "background": (17, 22, 32),
    "panel": (20, 26, 38),
    "card": (32, 41, 56),
    "active": (37, 83, 62),

    "text": (245, 247, 252),
    "muted": (160, 175, 195),

    "mint": (105, 230, 185),
    "blue": (235, 155, 85),
    "gold": (65, 195, 255),

    "success": (90, 220, 135),
    "warning": (75, 180, 255),

    "border": (63, 76, 96),
}


# ============================================================
# 2. GESTURE MAPPING
# ============================================================

# The left hand selects the root note.
#
# 0 fingers = E
# 1 finger  = C
# 2 fingers = G
# 3 fingers = D
# 4 fingers = A

ROOT_GESTURES = {
    0: "E",
    1: "C",
    2: "G",
    3: "D",
    4: "A",
}


# The right hand selects the chord type.
#
# 1 finger = Major
# 2 fingers = Minor
# 3 fingers = Dominant 7th

QUALITY_GESTURES = {
    1: "MAJOR",
    2: "MINOR",
    3: "7TH",
}


# ============================================================
# 3. CHORD DEFINITIONS
# ============================================================

CHORDS = {

    # ---------------- C CHORDS ----------------

    ("C", "MAJOR"): {
        "name": "C MAJOR",
        "notes": "C - E - G",
        "frequencies": [261.63, 329.63, 392.00],
        "fret": "x32010",
    },

    ("C", "MINOR"): {
        "name": "C MINOR",
        "notes": "C - Eb - G",
        "frequencies": [261.63, 311.13, 392.00],
        "fret": "x35543",
    },

    ("C", "7TH"): {
        "name": "C7",
        "notes": "C - E - G - Bb",
        "frequencies": [261.63, 329.63, 392.00, 466.16],
        "fret": "x32310",
    },


    # ---------------- G CHORDS ----------------

    ("G", "MAJOR"): {
        "name": "G MAJOR",
        "notes": "G - B - D",
        "frequencies": [196.00, 246.94, 293.66],
        "fret": "320003",
    },

    ("G", "MINOR"): {
        "name": "G MINOR",
        "notes": "G - Bb - D",
        "frequencies": [196.00, 233.08, 293.66],
        "fret": "355333",
    },

    ("G", "7TH"): {
        "name": "G7",
        "notes": "G - B - D - F",
        "frequencies": [196.00, 246.94, 293.66, 349.23],
        "fret": "320001",
    },


    # ---------------- D CHORDS ----------------

    ("D", "MAJOR"): {
        "name": "D MAJOR",
        "notes": "D - F# - A",
        "frequencies": [293.66, 369.99, 440.00],
        "fret": "xx0232",
    },

    ("D", "MINOR"): {
        "name": "D MINOR",
        "notes": "D - F - A",
        "frequencies": [293.66, 349.23, 440.00],
        "fret": "xx0231",
    },

    ("D", "7TH"): {
        "name": "D7",
        "notes": "D - F# - A - C",
        "frequencies": [293.66, 369.99, 440.00, 523.25],
        "fret": "xx0212",
    },


    # ---------------- A CHORDS ----------------

    ("A", "MAJOR"): {
        "name": "A MAJOR",
        "notes": "A - C# - E",
        "frequencies": [220.00, 277.18, 329.63],
        "fret": "x02220",
    },

    ("A", "MINOR"): {
        "name": "A MINOR",
        "notes": "A - C - E",
        "frequencies": [220.00, 261.63, 329.63],
        "fret": "x02210",
    },

    ("A", "7TH"): {
        "name": "A7",
        "notes": "A - C# - E - G",
        "frequencies": [220.00, 277.18, 329.63, 392.00],
        "fret": "x02020",
    },


    # ---------------- E CHORDS ----------------

    ("E", "MAJOR"): {
        "name": "E MAJOR",
        "notes": "E - G# - B",
        "frequencies": [164.81, 207.65, 246.94],
        "fret": "022100",
    },

    ("E", "MINOR"): {
        "name": "E MINOR",
        "notes": "E - G - B",
        "frequencies": [164.81, 196.00, 246.94],
        "fret": "022000",
    },

    ("E", "7TH"): {
        "name": "E7",
        "notes": "E - G# - B - D",
        "frequencies": [164.81, 207.65, 246.94, 293.66],
        "fret": "020100",
    },
}


# ============================================================
# 4. HAND LANDMARK CONNECTIONS
# ============================================================

# MediaPipe detects 21 landmarks on each hand.
# These connections draw the hand skeleton.

HAND_CONNECTIONS = [

    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    (0, 17),
]


# Four fingers used for chord recognition.

FINGER_LANDMARKS = {

    "INDEX": (5, 6, 7, 8),

    "MIDDLE": (9, 10, 11, 12),

    "RING": (13, 14, 15, 16),

    "PINKY": (17, 18, 19, 20),
}


# ============================================================
# 5. MEDIAPIPE MODEL
# ============================================================

def download_hand_model():
    """
    Download the MediaPipe hand-tracking model
    if it is not already present.
    """

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000:
        return

    if MODEL_PATH.exists():
        MODEL_PATH.unlink()

    temporary_path = MODEL_PATH.with_suffix(".task.part")

    request = urllib.request.Request(
        MODEL_URL,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    try:

        print("Downloading hand-tracking model...")

        with urllib.request.urlopen(
            request,
            timeout=90
        ) as response:

            with temporary_path.open("wb") as model_file:

                while True:

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    model_file.write(chunk)

        if temporary_path.stat().st_size <= 100000:
            raise RuntimeError(
                "The downloaded model file is incomplete."
            )

        temporary_path.replace(MODEL_PATH)

        print("Hand model downloaded successfully.")

    except Exception:

        temporary_path.unlink(missing_ok=True)

        raise


def create_hand_detector():
    """
    Configure and initialize MediaPipe Hand Landmarker.
    """

    download_hand_model()

    options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_hands=MAX_HANDS,

        min_hand_detection_confidence=DETECTION_CONFIDENCE,

        min_hand_presence_confidence=PRESENCE_CONFIDENCE,

        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# 6. HAND GEOMETRY AND FINGER RECOGNITION
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    """
    Calculate the angle between three points.

    The middle point is the vertex of the angle.
    """

    vector_a = point_a - point_b

    vector_c = point_c - point_b

    length_a = np.linalg.norm(vector_a)

    length_c = np.linalg.norm(vector_c)

    if length_a < 1e-6 or length_c < 1e-6:

        return 0.0

    cosine = np.dot(
        vector_a,
        vector_c
    ) / (length_a * length_c)

    cosine = np.clip(
        cosine,
        -1.0,
        1.0
    )

    return math.degrees(
        math.acos(cosine)
    )


def recognize_fingers(world_landmarks, previous_states):
    """
    Estimate which fingers are extended.

    Uses joint angles, finger distances and
    extension ratios.

    Returns:
        number of extended fingers
        individual finger states
        diagnostic measurements
    """

    points = np.array(

        [
            [point.x, point.y, point.z]
            for point in world_landmarks
        ],

        dtype=np.float64,
    )

    states = {}

    diagnostics = {}

    for name, indices in FINGER_LANDMARKS.items():

        mcp_index, pip_index, dip_index, tip_index = indices

        mcp = points[mcp_index]

        pip = points[pip_index]

        dip = points[dip_index]

        tip = points[tip_index]


        # Calculate joint angles.

        pip_angle = calculate_angle(
            mcp,
            pip,
            dip
        )

        dip_angle = calculate_angle(
            pip,
            dip,
            tip
        )


        # Calculate the distance between fingertips
        # and the base of each finger.

        tip_distance = np.linalg.norm(
            tip - mcp
        )

        pip_distance = np.linalg.norm(
            pip - mcp
        )

        extension_ratio = (
            tip_distance / max(pip_distance, 1e-6)
        )


        # Convert measurements into normalized scores.

        angle_score = np.clip(
            (pip_angle - 115) / 65,
            0,
            1
        )

        dip_score = np.clip(
            (dip_angle - 110) / 70,
            0,
            1
        )

        ratio_score = np.clip(
            (extension_ratio - 0.90) / 0.65,
            0,
            1
        )


        # Combine the measurements.

        score = (
            0.40 * angle_score
            + 0.25 * dip_score
            + 0.35 * ratio_score
        )


        # Apply hysteresis to reduce flickering.

        if previous_states.get(name, False):

            threshold = BENT_THRESHOLD

        else:

            threshold = EXTENSION_THRESHOLD


        is_extended = (
            score >= threshold
            and pip_angle >= 135
        )

        states[name] = bool(is_extended)

        diagnostics[name] = {

            "pip": pip_angle,

            "dip": dip_angle,

            "ratio": extension_ratio,

            "score": score,

            "extended": bool(is_extended),
        }


    extended_count = sum(states.values())

    return extended_count, states, diagnostics


# ============================================================
# 7. DRAW HAND SKELETON
# ============================================================

def draw_hand(frame, landmarks, states, role):
    """
    Draw hand landmarks and connections
    on the webcam frame.
    """

    height, width = frame.shape[:2]

    points = [

        (
            int(point.x * width),
            int(point.y * height)
        )

        for point in landmarks
    ]


    if role == "ROOT":

        line_color = COLORS["mint"]

    else:

        line_color = COLORS["blue"]


    # Draw connections between landmarks.

    for start, end in HAND_CONNECTIONS:

        cv2.line(

            frame,

            points[start],

            points[end],

            line_color,

            2,

            cv2.LINE_AA,
        )


    # Draw all landmarks.

    for point in points:

        cv2.circle(

            frame,

            point,

            3,

            (250, 250, 250),

            -1,

            cv2.LINE_AA,
        )


    # Highlight fingertips.

    fingertips = {

        "INDEX": 8,

        "MIDDLE": 12,

        "RING": 16,

        "PINKY": 20,
    }


    for name, index in fingertips.items():

        if states[name]:

            color = COLORS["success"]

        else:

            color = (90, 105, 125)


        cv2.circle(

            frame,

            points[index],

            8,

            color,

            2,

            cv2.LINE_AA,
        )


    # Add hand role label.

    wrist_x, wrist_y = points[0]

    label_x = max(
        5,
        min(wrist_x - 45, width - 120)
    )

    label_y = max(
        20,
        min(wrist_y + 28, height - 10)
    )

    label = (
        "ROOT HAND"
        if role == "ROOT"
        else "CHORD TYPE"
    )

    cv2.putText(

        frame,

        label,

        (label_x, label_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        line_color,

        1,

        cv2.LINE_AA,
    )


# ============================================================
# 8. DETECT AND ORDER HANDS
# ============================================================

def detect_hands(detector, frame, timestamp):
    """
    Detect hands using MediaPipe.

    Hands are sorted according to their horizontal
    screen position.

    The leftmost hand becomes the root selector.
    The rightmost hand becomes the chord selector.
    """

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb_frame = np.ascontiguousarray(
        rgb_frame
    )

    mp_image = mp.Image(

        image_format=mp.ImageFormat.SRGB,

        data=rgb_frame,
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    hands = []

    for index, image_landmarks in enumerate(
        result.hand_landmarks or []
    ):

        if index >= len(
            result.hand_world_landmarks or []
        ):

            continue


        world_landmarks = (
            result.hand_world_landmarks[index]
        )

        center_x = float(

            np.mean(

                [
                    point.x
                    for point in image_landmarks
                ]
            )
        )


        hands.append(

            (
                center_x,
                image_landmarks,
                world_landmarks
            )
        )


    hands.sort(
        key=lambda item: item[0]
    )

    return hands


# ============================================================
# 9. AUDIO GENERATION
# ============================================================

def generate_chord(frequencies, filename):
    """
    Generate a WAV file containing a chord.

    Several harmonics are combined to create
    a richer synthesized guitar-like tone.

    The generated file is designed to loop.
    """

    sample_count = int(
        SAMPLE_RATE * LOOP_DURATION
    )

    sample_indices = np.arange(
        sample_count,
        dtype=np.float64
    )

    mixed_audio = np.zeros(
        sample_count,
        dtype=np.float64
    )


    for frequency in frequencies:

        # Use an integer number of cycles
        # to help the audio loop smoothly.

        cycles = round(
            frequency * LOOP_DURATION
        )

        loop_frequency = (
            cycles / LOOP_DURATION
        )

        phase = (

            2
            * np.pi
            * loop_frequency
            * sample_indices
            / SAMPLE_RATE
        )


        # Fundamental frequency and harmonics.

        note = (

            np.sin(phase)

            + 0.30 * np.sin(2 * phase)

            + 0.12 * np.sin(3 * phase)

            + 0.05 * np.sin(4 * phase)
        )

        mixed_audio += (
            note / len(frequencies)
        )


    # Normalize volume.

    peak = np.max(
        np.abs(mixed_audio)
    )

    if peak > 0:

        mixed_audio = (
            mixed_audio / peak * 0.78
        )


    # Convert floating-point samples
    # into 16-bit PCM audio.

    pcm = (
        mixed_audio * 32767
    ).astype("<i2")


    with wave.open(
        str(filename),
        "wb"
    ) as audio_file:

        audio_file.setnchannels(1)

        audio_file.setsampwidth(2)

        audio_file.setframerate(
            SAMPLE_RATE
        )

        audio_file.writeframes(
            pcm.tobytes()
        )


# ============================================================
# 10. GENERATE ALL CHORD AUDIO FILES
# ============================================================

def create_chord_files():
    """
    Generate audio files for every chord.

    Returns a dictionary containing
    each chord and its WAV file path.
    """

    chord_files = {}

    for key, chord in CHORDS.items():

        filename = (

            SOUND_DIR
            / f"{chord['name'].replace(' ', '_')}.wav"
        )

        generate_chord(
            chord["frequencies"],
            filename
        )

        chord_files[key] = filename


    return chord_files


# ============================================================
# 11. AUDIO PLAYBACK
# ============================================================

def stop_chord():
    """
    Immediately stop the currently playing audio.
    """

    winsound.PlaySound(
        None,
        winsound.SND_PURGE
    )


def play_chord(chord_key, chord_files):
    """
    Play a chord continuously.

    The audio loops until another chord is selected
    or the hand gesture is released.
    """

    filename = chord_files.get(
        chord_key
    )

    if filename is None:

        return


    stop_chord()

    winsound.PlaySound(

        str(filename),

        winsound.SND_FILENAME
        | winsound.SND_ASYNC
        | winsound.SND_LOOP,
    )


# ============================================================
# 12. GESTURE STABILIZATION
# ============================================================

def stable_count(history):
    """
    Return the most frequently detected finger count.

    The result is accepted only when enough recent
    frames agree with one another.
    """

    if len(history) < HISTORY_SIZE:

        return None


    count, votes = Counter(
        history
    ).most_common(1)[0]


    if votes >= REQUIRED_VOTES:

        return count


    return None


# ============================================================
# 13. SELECT A CHORD
# ============================================================

def select_chord(root_count, quality_count):
    """
    Convert two finger counts into a chord key.

    Left hand:
        Selects root note.

    Right hand:
        Selects chord quality.
    """

    root = ROOT_GESTURES.get(
        root_count
    )

    quality = QUALITY_GESTURES.get(
        quality_count
    )


    if root is None or quality is None:

        return None


    key = (
        root,
        quality
    )

    if key not in CHORDS:

        return None


    return key


# ============================================================
# 14. INTERFACE DRAWING HELPERS
# ============================================================

def draw_panel(
    image,
    x1,
    y1,
    x2,
    y2,
    fill,
    border=None,
    radius=10
):
    """
    Draw a rounded rectangular panel.
    """

    if x2 <= x1 or y2 <= y1:

        return


    radius = max(

        1,

        min(
            radius,
            (x2 - x1) // 2,
            (y2 - y1) // 2
        )
    )


    cv2.rectangle(

        image,

        (x1 + radius, y1),

        (x2 - radius, y2),

        fill,

        -1
    )


    cv2.rectangle(

        image,

        (x1, y1 + radius),

        (x2, y2 - radius),

        fill,

        -1
    )


    corners = [

        (x1 + radius, y1 + radius),

        (x2 - radius, y1 + radius),

        (x1 + radius, y2 - radius),

        (x2 - radius, y2 - radius),
    ]


    for center in corners:

        cv2.circle(

            image,

            center,

            radius,

            fill,

            -1
        )


    if border is not None:

        cv2.rectangle(

            image,

            (x1 + radius, y1),

            (x2 - radius, y2),

            border,

            1
        )

        cv2.rectangle(

            image,

            (x1, y1 + radius),

            (x2, y2 - radius),

            border,

            1
        )


# ============================================================
# 15. FINGER INDICATORS
# ============================================================

def draw_finger_indicators(frame, x, y, states):
    """
    Display four small indicators for the fingers.
    """

    fingers = [

        ("INDEX", "I"),

        ("MIDDLE", "M"),

        ("RING", "R"),

        ("PINKY", "P"),
    ]


    for index, (full_name, short_name) in enumerate(
        fingers
    ):

        left = x + index * 49

        active = bool(
            states and states.get(full_name, False)
        )


        if active:

            fill = (50, 103, 78)

            text_color = COLORS["success"]

        else:

            fill = (42, 51, 67)

            text_color = COLORS["muted"]


        draw_panel(

            frame,

            left,

            y,

            left + 43,

            y + 21,

            fill,

            radius=6
        )


        cv2.putText(

            frame,

            short_name,

            (left + 16, y + 15),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.40,

            text_color,

            1,

            cv2.LINE_AA
        )


# ============================================================
# 16. HAND STATUS CARDS
# ============================================================

def draw_hand_card(
    frame,
    bounds,
    title,
    subtitle,
    count,
    states,
    accent
):
    """
    Draw a card showing the current hand status.
    """

    x1, y1, x2, y2 = bounds

    draw_panel(

        frame,

        x1,
        y1,
        x2,
        y2,

        COLORS["card"],

        COLORS["border"],

        12
    )


    cv2.line(

        frame,

        (x1 + 12, y1 + 9),

        (x1 + 12, y2 - 9),

        accent,

        3,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        title,

        (x1 + 25, y1 + 23),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.42,

        accent,

        1,

        cv2.LINE_AA
    )


    if count is not None:

        count_text = (
            f"{count} fingers detected"
        )

        count_color = COLORS["text"]

    else:

        count_text = "Waiting for hand..."

        count_color = COLORS["muted"]


    cv2.putText(

        frame,

        count_text,

        (x1 + 25, y1 + 48),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.55,

        count_color,

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        subtitle,

        (x1 + 25, y1 + 68),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.33,

        COLORS["muted"],

        1,

        cv2.LINE_AA
    )


    draw_finger_indicators(

        frame,

        x1 + 25,

        y1 + 74,

        states
    )


# ============================================================
# 17. CURRENT CHORD DISPLAY
# ============================================================

def draw_current_chord(frame, chord_key, top):
    """
    Display the selected chord, its notes
    and its guitar fret shape.
    """

    width = frame.shape[1]

    x1 = 18

    y1 = top + 151

    x2 = width - 18

    y2 = top + 230


    draw_panel(

        frame,

        x1,
        y1,
        x2,
        y2,

        (27, 37, 51),

        (71, 95, 109),

        12
    )


    cv2.putText(

        frame,

        "CURRENT CHORD",

        (x1 + 20, y1 + 20),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.36,

        COLORS["mint"],

        1,

        cv2.LINE_AA
    )


    if chord_key is None:

        cv2.putText(

            frame,

            "Waiting for both hands",

            (x1 + 20, y1 + 53),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.72,

            COLORS["text"],

            2,

            cv2.LINE_AA
        )


        cv2.putText(

            frame,

            "Left hand selects the root; right hand selects the chord type.",

            (x1 + 20, y1 + 73),

            cv2.FONT_HERSHEY_SIMPLEX,

            0.37,

            COLORS["muted"],

            1,

            cv2.LINE_AA
        )

        return


    chord = CHORDS[chord_key]


    cv2.putText(

        frame,

        chord["name"],

        (x1 + 20, y1 + 56),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.82,

        COLORS["text"],

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        f"NOTES: {chord['notes']}",

        (x1 + 22, y1 + 75),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.38,

        COLORS["muted"],

        1,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "GUITAR FRET SHAPE",

        (x2 - 215, y1 + 24),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        COLORS["gold"],

        1,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        chord["fret"],

        (x2 - 215, y1 + 62),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.82,

        COLORS["gold"],

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        frame,

        "Low E to high E",

        (x2 - 215, y1 + 77),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.30,

        COLORS["muted"],

        1,

        cv2.LINE_AA
    )


# ============================================================
# 18. CHORD LIBRARY
# ============================================================

def draw_chord_library(frame, active_chord, top):
    """
    Display all 15 available chords.

    The currently selected chord is highlighted.
    """

    width = frame.shape[1]

    x1 = 18

    y1 = top + 236

    x2 = width - 18

    y2 = top + 326


    draw_panel(

        frame,

        x1,
        y1,
        x2,
        y2,

        COLORS["background"],

        COLORS["border"],

        9
    )


    roots = [
        "C",
        "G",
        "D",
        "A",
        "E"
    ]

    qualities = [
        "MAJOR",
        "MINOR",
        "7TH"
    ]


    root_counts = {

        "C": 1,

        "G": 2,

        "D": 3,

        "A": 4,

        "E": 0
    }


    quality_counts = {

        "MAJOR": 1,

        "MINOR": 2,

        "7TH": 3
    }


    grid_x = x1 + 8

    grid_y = y1 + 6

    cell_width = (
        (x2 - x1 - 16) // 5
    )

    cell_height = (
        (y2 - y1 - 12) // 3
    )


    for row, quality in enumerate(qualities):

        for column, root in enumerate(roots):

            key = (
                root,
                quality
            )

            chord = CHORDS[key]


            left = (
                grid_x
                + column * cell_width
                + 2
            )

            right = (
                grid_x
                + (column + 1) * cell_width
                - 2
            )

            upper = (
                grid_y
                + row * cell_height
                + 1
            )

            lower = (
                grid_y
                + (row + 1) * cell_height
                - 1
            )


            selected = (
                key == active_chord
            )


            if selected:

                fill = COLORS["active"]

                border = COLORS["mint"]

                name_color = (190, 255, 220)

            else:

                fill = COLORS["card"]

                border = COLORS["border"]

                name_color = COLORS["text"]


            draw_panel(

                frame,

                left,
                upper,
                right,
                lower,

                fill,

                border,

                5
            )


            cv2.putText(

                frame,

                chord["name"],

                (left + 6, upper + 11),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.30,

                name_color,

                1,

                cv2.LINE_AA
            )


            control = (

                f"LEFT {root_counts[root]} "
                f"/ RIGHT {quality_counts[quality]}"
            )


            cv2.putText(

                frame,

                control,

                (left + 6, upper + 22),

                cv2.FONT_HERSHEY_SIMPLEX,

                0.25,

                COLORS["blue"],

                1,

                cv2.LINE_AA
            )


# ============================================================
# 19. COMPLETE DISPLAY
# ============================================================

def compose_display(camera_frame, state, fps):
    """
    Combine the webcam and the dashboard.

    The camera keeps its original proportions.
    The dashboard is placed underneath it.
    """

    canvas = np.zeros(

        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3
        ),

        dtype=np.uint8
    )

    canvas[:] = COLORS["background"]


    source_height, source_width = (
        camera_frame.shape[:2]
    )


    # Scale the camera image without stretching it.

    scale = min(

        DISPLAY_WIDTH / source_width,

        CAMERA_VIEW_HEIGHT / source_height
    )


    target_width = int(
        source_width * scale
    )

    target_height = int(
        source_height * scale
    )


    resized = cv2.resize(

        camera_frame,

        (target_width, target_height),

        interpolation=cv2.INTER_AREA
    )


    # Create a blurred background to fill
    # any empty side areas.

    background_scale = max(

        DISPLAY_WIDTH / source_width,

        CAMERA_VIEW_HEIGHT / source_height
    )


    background_width = int(
        source_width * background_scale
    )

    background_height = int(
        source_height * background_scale
    )


    enlarged = cv2.resize(

        camera_frame,

        (background_width, background_height)
    )


    crop_x = (
        background_width - DISPLAY_WIDTH
    ) // 2

    crop_y = (
        background_height - CAMERA_VIEW_HEIGHT
    ) // 2


    camera_background = enlarged[

        crop_y:crop_y + CAMERA_VIEW_HEIGHT,

        crop_x:crop_x + DISPLAY_WIDTH

    ].copy()


    camera_background = cv2.GaussianBlur(

        camera_background,

        (0, 0),

        22
    )


    camera_background = (

        camera_background.astype(np.float32)
        * 0.30

    ).astype(np.uint8)


    offset_x = (
        DISPLAY_WIDTH - target_width
    ) // 2

    offset_y = (
        CAMERA_VIEW_HEIGHT - target_height
    ) // 2


    camera_background[

        offset_y:offset_y + target_height,

        offset_x:offset_x + target_width

    ] = resized


    canvas[:CAMERA_VIEW_HEIGHT] = (
        camera_background
    )


    # ---------------- DASHBOARD HEADER ----------------

    top = CAMERA_VIEW_HEIGHT

    cv2.rectangle(

        canvas,

        (0, top),

        (DISPLAY_WIDTH, top + 39),

        COLORS["panel"],

        -1
    )


    cv2.line(

        canvas,

        (0, top),

        (DISPLAY_WIDTH, top),

        COLORS["mint"],

        2,

        cv2.LINE_AA
    )


    cv2.circle(

        canvas,

        (22, top + 20),

        7,

        COLORS["mint"],

        -1,

        cv2.LINE_AA
    )


    cv2.putText(

        canvas,

        PROJECT_NAME,

        (40, top + 25),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.48,

        COLORS["text"],

        2,

        cv2.LINE_AA
    )


    cv2.putText(

        canvas,

        "TWO-HAND PERFORMANCE MODE",

        (265, top + 24),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.34,

        COLORS["muted"],

        1,

        cv2.LINE_AA
    )


    hands_text = (
        f"{state['hand_count']}/2 HANDS TRACKED"
    )


    if state["hand_count"] == 2:

        hands_color = COLORS["success"]

    else:

        hands_color = COLORS["warning"]


    cv2.putText(

        canvas,

        hands_text,

        (DISPLAY_WIDTH - 180, top + 24),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.35,

        hands_color,

        1,

        cv2.LINE_AA
    )


    # ---------------- HAND STATUS CARDS ----------------

    margin = 18

    gap = 14

    card_width = (

        DISPLAY_WIDTH
        - 2 * margin
        - gap

    ) // 2


    card_top = top + 45

    card_bottom = top + 143


    draw_hand_card(

        canvas,

        (

            margin,

            card_top,

            margin + card_width,

            card_bottom
        ),

        "01  ROOT SELECTOR - SCREEN LEFT",

        "Fist E  |  1 C  |  2 G  |  3 D  |  4 A",

        state["root_count"],

        state["root_states"],

        COLORS["mint"]
    )


    right_left = (
        margin + card_width + gap
    )


    draw_hand_card(

        canvas,

        (

            right_left,

            card_top,

            DISPLAY_WIDTH - margin,

            card_bottom
        ),

        "02  CHORD QUALITY - SCREEN RIGHT",

        "1 Major  |  2 Minor  |  3 Dominant 7th",

        state["quality_count"],

        state["quality_states"],

        COLORS["blue"]
    )


    # ---------------- CURRENT CHORD ----------------

    draw_current_chord(

        canvas,

        state["active_chord"],

        top
    )


    # ---------------- CHORD LIBRARY ----------------

    draw_chord_library(

        canvas,

        state["active_chord"],

        top
    )


    # ---------------- FOOTER ----------------

    footer_y = DISPLAY_HEIGHT - 7


    cv2.putText(

        canvas,

        "Q QUIT     D DIAGNOSTICS     R RESET     SPACE REPLAY",

        (18, footer_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.29,

        COLORS["muted"],

        1,

        cv2.LINE_AA
    )


    cv2.putText(

        canvas,

        f"{fps:.0f} FPS",

        (DISPLAY_WIDTH - 75, footer_y),

        cv2.FONT_HERSHEY_SIMPLEX,

        0.29,

        COLORS["gold"],

        1,

        cv2.LINE_AA
    )


    # ---------------- DEBUG INFORMATION ----------------

    if state["debug_mode"]:

        overlay = canvas.copy()


        cv2.rectangle(

            overlay,

            (DISPLAY_WIDTH - 265, 12),

            (DISPLAY_WIDTH - 8, 145),

            COLORS["panel"],

            -1
        )


        cv2.addWeighted(

            overlay,

            0.90,

            canvas,

            0.10,

            0,

            canvas
        )


        y = 32


        for label, diagnostics in (

            ("LEFT", state["root_diagnostics"]),

            ("RIGHT", state["quality_diagnostics"])

        ):

            if not diagnostics:

                continue


            for finger, info in diagnostics.items():

                message = (

                    f"{label} {finger}: "
                    f"{info['score']:.2f} "
                    f"PIP {info['pip']:.0f}"
                )


                cv2.putText(

                    canvas,

                    message,

                    (DISPLAY_WIDTH - 253, y),

                    cv2.FONT_HERSHEY_SIMPLEX,

                    0.29,

                    COLORS["text"],

                    1,

                    cv2.LINE_AA
                )


                y += 15


    return canvas


# ============================================================
# 20. MAIN APPLICATION
# ============================================================

def run_application():
    """
    Initialize the webcam, MediaPipe detector,
    chord audio and main application loop.
    """

    camera = None

    detector = None

    chord_files = {}

    playing_chord = None

    last_valid_chord_time = 0.0


    try:

        print("\nAI Virtual Guitar")

        print("-----------------")

        print("Preparing chord audio...")


        # Generate all chord audio files.

        chord_files = create_chord_files()


        # Open the webcam.

        camera = cv2.VideoCapture(
            CAMERA_INDEX,
            cv2.CAP_DSHOW
        )


        if not camera.isOpened():

            camera = cv2.VideoCapture(
                CAMERA_INDEX
            )


        if not camera.isOpened():

            raise RuntimeError(
                "Could not open webcam. "
                "Check camera access and CAMERA_INDEX."
            )


        camera.set(

            cv2.CAP_PROP_FRAME_WIDTH,

            CAMERA_WIDTH
        )


        camera.set(

            cv2.CAP_PROP_FRAME_HEIGHT,

            CAMERA_HEIGHT
        )


        camera.set(

            cv2.CAP_PROP_BUFFERSIZE,

            1
        )


        # Initialize MediaPipe.

        detector = create_hand_detector()


        cv2.namedWindow(

            WINDOW_NAME,

            cv2.WINDOW_NORMAL
        )


        cv2.resizeWindow(

            WINDOW_NAME,

            1050,

            738
        )


        print("\nReady.")

        print("Screen-left hand: chord root.")

        print("Screen-right hand: chord quality.")

        print(
            "Q: quit | D: diagnostics | "
            "R: reset | SPACE: replay"
        )


        # Store recent finger counts.

        root_history = deque(
            maxlen=HISTORY_SIZE
        )

        quality_history = deque(
            maxlen=HISTORY_SIZE
        )


        # Initial finger states.

        root_states = {

            name: False

            for name in FINGER_LANDMARKS
        }


        quality_states = {

            name: False

            for name in FINGER_LANDMARKS
        }


        previous_timestamp = -1

        previous_frame_time = time.perf_counter()

        fps_history = deque(maxlen=15)

        debug_mode = False

        active_chord = None


        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            # Read webcam frame.

            success, frame = camera.read()


            if not success:

                raise RuntimeError(
                    "Could not read a frame from the webcam."
                )


            # Mirror the frame for natural interaction.

            frame = cv2.flip(
                frame,
                1
            )


            # MediaPipe VIDEO mode requires
            # an increasing timestamp.

            timestamp = max(

                int(time.perf_counter() * 1000),

                previous_timestamp + 1
            )

            previous_timestamp = timestamp


            # Detect hands.

            hands = detect_hands(

                detector,

                frame,

                timestamp
            )


            root_count = None

            quality_count = None

            root_diagnostics = None

            quality_diagnostics = None

            active_chord = None


            # =================================================
            # PROCESS BOTH HANDS
            # =================================================

            if len(hands) >= 2:

                # Hands are sorted from left to right.

                _, root_image, root_world = hands[0]

                _, quality_image, quality_world = hands[1]


                # Recognize fingers on the left hand.

                (

                    root_count,

                    root_states,

                    root_diagnostics

                ) = recognize_fingers(

                    root_world,

                    root_states
                )


                # Recognize fingers on the right hand.

                (

                    quality_count,

                    quality_states,

                    quality_diagnostics

                ) = recognize_fingers(

                    quality_world,

                    quality_states
                )


                # Draw both hand skeletons.

                draw_hand(

                    frame,

                    root_image,

                    root_states,

                    "ROOT"
                )


                draw_hand(

                    frame,

                    quality_image,

                    quality_states,

                    "QUALITY"
                )


                # Store recent finger counts.

                root_history.append(
                    root_count
                )

                quality_history.append(
                    quality_count
                )


                # Stabilize recognition.

                stable_root = stable_count(
                    root_history
                )

                stable_quality = stable_count(
                    quality_history
                )


                # Select a chord when both hands
                # have stable gesture counts.

                if (

                    stable_root is not None

                    and stable_quality is not None

                ):

                    active_chord = select_chord(

                        stable_root,

                        stable_quality
                    )


            else:

                # Reset recognition when both hands
                # are no longer available.

                root_history.clear()

                quality_history.clear()


                root_states = {

                    name: False

                    for name in FINGER_LANDMARKS
                }


                quality_states = {

                    name: False

                    for name in FINGER_LANDMARKS
                }


            # =================================================
            # SUSTAIN AND RELEASE AUDIO
            # =================================================

            now = time.monotonic()


            if active_chord is not None:

                # Keep track of the last valid gesture.

                last_valid_chord_time = now


                # Change audio only when the chord changes.

                if active_chord != playing_chord:

                    play_chord(

                        active_chord,

                        chord_files
                    )


                    playing_chord = active_chord


                    print(

                        "Playing:",

                        CHORDS[active_chord]["name"]
                    )


            elif (

                playing_chord is not None

                and now - last_valid_chord_time
                >= RELEASE_GRACE_SECONDS

            ):

                # Stop audio after the gesture
                # has been missing for long enough.

                stop_chord()

                print("Chord released.")

                playing_chord = None


            # =================================================
            # FPS CALCULATION
            # =================================================

            frame_time = time.perf_counter()

            elapsed = (
                frame_time - previous_frame_time
            )

            previous_frame_time = frame_time


            if elapsed > 0:

                fps_history.append(
                    1.0 / elapsed
                )


            if fps_history:

                fps = (
                    sum(fps_history)
                    / len(fps_history)
                )

            else:

                fps = 0.0


            # =================================================
            # DISPLAY STATE
            # =================================================

            state = {

                "hand_count": len(hands),

                "root_count": root_count,

                "quality_count": quality_count,


                "root_states": (

                    root_states
                    if len(hands) >= 2
                    else None
                ),


                "quality_states": (

                    quality_states
                    if len(hands) >= 2
                    else None
                ),


                "active_chord": (

                    active_chord
                    if active_chord is not None
                    else playing_chord
                ),


                "root_diagnostics": root_diagnostics,

                "quality_diagnostics": quality_diagnostics,

                "debug_mode": debug_mode,
            }


            # Build and show the interface.

            display = compose_display(

                frame,

                state,

                fps
            )


            cv2.imshow(

                WINDOW_NAME,

                display
            )


            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = cv2.waitKey(1) & 0xFF


            # Quit.

            if key == ord("q"):

                break


            # Toggle diagnostics.

            if key == ord("d"):

                debug_mode = not debug_mode


            # Reset gesture recognition.

            elif key == ord("r"):

                root_history.clear()

                quality_history.clear()


                root_states = {

                    name: False

                    for name in FINGER_LANDMARKS
                }


                quality_states = {

                    name: False

                    for name in FINGER_LANDMARKS
                }


                active_chord = None

                playing_chord = None

                last_valid_chord_time = 0.0


                stop_chord()


                print(
                    "Gesture recognition reset; audio stopped."
                )


            # Replay the currently selected chord.

            elif key == 32:

                chord_to_replay = (
                    active_chord or playing_chord
                )


                if chord_to_replay is not None:

                    play_chord(

                        chord_to_replay,

                        chord_files
                    )


                    playing_chord = chord_to_replay

                    last_valid_chord_time = time.monotonic()


    finally:

        # Stop any audio when the application closes.

        try:

            stop_chord()

        except Exception:

            pass


        # Release the webcam.

        if camera is not None:

            camera.release()


        # Close MediaPipe.

        if detector is not None:

            detector.close()


        # Close all OpenCV windows.

        cv2.destroyAllWindows()


        print("AI Virtual Guitar closed.")


# ============================================================
# 21. APPLICATION ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        run_application()


    except KeyboardInterrupt:

        print("\nApplication interrupted.")


    except Exception as error:

        print(
            f"\nAPPLICATION ERROR: "
            f"{type(error).__name__}: {error}"
        )