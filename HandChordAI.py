"""
Two-hand virtual guitar with 21 chords and a split-screen interface.

Install:
    python -m pip install mediapipe opencv-python numpy

Run:
    python HandChordAI.py

Keys:
    Q      Quit
    D      Toggle diagnostics
    R      Reset gestures and stop audio
    SPACE  Replay the selected chord

Left hand selects the chord root.
Right hand selects Major, Minor or 7th.
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
# 1. SETTINGS
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720

CAMERA_VIEW_WIDTH = DISPLAY_WIDTH // 2
PANEL_LEFT = CAMERA_VIEW_WIDTH

SAMPLE_RATE = 44100
LOOP_DURATION = 2.0

MAX_HANDS = 2

HISTORY_SIZE = 9
REQUIRED_VOTES = 6

RELEASE_GRACE_SECONDS = 0.45

FINGER_EXTEND_THRESHOLD = 0.58
FINGER_BENT_THRESHOLD = 0.46

# Thumb recognition thresholds.
# Adjust these if your webcam has trouble detecting the thumb.

THUMB_ANGLE_ENTER = 135.0
THUMB_ANGLE_KEEP = 125.0

THUMB_SPAN_ENTER = 0.48
THUMB_SPAN_KEEP = 0.40

DETECTION_CONFIDENCE = 0.45
PRESENCE_CONFIDENCE = 0.45
TRACKING_CONFIDENCE = 0.45

BASE_DIR = Path(__file__).resolve().parent

SOUND_DIR = BASE_DIR / "sounds"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

SOUND_DIR.mkdir(exist_ok=True)

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

WINDOW_NAME = "AI Virtual Guitar"


# OpenCV uses BGR instead of RGB.

COLORS = {
    "background": (17, 22, 32),
    "panel": (25, 33, 47),
    "card": (34, 44, 60),
    "active": (37, 83, 62),

    "text": (245, 247, 252),
    "muted": (160, 175, 195),

    "mint": (105, 230, 185),
    "blue": (235, 155, 85),
    "gold": (65, 195, 255),

    "success": (90, 220, 135),
    "border": (63, 76, 96),
}


# ============================================================
# 2. CHORDS AND GESTURES
# ============================================================

# LEFT HAND:
#
# Thumb folded:
# 0 fingers = C
# 1 finger  = D
# 2 fingers = E
# 3 fingers = F
# 4 fingers = G
#
# Thumb extended:
# 0 fingers = A
# 1 finger  = B

ROOT_GESTURES = {
    (False, 0): "C",
    (False, 1): "D",
    (False, 2): "E",
    (False, 3): "F",
    (False, 4): "G",
    (True, 0): "A",
    (True, 1): "B",
}


# RIGHT HAND:
#
# 1 finger = Major
# 2 fingers = Minor
# 3 fingers = Dominant 7th

QUALITY_GESTURES = {
    1: "MAJOR",
    2: "MINOR",
    3: "7TH",
}


ROOTS = ["C", "D", "E", "F", "G", "A", "B"]

QUALITIES = [
    "MAJOR",
    "MINOR",
    "7TH",
]


# Semitone positions for the seven roots.

ROOT_SEMITONES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}


# Fundamental frequencies of the chord roots.

ROOT_FREQUENCIES = {
    "C": 261.63,
    "D": 293.66,
    "E": 164.81,
    "F": 174.61,
    "G": 196.00,
    "A": 220.00,
    "B": 246.94,
}


# Intervals used to construct each chord.

CHORD_INTERVALS = {
    "MAJOR": (0, 4, 7),
    "MINOR": (0, 3, 7),
    "7TH": (0, 4, 7, 10),
}


NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


# Guitar fret shapes, from low E string to high E string.

FRET_SHAPES = {

    ("C", "MAJOR"): "x32010",
    ("C", "MINOR"): "x35543",
    ("C", "7TH"): "x32310",

    ("D", "MAJOR"): "xx0232",
    ("D", "MINOR"): "xx0231",
    ("D", "7TH"): "xx0212",

    ("E", "MAJOR"): "022100",
    ("E", "MINOR"): "022000",
    ("E", "7TH"): "020100",

    ("F", "MAJOR"): "133211",
    ("F", "MINOR"): "133111",
    ("F", "7TH"): "131211",

    ("G", "MAJOR"): "320003",
    ("G", "MINOR"): "355333",
    ("G", "7TH"): "320001",

    ("A", "MAJOR"): "x02220",
    ("A", "MINOR"): "x02210",
    ("A", "7TH"): "x02020",

    ("B", "MAJOR"): "x24442",
    ("B", "MINOR"): "x24432",
    ("B", "7TH"): "x21202",
}


def build_chords():
    """
    Generate all 21 chords using their musical intervals.

    Each chord contains:
        Name
        Musical notes
        Frequencies
        Guitar fret shape
    """

    chords = {}

    for root in ROOTS:

        for quality in QUALITIES:

            intervals = CHORD_INTERVALS[quality]

            root_frequency = ROOT_FREQUENCIES[root]

            root_semitone = ROOT_SEMITONES[root]

            frequencies = [
                root_frequency * (2 ** (interval / 12))
                for interval in intervals
            ]

            note_names = [
                NOTE_NAMES[
                    (root_semitone + interval) % 12
                ]
                for interval in intervals
            ]

            if quality == "7TH":
                name = root + "7"
            else:
                name = f"{root} {quality}"

            chords[(root, quality)] = {
                "name": name,
                "notes": " - ".join(note_names),
                "frequencies": frequencies,
                "fret": FRET_SHAPES[(root, quality)],
            }

    return chords


CHORDS = build_chords()


# ============================================================
# 3. HAND LANDMARKS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17),
]


FINGER_LANDMARKS = {

    "INDEX": (5, 6, 7, 8),

    "MIDDLE": (9, 10, 11, 12),

    "RING": (13, 14, 15, 16),

    "PINKY": (17, 18, 19, 20),
}


def empty_finger_states():
    """Create the initial state for all five fingers."""

    return {
        name: False
        for name in ("THUMB", *FINGER_LANDMARKS)
    }


# ============================================================
# 4. MEDIAPIPE MODEL
# ============================================================

def download_hand_model():
    """Download the hand model if it is not already available."""

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000:
        return

    MODEL_PATH.unlink(missing_ok=True)

    temporary_path = MODEL_PATH.with_suffix(".task.part")

    request = urllib.request.Request(
        MODEL_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    try:

        print("Downloading hand-tracking model...")

        with urllib.request.urlopen(request, timeout=90) as response:

            with temporary_path.open("wb") as model_file:

                while True:

                    chunk = response.read(1024 * 1024)

                    if not chunk:
                        break

                    model_file.write(chunk)

        if temporary_path.stat().st_size <= 100000:

            raise RuntimeError(
                "The downloaded hand model is incomplete."
            )

        temporary_path.replace(MODEL_PATH)

        print("Hand model downloaded successfully.")

    except Exception:

        temporary_path.unlink(missing_ok=True)

        raise


def create_hand_detector():
    """Initialize MediaPipe Hand Landmarker."""

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
# 5. FINGER RECOGNITION
# ============================================================

def calculate_angle(point_a, point_b, point_c):
    """Calculate the angle at the middle point."""

    vector_a = point_a - point_b

    vector_c = point_c - point_b

    length_a = np.linalg.norm(vector_a)

    length_c = np.linalg.norm(vector_c)

    if length_a < 1e-6 or length_c < 1e-6:

        return 0.0

    cosine = np.dot(vector_a, vector_c) / (
        length_a * length_c
    )

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
    Recognize the thumb and four main fingers.

    The four main fingers use joint angles and extension ratios.
    The thumb uses its joint angle and distance from the palm.
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

    # Recognize index, middle, ring and pinky fingers.

    for name, indices in FINGER_LANDMARKS.items():

        mcp, pip, dip, tip = (
            points[index]
            for index in indices
        )

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

        tip_distance = np.linalg.norm(
            tip - mcp
        )

        pip_distance = np.linalg.norm(
            pip - mcp
        )

        extension_ratio = (
            tip_distance / max(pip_distance, 1e-6)
        )

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

        score = (
            0.40 * angle_score
            + 0.25 * dip_score
            + 0.35 * ratio_score
        )

        threshold = (

            FINGER_BENT_THRESHOLD
            if previous_states.get(name, False)

            else FINGER_EXTEND_THRESHOLD
        )

        extended = (
            score >= threshold
            and pip_angle >= 135
        )

        states[name] = bool(extended)

        diagnostics[name] = {
            "pip": float(pip_angle),
            "dip": float(dip_angle),
            "ratio": float(extension_ratio),
            "score": float(score),
            "extended": bool(extended),
        }

    # THUMB RECOGNITION
    #
    # Landmark 2 = thumb MCP joint
    # Landmark 3 = thumb IP joint
    # Landmark 4 = thumb tip
    # Landmark 5 = index MCP
    # Landmark 17 = pinky MCP

    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    palm_width = np.linalg.norm(
        points[5] - points[17]
    )

    thumb_span = np.linalg.norm(
        points[4] - points[5]
    )

    thumb_ratio = (
        thumb_span / max(palm_width, 1e-6)
    )

    # Hysteresis prevents the thumb state from flickering.

    if previous_states.get("THUMB", False):

        thumb_extended = (
            thumb_angle >= THUMB_ANGLE_KEEP
            and thumb_ratio >= THUMB_SPAN_KEEP
        )

    else:

        thumb_extended = (
            thumb_angle >= THUMB_ANGLE_ENTER
            and thumb_ratio >= THUMB_SPAN_ENTER
        )

    states["THUMB"] = bool(thumb_extended)

    diagnostics["THUMB"] = {
        "pip": float(thumb_angle),
        "dip": 0.0,
        "ratio": float(thumb_ratio),
        "score": float(thumb_ratio),
        "extended": bool(thumb_extended),
    }

    # Count the four main fingers separately from the thumb.

    count = sum(
        states[name]
        for name in FINGER_LANDMARKS
    )

    return count, states, diagnostics


# ============================================================
# 6. HAND DETECTION
# ============================================================

def detect_hands(detector, frame, timestamp):
    """Detect hands and sort them from screen-left to screen-right."""

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    rgb_frame = np.ascontiguousarray(
        rgb_frame
    )

    image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(
        image,
        timestamp
    )

    hands = []

    for index, image_landmarks in enumerate(
        result.hand_landmarks or []
    ):

        if index >= len(result.hand_world_landmarks or []):
            continue

        world_landmarks = result.hand_world_landmarks[index]

        center_x = float(
            np.mean([
                point.x
                for point in image_landmarks
            ])
        )

        hands.append(
            (
                center_x,
                image_landmarks,
                world_landmarks
            )
        )

    hands.sort(
        key=lambda hand: hand[0]
    )

    return hands


def draw_hand(frame, landmarks, states, role):
    """Draw the hand skeleton and highlight the fingertips."""

    height, width = frame.shape[:2]

    points = [

        (
            int(point.x * width),
            int(point.y * height)
        )

        for point in landmarks
    ]

    line_color = (
        COLORS["mint"]
        if role == "ROOT"
        else COLORS["blue"]
    )

    for start, end in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start],
            points[end],
            line_color,
            2,
            cv2.LINE_AA
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            3,
            (250, 250, 250),
            -1,
            cv2.LINE_AA
        )

    fingertips = {

        "THUMB": 4,
        "INDEX": 8,
        "MIDDLE": 12,
        "RING": 16,
        "PINKY": 20,
    }

    for name, index in fingertips.items():

        color = (
            COLORS["success"]
            if states.get(name, False)
            else (90, 105, 125)
        )

        cv2.circle(
            frame,
            points[index],
            8,
            color,
            2,
            cv2.LINE_AA
        )

    wrist_x, wrist_y = points[0]

    label_x = max(
        5,
        min(wrist_x - 45, width - 120)
    )

    label_y = max(
        20,
        min(wrist_y + 28, height - 10)
    )

    cv2.putText(
        frame,
        "ROOT HAND" if role == "ROOT" else "CHORD TYPE",
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        line_color,
        1,
        cv2.LINE_AA
    )


# ============================================================
# 7. AUDIO GENERATION
# ============================================================

def generate_chord(frequencies, filename):
    """Create a harmonically rich WAV sound that can be looped."""

    sample_count = int(
        SAMPLE_RATE * LOOP_DURATION
    )

    sample_indices = np.arange(
        sample_count,
        dtype=np.float64
    )

    audio = np.zeros(
        sample_count,
        dtype=np.float64
    )

    for frequency in frequencies:

        cycles = round(
            frequency * LOOP_DURATION
        )

        loop_frequency = (
            cycles / LOOP_DURATION
        )

        phase = (
            2 * np.pi
            * loop_frequency
            * sample_indices
            / SAMPLE_RATE
        )

        note = (
            np.sin(phase)
            + 0.30 * np.sin(2 * phase)
            + 0.12 * np.sin(3 * phase)
            + 0.05 * np.sin(4 * phase)
        )

        audio += (
            note / len(frequencies)
        )

    peak = np.max(
        np.abs(audio)
    )

    if peak > 0:

        audio = (
            audio / peak * 0.78
        )

    pcm = (
        audio * 32767
    ).astype("<i2")

    with wave.open(str(filename), "wb") as output:

        output.setnchannels(1)

        output.setsampwidth(2)

        output.setframerate(SAMPLE_RATE)

        output.writeframes(
            pcm.tobytes()
        )


def create_chord_files():
    """Generate WAV files for all 21 chords."""

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


def stop_chord():
    """Stop the currently playing chord."""

    winsound.PlaySound(
        None,
        winsound.SND_PURGE
    )


def play_chord(chord_key, chord_files):
    """Play a chord continuously until it is stopped."""

    filename = chord_files.get(chord_key)

    if filename is None:
        return

    stop_chord()

    winsound.PlaySound(
        str(filename),
        winsound.SND_FILENAME
        | winsound.SND_ASYNC
        | winsound.SND_LOOP
    )


# ============================================================
# 8. GESTURE STABILIZATION
# ============================================================

def stable_count(history):
    """Accept a gesture only when enough recent frames agree."""

    if len(history) < HISTORY_SIZE:
        return None

    value, votes = Counter(
        history
    ).most_common(1)[0]

    return (
        value
        if votes >= REQUIRED_VOTES
        else None
    )


def select_chord(
    root_count,
    quality_count,
    thumb_extended=False
):
    """Convert both hand gestures into a chord."""

    root = ROOT_GESTURES.get(
        (bool(thumb_extended), root_count)
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

    return key if key in CHORDS else None


# ============================================================
# 9. INTERFACE DRAWING
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
    """Draw a rounded rectangular panel."""

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


def draw_debug(canvas, state):
    """Display finger and thumb measurements for calibration."""

    overlay = canvas.copy()

    cv2.rectangle(
        overlay,
        (12, 58),
        (CAMERA_VIEW_WIDTH - 12, 280),
        COLORS["panel"],
        -1
    )

    cv2.addWeighted(
        overlay,
        0.86,
        canvas,
        0.14,
        0,
        canvas
    )

    y = 82

    for label, diagnostics in (

        ("LEFT", state["root_diagnostics"]),

        ("RIGHT", state["quality_diagnostics"]),

    ):

        if not diagnostics:
            continue

        for finger, info in diagnostics.items():

            message = (
                f"{label} {finger}: "
                f"score={info['score']:.2f} "
                f"angle={info['pip']:.0f} "
                f"ratio={info['ratio']:.2f}"
            )

            cv2.putText(
                canvas,
                message,
                (23, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.36,
                COLORS["text"],
                1,
                cv2.LINE_AA
            )

            y += 19


# ============================================================
# 10. SPLIT-SCREEN INTERFACE
# ============================================================

def compose_display(camera_frame, state, fps):
    """Show the camera on the left and the chord library on the right."""

    canvas = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    canvas[:] = COLORS["background"]

    # --------------------------------------------------------
    # LEFT HALF: CAMERA
    # --------------------------------------------------------

    source_height, source_width = camera_frame.shape[:2]

    # Fit the webcam image without stretching it.

    scale = min(
        CAMERA_VIEW_WIDTH / source_width,
        DISPLAY_HEIGHT / source_height
    )

    target_width = max(
        1,
        int(source_width * scale)
    )

    target_height = max(
        1,
        int(source_height * scale)
    )

    resized = cv2.resize(
        camera_frame,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA
    )

    # Create a blurred background to fill empty space.

    background_scale = max(
        CAMERA_VIEW_WIDTH / source_width,
        DISPLAY_HEIGHT / source_height
    )

    background_width = max(
        1,
        int(source_width * background_scale)
    )

    background_height = max(
        1,
        int(source_height * background_scale)
    )

    enlarged = cv2.resize(
        camera_frame,
        (background_width, background_height)
    )

    crop_x = max(
        0,
        (background_width - CAMERA_VIEW_WIDTH) // 2
    )

    crop_y = max(
        0,
        (background_height - DISPLAY_HEIGHT) // 2
    )

    camera_background = enlarged[
        crop_y:crop_y + DISPLAY_HEIGHT,
        crop_x:crop_x + CAMERA_VIEW_WIDTH
    ].copy()

    camera_background = cv2.GaussianBlur(
        camera_background,
        (0, 0),
        20
    )

    camera_background = (
        camera_background.astype(np.float32) * 0.32
    ).astype(np.uint8)

    offset_x = (
        CAMERA_VIEW_WIDTH - target_width
    ) // 2

    offset_y = (
        DISPLAY_HEIGHT - target_height
    ) // 2

    camera_background[
        offset_y:offset_y + target_height,
        offset_x:offset_x + target_width
    ] = resized

    canvas[:, :CAMERA_VIEW_WIDTH] = camera_background

    # Camera heading.

    cv2.rectangle(
        canvas,
        (14, 14),
        (250, 51),
        COLORS["panel"],
        -1
    )

    cv2.putText(
        canvas,
        "LIVE HAND TRACKING",
        (27, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLORS["mint"],
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        f"HANDS: {state['hand_count']} / 2    FPS: {fps:.0f}",
        (18, DISPLAY_HEIGHT - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        COLORS["text"],
        1,
        cv2.LINE_AA
    )

    if state["debug_mode"]:
        draw_debug(canvas, state)

    # --------------------------------------------------------
    # DIVIDER
    # --------------------------------------------------------

    cv2.line(
        canvas,
        (CAMERA_VIEW_WIDTH, 0),
        (CAMERA_VIEW_WIDTH, DISPLAY_HEIGHT),
        COLORS["mint"],
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # RIGHT HALF: CHORD LIBRARY
    # --------------------------------------------------------

    left = PANEL_LEFT
    right = DISPLAY_WIDTH

    cv2.rectangle(
        canvas,
        (left + 1, 0),
        (right, DISPLAY_HEIGHT),
        COLORS["background"],
        -1
    )

    cv2.putText(
        canvas,
        "VIRTUAL GUITAR",
        (left + 23, 39),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        COLORS["text"],
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "CHORD LIBRARY | 21 CHORDS",
        (left + 24, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLORS["muted"],
        1,
        cv2.LINE_AA
    )

    # Active chord card.

    draw_panel(
        canvas,
        left + 17,
        83,
        right - 17,
        205,
        (27, 45, 54),
        COLORS["mint"],
        12
    )

    cv2.putText(
        canvas,
        "NOW PLAYING",
        (left + 34, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        COLORS["mint"],
        1,
        cv2.LINE_AA
    )

    if state["active_chord"] is None:

        chord_name = "SHOW BOTH HANDS"

        chord_notes = "Choose a root with your left hand"

        chord_fret = "Choose Major / Minor / 7th with your right"

    else:

        info = CHORDS[state["active_chord"]]

        chord_name = info["name"]

        chord_notes = f"NOTES: {info['notes']}"

        chord_fret = f"GUITAR FRET: {info['fret']}"

    cv2.putText(
        canvas,
        chord_name,
        (left + 34, 151),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.90,
        COLORS["text"],
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        chord_notes,
        (left + 36, 176),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        COLORS["muted"],
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        chord_fret,
        (left + 36, 193),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        COLORS["gold"],
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "AVAILABLE CHORDS",
        (left + 22, 235),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        COLORS["text"],
        1,
        cv2.LINE_AA
    )

    # Seven rows, three chord-quality columns.

    grid_left = left + 17
    grid_right = right - 17

    grid_top = 266
    grid_bottom = 661

    root_width = 67
    gap = 6
    row_gap = 4

    row_height = (
        grid_bottom - grid_top - 6 * row_gap
    ) // 7

    chord_width = (
        grid_right - grid_left - root_width - 3 * gap
    ) // 3

    for column, quality in enumerate(QUALITIES):

        x = (
            grid_left
            + root_width
            + gap
            + column * (chord_width + gap)
        )

        cv2.putText(
            canvas,
            quality,
            (x + 5, 258),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            COLORS["muted"],
            1,
            cv2.LINE_AA
        )

    gesture_labels = {
        "C": "FIST",
        "D": "1",
        "E": "2",
        "F": "3",
        "G": "4",
        "A": "THUMB",
        "B": "TH+1",
    }

    for row, root in enumerate(ROOTS):

        y1 = (
            grid_top
            + row * (row_height + row_gap)
        )

        y2 = y1 + row_height

        draw_panel(
            canvas,
            grid_left,
            y1,
            grid_left + root_width - 2,
            y2,
            (42, 55, 72),
            COLORS["border"],
            6
        )

        cv2.putText(
            canvas,
            root,
            (grid_left + 10, y1 + 29),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.77,
            COLORS["mint"],
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            gesture_labels[root],
            (grid_left + 10, y1 + 47),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.28,
            COLORS["muted"],
            1,
            cv2.LINE_AA
        )

        for column, quality in enumerate(QUALITIES):

            key = (root, quality)

            chord = CHORDS[key]

            x1 = (
                grid_left
                + root_width
                + gap
                + column * (chord_width + gap)
            )

            x2 = x1 + chord_width

            selected = key == state["active_chord"]

            draw_panel(
                canvas,
                x1,
                y1,
                x2,
                y2,
                COLORS["active"] if selected else COLORS["card"],
                COLORS["mint"] if selected else COLORS["border"],
                6
            )

            cv2.putText(
                canvas,
                chord["name"],
                (x1 + 8, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (190, 255, 220) if selected else COLORS["text"],
                2 if selected else 1,
                cv2.LINE_AA
            )

    # Instructions.

    cv2.putText(
        canvas,
        "RIGHT HAND: 1 Major   2 Minor   3 Seventh",
        (left + 19, 689),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        COLORS["muted"],
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "Q Quit  |  D Debug  |  R Reset  |  SPACE Replay",
        (left + 19, 710),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.29,
        COLORS["muted"],
        1,
        cv2.LINE_AA
    )

    return canvas


# ============================================================
# 11. MAIN APPLICATION
# ============================================================

def run_application():

    camera = None
    detector = None

    playing_chord = None
    last_valid_chord_time = 0.0

    try:

        print("Preparing the 21 chord sounds...")

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
                "Could not open webcam. Check camera permissions."
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
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        print("Virtual Guitar is ready.")
        print("Left hand selects the root.")
        print("Right hand selects chord quality.")
        print("Q: quit | D: diagnostics | R: reset | SPACE: replay")

        # Gesture histories.

        root_history = deque(
            maxlen=HISTORY_SIZE
        )

        quality_history = deque(
            maxlen=HISTORY_SIZE
        )

        root_states = empty_finger_states()

        quality_states = empty_finger_states()

        previous_timestamp = -1

        previous_frame_time = time.perf_counter()

        fps_history = deque(maxlen=15)

        debug_mode = False

        # Main webcam loop.

        while True:

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

            timestamp = max(
                int(time.perf_counter() * 1000),
                previous_timestamp + 1
            )

            previous_timestamp = timestamp

            # Detect both hands.

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

            # ------------------------------------------------
            # PROCESS BOTH HANDS
            # ------------------------------------------------

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

                # Draw the hand skeletons.

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

                # Save the left-hand finger count and thumb state.

                root_history.append(
                    (
                        root_count,
                        root_states["THUMB"]
                    )
                )

                quality_history.append(
                    quality_count
                )

                # Stabilize both gestures.

                stable_root = stable_count(
                    root_history
                )

                stable_quality = stable_count(
                    quality_history
                )

                # Select a chord only when both gestures are stable.

                if (
                    stable_root is not None
                    and stable_quality is not None
                ):

                    stable_root_count, stable_thumb = stable_root

                    active_chord = select_chord(
                        stable_root_count,
                        stable_quality,
                        stable_thumb
                    )

            else:

                # Reset histories when both hands are not visible.

                root_history.clear()

                quality_history.clear()

                root_states = empty_finger_states()

                quality_states = empty_finger_states()

            # ------------------------------------------------
            # SUSTAINED AUDIO PLAYBACK
            # ------------------------------------------------

            now = time.monotonic()

            if active_chord is not None:

                last_valid_chord_time = now

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

                stop_chord()

                playing_chord = None

                print("Chord released.")

            # ------------------------------------------------
            # FPS CALCULATION
            # ------------------------------------------------

            frame_time = time.perf_counter()

            elapsed = (
                frame_time - previous_frame_time
            )

            previous_frame_time = frame_time

            if elapsed > 0:

                fps_history.append(
                    1.0 / elapsed
                )

            fps = (
                sum(fps_history) / len(fps_history)
                if fps_history
                else 0.0
            )

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

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

            display = compose_display(
                frame,
                state,
                fps
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # ------------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("d"):

                debug_mode = not debug_mode

            elif key == ord("r"):

                root_history.clear()
                quality_history.clear()

                root_states = empty_finger_states()
                quality_states = empty_finger_states()

                playing_chord = None
                last_valid_chord_time = 0.0

                stop_chord()

                print("Gesture recognition reset.")

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

        # Stop audio when the application closes.

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

        cv2.destroyAllWindows()

        print("Virtual Guitar closed.")


# ============================================================
# 12. APPLICATION ENTRY POINT
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