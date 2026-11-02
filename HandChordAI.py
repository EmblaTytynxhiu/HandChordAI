import cv2
import numpy as np
import mediapipe as mp
import math
import time
import wave
import urllib.request
import winsound

from pathlib import Path
from collections import deque, Counter


# ============================================================
# 1. PROJECT CONFIGURATION
# ============================================================

PROJECT_NAME = "AI VIRTUAL GUITAR"

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Fixed display dimensions: approximately 16:9
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 760

SAMPLE_RATE = 44100
CHORD_DURATION = 1.4

MAX_HANDS = 2

DETECTION_CONFIDENCE = 0.45
PRESENCE_CONFIDENCE = 0.45
TRACKING_CONFIDENCE = 0.45

EXTENSION_THRESHOLD = 0.58
BENT_THRESHOLD = 0.46

HISTORY_SIZE = 9
REQUIRED_VOTES = 6

CHORD_COOLDOWN = 0.65

BASE_DIR = Path(__file__).resolve().parent

SOUND_DIR = BASE_DIR / "sounds"
SOUND_DIR.mkdir(exist_ok=True)

MODEL_PATH = BASE_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/"
    "mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

WINDOW_NAME = "AI Virtual Guitar - Studio"


# ============================================================
# 2. INTERFACE COLORS - BGR FORMAT
# ============================================================

BG_MAIN = (17, 22, 32)
BG_PANEL = (20, 26, 38)
BG_CARD = (32, 41, 56)
BG_CARD_ACTIVE = (37, 83, 62)

TEXT_MAIN = (245, 247, 252)
TEXT_MUTED = (160, 175, 195)

ACCENT_MINT = (105, 230, 185)
ACCENT_BLUE = (235, 155, 85)
ACCENT_GOLD = (65, 195, 255)

SUCCESS = (90, 220, 135)
WARNING = (75, 180, 255)

BORDER = (65, 77, 96)


# ============================================================
# 3. TWO-HAND GESTURE CONTROLS
# ============================================================

# SCREEN-LEFT HAND:
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
    4: "A"
}


# SCREEN-RIGHT HAND:
# 1 finger = Major
# 2 fingers = Minor
# 3 fingers = Dominant 7th

QUALITY_GESTURES = {
    1: "MAJOR",
    2: "MINOR",
    3: "7TH"
}


# ============================================================
# 4. CHORD LIBRARY
# ============================================================

CHORDS = {

    ("C", "MAJOR"): {
        "name": "C MAJOR",
        "notes": "C - E - G",
        "frequencies": [261.63, 329.63, 392.00],
        "fret": "x32010"
    },

    ("C", "MINOR"): {
        "name": "C MINOR",
        "notes": "C - Eb - G",
        "frequencies": [261.63, 311.13, 392.00],
        "fret": "x35543"
    },

    ("C", "7TH"): {
        "name": "C7",
        "notes": "C - E - G - Bb",
        "frequencies": [261.63, 329.63, 392.00, 466.16],
        "fret": "x32310"
    },

    ("G", "MAJOR"): {
        "name": "G MAJOR",
        "notes": "G - B - D",
        "frequencies": [196.00, 246.94, 293.66],
        "fret": "320003"
    },

    ("G", "MINOR"): {
        "name": "G MINOR",
        "notes": "G - Bb - D",
        "frequencies": [196.00, 233.08, 293.66],
        "fret": "355333"
    },

    ("G", "7TH"): {
        "name": "G7",
        "notes": "G - B - D - F",
        "frequencies": [196.00, 246.94, 293.66, 349.23],
        "fret": "320001"
    },

    ("D", "MAJOR"): {
        "name": "D MAJOR",
        "notes": "D - F# - A",
        "frequencies": [293.66, 369.99, 440.00],
        "fret": "xx0232"
    },

    ("D", "MINOR"): {
        "name": "D MINOR",
        "notes": "D - F - A",
        "frequencies": [293.66, 349.23, 440.00],
        "fret": "xx0231"
    },

    ("D", "7TH"): {
        "name": "D7",
        "notes": "D - F# - A - C",
        "frequencies": [293.66, 369.99, 440.00, 523.25],
        "fret": "xx0212"
    },

    ("A", "MAJOR"): {
        "name": "A MAJOR",
        "notes": "A - C# - E",
        "frequencies": [220.00, 277.18, 329.63],
        "fret": "x02220"
    },

    ("A", "MINOR"): {
        "name": "A MINOR",
        "notes": "A - C - E",
        "frequencies": [220.00, 261.63, 329.63],
        "fret": "x02210"
    },

    ("A", "7TH"): {
        "name": "A7",
        "notes": "A - C# - E - G",
        "frequencies": [220.00, 277.18, 329.63, 392.00],
        "fret": "x02020"
    },

    ("E", "MAJOR"): {
        "name": "E MAJOR",
        "notes": "E - G# - B",
        "frequencies": [164.81, 207.65, 246.94],
        "fret": "022100"
    },

    ("E", "MINOR"): {
        "name": "E MINOR",
        "notes": "E - G - B",
        "frequencies": [164.81, 196.00, 246.94],
        "fret": "022000"
    },

    ("E", "7TH"): {
        "name": "E7",
        "notes": "E - G# - B - D",
        "frequencies": [164.81, 207.65, 246.94, 293.66],
        "fret": "020100"
    }
}


# ============================================================
# 5. HAND LANDMARK DEFINITIONS
# ============================================================

HAND_CONNECTIONS = [

    (0, 1), (1, 2), (2, 3), (3, 4),

    (0, 5), (5, 6), (6, 7), (7, 8),

    (5, 9), (9, 10), (10, 11), (11, 12),

    (9, 13), (13, 14), (14, 15), (15, 16),

    (13, 17), (17, 18), (18, 19), (19, 20),

    (0, 17)
]


FINGER_LANDMARKS = {

    "INDEX": (5, 6, 7, 8),

    "MIDDLE": (9, 10, 11, 12),

    "RING": (13, 14, 15, 16),

    "PINKY": (17, 18, 19, 20)

}


# ============================================================
# 6. DOWNLOAD HAND MODEL
# ============================================================

def download_hand_model():

    if MODEL_PATH.exists():

        if MODEL_PATH.stat().st_size > 100000:

            return

        MODEL_PATH.unlink()

    print("Downloading hand detection model...")

    temporary_file = Path(
        str(MODEL_PATH) + ".part"
    )

    try:

        request = urllib.request.Request(
            MODEL_URL,
            headers={
                "User-Agent": "Mozilla/5.0"
            }
        )

        with urllib.request.urlopen(
            request,
            timeout=90
        ) as response:

            with open(
                temporary_file,
                "wb"
            ) as output:

                while True:

                    data = response.read(1024 * 1024)

                    if not data:

                        break

                    output.write(data)

        if temporary_file.stat().st_size < 100000:

            raise RuntimeError("Incomplete model file.")

        temporary_file.replace(MODEL_PATH)

        print("Hand model downloaded.")

    except Exception as error:

        if temporary_file.exists():

            temporary_file.unlink()

        raise RuntimeError(
            "Could not download the hand model. "
            "Check your internet connection."
        ) from error


# ============================================================
# 7. INITIALIZE MEDIAPIPE
# ============================================================

def create_hand_detector():

    download_hand_model()

    options = mp.tasks.vision.HandLandmarkerOptions(

        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        num_hands=MAX_HANDS,

        min_hand_detection_confidence=DETECTION_CONFIDENCE,

        min_hand_presence_confidence=PRESENCE_CONFIDENCE,

        min_tracking_confidence=TRACKING_CONFIDENCE

    )

    detector = (
        mp.tasks.vision.HandLandmarker
        .create_from_options(options)
    )

    print("Two-hand tracking initialized.")

    return detector


# ============================================================
# 8. FINGER RECOGNITION
# ============================================================

def calculate_angle(point_a, point_b, point_c):

    vector_a = point_a - point_b

    vector_c = point_c - point_b

    length_a = np.linalg.norm(vector_a)

    length_c = np.linalg.norm(vector_c)

    if length_a < 0.000001 or length_c < 0.000001:

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

    points = np.array(

        [
            [landmark.x, landmark.y, landmark.z]

            for landmark in world_landmarks
        ],

        dtype=np.float64

    )

    states = {}

    diagnostics = {}

    for finger_name, indices in FINGER_LANDMARKS.items():

        mcp_index, pip_index, dip_index, tip_index = indices

        mcp = points[mcp_index]

        pip = points[pip_index]

        dip = points[dip_index]

        tip = points[tip_index]

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

        ratio = tip_distance / max(
            pip_distance,
            0.000001
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
            (ratio - 0.90) / 0.65,
            0,
            1
        )

        score = (
            0.40 * angle_score
            + 0.25 * dip_score
            + 0.35 * ratio_score
        )

        was_extended = previous_states.get(
            finger_name,
            False
        )

        threshold = (
            BENT_THRESHOLD
            if was_extended
            else EXTENSION_THRESHOLD
        )

        is_extended = (
            score >= threshold
            and pip_angle >= 135
        )

        states[finger_name] = bool(is_extended)

        diagnostics[finger_name] = {
            "pip": pip_angle,
            "dip": dip_angle,
            "ratio": ratio,
            "score": score,
            "extended": is_extended
        }

    finger_count = sum(
        states.values()
    )

    return finger_count, states, diagnostics


# ============================================================
# 9. DRAW HAND SKELETON
# ============================================================

def draw_hand(frame, image_landmarks, states, role):

    height, width = frame.shape[:2]

    points = []

    for landmark in image_landmarks:

        x = int(landmark.x * width)

        y = int(landmark.y * height)

        points.append((x, y))

    if role == "ROOT":

        line_color = ACCENT_MINT

        label = "ROOT HAND"

    else:

        line_color = ACCENT_BLUE

        label = "CHORD TYPE"

    for start_index, end_index in HAND_CONNECTIONS:

        cv2.line(
            frame,
            points[start_index],
            points[end_index],
            line_color,
            2,
            cv2.LINE_AA
        )

    for point in points:

        cv2.circle(
            frame,
            point,
            3,
            (255, 255, 255),
            -1,
            cv2.LINE_AA
        )

    fingertips = {
        "INDEX": 8,
        "MIDDLE": 12,
        "RING": 16,
        "PINKY": 20
    }

    for name, index in fingertips.items():

        color = (
            SUCCESS
            if states[name]
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
        min(wrist_y + 29, height - 10)
    )

    cv2.putText(
        frame,
        label,
        (label_x, label_y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        line_color,
        1,
        cv2.LINE_AA
    )


# ============================================================
# 10. AUDIO GENERATION
# ============================================================

def generate_chord(frequencies, filename):

    total_samples = int(
        SAMPLE_RATE * CHORD_DURATION
    )

    mixed_audio = np.zeros(
        total_samples,
        dtype=np.float64
    )

    strum_delay = 0.035

    for note_index, frequency in enumerate(frequencies):

        start = int(
            note_index * strum_delay * SAMPLE_RATE
        )

        remaining = total_samples - start

        t = np.arange(
            remaining,
            dtype=np.float64
        ) / SAMPLE_RATE

        fundamental = np.sin(
            2 * np.pi * frequency * t
        )

        harmonic_2 = 0.30 * np.sin(
            2 * np.pi * frequency * 2 * t
        )

        harmonic_3 = 0.12 * np.sin(
            2 * np.pi * frequency * 3 * t
        )

        harmonic_4 = 0.05 * np.sin(
            2 * np.pi * frequency * 4 * t
        )

        attack = 1 - np.exp(-65 * t)

        decay = np.exp(-2.5 * t)

        envelope = attack * decay

        note = (
            fundamental
            + harmonic_2
            + harmonic_3
            + harmonic_4
        ) * envelope

        mixed_audio[start:] += note * 0.25

    peak = np.max(
        np.abs(mixed_audio)
    )

    if peak > 0:

        mixed_audio /= peak

        mixed_audio *= 0.85

    fade_length = int(
        SAMPLE_RATE * 0.08
    )

    mixed_audio[-fade_length:] *= np.linspace(
        1.0,
        0.0,
        fade_length
    )

    pcm = (
        mixed_audio * 32767
    ).astype("<i2")

    with wave.open(
        str(filename),
        "wb"
    ) as audio:

        audio.setnchannels(1)

        audio.setsampwidth(2)

        audio.setframerate(SAMPLE_RATE)

        audio.writeframes(
            pcm.tobytes()
        )


# ============================================================
# 11. CREATE AND PLAY CHORDS
# ============================================================

def create_chord_files():

    chord_files = {}

    print("\nGenerating 15 guitar chords...\n")

    for key, chord in CHORDS.items():

        filename = SOUND_DIR / (
            chord["name"].replace(" ", "_") + ".wav"
        )

        generate_chord(
            chord["frequencies"],
            filename
        )

        chord_files[key] = filename

        print("Created:", chord["name"])

    return chord_files


def play_chord(chord_key, chord_files):

    filename = chord_files.get(chord_key)

    if filename is not None:

        winsound.PlaySound(
            str(filename),
            winsound.SND_FILENAME | winsound.SND_ASYNC
        )


# ============================================================
# 12. GESTURE STABILIZATION
# ============================================================

def get_stable_count(history):

    if len(history) < HISTORY_SIZE:

        return None

    count, votes = Counter(
        history
    ).most_common(1)[0]

    if votes >= REQUIRED_VOTES:

        return count

    return None


def select_chord(root_count, quality_count):

    root = ROOT_GESTURES.get(root_count)

    quality = QUALITY_GESTURES.get(quality_count)

    if root is None or quality is None:

        return None

    key = (root, quality)

    return key if key in CHORDS else None


# ============================================================
# 13. ROUNDED PANEL HELPER
# ============================================================

def draw_panel(
    image,
    x1,
    y1,
    x2,
    y2,
    color,
    border=None,
    radius=9
):

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
        color,
        -1
    )

    cv2.rectangle(
        image,
        (x1, y1 + radius),
        (x2, y2 - radius),
        color,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y1 + radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x1 + radius, y2 - radius),
        radius,
        color,
        -1
    )

    cv2.circle(
        image,
        (x2 - radius, y2 - radius),
        radius,
        color,
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
# 14. DRAW FINGER INDICATORS
# ============================================================

def draw_finger_pills(frame, x, y, states):

    names = [
        ("INDEX", "I"),
        ("MIDDLE", "M"),
        ("RING", "R"),
        ("PINKY", "P")
    ]

    pill_width = 43
    pill_height = 20
    gap = 6

    for index, (full_name, short_name) in enumerate(names):

        px = x + index * (pill_width + gap)

        active = (
            states is not None
            and states.get(full_name, False)
        )

        fill = (
            (50, 103, 78)
            if active
            else (42, 51, 67)
        )

        text_color = (
            SUCCESS
            if active
            else TEXT_MUTED
        )

        draw_panel(
            frame,
            px,
            y,
            px + pill_width,
            y + pill_height,
            fill,
            None,
            6
        )

        cv2.putText(
            frame,
            short_name,
            (px + 16, y + 14),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            text_color,
            1,
            cv2.LINE_AA
        )


# ============================================================
# 15. DRAW HAND STATUS CARDS
# ============================================================

def draw_hand_card(
    frame,
    x1,
    y1,
    x2,
    y2,
    title,
    subtitle,
    count,
    states,
    accent
):

    draw_panel(
        frame,
        x1,
        y1,
        x2,
        y2,
        BG_CARD,
        BORDER,
        10
    )

    cv2.line(
        frame,
        (x1 + 11, y1 + 9),
        (x1 + 11, y2 - 9),
        accent,
        3,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        title,
        (x1 + 24, y1 + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        accent,
        1,
        cv2.LINE_AA
    )

    if count is None:

        count_text = "Waiting for hand..."

        count_color = TEXT_MUTED

    else:

        count_text = f"{count} fingers detected"

        count_color = TEXT_MAIN

    cv2.putText(
        frame,
        count_text,
        (x1 + 24, y1 + 49),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.56,
        count_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        subtitle,
        (x1 + 24, y1 + 69),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        TEXT_MUTED,
        1,
        cv2.LINE_AA
    )

    draw_finger_pills(
        frame,
        x1 + 24,
        y1 + 75,
        states
    )


# ============================================================
# 16. CURRENT CHORD HERO PANEL
# ============================================================

def draw_hero_card(frame, active_chord, panel_top):

    width = frame.shape[1]

    x1 = 18
    y1 = panel_top + 149

    x2 = width - 18
    y2 = panel_top + 247

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
        "CURRENTLY SELECTED",
        (x1 + 20, y1 + 23),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        ACCENT_MINT,
        1,
        cv2.LINE_AA
    )

    if active_chord is None:

        cv2.putText(
            frame,
            "Waiting for both hands",
            (x1 + 20, y1 + 59),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            TEXT_MAIN,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "Select a root with your left hand and chord quality with your right hand.",
            (x1 + 20, y1 + 81),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            TEXT_MUTED,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            "15 CHORDS",
            (x2 - 145, y1 + 63),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            ACCENT_GOLD,
            1,
            cv2.LINE_AA
        )

        return

    chord = CHORDS[active_chord]

    cv2.putText(
        frame,
        chord["name"],
        (x1 + 20, y1 + 58),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        TEXT_MAIN,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "NOTES: " + chord["notes"],
        (x1 + 22, y1 + 81),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.39,
        TEXT_MUTED,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "GUITAR FRET SHAPE",
        (x2 - 220, y1 + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.35,
        ACCENT_GOLD,
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        chord["fret"],
        (x2 - 220, y1 + 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.85,
        ACCENT_GOLD,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        frame,
        "Low E to high E",
        (x2 - 220, y1 + 82),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        TEXT_MUTED,
        1,
        cv2.LINE_AA
    )


# ============================================================
# 17. DRAW CHORD LIBRARY
# ============================================================

def draw_chord_library(frame, active_chord, panel_top):

    width = frame.shape[1]

    x1 = 18
    y1 = panel_top + 255

    x2 = width - 18
    y2 = panel_top + 397

    draw_panel(
        frame,
        x1,
        y1,
        x2,
        y2,
        BG_MAIN,
        BORDER,
        10
    )

    cv2.putText(
        frame,
        "CHORD LIBRARY",
        (x1 + 14, y1 + 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        TEXT_MAIN,
        1,
        cv2.LINE_AA
    )

    root_order = ["C", "G", "D", "A", "E"]

    quality_order = [
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

    grid_x = x1 + 9
    grid_y = y1 + 25

    grid_width = x2 - x1 - 18
    grid_height = y2 - grid_y - 5

    cell_width = grid_width // 5
    cell_height = grid_height // 3

    for row, quality in enumerate(quality_order):

        for column, root in enumerate(root_order):

            key = (root, quality)

            chord = CHORDS[key]

            cx1 = grid_x + column * cell_width + 2

            cx2 = grid_x + (column + 1) * cell_width - 2

            cy1 = grid_y + row * cell_height + 2

            cy2 = grid_y + (row + 1) * cell_height - 2

            active = key == active_chord

            if active:

                card_color = BG_CARD_ACTIVE

                card_border = ACCENT_MINT

                name_color = (190, 255, 220)

            else:

                card_color = BG_CARD

                card_border = (48, 61, 79)

                name_color = TEXT_MAIN

            draw_panel(
                frame,
                cx1,
                cy1,
                cx2,
                cy2,
                card_color,
                card_border,
                6
            )

            cv2.putText(
                frame,
                chord["name"],
                (cx1 + 7, cy1 + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                name_color,
                1,
                cv2.LINE_AA
            )

            gesture = (
                f"LEFT {root_counts[root]} / RIGHT {quality_counts[quality]}"
            )

            cv2.putText(
                frame,
                gesture,
                (cx1 + 7, cy1 + 29),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.29,
                ACCENT_BLUE,
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                frame,
                chord["notes"] + " | " + chord["fret"],
                (cx1 + 7, cy1 + 43),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.27,
                TEXT_MUTED,
                1,
                cv2.LINE_AA
            )


# ============================================================
# 18. COMPLETE RESPONSIVE CAMERA INTERFACE
# ============================================================

def draw_interface(
    frame,
    root_count,
    quality_count,
    root_states,
    quality_states,
    active_chord,
    fps,
    hand_count,
    debug_mode,
    root_diagnostics=None,
    quality_diagnostics=None
):

    # Create a fixed landscape canvas.
    # The camera image is fitted without stretching.

    canvas = np.zeros(
        (
            DISPLAY_HEIGHT,
            DISPLAY_WIDTH,
            3
        ),
        dtype=np.uint8
    )

    canvas[:] = BG_MAIN

    frame_height, frame_width = frame.shape[:2]

    scale = min(
        DISPLAY_WIDTH / frame_width,
        DISPLAY_HEIGHT / frame_height
    )

    new_width = int(
        frame_width * scale
    )

    new_height = int(
        frame_height * scale
    )

    # Resize while preserving aspect ratio.
    resized_frame = cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )

    # Center the complete camera image.
    offset_x = (
        DISPLAY_WIDTH - new_width
    ) // 2

    offset_y = (
        DISPLAY_HEIGHT - new_height
    ) // 2

    canvas[
        offset_y:offset_y + new_height,
        offset_x:offset_x + new_width
    ] = resized_frame

    # ========================================================
    # TOP HEADER
    # ========================================================

    header_overlay = canvas.copy()

    cv2.rectangle(
        header_overlay,
        (0, 0),
        (DISPLAY_WIDTH, 67),
        BG_PANEL,
        -1
    )

    cv2.addWeighted(
        header_overlay,
        0.92,
        canvas,
        0.08,
        0,
        canvas
    )

    cv2.line(
        canvas,
        (0, 66),
        (DISPLAY_WIDTH, 66),
        ACCENT_MINT,
        2
    )

    # Brand icon
    cv2.circle(
        canvas,
        (27, 27),
        10,
        ACCENT_MINT,
        -1,
        cv2.LINE_AA
    )

    cv2.circle(
        canvas,
        (27, 27),
        4,
        BG_PANEL,
        -1,
        cv2.LINE_AA
    )

    # Title
    cv2.putText(
        canvas,
        PROJECT_NAME,
        (47, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.60,
        TEXT_MAIN,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        canvas,
        "TWO-HAND PERFORMANCE MODE",
        (48, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.32,
        TEXT_MUTED,
        1,
        cv2.LINE_AA
    )

    # Root status
    root_name = (
        ROOT_GESTURES.get(root_count, "INVALID")
        if root_count is not None
        else "WAITING"
    )

    draw_panel(
        canvas,
        420,
        12,
        605,
        53,
        BG_CARD,
        BORDER,
        9
    )

    cv2.putText(
        canvas,
        "ROOT HAND",
        (433, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        ACCENT_MINT,
        1
    )

    cv2.putText(
        canvas,
        root_name,
        (433, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_MAIN,
        1
    )

    # Quality status
    quality_name = (
        QUALITY_GESTURES.get(quality_count, "INVALID")
        if quality_count is not None
        else "WAITING"
    )

    draw_panel(
        canvas,
        619,
        12,
        812,
        53,
        BG_CARD,
        BORDER,
        9
    )

    cv2.putText(
        canvas,
        "CHORD QUALITY",
        (632, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.31,
        ACCENT_BLUE,
        1
    )

    cv2.putText(
        canvas,
        quality_name,
        (632, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        TEXT_MAIN,
        1
    )

    # Hand count and FPS
    cv2.putText(
        canvas,
        f"{hand_count}/2 HANDS",
        (1000, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        SUCCESS if hand_count == 2 else WARNING,
        1
    )

    cv2.putText(
        canvas,
        f"{fps:.0f} FPS",
        (1000, 47),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        TEXT_MUTED,
        1
    )

    # ========================================================
    # BOTTOM TRANSLUCENT CHORD PANEL
    # ========================================================

    panel_top = 542

    overlay = canvas.copy()

    cv2.rectangle(
        overlay,
        (0, panel_top),
        (DISPLAY_WIDTH, DISPLAY_HEIGHT),
        BG_PANEL,
        -1
    )

    cv2.addWeighted(
        overlay,
        0.94,
        canvas,
        0.06,
        0,
        canvas
    )

    # Panel accent
    cv2.line(
        canvas,
        (0, panel_top),
        (DISPLAY_WIDTH, panel_top),
        ACCENT_MINT,
        2
    )

    # ========================================================
    # CURRENT CHORD INFORMATION
    # ========================================================

    if active_chord is not None:

        chord = CHORDS[active_chord]

        current_text = chord["name"]

        cv2.putText(
            canvas,
            current_text,
            (18, panel_top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.57,
            SUCCESS,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            f"NOTES: {chord['notes']}     GUITAR: {chord['fret']}",
            (215, panel_top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            TEXT_MUTED,
            1,
            cv2.LINE_AA
        )

    else:

        cv2.putText(
            canvas,
            "CHORD LIBRARY",
            (18, panel_top + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            TEXT_MAIN,
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            canvas,
            "15 combinations  |  Select using both hands",
            (215, panel_top + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.39,
            TEXT_MUTED,
            1,
            cv2.LINE_AA
        )

    cv2.putText(
        canvas,
        "Q QUIT  |  D DEBUG  |  R RESET  |  SPACE REPLAY",
        (900, panel_top + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.30,
        TEXT_MUTED,
        1,
        cv2.LINE_AA
    )

    # ========================================================
    # FIVE COLUMNS, THREE ROWS
    # ========================================================

    root_order = ["C", "G", "D", "A", "E"]

    quality_order = [
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

    grid_x = 14
    grid_y = panel_top + 38

    grid_width = DISPLAY_WIDTH - 28

    grid_height = DISPLAY_HEIGHT - grid_y - 8

    column_width = grid_width // 5

    row_height = grid_height // 3

    for row, quality in enumerate(quality_order):

        for column, root in enumerate(root_order):

            key = (root, quality)

            chord = CHORDS[key]

            x1 = grid_x + column * column_width + 3

            x2 = grid_x + (column + 1) * column_width - 3

            y1 = grid_y + row * row_height + 2

            y2 = grid_y + (row + 1) * row_height - 2

            selected = key == active_chord

            if selected:

                card_color = BG_CARD_ACTIVE

                border_color = ACCENT_MINT

                name_color = (195, 255, 220)

            else:

                card_color = BG_CARD

                border_color = BORDER

                name_color = TEXT_MAIN

            draw_panel(
                canvas,
                x1,
                y1,
                x2,
                y2,
                card_color,
                border_color,
                8
            )

            # Chord name
            cv2.putText(
                canvas,
                chord["name"],
                (x1 + 9, y1 + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.43,
                name_color,
                1,
                cv2.LINE_AA
            )

            # Gesture
            control = (
                f"LEFT {root_counts[root]}   /   "
                f"RIGHT {quality_counts[quality]}"
            )

            cv2.putText(
                canvas,
                control,
                (x1 + 9, y1 + 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                ACCENT_BLUE,
                1,
                cv2.LINE_AA
            )

            # Notes and guitar fret
            detail = (
                chord["notes"] + "     " + chord["fret"]
            )

            cv2.putText(
                canvas,
                detail,
                (x1 + 9, y1 + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.34,
                TEXT_MUTED,
                1,
                cv2.LINE_AA
            )

    # ========================================================
    # DEBUG INFORMATION
    # ========================================================

    if debug_mode:

        debug_overlay = canvas.copy()

        cv2.rectangle(
            debug_overlay,
            (DISPLAY_WIDTH - 280, 80),
            (DISPLAY_WIDTH - 8, 280),
            BG_PANEL,
            -1
        )

        cv2.addWeighted(
            debug_overlay,
            0.90,
            canvas,
            0.10,
            0,
            canvas
        )

        y = 103

        for label, diagnostics in [
            ("ROOT", root_diagnostics),
            ("TYPE", quality_diagnostics)
        ]:

            if not diagnostics:
                continue

            for finger, info in diagnostics.items():

                message = (
                    f"{label} {finger}: "
                    f"{info['score']:.2f}  "
                    f"PIP {info['pip']:.0f}"
                )

                cv2.putText(
                    canvas,
                    message,
                    (DISPLAY_WIDTH - 268, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.30,
                    TEXT_MAIN,
                    1
                )

                y += 17

    return canvas


# ============================================================
# 19. MAIN APPLICATION
# ============================================================

def main():

    camera = None

    detector = None

    try:

        print("\n======================================")
        print("          AI VIRTUAL GUITAR")
        print("======================================")

        # Create musical sounds
        chord_files = {}

        print("\nGenerating guitar chords...\n")

        for key, chord in CHORDS.items():

            filename = SOUND_DIR / (
                chord["name"].replace(" ", "_") + ".wav"
            )

            generate_chord(
                chord["frequencies"],
                filename
            )

            chord_files[key] = filename

            print("Created:", chord["name"])

        # Initialize webcam
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
                "Could not open webcam."
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

        # Initialize MediaPipe
        detector = create_hand_detector()

        # Create window
        cv2.namedWindow(
            WINDOW_NAME,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            WINDOW_NAME,
            DISPLAY_WIDTH,
            DISPLAY_HEIGHT
        )

        print("\nVirtual guitar is ready.")

        print("LEFT HAND = ROOT")
        print("RIGHT HAND = CHORD QUALITY")

        print("\nKEYBOARD:")
        print("Q = Quit")
        print("D = Diagnostics")
        print("R = Reset")
        print("SPACE = Replay chord")

        # Gesture histories
        root_history = deque(
            maxlen=HISTORY_SIZE
        )

        quality_history = deque(
            maxlen=HISTORY_SIZE
        )

        # Finger states
        root_finger_states = {

            name: False

            for name in FINGER_LANDMARKS

        }

        quality_finger_states = {

            name: False

            for name in FINGER_LANDMARKS

        }

        active_chord = None

        previous_chord = None

        last_play_time = 0.0

        previous_timestamp = -1

        previous_frame_time = time.perf_counter()

        fps_history = deque(
            maxlen=15
        )

        debug_mode = False

        # ====================================================
        # MAIN LOOP
        # ====================================================

        while True:

            # Read webcam
            success, frame = camera.read()

            if not success:

                print("Could not read camera frame.")

                break

            # Mirror webcam
            frame = cv2.flip(
                frame,
                1
            )

            # Prepare image
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB
            )

            rgb_frame = np.ascontiguousarray(
                rgb_frame
            )

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=rgb_frame
            )

            # Increasing timestamp
            timestamp = int(
                time.perf_counter() * 1000
            )

            timestamp = max(
                timestamp,
                previous_timestamp + 1
            )

            previous_timestamp = timestamp

            # Detect both hands
            result = detector.detect_for_video(
                mp_image,
                timestamp
            )

            detected_hands = []

            if result.hand_landmarks:

                for index, image_landmarks in enumerate(
                    result.hand_landmarks
                ):

                    if (
                        not result.hand_world_landmarks
                        or index >= len(result.hand_world_landmarks)
                    ):

                        continue

                    world_landmarks = (
                        result.hand_world_landmarks[index]
                    )

                    center_x = np.mean(
                        [
                            landmark.x

                            for landmark in image_landmarks
                        ]
                    )

                    detected_hands.append(
                        (
                            center_x,
                            image_landmarks,
                            world_landmarks
                        )
                    )

            # Sort by screen position
            detected_hands.sort(
                key=lambda item: item[0]
            )

            root_count = None

            quality_count = None

            root_diagnostics = None

            quality_diagnostics = None

            active_chord = None

            # =================================================
            # PROCESS BOTH HANDS
            # =================================================

            if len(detected_hands) >= 2:

                # Screen-left hand = root
                (
                    _,
                    root_image,
                    root_world
                ) = detected_hands[0]

                # Screen-right hand = quality
                (
                    _,
                    quality_image,
                    quality_world
                ) = detected_hands[1]

                # Recognize left hand
                (
                    root_count,
                    root_finger_states,
                    root_diagnostics
                ) = recognize_fingers(
                    root_world,
                    root_finger_states
                )

                # Recognize right hand
                (
                    quality_count,
                    quality_finger_states,
                    quality_diagnostics
                ) = recognize_fingers(
                    quality_world,
                    quality_finger_states
                )

                # Draw left hand
                draw_hand(
                    frame,
                    root_image,
                    root_finger_states,
                    "ROOT"
                )

                # Draw right hand
                draw_hand(
                    frame,
                    quality_image,
                    quality_finger_states,
                    "QUALITY"
                )

                # Stabilize gestures
                root_history.append(
                    root_count
                )

                quality_history.append(
                    quality_count
                )

                stable_root = get_stable_count(
                    root_history
                )

                stable_quality = get_stable_count(
                    quality_history
                )

                # Select chord
                if (
                    stable_root is not None

                    and stable_quality is not None
                ):

                    active_chord = select_chord(
                        stable_root,
                        stable_quality
                    )

            else:

                # Reset if two hands are not present
                root_history.clear()

                quality_history.clear()

                for name in FINGER_LANDMARKS:

                    root_finger_states[name] = False

                    quality_finger_states[name] = False

            # =================================================
            # PLAY CHORD
            # =================================================

            current_time = time.time()

            if active_chord is not None:

                if (
                    active_chord != previous_chord

                    and current_time - last_play_time >= CHORD_COOLDOWN
                ):

                    play_chord(
                        active_chord,
                        chord_files
                    )

                    print(
                        "Playing:",
                        CHORDS[active_chord]["name"]
                    )

                    previous_chord = active_chord

                    last_play_time = current_time

            else:

                previous_chord = None

            # =================================================
            # FPS
            # =================================================

            now = time.perf_counter()

            elapsed = now - previous_frame_time

            previous_frame_time = now

            if elapsed > 0:

                fps_history.append(
                    1.0 / elapsed
                )

            fps = (
                sum(fps_history) / len(fps_history)

                if fps_history

                else 0.0
            )

            # =================================================
            # RENDER INTERFACE
            # =================================================

            display = draw_interface(
                frame,
                root_count,
                quality_count,

                root_finger_states
                if len(detected_hands) >= 2
                else None,

                quality_finger_states
                if len(detected_hands) >= 2
                else None,

                active_chord,
                fps,
                len(detected_hands),
                debug_mode,
                root_diagnostics,
                quality_diagnostics
            )

            cv2.imshow(
                WINDOW_NAME,
                display
            )

            # =================================================
            # KEYBOARD CONTROLS
            # =================================================

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("d"):

                debug_mode = not debug_mode

                print(
                    "Diagnostics:",
                    "ON" if debug_mode else "OFF"
                )

            elif key == ord("r"):

                root_history.clear()

                quality_history.clear()

                active_chord = None

                previous_chord = None

                print("Gesture recognition reset.")

            elif key == 32:

                if active_chord is not None:

                    play_chord(
                        active_chord,
                        chord_files
                    )

                    print(
                        "Replaying:",
                        CHORDS[active_chord]["name"]
                    )

    finally:

        # Stop audio
        try:

            winsound.PlaySound(
                None,
                winsound.SND_PURGE
            )

        except Exception:

            pass

        # Release webcam
        if camera is not None:

            camera.release()

        # Close detector
        if detector is not None:

            detector.close()

        # Close windows
        cv2.destroyAllWindows()

        print("\nAI Virtual Guitar closed successfully.")


# ============================================================
# 20. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        main()

    except KeyboardInterrupt:

        print("\nProgram interrupted by user.")

    except Exception as error:

        print("\nAPPLICATION ERROR:")

        print(
            type(error).__name__,
            ":",
            error
        )
