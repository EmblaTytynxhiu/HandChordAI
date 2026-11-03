
from __future__ import annotations

import time
import urllib.request
import wave
from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

try:
    import winsound
except ImportError:
    winsound = None


# ============================================================
# 1. SETTINGS
# ============================================================

CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_VIEW_WIDTH = 640

SAMPLE_RATE = 44100
LOOP_DURATION = 2.0

MAX_HANDS = 2

HISTORY_SIZE = 7
FINGER_HISTORY_SIZE = 5
REQUIRED_VOTES = 4

RELEASE_GRACE_SECONDS = 0.5
HAND_LOST_GRACE_SECONDS = 0.7

BASE_DIR = Path(__file__).resolve().parent
SOUND_DIR = BASE_DIR / "sounds"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

DETECTION_CONFIDENCE = 0.55
PRESENCE_CONFIDENCE = 0.55
TRACKING_CONFIDENCE = 0.55


# ============================================================
# 2. COLORS
# ============================================================

BACKGROUND = (17, 22, 32)
PANEL = (25, 33, 47)
CARD = (34, 44, 60)
ACTIVE = (37, 83, 62)

TEXT = (245, 247, 252)
MUTED = (160, 175, 195)

MINT = (105, 230, 185)
BLUE = (235, 155, 85)
GOLD = (65, 195, 255)

SUCCESS = (90, 220, 135)
DANGER = (75, 90, 235)
BORDER = (63, 76, 96)


# ============================================================
# 3. CHORD DEFINITIONS
# ============================================================

ROOTS = ["C", "D", "E", "F", "G", "A", "B"]
QUALITIES = ["MAJOR", "MINOR", "7TH"]

ROOT_GESTURES = {
    (False, 0): "C",
    (False, 1): "D",
    (False, 2): "E",
    (False, 3): "F",
    (False, 4): "G",
    (True, 0): "A",
    (True, 1): "B",
}

QUALITY_GESTURES = {
    1: "MAJOR",
    2: "MINOR",
    3: "7TH",
}

ROOT_SEMITONES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

ROOT_FREQUENCIES = {
    "C": 130.81,
    "D": 146.83,
    "E": 164.81,
    "F": 174.61,
    "G": 196.00,
    "A": 220.00,
    "B": 246.94,
}

NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

CHORD_INTERVALS = {
    "MAJOR": (0, 4, 7),
    "MINOR": (0, 3, 7),
    "7TH": (0, 4, 7, 10),
}

FRET_SHAPES = {
    "C": {
        "MAJOR": "x32010",
        "MINOR": "x35543",
        "7TH": "x32310",
    },
    "D": {
        "MAJOR": "xx0232",
        "MINOR": "xx0231",
        "7TH": "xx0212",
    },
    "E": {
        "MAJOR": "022100",
        "MINOR": "022000",
        "7TH": "020100",
    },
    "F": {
        "MAJOR": "133211",
        "MINOR": "133111",
        "7TH": "131211",
    },
    "G": {
        "MAJOR": "320003",
        "MINOR": "355333",
        "7TH": "320001",
    },
    "A": {
        "MAJOR": "x02220",
        "MINOR": "x02210",
        "7TH": "x02020",
    },
    "B": {
        "MAJOR": "x24442",
        "MINOR": "x24432",
        "7TH": "x21202",
    },
}


# ============================================================
# 4. GUITAR FINGERINGS
# ============================================================

# String order:
# Low E, A, D, G, B, high e.
#
# 0 = open string
# 1 = index finger
# 2 = middle finger
# 3 = ring finger
# 4 = pinky
# None = muted string

CHORD_FINGERINGS = {

    "CMAJOR": (None, 3, 2, 0, 1, 0),
    "CMINOR": (None, 1, 3, 4, 2, 1),
    "C7":     (None, 3, 2, 4, 1, 0),

    "DMAJOR": (None, None, 0, 1, 3, 2),
    "DMINOR": (None, None, 0, 2, 3, 1),
    "D7":     (None, None, 0, 2, 1, 3),

    "EMAJOR": (0, 2, 3, 1, 0, 0),
    "EMINOR": (0, 2, 3, 0, 0, 0),
    "E7":     (0, 2, 0, 1, 0, 0),

    "FMAJOR": (1, 3, 4, 2, 1, 1),
    "FMINOR": (1, 3, 4, 1, 1, 1),
    "F7":     (1, 3, 1, 2, 1, 1),

    "GMAJOR": (2, 1, 0, 0, 0, 3),
    "GMINOR": (1, 3, 4, 1, 1, 1),
    "G7":     (3, 2, 0, 0, 0, 1),

    "AMAJOR": (None, 0, 1, 2, 3, 0),
    "AMINOR": (None, 0, 2, 3, 1, 0),
    "A7":     (None, 0, 1, 0, 2, 0),

    "BMAJOR": (None, 1, 3, 3, 3, 1),
    "BMINOR": (None, 1, 3, 4, 2, 1),
    "B7":     (None, 2, 1, 3, 0, 4),
}

STRING_NAMES = ["E", "A", "D", "G", "B", "e"]

FINGER_COLORS = {
    1: SUCCESS,
    2: BLUE,
    3: GOLD,
    4: (220, 130, 240),
}


# ============================================================
# 5. BUILD ALL 21 CHORDS
# ============================================================

def build_chords():

    chords = {}

    for root in ROOTS:

        for quality in QUALITIES:

            root_note = ROOT_SEMITONES[root]
            root_frequency = ROOT_FREQUENCIES[root]

            notes = []
            frequencies = []

            for interval in CHORD_INTERVALS[quality]:

                note_index = (
                    root_note + interval
                ) % 12

                notes.append(
                    NOTE_NAMES[note_index]
                )

                frequencies.append(
                    root_frequency * (2 ** (interval / 12))
                )

            chord_name = root + quality

            # IMPORTANT FIX:
            # Seventh chord names are C7TH, D7TH, etc.
            # The fingering dictionary uses C7, D7, etc.
            #
            # Convert the key before accessing the dictionary.

            if quality == "7TH":
                finger_key = root + "7"
            else:
                finger_key = root + quality

            # Verify that a fingering exists.
            if finger_key not in CHORD_FINGERINGS:
                raise KeyError(
                    f"Missing guitar fingering: {finger_key}"
                )

            chords[chord_name] = {
                "root": root,
                "quality": quality,
                "notes": notes,
                "frequencies": frequencies,
                "fret": FRET_SHAPES[root][quality],
                "fingers": CHORD_FINGERINGS[finger_key],
            }

    return chords


CHORDS = build_chords()

assert len(CHORDS) == 21, "There should be exactly 21 chords."


# ============================================================
# 6. HAND LANDMARKS
# ============================================================

FINGER_LANDMARKS = {
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}

TRACE_LANDMARKS = {
    "THUMB": (1, 2, 3, 4),
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]


# ============================================================
# 7. DOWNLOAD MEDIAPIPE MODEL
# ============================================================

def download_model():

    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100000:
        print("Hand detection model already exists.")
        return

    print("Downloading hand detection model...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            str(MODEL_PATH)
        )

        if MODEL_PATH.stat().st_size < 100000:
            raise RuntimeError("Downloaded model is incomplete.")

        print("Model downloaded successfully.")

    except Exception as error:

        raise RuntimeError(
            f"Could not download MediaPipe model: {error}"
        )


# ============================================================
# 8. CREATE HAND DETECTOR
# ============================================================

def create_hand_detector():

    download_model()

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
# 9. ANGLE AND SCORE FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):

    vector1 = a - b
    vector2 = c - b

    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    cosine = np.dot(vector1, vector2) / (norm1 * norm2)

    cosine = np.clip(cosine, -1.0, 1.0)

    return float(
        np.degrees(np.arccos(cosine))
    )


def clamp_score(value):

    return float(
        np.clip(value, 0.0, 1.0)
    )


# ============================================================
# 10. STRICT FINGER RECOGNITION
# ============================================================

def recognize_fingers(world_landmarks, previous_states=None):

    if previous_states is None:
        previous_states = {}

    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in world_landmarks
        ],
        dtype=np.float64
    )

    states = {}
    diagnostics = {}

    wrist = points[0]

    # --------------------------------------------------------
    # INDEX, MIDDLE, RING AND PINKY
    # --------------------------------------------------------

    for name, indices in FINGER_LANDMARKS.items():

        mcp, pip, dip, tip = (
            points[index]
            for index in indices
        )

        pip_angle = calculate_angle(
            mcp, pip, dip
        )

        dip_angle = calculate_angle(
            pip, dip, tip
        )

        mcp_to_pip = np.linalg.norm(
            pip - mcp
        )

        mcp_to_tip = np.linalg.norm(
            tip - mcp
        )

        extension_ratio = (
            mcp_to_tip / max(mcp_to_pip, 1e-6)
        )

        mcp_to_wrist = np.linalg.norm(
            mcp - wrist
        )

        tip_to_wrist = np.linalg.norm(
            tip - wrist
        )

        reach_ratio = (
            tip_to_wrist / max(mcp_to_wrist, 1e-6)
        )

        pip_score = clamp_score(
            (pip_angle - 115) / 60
        )

        dip_score = clamp_score(
            (dip_angle - 105) / 70
        )

        extension_score = clamp_score(
            (extension_ratio - 1.0) / 0.85
        )

        reach_score = clamp_score(
            (reach_ratio - 1.0) / 0.50
        )

        score = (
            0.36 * pip_score
            + 0.24 * dip_score
            + 0.25 * extension_score
            + 0.15 * reach_score
        )

        was_open = previous_states.get(
            name,
            False
        )

        if was_open:

            score_threshold = 0.58
            pip_minimum = 132
            extension_minimum = 1.18

        else:

            score_threshold = 0.68
            pip_minimum = 142
            extension_minimum = 1.30

        is_open = (
            score >= score_threshold
            and pip_angle >= pip_minimum
            and extension_ratio >= extension_minimum
        )

        states[name] = bool(is_open)

        diagnostics[name] = {
            "pip": float(pip_angle),
            "dip": float(dip_angle),
            "ratio": float(extension_ratio),
            "reach": float(reach_ratio),
            "score": float(score),
            "open": bool(is_open),
        }

    # --------------------------------------------------------
    # THUMB RECOGNITION
    # --------------------------------------------------------

    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    palm_width = np.linalg.norm(
        points[5] - points[17]
    )

    thumb_distance = np.linalg.norm(
        points[4] - points[5]
    )

    thumb_ratio = (
        thumb_distance / max(palm_width, 1e-6)
    )

    was_thumb_open = previous_states.get(
        "THUMB",
        False
    )

    if was_thumb_open:

        angle_threshold = 138
        distance_threshold = 0.56

    else:

        angle_threshold = 150
        distance_threshold = 0.66

    thumb_open = (
        thumb_angle >= angle_threshold
        and thumb_ratio >= distance_threshold
    )

    states["THUMB"] = bool(thumb_open)

    diagnostics["THUMB"] = {
        "pip": float(thumb_angle),
        "dip": 0.0,
        "ratio": float(thumb_ratio),
        "reach": 0.0,
        "score": 1.0 if thumb_open else 0.0,
        "open": bool(thumb_open),
    }

    finger_count = sum(
        states[name]
        for name in FINGER_LANDMARKS
    )

    return finger_count, states, diagnostics


# ============================================================
# 11. TEMPORAL FINGER SMOOTHING
# ============================================================

def create_finger_histories():

    return {
        name: deque(maxlen=FINGER_HISTORY_SIZE)
        for name in [
            "THUMB",
            "INDEX",
            "MIDDLE",
            "RING",
            "PINKY",
        ]
    }


def stabilize_fingers(raw_states, histories):

    stable_states = {}

    for name, raw_value in raw_states.items():

        histories[name].append(
            bool(raw_value)
        )

        open_votes = sum(
            histories[name]
        )

        closed_votes = (
            len(histories[name]) - open_votes
        )

        if open_votes > closed_votes:

            stable_states[name] = True

        elif closed_votes > open_votes:

            stable_states[name] = False

        else:

            stable_states[name] = bool(raw_value)

    return stable_states


# ============================================================
# 12. DETECT HANDS
# ============================================================

def detect_hands(detector, frame, timestamp):

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    hands = []

    if not result.hand_landmarks:
        return hands

    for index, image_landmarks in enumerate(
        result.hand_landmarks
    ):

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


# ============================================================
# 13. FINGER TRACING
# ============================================================

def create_trails():

    return {
        name: deque(maxlen=12)
        for name in TRACE_LANDMARKS
    }


def draw_hand_landmarks(
    frame,
    landmarks,
    states,
    trails
):

    height, width = frame.shape[:2]

    # Draw palm connections.
    for start, end in HAND_CONNECTIONS:

        p1 = landmarks[start]
        p2 = landmarks[end]

        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (110, 125, 145),
            2,
            cv2.LINE_AA
        )

    # Draw individual fingers.
    for finger_name, indices in TRACE_LANDMARKS.items():

        is_open = states.get(
            finger_name,
            False
        )

        color = (
            SUCCESS
            if is_open
            else DANGER
        )

        for i in range(len(indices) - 1):

            p1 = landmarks[indices[i]]
            p2 = landmarks[indices[i + 1]]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                4,
                cv2.LINE_AA
            )

        tip = landmarks[indices[-1]]

        tip_position = (
            int(tip.x * width),
            int(tip.y * height)
        )

        trails[finger_name].append(
            tip_position
        )

        trail_points = list(
            trails[finger_name]
        )

        if len(trail_points) >= 2:

            cv2.polylines(
                frame,
                [
                    np.array(
                        trail_points,
                        dtype=np.int32
                    )
                ],
                False,
                color,
                2,
                cv2.LINE_AA
            )

        cv2.circle(
            frame,
            tip_position,
            8,
            color,
            -1
        )

        cv2.circle(
            frame,
            tip_position,
            10,
            TEXT,
            1
        )

        # Display finger name beside the tip.
        label_x = min(
            max(tip_position[0] + 8, 0),
            width - 80
        )

        label_y = max(
            tip_position[1] - 8,
            20
        )

        cv2.putText(
            frame,
            finger_name[:3],
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.35,
            color,
            1,
            cv2.LINE_AA
        )


# ============================================================
# 14. STABILIZE COMPLETE GESTURES
# ============================================================

def stable_value(history):

    if not history:
        return None

    counts = Counter(history)

    value, votes = counts.most_common(1)[0]

    if votes >= REQUIRED_VOTES:
        return value

    return None


# ============================================================
# 15. SELECT CHORD
# ============================================================

def select_chord(root_gesture, quality_count):

    if root_gesture is None or quality_count is None:
        return None

    root_count, thumb_open = root_gesture

    root = ROOT_GESTURES.get(
        (bool(thumb_open), root_count)
    )

    quality = QUALITY_GESTURES.get(
        quality_count
    )

    if root is None or quality is None:
        return None

    chord = root + quality

    if chord in CHORDS:
        return chord

    return None


# ============================================================
# 16. GENERATE CHORD AUDIO
# ============================================================

def generate_chord(frequencies, filename):

    sample_count = int(
        SAMPLE_RATE * LOOP_DURATION
    )

    time_values = (
        np.arange(sample_count) / SAMPLE_RATE
    )

    audio = np.zeros(
        sample_count,
        dtype=np.float64
    )

    for frequency in frequencies:

        note = (
            np.sin(2 * np.pi * frequency * time_values)
            + 0.30 * np.sin(2 * np.pi * frequency * 2 * time_values)
            + 0.12 * np.sin(2 * np.pi * frequency * 3 * time_values)
            + 0.05 * np.sin(2 * np.pi * frequency * 4 * time_values)
        )

        audio += note

    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:
        audio /= maximum

    audio *= 0.65

    fade_size = int(
        SAMPLE_RATE * 0.01
    )

    if fade_size > 0:

        fade = np.linspace(
            0,
            1,
            fade_size
        )

        audio[:fade_size] *= fade
        audio[-fade_size:] *= fade[::-1]

    audio = (
        audio * 32767
    ).astype(np.int16)

    with wave.open(
        str(filename),
        "wb"
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        wav_file.writeframes(
            audio.tobytes()
        )


def create_chord_files():

    SOUND_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Preparing chord sounds...")

    for chord_name, chord_data in CHORDS.items():

        filename = SOUND_DIR / f"{chord_name}.wav"

        if not filename.exists():

            generate_chord(
                chord_data["frequencies"],
                filename
            )

    print("All 21 chord sounds are ready.")


# ============================================================
# 17. PLAY AND STOP AUDIO
# ============================================================

def stop_chord():

    if winsound is None:
        return

    try:

        winsound.PlaySound(
            None,
            winsound.SND_PURGE
        )

    except Exception:
        pass


def play_chord(chord_name):

    if winsound is None:
        return

    filename = SOUND_DIR / f"{chord_name}.wav"

    if not filename.exists():
        return

    try:

        winsound.PlaySound(
            str(filename),
            winsound.SND_FILENAME
            | winsound.SND_ASYNC
            | winsound.SND_LOOP
        )

    except Exception as error:

        print("Audio playback error:", error)


# ============================================================
# 18. DRAW TEXT
# ============================================================

def draw_text(
    image,
    message,
    position,
    scale=0.6,
    color=TEXT,
    thickness=1
):

    cv2.putText(
        image,
        str(message),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 19. PREPARE CAMERA VIEW
# ============================================================

def prepare_camera_view(frame):

    height, width = frame.shape[:2]

    scale = min(
        CAMERA_VIEW_WIDTH / width,
        DISPLAY_HEIGHT / height
    )

    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    resized = cv2.resize(
        frame,
        (new_width, new_height)
    )

    background = cv2.resize(
        frame,
        (CAMERA_VIEW_WIDTH, DISPLAY_HEIGHT)
    )

    background = cv2.GaussianBlur(
        background,
        (51, 51),
        0
    )

    canvas = background.copy()

    x = (CAMERA_VIEW_WIDTH - new_width) // 2
    y = (DISPLAY_HEIGHT - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


# ============================================================
# 20. DRAW DIAGNOSTICS
# ============================================================

def draw_debug(
    frame,
    root_data,
    quality_data
):

    draw_text(
        frame,
        "FINGER DETECTION",
        (10, 105),
        0.48,
        GOLD,
        2
    )

    y = 132

    for hand_name, data in [
        ("ROOT HAND", root_data),
        ("QUALITY HAND", quality_data),
    ]:

        if data is None:
            continue

        draw_text(
            frame,
            hand_name,
            (10, y),
            0.40,
            MINT
        )

        y += 21

        for name, values in data.items():

            status = (
                "OPEN"
                if values["open"]
                else "CLOSED"
            )

            color = (
                SUCCESS
                if values["open"]
                else MUTED
            )

            message = (
                f"{name}: {status} "
                f"A:{values['pip']:.0f} "
                f"R:{values['ratio']:.2f} "
                f"S:{values['score']:.2f}"
            )

            draw_text(
                frame,
                message,
                (10, y),
                0.33,
                color
            )

            y += 18

        y += 6


# ============================================================
# 21. DRAW CHORD FINGER GUIDE
# ============================================================

def draw_finger_guide(panel, chord_data):

    draw_text(
        panel,
        "FINGER GUIDE",
        (275, 137),
        0.48,
        MINT,
        2
    )

    start_x = 295
    spacing = 46

    # String labels.
    for i, string_name in enumerate(STRING_NAMES):

        x = start_x + i * spacing

        draw_text(
            panel,
            string_name,
            (x, 163),
            0.43,
            GOLD,
            2
        )

    draw_text(
        panel,
        "FRET",
        (260, 190),
        0.36,
        MUTED
    )

    draw_text(
        panel,
        "FINGER",
        (260, 218),
        0.36,
        MUTED
    )

    # Show fret numbers.
    for i, value in enumerate(chord_data["fret"]):

        x = start_x + i * spacing

        if value.lower() == "x":

            color = DANGER

        elif value == "0":

            color = MUTED

        else:

            color = TEXT

        draw_text(
            panel,
            value,
            (x, 190),
            0.48,
            color,
            2
        )

    # Show finger assignments.
    for i, finger in enumerate(chord_data["fingers"]):

        x = start_x + i * spacing

        if finger is None:

            value = "X"
            color = DANGER

        elif finger == 0:

            value = "-"
            color = MUTED

        else:

            value = str(finger)
            color = FINGER_COLORS[finger]

        draw_text(
            panel,
            value,
            (x, 218),
            0.50,
            color,
            2
        )

    draw_text(
        panel,
        "1 Index   2 Middle",
        (275, 243),
        0.36,
        TEXT
    )

    draw_text(
        panel,
        "3 Ring    4 Pinky",
        (445, 243),
        0.36,
        TEXT
    )

    draw_text(
        panel,
        "X = mute   - = open",
        (275, 260),
        0.34,
        MUTED
    )


# ============================================================
# 22. DRAW RIGHT PANEL
# ============================================================

def draw_panel(
    display,
    active_chord,
    playing_chord,
    root_gesture,
    quality_count,
    hand_count
):

    panel = display[:, CAMERA_VIEW_WIDTH:]

    panel[:] = PANEL

    # HEADER
    draw_text(
        panel,
        "VIRTUAL GUITAR",
        (22, 43),
        0.90,
        TEXT,
        2
    )

    draw_text(
        panel,
        "TWO-HAND CHORD CONTROLLER",
        (24, 70),
        0.40,
        MUTED
    )

    cv2.line(
        panel,
        (20, 88),
        (620, 88),
        BORDER,
        1
    )

    # CURRENT CHORD CARD
    cv2.rectangle(
        panel,
        (18, 100),
        (622, 268),
        CARD,
        -1
    )

    cv2.rectangle(
        panel,
        (18, 100),
        (622, 268),
        BORDER,
        1
    )

    draw_text(
        panel,
        "CURRENT CHORD",
        (35, 127),
        0.40,
        MUTED
    )

    if active_chord:

        chord_data = CHORDS[active_chord]

        draw_text(
            panel,
            active_chord,
            (35, 181),
            1.10,
            MINT,
            3
        )

        draw_text(
            panel,
            chord_data["quality"],
            (35, 207),
            0.48,
            TEXT
        )

        draw_text(
            panel,
            "Fret: " + chord_data["fret"],
            (35, 235),
            0.44,
            MUTED
        )

        draw_finger_guide(
            panel,
            chord_data
        )

    else:

        draw_text(
            panel,
            "Waiting for gestures...",
            (35, 185),
            0.55,
            MUTED
        )

        draw_text(
            panel,
            "Finger guide appears when a chord is selected.",
            (35, 222),
            0.34,
            MUTED
        )

    if playing_chord:

        draw_text(
            panel,
            "PLAYING",
            (35, 255),
            0.35,
            SUCCESS
        )

    else:

        draw_text(
            panel,
            "SOUND OFF",
            (35, 255),
            0.35,
            MUTED
        )

    # HAND STATUS
    draw_text(
        panel,
        f"HANDS DETECTED: {hand_count}/2",
        (22, 293),
        0.43,
        SUCCESS if hand_count >= 2 else GOLD,
        2
    )

    draw_text(
        panel,
        "LEFT HAND: ROOT NOTE",
        (22, 321),
        0.40,
        BLUE
    )

    draw_text(
        panel,
        "RIGHT HAND: CHORD TYPE",
        (320, 321),
        0.40,
        GOLD
    )

    if root_gesture is not None:

        root_count, thumb_open = root_gesture

        thumb_text = (
            "OPEN"
            if thumb_open
            else "CLOSED"
        )

        root_text = (
            f"Fingers: {root_count} | Thumb: {thumb_text}"
        )

    else:

        root_text = "Waiting for root hand..."

    draw_text(
        panel,
        root_text,
        (22, 345),
        0.37,
        TEXT
    )

    quality_text = (
        f"Fingers: {quality_count}"
        if quality_count is not None
        else "Waiting for quality hand..."
    )

    draw_text(
        panel,
        quality_text,
        (320, 345),
        0.37,
        TEXT
    )

    # CHORD LIBRARY
    draw_text(
        panel,
        "CHORD LIBRARY",
        (22, 379),
        0.53,
        TEXT,
        2
    )

    grid_left = 18
    root_width = 65
    gap = 6

    grid_top = 393
    row_height = 35
    row_gap = 1

    grid_right = 622

    chord_width = (
        grid_right
        - grid_left
        - root_width
        - 3 * gap
    ) // 3

    for row, root in enumerate(ROOTS):

        y1 = grid_top + row * (
            row_height + row_gap
        )

        y2 = y1 + row_height

        cv2.rectangle(
            panel,
            (grid_left, y1),
            (grid_left + root_width, y2),
            CARD,
            -1
        )

        draw_text(
            panel,
            root,
            (grid_left + 24, y1 + 24),
            0.55,
            MINT,
            2
        )

        for column, quality in enumerate(QUALITIES):

            x1 = (
                grid_left
                + root_width
                + gap
                + column * (chord_width + gap)
            )

            x2 = x1 + chord_width

            chord_name = root + quality

            cell_color = (
                ACTIVE
                if active_chord == chord_name
                else CARD
            )

            cv2.rectangle(
                panel,
                (x1, y1),
                (x2, y2),
                cell_color,
                -1
            )

            cv2.rectangle(
                panel,
                (x1, y1),
                (x2, y2),
                BORDER,
                1
            )

            label = {
                "MAJOR": "Major",
                "MINOR": "Minor",
                "7TH": "7th",
            }[quality]

            draw_text(
                panel,
                label,
                (x1 + 9, y1 + 23),
                0.40,
                TEXT
            )

    # FOOTER
    draw_text(
        panel,
        "SPACE: Replay | R: Reset | D: Diagnostics | Q: Quit",
        (18, 680),
        0.35,
        MUTED
    )

    draw_text(
        panel,
        "21 AVAILABLE CHORDS",
        (18, 708),
        0.35,
        MINT
    )


# ============================================================
# 23. COMPOSE DISPLAY
# ============================================================

def compose_display(
    frame,
    active_chord,
    playing_chord,
    root_gesture,
    quality_count,
    hand_count,
    debug_enabled,
    root_diagnostics,
    quality_diagnostics
):

    camera_view = prepare_camera_view(frame)

    display = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    display[:, :CAMERA_VIEW_WIDTH] = camera_view

    display[:, CAMERA_VIEW_WIDTH:] = PANEL

    draw_text(
        display,
        "HAND TRACKING",
        (20, 35),
        0.60,
        TEXT,
        2
    )

    draw_text(
        display,
        "Green = extended | Red = folded",
        (20, 63),
        0.39,
        MUTED
    )

    if debug_enabled:

        draw_debug(
            display[:, :CAMERA_VIEW_WIDTH],
            root_diagnostics,
            quality_diagnostics
        )

    draw_panel(
        display,
        active_chord,
        playing_chord,
        root_gesture,
        quality_count,
        hand_count
    )

    return display


# ============================================================
# 24. MAIN APPLICATION
# ============================================================

def run_application():

    print("=" * 45)
    print("VIRTUAL GUITAR")
    print("=" * 45)

    print("Total chords:", len(CHORDS))

    if winsound is None:
        print("WARNING: Windows audio playback unavailable.")

    create_chord_files()

    print("Opening camera...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Camera could not be opened. Check CAMERA_INDEX."
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

    detector = create_hand_detector()

    cv2.namedWindow(
        "VIRTUAL GUITAR",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "VIRTUAL GUITAR",
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    # Gesture histories.
    root_history = deque(
        maxlen=HISTORY_SIZE
    )

    quality_history = deque(
        maxlen=HISTORY_SIZE
    )

    # Individual finger histories.
    root_finger_histories = create_finger_histories()

    quality_finger_histories = create_finger_histories()

    root_previous_states = {}
    quality_previous_states = {}

    # Finger trails.
    root_trails = create_trails()
    quality_trails = create_trails()

    root_gesture = None
    quality_count = None

    root_diagnostics = None
    quality_diagnostics = None

    active_chord = None
    playing_chord = None

    debug_enabled = False

    last_timestamp = 0

    last_valid_chord_time = 0
    last_two_hand_time = 0

    print("\nROOT HAND:")
    print("Thumb CLOSED + 0 fingers = C")
    print("Thumb CLOSED + 1 finger  = D")
    print("Thumb CLOSED + 2 fingers = E")
    print("Thumb CLOSED + 3 fingers = F")
    print("Thumb CLOSED + 4 fingers = G")
    print("Thumb OPEN + 0 fingers   = A")
    print("Thumb OPEN + 1 finger    = B")

    print("\nQUALITY HAND:")
    print("1 finger = MAJOR")
    print("2 fingers = MINOR")
    print("3 fingers = 7TH")

    print("\nCONTROLS:")
    print("D = Diagnostics")
    print("R = Reset")
    print("SPACE = Replay")
    print("Q = Quit")

    try:

        while True:

            success, frame = camera.read()

            if not success or frame is None:

                print("Could not read camera frame.")
                break

            frame = cv2.flip(
                frame,
                1
            )

            current_time = time.monotonic()

            timestamp = int(
                current_time * 1000
            )

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp

            hands = detect_hands(
                detector,
                frame,
                timestamp
            )

            hand_count = len(hands)

            # ------------------------------------------------
            # TWO HANDS DETECTED
            # ------------------------------------------------

            if hand_count >= 2:

                last_two_hand_time = current_time

                root_hand = hands[0]
                quality_hand = hands[1]

                root_image = root_hand[1]
                root_world = root_hand[2]

                quality_image = quality_hand[1]
                quality_world = quality_hand[2]

                # ROOT HAND: RAW RECOGNITION.
                (
                    _,
                    root_raw_states,
                    root_diagnostics
                ) = recognize_fingers(
                    root_world,
                    root_previous_states
                )

                # ROOT HAND: TEMPORAL SMOOTHING.
                root_states = stabilize_fingers(
                    root_raw_states,
                    root_finger_histories
                )

                root_previous_states = root_states

                root_count = sum(
                    root_states[name]
                    for name in FINGER_LANDMARKS
                )

                # QUALITY HAND: RAW RECOGNITION.
                (
                    _,
                    quality_raw_states,
                    quality_diagnostics
                ) = recognize_fingers(
                    quality_world,
                    quality_previous_states
                )

                # QUALITY HAND: TEMPORAL SMOOTHING.
                quality_states = stabilize_fingers(
                    quality_raw_states,
                    quality_finger_histories
                )

                quality_previous_states = quality_states

                quality_detected = sum(
                    quality_states[name]
                    for name in FINGER_LANDMARKS
                )

                # DRAW FINGER TRACES.
                draw_hand_landmarks(
                    frame,
                    root_image,
                    root_states,
                    root_trails
                )

                draw_hand_landmarks(
                    frame,
                    quality_image,
                    quality_states,
                    quality_trails
                )

                # SAVE GESTURES.
                root_history.append(
                    (
                        root_count,
                        root_states["THUMB"]
                    )
                )

                quality_history.append(
                    quality_detected
                )

                # STABILIZE COMPLETE GESTURES.
                root_gesture = stable_value(
                    root_history
                )

                quality_count = stable_value(
                    quality_history
                )

                # SELECT CHORD.
                selected_chord = select_chord(
                    root_gesture,
                    quality_count
                )

                if selected_chord is not None:

                    active_chord = selected_chord

                    last_valid_chord_time = current_time

                    if playing_chord != active_chord:

                        stop_chord()

                        play_chord(
                            active_chord
                        )

                        playing_chord = active_chord

            # ------------------------------------------------
            # HANDS TEMPORARILY LOST
            # ------------------------------------------------

            elif (
                current_time - last_two_hand_time
                > HAND_LOST_GRACE_SECONDS
            ):

                root_history.clear()
                quality_history.clear()

                root_previous_states.clear()
                quality_previous_states.clear()

                root_finger_histories = create_finger_histories()

                quality_finger_histories = create_finger_histories()

                root_trails = create_trails()
                quality_trails = create_trails()

                root_gesture = None
                quality_count = None

                root_diagnostics = None
                quality_diagnostics = None

            # ------------------------------------------------
            # STOP AUDIO AFTER RELEASE
            # ------------------------------------------------

            if (
                playing_chord is not None
                and current_time - last_valid_chord_time
                > RELEASE_GRACE_SECONDS
            ):

                stop_chord()

                playing_chord = None

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = compose_display(
                frame,
                active_chord,
                playing_chord,
                root_gesture,
                quality_count,
                hand_count,
                debug_enabled,
                root_diagnostics,
                quality_diagnostics
            )

            cv2.imshow(
                "VIRTUAL GUITAR",
                display
            )

            # ------------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):

                break

            elif key == ord("d"):

                debug_enabled = not debug_enabled

                print(
                    "Diagnostics:",
                    "ON" if debug_enabled else "OFF"
                )

            elif key == ord("r"):

                root_history.clear()
                quality_history.clear()

                root_previous_states.clear()
                quality_previous_states.clear()

                root_finger_histories = create_finger_histories()

                quality_finger_histories = create_finger_histories()

                root_trails = create_trails()
                quality_trails = create_trails()

                root_gesture = None
                quality_count = None

                active_chord = None
                playing_chord = None

                root_diagnostics = None
                quality_diagnostics = None

                last_valid_chord_time = 0

                stop_chord()

                print("Reset complete.")

            elif key == 32:

                if active_chord is not None:

                    stop_chord()

                    play_chord(
                        active_chord
                    )

                    playing_chord = active_chord

                    last_valid_chord_time = current_time

    finally:

        print("Closing Virtual Guitar...")

        stop_chord()

        camera.release()
        detector.close()

        cv2.destroyAllWindows()

        print("Application closed.")


# ============================================================
# 25. START PROGRAM
# ============================================================

if __name__ == "__main__":

    try:

        run_application()

    except KeyboardInterrupt:

        print("\nApplication interrupted.")

        stop_chord()
        cv2.destroyAllWindows()

    except Exception as error:

        print("\nAPPLICATION ERROR:")
        print(error)

        stop_chord()
        cv2.destroyAllWindows()

        raise
