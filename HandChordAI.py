
"""Gesture-controlled virtual guitar with personal hand calibration.

The application combines:
1. MediaPipe hand detection and 21 hand landmarks.
2. Geometric finger-feature extraction.
3. Personalized calibration for both hands.
4. Temporal smoothing and gesture recognition.
5. Generated WAV audio for 21 guitar chords.
6. An OpenCV interface with live finger tracing and chord diagrams.

CONTROLS
--------
C       Start/restart personal calibration.
D       Toggle finger diagnostics.
R       Reset the selected chord and gesture history.
SPACE   Replay the selected chord.
Q/ESC   Quit.

HAND ROLES
----------
Screen-left hand  = root note.
Screen-right hand = chord quality.

ROOT HAND
---------
Thumb closed + 0 fingers = C
Thumb closed + 1 finger  = D
Thumb closed + 2 fingers = E
Thumb closed + 3 fingers = F
Thumb closed + 4 fingers = G
Thumb open   + 0 fingers = A
Thumb open   + 1 finger  = B

QUALITY HAND
------------
1 finger = Major
2 fingers = Minor
3 fingers = 7th

Calibration is saved in finger_calibration.json. It adapts the geometric
classifier to the user's measurements; it does not retrain MediaPipe itself.
"""

from __future__ import annotations

import json
import time
import urllib.request
import wave

from collections import Counter, deque
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# ============================================================
# 1. IMPORT WINDOWS AUDIO
# ============================================================

# winsound is included with Windows Python.
# Keep the import optional so the rest of the program can still load
# on systems where winsound is unavailable.

try:
    import winsound
except ImportError:
    winsound = None


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Camera settings.
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

# Application window dimensions.
DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_VIEW_WIDTH = 640

# Audio settings.
SAMPLE_RATE = 44100
LOOP_DURATION = 2.0

# Maximum number of hands MediaPipe should track.
MAX_HANDS = 2

# Gesture history settings.
GESTURE_HISTORY_SIZE = 7
FINGER_HISTORY_SIZE = 5
REQUIRED_GESTURE_VOTES = 4

# Audio and hand-detection grace periods.
RELEASE_GRACE_SECONDS = 0.50
HAND_LOST_GRACE_SECONDS = 0.70

# Calibration settings.
CALIBRATION_SETTLE_SECONDS = 0.50
CALIBRATION_CAPTURE_SECONDS = 2.50
CALIBRATION_MIN_SAMPLES = 12

# Store generated files beside this Python script.
BASE_DIR = Path(__file__).resolve().parent

SOUND_DIR = BASE_DIR / "sounds"
MODEL_PATH = BASE_DIR / "hand_landmarker.task"
CALIBRATION_PATH = BASE_DIR / "finger_calibration.json"

# Official MediaPipe hand-landmarker model.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/"
    "hand_landmarker.task"
)

# Minimum MediaPipe detection confidence.
DETECTION_CONFIDENCE = 0.55
PRESENCE_CONFIDENCE = 0.55
TRACKING_CONFIDENCE = 0.55


# ============================================================
# 3. COLORS
# ============================================================

# OpenCV uses BGR colors instead of RGB.

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
WARNING = (0, 190, 255)

BORDER = (63, 76, 96)

# Colors used to identify individual fingers.
FINGER_COLORS = {
    "THUMB": (255, 180, 70),
    "INDEX": SUCCESS,
    "MIDDLE": BLUE,
    "RING": (220, 130, 240),
    "PINKY": (255, 110, 110),
}

# Colors used for numbered guitar fingering dots.
FINGER_NUMBER_COLORS = {
    1: SUCCESS,
    2: BLUE,
    3: GOLD,
    4: (220, 130, 240),
}


# ============================================================
# 4. MUSICAL DEFINITIONS
# ============================================================

# Available root notes.
ROOTS = ["C", "D", "E", "F", "G", "A", "B"]

# Available chord qualities.
QUALITIES = ["MAJOR", "MINOR", "7TH"]

# Root-hand gesture mapping.
#
# A tuple contains:
# (thumb_is_open, number_of_open_non_thumb_fingers)

ROOT_GESTURES = {
    (False, 0): "C",
    (False, 1): "D",
    (False, 2): "E",
    (False, 3): "F",
    (False, 4): "G",
    (True, 0): "A",
    (True, 1): "B",
}

# Quality-hand gesture mapping.
QUALITY_GESTURES = {
    1: "MAJOR",
    2: "MINOR",
    3: "7TH",
}

# Semitone positions of natural notes relative to C.
ROOT_SEMITONES = {
    "C": 0,
    "D": 2,
    "E": 4,
    "F": 5,
    "G": 7,
    "A": 9,
    "B": 11,
}

# Fundamental frequencies of the root notes.
# All notes are kept in a similar lower register.
ROOT_FREQUENCIES = {
    "C": 130.81,
    "D": 146.83,
    "E": 164.81,
    "F": 174.61,
    "G": 196.00,
    "A": 220.00,
    "B": 246.94,
}

# Chromatic note names.
NOTE_NAMES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B"
]

# Chord intervals measured in semitones.
CHORD_INTERVALS = {
    "MAJOR": (0, 4, 7),
    "MINOR": (0, 3, 7),
    "7TH": (0, 4, 7, 10),
}


# ============================================================
# 5. GUITAR FRET SHAPES
# ============================================================

# Every shape contains six characters.
#
# The order is:
# Low E, A, D, G, B, high e.
#
# X = muted string.
# 0 = open string.
# Other digits = fret number.

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
# 6. GUITAR FINGER ASSIGNMENTS
# ============================================================

# Guitar finger numbers:
#
# 1 = index finger
# 2 = middle finger
# 3 = ring finger
# 4 = pinky
# 0 = open string
# None = muted string
#
# The entries use the same low-E-to-high-e string order.

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

# Guitar strings from thickest to thinnest.
STRING_NAMES = ["E", "A", "D", "G", "B", "e"]


# ============================================================
# 7. MEDIAPIPE LANDMARK DEFINITIONS
# ============================================================

# MediaPipe provides 21 hand landmarks.
#
# Each non-thumb finger has:
# MCP = base knuckle
# PIP = middle joint
# DIP = joint near the fingertip
# TIP = fingertip

FINGER_LANDMARKS = {
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}

# Landmark paths used to draw the fingers.
TRACE_LANDMARKS = {
    "THUMB": (1, 2, 3, 4),
    "INDEX": (5, 6, 7, 8),
    "MIDDLE": (9, 10, 11, 12),
    "RING": (13, 14, 15, 16),
    "PINKY": (17, 18, 19, 20),
}

# Connections between landmarks used to draw the hand skeleton.
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
]

# Consistent finger and hand-role ordering.
FINGER_NAMES = [
    "THUMB", "INDEX", "MIDDLE", "RING", "PINKY"
]

ROLE_NAMES = ["ROOT", "QUALITY"]


# ============================================================
# 8. BUILD THE 21 CHORDS
# ============================================================

def build_chords():
    """Create all chord entries, including notes, frequencies, and fingerings.

    Seventh chords use names such as C7TH internally. The fingering table uses
    C7, so the key is translated before looking up the finger assignments.
    """

    chords = {}

    # Process every root note and every chord quality.
    for root in ROOTS:

        for quality in QUALITIES:

            notes = []
            frequencies = []

            # Calculate all notes belonging to this chord.
            for interval in CHORD_INTERVALS[quality]:

                note_index = (
                    ROOT_SEMITONES[root] + interval
                ) % 12

                notes.append(
                    NOTE_NAMES[note_index]
                )

                frequencies.append(
                    ROOT_FREQUENCIES[root]
                    * (2 ** (interval / 12))
                )

            # Internal chord name.
            chord_name = root + quality

            # Convert C7TH to C7, D7TH to D7, and so on.
            finger_key = (
                root + "7"
                if quality == "7TH"
                else root + quality
            )

            # Validate that the chord has a fingering entry.
            if finger_key not in CHORD_FINGERINGS:
                raise KeyError(
                    f"Missing guitar fingering: {finger_key}"
                )

            # Retrieve the fret shape and finger assignments.
            fret_shape = FRET_SHAPES[root][quality]
            fingering = CHORD_FINGERINGS[finger_key]

            # Each guitar chord must contain exactly six strings.
            if len(fret_shape) != 6 or len(fingering) != 6:
                raise ValueError(
                    f"Invalid guitar shape for {chord_name}"
                )

            # Store everything needed by the audio and UI.
            chords[chord_name] = {
                "root": root,
                "quality": quality,
                "notes": notes,
                "frequencies": frequencies,
                "fret": fret_shape,
                "fingers": fingering,
            }

    # There must be seven roots multiplied by three qualities.
    if len(chords) != 21:
        raise ValueError(
            f"Expected 21 chords, but created {len(chords)}"
        )

    return chords


# Create the chord dictionary once.
CHORDS = build_chords()


# ============================================================
# 9. DOWNLOAD THE MEDIAPIPE MODEL
# ============================================================

def download_model():
    """Download the MediaPipe model if it is not already available."""

    # Reuse the existing model if it appears to be a valid file.
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return

    print("Downloading MediaPipe hand model...")

    try:

        urllib.request.urlretrieve(
            MODEL_URL,
            str(MODEL_PATH)
        )

        # Reject an empty or incomplete download.
        if MODEL_PATH.stat().st_size <= 100_000:
            raise RuntimeError(
                "Downloaded model file is too small."
            )

    except Exception as error:

        raise RuntimeError(
            f"Could not download the hand model: {error}"
        ) from error


# ============================================================
# 10. CREATE THE HAND DETECTOR
# ============================================================

def create_hand_detector():
    """Create a MediaPipe hand detector configured for video input."""

    # Make sure the neural-network model has been downloaded.
    download_model()

    # VIDEO mode supports tracking between consecutive frames.
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

    # Return the configured detector.
    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# 11. CALCULATE JOINT ANGLES
# ============================================================

def calculate_angle(a, b, c):
    """Calculate the 3D angle at point b."""

    # Create vectors from the joint to its neighboring landmarks.
    vector1 = a - b
    vector2 = c - b

    # Calculate vector lengths.
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # Avoid division by zero.
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    # Normalize the dot product to obtain the angle.
    cosine = np.dot(vector1, vector2) / (norm1 * norm2)

    # Protect arccos from small floating-point errors.
    cosine = np.clip(cosine, -1.0, 1.0)

    # Return the angle in degrees.
    return float(
        np.degrees(np.arccos(cosine))
    )


# ============================================================
# 12. EXTRACT FINGER FEATURES
# ============================================================

def extract_features(world_landmarks):
    """Extract measurements describing the position of each finger.

    Non-thumb feature vector:
        PIP angle, DIP angle, extension ratio, wrist reach.

    Thumb feature vector:
        Thumb angle, distance from index base, distance from wrist.

    The same feature extraction is used during calibration and normal tracking.
    """

    # Convert MediaPipe landmark objects into a NumPy array.
    points = np.asarray(
        [
            [point.x, point.y, point.z]
            for point in world_landmarks
        ],
        dtype=np.float64
    )

    # Palm width is used to normalize distances.
    palm_width = max(
        float(np.linalg.norm(points[5] - points[17])),
        1e-6
    )

    # Landmark 0 represents the wrist.
    wrist = points[0]

    features = {}

    # --------------------------------------------------------
    # INDEX, MIDDLE, RING AND PINKY
    # --------------------------------------------------------

    for name, indices in FINGER_LANDMARKS.items():

        # Retrieve the four landmarks for this finger.
        mcp_i, pip_i, dip_i, tip_i = indices

        mcp = points[mcp_i]
        pip = points[pip_i]
        dip = points[dip_i]
        tip = points[tip_i]

        # Measure the PIP and DIP joint angles.
        pip_angle = calculate_angle(
            mcp, pip, dip
        )

        dip_angle = calculate_angle(
            pip, dip, tip
        )

        # Measure the first finger segment.
        first_segment = max(
            float(np.linalg.norm(pip - mcp)),
            1e-6
        )

        # Compare fingertip distance with the first segment.
        extension_ratio = (
            float(np.linalg.norm(tip - mcp))
            / first_segment
        )

        # Measure the fingertip's distance from the wrist.
        wrist_reach = (
            float(np.linalg.norm(tip - wrist))
            / palm_width
        )

        # Store all four measurements.
        features[name] = np.array(
            [
                pip_angle,
                dip_angle,
                extension_ratio,
                wrist_reach
            ],
            dtype=np.float64
        )

    # --------------------------------------------------------
    # THUMB
    # --------------------------------------------------------

    # Measure the thumb joint angle.
    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    # Measure distance from the thumb tip to the index base.
    thumb_from_index = (
        float(np.linalg.norm(points[4] - points[5]))
        / palm_width
    )

    # Measure thumb distance from the wrist.
    thumb_from_wrist = (
        float(np.linalg.norm(points[4] - wrist))
        / palm_width
    )

    # Save the thumb measurements.
    features["THUMB"] = np.array(
        [
            thumb_angle,
            thumb_from_index,
            thumb_from_wrist
        ],
        dtype=np.float64
    )

    return features


# ============================================================
# 13. GENERIC FINGER CLASSIFICATION
# ============================================================

def fallback_classification(
    feature_vector,
    finger_name,
    previous_state
):
    """Classify a finger using default geometric rules.

    These thresholds are a fallback for users who have not calibrated.
    Personalized calibration is recommended for better recognition.
    """

    # The thumb uses a separate feature vector.
    if finger_name == "THUMB":

        angle, thumb_ratio, wrist_ratio = feature_vector

        # Use lower thresholds when the thumb was already open.
        if previous_state:

            is_open = (
                angle >= 135
                and thumb_ratio >= 0.55
            )

        else:

            is_open = (
                angle >= 148
                and thumb_ratio >= 0.68
            )

        # This is a rough geometric confidence value.
        confidence = min(
            1.0,
            max(0.0, (thumb_ratio - 0.35) / 0.70)
        )

        return bool(is_open), confidence

    # Extract the four measurements for a normal finger.
    (
        pip_angle,
        dip_angle,
        extension_ratio,
        wrist_reach
    ) = feature_vector

    # Lower thresholds prevent an already open finger from flickering closed.
    if previous_state:

        is_open = (
            pip_angle >= 140
            and dip_angle >= 125
            and extension_ratio >= 1.22
        )

    else:

        is_open = (
            pip_angle >= 150
            and dip_angle >= 138
            and extension_ratio >= 1.35
        )

    # Calculate a rough confidence from angle and extension.
    angle_part = np.clip(
        (pip_angle - 110) / 70,
        0.0,
        1.0
    )

    extension_part = np.clip(
        (extension_ratio - 0.9) / 0.8,
        0.0,
        1.0
    )

    confidence = float(
        (angle_part + extension_part) / 2.0
    )

    return bool(is_open), confidence


# ============================================================
# 14. CALIBRATED FINGER CLASSIFICATION
# ============================================================

def classify_with_calibration(
    feature_vector,
    finger_name,
    profile,
    previous_state
):
    """Compare a live finger measurement with its calibrated poses.

    Calibration provides two feature centers:
    - one for a closed finger;
    - one for an extended finger.

    The current measurement is compared with both centers. A small uncertain
    region preserves the previous state instead of rapidly changing the finger.
    """

    # Retrieve this finger's calibration entry.
    entry = profile[finger_name]

    closed_center = np.asarray(
        entry["closed"],
        dtype=np.float64
    )

    open_center = np.asarray(
        entry["open"],
        dtype=np.float64
    )

    # Scale prevents one feature with large numerical values from dominating.
    scale = np.maximum(
        np.asarray(entry["scale"], dtype=np.float64),
        1e-6
    )

    # Calculate normalized squared distance from the closed calibration center.
    distance_closed = float(
        np.mean(
            ((feature_vector - closed_center) / scale) ** 2
        )
    )

    # Calculate normalized squared distance from the open calibration center.
    distance_open = float(
        np.mean(
            ((feature_vector - open_center) / scale) ** 2
        )
    )

    # A measurement close to open has a small open distance and a larger
    # closed distance. This produces a score close to 1.
    denominator = max(
        distance_closed + distance_open,
        1e-8
    )

    open_score = (
        distance_closed / denominator
    )

    # Convert the distance from the ambiguous midpoint into a confidence score.
    confidence = float(
        min(
            1.0,
            abs(open_score - 0.5) * 2.0
        )
    )

    # Decide whether the finger is open or closed.
    if open_score >= 0.60:

        state = True

    elif open_score <= 0.40:

        state = False

    else:

        # Keep the previous state if the feature is ambiguous.
        state = bool(previous_state)

    return bool(state), confidence


# ============================================================
# 15. RECOGNIZE ALL FIVE FINGERS
# ============================================================

def recognize_fingers(
    world_landmarks,
    previous_states=None,
    calibration=None
):
    """Recognize each finger and return its count, states, and diagnostics.

    The calibration argument must be the profile for one hand role:
    ROOT or QUALITY. The two hands have separate profiles.
    """

    # If this is the first frame, assume all fingers are folded.
    if previous_states is None:
        previous_states = {}

    # Extract measurements from all hand landmarks.
    features = extract_features(world_landmarks)

    states = {}
    diagnostics = {}

    # Classify every finger independently.
    for finger_name in FINGER_NAMES:

        previous_state = bool(
            previous_states.get(finger_name, False)
        )

        # Use personalized recognition when a profile is available.
        if calibration is not None:

            state, confidence = classify_with_calibration(
                features[finger_name],
                finger_name,
                calibration,
                previous_state
            )

        # Otherwise use generic fallback thresholds.
        else:

            state, confidence = fallback_classification(
                features[finger_name],
                finger_name,
                previous_state
            )

        states[finger_name] = bool(state)

        vector = features[finger_name]

        # Keep measurements for the diagnostics panel.
        diagnostics[finger_name] = {
            "pip": float(vector[0]),

            "dip": (
                float(vector[1])
                if finger_name != "THUMB"
                else 0.0
            ),

            "ratio": (
                float(vector[2])
                if finger_name != "THUMB"
                else float(vector[1])
            ),

            "score": float(confidence),
            "confidence": float(confidence),
            "open": bool(state),
        }

    # The thumb is NOT included in the four-finger count.
    count = sum(
        1
        for name in FINGER_LANDMARKS
        if states[name]
    )

    return count, states, diagnostics


# ============================================================
# 16. CALIBRATION SESSION
# ============================================================

def create_calibration_session():
    """Create the state dictionary for the two-pose calibration wizard."""

    return {
        "active": False,
        "phase": 0,
        "phase_start": 0.0,
        "samples": [],
        "closed_samples": [],
        "open_samples": [],
        "notice": "",
    }


def start_calibration(session, now):
    """Start a new calibration and clear previous temporary samples."""

    session.update({
        "active": True,
        "phase": 0,
        "phase_start": now,
        "samples": [],
        "closed_samples": [],
        "open_samples": [],
        "notice": "",
    })

    print(
        "Calibration started: hold BOTH hands in relaxed fists."
    )


def cancel_calibration(session):
    """Cancel the current calibration without deleting the saved profile."""

    session["active"] = False
    session["samples"] = []
    session["notice"] = "Calibration cancelled."


def collect_calibration_sample(hands):
    """Collect features from the left and right hands separately.

    The left-most hand is the ROOT role and the right-most is the QUALITY role.
    The profiles are deliberately not averaged together.
    """

    return {
        "ROOT": extract_features(hands[0][2]),
        "QUALITY": extract_features(hands[1][2]),
    }


# ============================================================
# 17. BUILD A PERSONAL CALIBRATION PROFILE
# ============================================================

def make_calibration_profile(
    closed_samples,
    open_samples
):
    """Calculate median closed/open feature centers and robust feature scales.

    Median measurements reduce the influence of occasional bad landmark frames.
    A minimum scale prevents tiny variations from making the classifier unstable.
    """

    profile = {}

    # Minimum variation allowed for each measurement.
    scale_floors = {

        "THUMB": np.array([
            8.0, 0.08, 0.08
        ]),

        "INDEX": np.array([
            8.0, 8.0, 0.08, 0.08
        ]),

        "MIDDLE": np.array([
            8.0, 8.0, 0.08, 0.08
        ]),

        "RING": np.array([
            8.0, 8.0, 0.08, 0.08
        ]),

        "PINKY": np.array([
            8.0, 8.0, 0.08, 0.08
        ]),
    }

    for finger in FINGER_NAMES:

        # Convert all samples into numerical arrays.
        closed_array = np.asarray(
            [sample[finger] for sample in closed_samples],
            dtype=np.float64
        )

        open_array = np.asarray(
            [sample[finger] for sample in open_samples],
            dtype=np.float64
        )

        # The median is used instead of the mean to reduce outlier influence.
        closed_center = np.median(
            closed_array,
            axis=0
        )

        open_center = np.median(
            open_array,
            axis=0
        )

        # Combine both pose groups for spread estimation.
        all_values = np.vstack(
            (closed_array, open_array)
        )

        combined_median = np.median(
            all_values,
            axis=0
        )

        # Median absolute deviation provides a robust spread estimate.
        mad = np.median(
            np.abs(all_values - combined_median),
            axis=0
        )

        robust_spread = 1.4826 * mad

        # How far apart are the two pose centers?
        class_separation = np.abs(
            open_center - closed_center
        )

        # Scale each feature using noise, pose separation, and its minimum floor.
        scale = np.maximum.reduce((
            scale_floors[finger],
            robust_spread * 2.0,
            class_separation * 0.35,
        ))

        # Save all three values needed for later recognition.
        profile[finger] = {
            "closed": closed_center.tolist(),
            "open": open_center.tolist(),
            "scale": scale.tolist(),
        }

    return profile


# ============================================================
# 18. SAVE CALIBRATION
# ============================================================

def save_calibration(profile):
    """Save the personal calibration profile as JSON."""

    # Version the file so incompatible future formats can be detected.
    document = {
        "version": 2,
        "roles": profile,
    }

    try:

        CALIBRATION_PATH.write_text(
            json.dumps(document, indent=2),
            encoding="utf-8"
        )

        return True

    except OSError as error:

        print(
            f"Could not save calibration: {error}"
        )

        return False


# ============================================================
# 19. LOAD CALIBRATION
# ============================================================

def load_calibration():
    """Load and validate the saved root-hand and quality-hand profiles."""

    if not CALIBRATION_PATH.exists():
        return None

    try:

        document = json.loads(
            CALIBRATION_PATH.read_text(
                encoding="utf-8"
            )
        )

        # Only version 2 has the separate ROOT and QUALITY format.
        if document.get("version") != 2:
            raise ValueError(
                "Unsupported calibration file version."
            )

        roles = document["roles"]

        # Verify every role and finger exists.
        for role in ROLE_NAMES:

            role_profile = roles[role]

            for finger in FINGER_NAMES:

                entry = role_profile[finger]

                expected_length = (
                    3 if finger == "THUMB" else 4
                )

                # Every feature vector must have its expected length.
                for key in ("closed", "open", "scale"):

                    if len(entry[key]) != expected_length:

                        raise ValueError(
                            f"Invalid {role}/{finger}/{key} feature length."
                        )

        return roles

    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError
    ) as error:

        print(
            "Saved calibration could not be loaded:",
            error
        )

        return None


# ============================================================
# 20. UPDATE CALIBRATION
# ============================================================

def update_calibration(
    session,
    hands,
    now
):
    """Capture the fist pose, then the open-palm pose.

    Both hands must be detected during capture. If there are not enough samples,
    the current phase is repeated rather than saving an incomplete profile.
    """

    if not session["active"]:
        return None

    elapsed = now - session["phase_start"]

    phase_duration = (
        CALIBRATION_SETTLE_SECONDS
        + CALIBRATION_CAPTURE_SECONDS
    )

    # Start collecting after the user has had time to position their hands.
    if (
        len(hands) >= 2
        and elapsed >= CALIBRATION_SETTLE_SECONDS
    ):

        session["samples"].append(
            collect_calibration_sample(hands)
        )

    # Wait for the whole capture period to finish.
    if elapsed < phase_duration:
        return None

    # Repeat the phase if too few valid frames were collected.
    if len(session["samples"]) < CALIBRATION_MIN_SAMPLES:

        session["phase_start"] = now
        session["samples"].clear()

        session["notice"] = (
            "Show BOTH hands clearly; repeating this pose."
        )

        print(session["notice"])

        return None

    # Save the first capture as the folded-finger reference.
    if session["phase"] == 0:

        session["closed_samples"] = (
            session["samples"][:]
        )

        session["samples"].clear()

        # Move to the open-palm phase.
        session["phase"] = 1
        session["phase_start"] = now

        session["notice"] = (
            "Good. Now spread all fingers on BOTH hands."
        )

        print(session["notice"])

        return None

    # Save the second capture as the extended-finger reference.
    session["open_samples"] = (
        session["samples"][:]
    )

    completed_profile = {}

    # Build an independent classifier for each hand role.
    for role in ROLE_NAMES:

        closed_for_role = [
            sample[role]
            for sample in session["closed_samples"]
        ]

        open_for_role = [
            sample[role]
            for sample in session["open_samples"]
        ]

        completed_profile[role] = make_calibration_profile(
            closed_for_role,
            open_for_role
        )

    # Calibration is complete.
    session["active"] = False

    session["notice"] = (
        "Calibration complete and saved."
    )

    save_calibration(completed_profile)

    print(
        "Calibration complete. Saved to:",
        CALIBRATION_PATH
    )

    return completed_profile


# ============================================================
# 21. DETECT HANDS
# ============================================================

def detect_hands(
    detector,
    frame,
    timestamp
):
    """Detect hands and return them sorted from left to right."""

    # OpenCV supplies BGR; MediaPipe expects RGB.
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Convert the image into MediaPipe's image representation.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run the video-mode detector.
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    hands = []

    # If there are no hands, return an empty list.
    if not result.hand_landmarks:
        return hands

    # Save image-space and world-space landmarks for each hand.
    for index, image_landmarks in enumerate(
        result.hand_landmarks
    ):

        if index >= len(result.hand_world_landmarks):
            continue

        world_landmarks = (
            result.hand_world_landmarks[index]
        )

        # The average x coordinate identifies the hand's horizontal position.
        center_x = float(
            np.mean([
                point.x
                for point in image_landmarks
            ])
        )

        hands.append((
            center_x,
            image_landmarks,
            world_landmarks
        ))

    # Screen-left hand is ROOT; screen-right hand is QUALITY.
    hands.sort(
        key=lambda hand: hand[0]
    )

    return hands


# ============================================================
# 22. FINGER HISTORY AND SMOOTHING
# ============================================================

def create_finger_histories():
    """Create an independent short history for every finger."""

    return {
        name: deque(maxlen=FINGER_HISTORY_SIZE)
        for name in FINGER_NAMES
    }


def stabilize_fingers(raw_states, histories):
    """Use majority voting to smooth each finger independently."""

    stable_states = {}

    for name, raw_value in raw_states.items():

        # Store this frame's open/closed result.
        histories[name].append(
            bool(raw_value)
        )

        # Count open and closed observations.
        open_votes = sum(
            histories[name]
        )

        closed_votes = (
            len(histories[name]) - open_votes
        )

        # The majority result becomes the stable finger state.
        if open_votes > closed_votes:

            stable_states[name] = True

        elif closed_votes > open_votes:

            stable_states[name] = False

        else:

            # Use the newest result if the votes are tied.
            stable_states[name] = bool(raw_value)

    return stable_states


def stable_value(history):
    """Return a complete gesture only when enough recent votes agree."""

    if not history:
        return None

    value, votes = Counter(history).most_common(1)[0]

    return (
        value
        if votes >= REQUIRED_GESTURE_VOTES
        else None
    )


# ============================================================
# 23. FINGERTIP TRAILS
# ============================================================

def create_trails():
    """Create small queues for recent fingertip positions."""

    return {
        name: deque(maxlen=12)
        for name in TRACE_LANDMARKS
    }


# ============================================================
# 24. DRAW HAND AND FINGER TRACING
# ============================================================

def draw_hand_landmarks(
    frame,
    landmarks,
    states,
    trails,
    label
):
    """Draw the hand skeleton, colored finger bones, and fingertip trails.

    Green means extended; red means folded. Different finger names are shown
    beside the fingertips so the user can see what the program is recognizing.
    """

    height, width = frame.shape[:2]

    # --------------------------------------------------------
    # DRAW THE PALM SKELETON
    # --------------------------------------------------------

    for start, end in HAND_CONNECTIONS:

        p1 = landmarks[start]
        p2 = landmarks[end]

        start_point = (
            int(p1.x * width),
            int(p1.y * height)
        )

        end_point = (
            int(p2.x * width),
            int(p2.y * height)
        )

        cv2.line(
            frame,
            start_point,
            end_point,
            (105, 120, 140),
            2,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # DRAW EACH FINGER
    # --------------------------------------------------------

    for finger_name, indices in TRACE_LANDMARKS.items():

        # Determine the recognized state of this finger.
        is_open = states.get(
            finger_name,
            False
        )

        # Assign green to open fingers and red to folded fingers.
        color = (
            FINGER_COLORS[finger_name]
            if is_open
            else DANGER
        )

        # Draw all three bones making up the finger.
        for index in range(len(indices) - 1):

            p1 = landmarks[indices[index]]
            p2 = landmarks[indices[index + 1]]

            point1 = (
                int(p1.x * width),
                int(p1.y * height)
            )

            point2 = (
                int(p2.x * width),
                int(p2.y * height)
            )

            cv2.line(
                frame,
                point1,
                point2,
                color,
                4,
                cv2.LINE_AA
            )

        # Obtain the fingertip position.
        tip = landmarks[indices[-1]]

        tip_position = (
            int(tip.x * width),
            int(tip.y * height)
        )

        # Add this position to the movement history.
        trails[finger_name].append(
            tip_position
        )

        trail_points = list(
            trails[finger_name]
        )

        # Draw the recent fingertip path.
        if len(trail_points) >= 2:

            cv2.polylines(
                frame,
                [
                    np.asarray(
                        trail_points,
                        dtype=np.int32
                    )
                ],
                False,
                color,
                2,
                cv2.LINE_AA
            )

        # Draw a colored fingertip marker.
        cv2.circle(
            frame,
            tip_position,
            7,
            color,
            -1,
            cv2.LINE_AA
        )

        # Add a white outline.
        cv2.circle(
            frame,
            tip_position,
            9,
            TEXT,
            1,
            cv2.LINE_AA
        )

        # Position the three-letter finger label near the fingertip.
        label_x = min(
            max(tip_position[0] + 7, 0),
            max(0, width - 65)
        )

        label_y = max(
            tip_position[1] - 8,
            18
        )

        cv2.putText(
            frame,
            finger_name[:3],
            (label_x, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # LABEL THE HAND ROLE
    # --------------------------------------------------------

    wrist = landmarks[0]

    wrist_position = (
        int(wrist.x * width),
        int(wrist.y * height)
    )

    cv2.putText(
        frame,
        label,
        (
            max(5, wrist_position[0] - 35),
            max(20, wrist_position[1] - 18)
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        GOLD,
        1,
        cv2.LINE_AA
    )


# ============================================================
# 25. DRAW DIAGNOSTICS
# ============================================================

def draw_debug(
    frame,
    root_data,
    quality_data,
    calibrated
):
    """Display the raw recognition measurements for both hands."""

    # The first line tells the user whether their personal profile is active.
    lines = [
        "CALIBRATED"
        if calibrated
        else "FALLBACK - PRESS C TO CALIBRATE"
    ]

    # Add one section for each detected hand.
    for hand_label, data in (
        ("ROOT", root_data),
        ("QUALITY", quality_data)
    ):

        if data is None:
            continue

        lines.append(
            f"{hand_label} HAND"
        )

        # Show state, joint angle, extension ratio, and confidence.
        for finger, values in data.items():

            state_text = (
                "OPEN"
                if values["open"]
                else "CLOSED"
            )

            lines.append(
                f"{finger[:3]} {state_text} "
                f"A:{values['pip']:.0f} "
                f"R:{values['ratio']:.2f} "
                f"C:{values['confidence']:.2f}"
            )

    # Draw the collected diagnostic lines onto the camera image.
    y = 105

    for line in lines:

        if line.startswith("FALLBACK"):
            color = WARNING

        elif "OPEN" in line:
            color = SUCCESS

        elif "CLOSED" in line:
            color = MUTED

        else:
            color = GOLD if "HAND" in line else TEXT

        cv2.putText(
            frame,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            color,
            1,
            cv2.LINE_AA
        )

        y += 17


# ============================================================
# 26. SELECT A CHORD
# ============================================================

def select_chord(
    root_gesture,
    quality_count
):
    """Translate the two stable hand gestures into a chord name."""

    # Both hands need valid gestures before a chord can be selected.
    if root_gesture is None or quality_count is None:
        return None

    # Unpack the root-hand gesture.
    root_count, thumb_open = root_gesture

    # Find the musical root note.
    root = ROOT_GESTURES.get(
        (bool(thumb_open), root_count)
    )

    # Find the chord quality.
    quality = QUALITY_GESTURES.get(
        quality_count
    )

    # Do not guess if either gesture is invalid.
    if root is None or quality is None:
        return None

    chord_name = root + quality

    return (
        chord_name
        if chord_name in CHORDS
        else None
    )


# ============================================================
# 27. GENERATE CHORD AUDIO
# ============================================================

def generate_chord(
    frequencies,
    filename
):
    """Generate a harmonically enriched chord and save it as a WAV file."""

    sample_count = int(
        SAMPLE_RATE * LOOP_DURATION
    )

    # Create the time coordinate for every audio sample.
    time_values = (
        np.arange(sample_count, dtype=np.float64)
        / SAMPLE_RATE
    )

    # Start with silence.
    audio = np.zeros(
        sample_count,
        dtype=np.float64
    )

    # Add the fundamental and three harmonics of each chord note.
    for frequency in frequencies:

        audio += (
            np.sin(
                2 * np.pi * frequency * time_values
            )
            + 0.30 * np.sin(
                2 * np.pi * frequency * 2 * time_values
            )
            + 0.12 * np.sin(
                2 * np.pi * frequency * 3 * time_values
            )
            + 0.05 * np.sin(
                2 * np.pi * frequency * 4 * time_values
            )
        )

    # Normalize the audio to prevent clipping.
    peak = float(
        np.max(np.abs(audio))
    )

    if peak > 0:
        audio /= peak

    # Reduce the overall output volume.
    audio *= 0.65

    # Fade the beginning and ending to reduce audible clicks.
    fade_size = int(
        SAMPLE_RATE * 0.015
    )

    if fade_size > 0:

        fade = np.linspace(
            0.0,
            1.0,
            fade_size
        )

        audio[:fade_size] *= fade
        audio[-fade_size:] *= fade[::-1]

    # Convert floating-point samples to 16-bit PCM.
    pcm = np.clip(
        audio * 32767,
        -32768,
        32767
    ).astype(np.int16)

    # Write the WAV file.
    with wave.open(
        str(filename),
        "wb"
    ) as wav_file:

        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(SAMPLE_RATE)

        wav_file.writeframes(
            pcm.tobytes()
        )


def create_chord_files():
    """Create WAV files for all chords that do not already have audio."""

    # Create the sound directory if it does not exist.
    SOUND_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Preparing 21 chord sounds...")

    for chord_name, chord_data in CHORDS.items():

        audio_path = (
            SOUND_DIR / f"{chord_name}.wav"
        )

        # Generate the sound only if it is missing.
        if not audio_path.exists():

            generate_chord(
                chord_data["frequencies"],
                audio_path
            )

    print("Chord sounds ready.")


# ============================================================
# 28. PLAY AND STOP CHORD AUDIO
# ============================================================

def stop_chord():
    """Stop any sound currently being played by winsound."""

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
    """Loop a valid chord WAV file asynchronously."""

    if winsound is None or chord_name not in CHORDS:
        return

    audio_path = (
        SOUND_DIR / f"{chord_name}.wav"
    )

    if not audio_path.exists():

        print(
            f"Missing audio file: {audio_path}"
        )

        return

    try:

        winsound.PlaySound(
            str(audio_path),
            winsound.SND_FILENAME
            | winsound.SND_ASYNC
            | winsound.SND_LOOP
        )

    except Exception as error:

        print(
            "Audio playback error:",
            error
        )


# ============================================================
# 29. DRAW TEXT
# ============================================================

def draw_text(
    image,
    text,
    position,
    scale=0.6,
    color=TEXT,
    thickness=1
):
    """Draw anti-aliased text using OpenCV."""

    cv2.putText(
        image,
        str(text),
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# 30. PREPARE CAMERA VIEW
# ============================================================

def prepare_camera_view(frame):
    """Fit the camera image into the left panel without stretching it."""

    height, width = frame.shape[:2]

    # Choose the scale that fits both target dimensions.
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

    # Keep the sharp camera image at its original aspect ratio.
    resized = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # Create a blurred background that fills the whole panel.
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

    # Center the sharp image.
    x = (CAMERA_VIEW_WIDTH - new_width) // 2
    y = (DISPLAY_HEIGHT - new_height) // 2

    canvas[
        y:y + new_height,
        x:x + new_width
    ] = resized

    return canvas


# ============================================================
# 31. DRAW THE GUITAR FRETBOARD
# ============================================================

def draw_fretboard(
    panel,
    chord_data
):
    """Draw six strings, fret lines, open/muted markers, and finger numbers."""

    fret_shape = chord_data["fret"]
    fingers = chord_data["fingers"]

    # Find the frets actually used by the chord.
    fretted_notes = [
        int(value)
        for value in fret_shape
        if value.isdigit() and value != "0"
    ]

    # Most chords start at fret 1.
    first_fret = 1

    # Shift the diagram for chords positioned higher on the neck.
    if fretted_notes and min(fretted_notes) > 1:
        first_fret = min(fretted_notes)

    # The diagram shows four fret spaces.
    if (
        fretted_notes
        and max(fretted_notes) - first_fret > 3
    ):
        first_fret = max(
            1,
            max(fretted_notes) - 3
        )

    # Diagram coordinates.
    x_left = 300
    x_right = 590
    top = 151

    fret_spacing = 25

    string_spacing = (
        x_right - x_left
    ) / 5

    draw_text(
        panel,
        "FRETBOARD",
        (300, 133),
        0.42,
        MINT,
        2
    )

    # --------------------------------------------------------
    # DRAW SIX GUITAR STRINGS
    # --------------------------------------------------------

    for index, string_name in enumerate(STRING_NAMES):

        x = int(
            x_left + index * string_spacing
        )

        # Label the string.
        draw_text(
            panel,
            string_name,
            (x - 4, top - 9),
            0.35,
            GOLD,
            1
        )

        # Draw its vertical line.
        cv2.line(
            panel,
            (x, top),
            (x, top + 4 * fret_spacing),
            (185, 195, 210),
            1,
            cv2.LINE_AA
        )

    # --------------------------------------------------------
    # DRAW FIVE HORIZONTAL FRET LINES
    # --------------------------------------------------------

    for row in range(5):

        y = top + row * fret_spacing

        # Use a thicker nut line for chords starting at fret 1.
        thickness = (
            3
            if row == 0 and first_fret == 1
            else 1
        )

        cv2.line(
            panel,
            (x_left, y),
            (x_right, y),
            (185, 195, 210),
            thickness,
            cv2.LINE_AA
        )

    # Display the starting fret number when needed.
    if first_fret > 1:

        draw_text(
            panel,
            str(first_fret),
            (278, top + 16),
            0.36,
            TEXT
        )

    # --------------------------------------------------------
    # DETECT BARRE GROUPS
    # --------------------------------------------------------

    # Group strings held by the same finger at the same fret.
    barre_groups = {}

    for string_index, (fret_value, finger) in enumerate(
        zip(fret_shape, fingers)
    ):

        if (
            fret_value.isdigit()
            and fret_value != "0"
            and finger not in (None, 0)
        ):

            key = (
                int(fret_value),
                finger
            )

            barre_groups.setdefault(
                key,
                []
            ).append(string_index)

    # Draw horizontal bars for adjacent strings.
    for (barre_fret, finger), indices in barre_groups.items():

        contiguous_groups = []
        current_group = []

        for string_index in indices:

            if (
                current_group
                and string_index != current_group[-1] + 1
            ):

                contiguous_groups.append(
                    current_group
                )

                current_group = []

            current_group.append(
                string_index
            )

        if current_group:
            contiguous_groups.append(current_group)

        # Only draw a barre when two or more adjacent strings share it.
        for group in contiguous_groups:

            if len(group) < 2:
                continue

            y = int(
                top
                + (barre_fret - first_fret + 0.5)
                * fret_spacing
            )

            x1 = int(
                x_left + group[0] * string_spacing
            )

            x2 = int(
                x_left + group[-1] * string_spacing
            )

            cv2.line(
                panel,
                (x1, y),
                (x2, y),
                FINGER_NUMBER_COLORS[finger],
                7,
                cv2.LINE_AA
            )

    # --------------------------------------------------------
    # DRAW OPEN STRINGS, MUTED STRINGS AND FINGER DOTS
    # --------------------------------------------------------

    for index, (fret_value, finger) in enumerate(
        zip(fret_shape, fingers)
    ):

        x = int(
            x_left + index * string_spacing
        )

        # Muted strings are marked with X.
        if fret_value.lower() == "x":

            draw_text(
                panel,
                "X",
                (x - 5, top - 20),
                0.40,
                DANGER,
                2
            )

            continue

        fret_number = int(fret_value)

        # Open strings receive a hollow circle above the nut.
        if fret_number == 0:

            cv2.circle(
                panel,
                (x, top - 16),
                5,
                MUTED,
                1,
                cv2.LINE_AA
            )

            continue

        # Find the displayed fret position.
        relative_fret = (
            fret_number - first_fret
        )

        if not 0 <= relative_fret <= 3:
            continue

        # Finger dots are placed between the two fret lines.
        y = int(
            top + (relative_fret + 0.5) * fret_spacing
        )

        finger_color = FINGER_NUMBER_COLORS.get(
            finger,
            TEXT
        )

        cv2.circle(
            panel,
            (x, y),
            9,
            finger_color,
            -1,
            cv2.LINE_AA
        )

        # Write the finger number inside the dot.
        if finger is not None:

            draw_text(
                panel,
                str(finger),
                (x - 4, y + 4),
                0.30,
                BACKGROUND,
                1
            )

    # Finger legend.
    draw_text(
        panel,
        "1 Index  2 Middle  3 Ring  4 Pinky",
        (286, 267),
        0.31,
        TEXT
    )

    draw_text(
        panel,
        "X muted   O open",
        (286, 282),
        0.30,
        MUTED
    )


# ============================================================
# 32. DRAW THE FINGER GUIDE
# ============================================================

def draw_finger_guide(
    panel,
    chord_data
):
    """Draw the title and graphical fretboard for the selected chord."""

    draw_text(
        panel,
        "FINGER GUIDE",
        (278, 119),
        0.43,
        MINT,
        2
    )

    draw_fretboard(
        panel,
        chord_data
    )


# ============================================================
# 33. CALIBRATION CARD
# ============================================================

def draw_calibration_card(
    panel,
    session,
    now
):
    """Display calibration instructions and its progress bar."""

    cv2.rectangle(
        panel,
        (18, 100),
        (622, 286),
        CARD,
        -1
    )

    cv2.rectangle(
        panel,
        (18, 100),
        (622, 286),
        BORDER,
        1
    )

    draw_text(
        panel,
        "PERSONAL HAND CALIBRATION",
        (35, 132),
        0.56,
        MINT,
        2
    )

    # Show the instruction matching the current calibration phase.
    if session["phase"] == 0:

        instruction = (
            "1/2  Make a relaxed fist with BOTH hands."
        )

        detail = (
            "Keep both thumbs tucked in. Hold still."
        )

    else:

        instruction = (
            "2/2  Open BOTH hands and spread all fingers."
        )

        detail = (
            "Face your palms toward the camera. Hold still."
        )

    draw_text(
        panel,
        instruction,
        (35, 175),
        0.43,
        TEXT
    )

    draw_text(
        panel,
        detail,
        (35, 203),
        0.40,
        MUTED
    )

    # Calculate the progress after the settling interval.
    elapsed = now - session["phase_start"]

    progress = min(
        1.0,
        max(
            0.0,
            (
                elapsed - CALIBRATION_SETTLE_SECONDS
            ) / CALIBRATION_CAPTURE_SECONDS
        )
    )

    # Draw the progress-bar background.
    bar_x1 = 35
    bar_y1 = 229
    bar_x2 = 600
    bar_y2 = 249

    cv2.rectangle(
        panel,
        (bar_x1, bar_y1),
        (bar_x2, bar_y2),
        BORDER,
        -1
    )

    # Calculate the width of the filled portion.
    fill_x = (
        bar_x1
        + int((bar_x2 - bar_x1) * progress)
    )

    cv2.rectangle(
        panel,
        (bar_x1, bar_y1),
        (fill_x, bar_y2),
        SUCCESS,
        -1
    )

    # Show the number of usable samples collected so far.
    draw_text(
        panel,
        f"Samples collected: {len(session['samples'])}",
        (35, 273),
        0.36,
        TEXT
    )


# ============================================================
# 34. DRAW THE RIGHT-SIDE PANEL
# ============================================================

def draw_panel(
    display,
    active_chord,
    playing_chord,
    root_gesture,
    quality_count,
    hand_count,
    calibration_session,
    calibration_loaded,
    now
):
    """Draw the chord card, chord guide, library, and application status."""

    # Work directly on the right half of the display.
    panel = display[:, CAMERA_VIEW_WIDTH:]

    panel[:] = PANEL

    # --------------------------------------------------------
    # HEADER
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # CALIBRATION OR CURRENT CHORD
    # --------------------------------------------------------

    if calibration_session["active"]:

        draw_calibration_card(
            panel,
            calibration_session,
            now
        )

    else:

        # Draw the current-chord card.
        cv2.rectangle(
            panel,
            (18, 100),
            (622, 286),
            CARD,
            -1
        )

        cv2.rectangle(
            panel,
            (18, 100),
            (622, 286),
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

            # Current chord name and quality.
            draw_text(
                panel,
                active_chord,
                (35, 181),
                1.05,
                MINT,
                3
            )

            draw_text(
                panel,
                chord_data["quality"],
                (35, 207),
                0.46,
                TEXT
            )

            draw_text(
                panel,
                "Shape: " + chord_data["fret"],
                (35, 235),
                0.40,
                MUTED
            )

            # Show the graphical fretboard.
            draw_finger_guide(
                panel,
                chord_data
            )

        else:

            # Instructions displayed before a chord is selected.
            draw_text(
                panel,
                "Waiting for gestures...",
                (35, 183),
                0.55,
                MUTED
            )

            draw_text(
                panel,
                "Press C to calibrate your hands",
                (35, 220),
                0.40,
                GOLD
            )

            draw_text(
                panel,
                "or use default recognition",
                (35, 245),
                0.36,
                MUTED
            )

        # Show whether a personal calibration profile is loaded.
        status = (
            "CALIBRATED"
            if calibration_loaded
            else "NOT CALIBRATED - PRESS C"
        )

        status_color = (
            SUCCESS
            if calibration_loaded
            else GOLD
        )

        draw_text(
            panel,
            status,
            (35, 270),
            0.31,
            status_color
        )

    # --------------------------------------------------------
    # AUDIO STATUS
    # --------------------------------------------------------

    if playing_chord:

        draw_text(
            panel,
            "PLAYING",
            (500, 115),
            0.32,
            SUCCESS,
            2
        )

    elif not calibration_session["active"]:

        draw_text(
            panel,
            "SOUND OFF",
            (500, 115),
            0.32,
            MUTED
        )

    # --------------------------------------------------------
    # HAND STATUS AND CURRENT GESTURES
    # --------------------------------------------------------

    draw_text(
        panel,
        f"HANDS: {hand_count}/2",
        (22, 309),
        0.42,
        SUCCESS if hand_count >= 2 else GOLD,
        2
    )

    draw_text(
        panel,
        "LEFT HAND: ROOT",
        (22, 337),
        0.38,
        BLUE
    )

    draw_text(
        panel,
        "RIGHT HAND: QUALITY",
        (320, 337),
        0.38,
        GOLD
    )

    # Display the root-hand count and thumb state.
    if root_gesture is None:

        root_text = "Waiting..."

    else:

        count, thumb_open = root_gesture

        thumb_label = (
            "OPEN"
            if thumb_open
            else "CLOSED"
        )

        root_text = (
            f"Fingers: {count} | Thumb: {thumb_label}"
        )

    draw_text(
        panel,
        root_text,
        (22, 360),
        0.35,
        TEXT
    )

    # Display the quality-hand count.
    quality_text = (
        "Waiting..."
        if quality_count is None
        else f"Fingers: {quality_count}"
    )

    draw_text(
        panel,
        quality_text,
        (320, 360),
        0.35,
        TEXT
    )

    # --------------------------------------------------------
    # 21-CHORD LIBRARY
    # --------------------------------------------------------

    draw_text(
        panel,
        "CHORD LIBRARY",
        (22, 391),
        0.50,
        TEXT,
        2
    )

    grid_left = 18
    root_width = 65
    gap = 6

    grid_top = 402
    row_height = 34
    row_gap = 1

    grid_right = 622

    # Divide the available width among the three qualities.
    chord_width = (
        grid_right
        - grid_left
        - root_width
        - 3 * gap
    ) // 3

    # Each root note receives one row.
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
            (grid_left + 24, y1 + 23),
            0.53,
            MINT,
            2
        )

        # Draw Major, Minor, and 7th cells.
        for column, quality in enumerate(QUALITIES):

            x1 = (
                grid_left
                + root_width
                + gap
                + column * (chord_width + gap)
            )

            x2 = x1 + chord_width

            chord_name = root + quality

            # Highlight the selected chord.
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
                (x1 + 9, y1 + 22),
                0.39,
                TEXT
            )

    # --------------------------------------------------------
    # FOOTER
    # --------------------------------------------------------

    draw_text(
        panel,
        "C: Calibrate | D: Debug | R: Reset | SPACE: Replay | Q: Quit",
        (18, 679),
        0.32,
        MUTED
    )

    draw_text(
        panel,
        "Green = extended   Red = folded   Orange = thumb",
        (18, 706),
        0.30,
        MUTED
    )


# ============================================================
# 35. COMPOSE THE COMPLETE DISPLAY
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
    quality_diagnostics,
    calibration_session,
    calibration_loaded,
    now
):
    """Combine the camera view, tracing overlay, and right-side UI."""

    # Prepare the camera image for the left-side panel.
    camera_view = prepare_camera_view(frame)

    # Create the complete display canvas.
    display = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    display[:, :CAMERA_VIEW_WIDTH] = camera_view
    display[:, CAMERA_VIEW_WIDTH:] = PANEL

    # Camera-panel heading.
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

    # Show calibration instructions over the camera during the wizard.
    if calibration_session["active"]:

        pose = (
            "FISTS"
            if calibration_session["phase"] == 0
            else "OPEN PALMS"
        )

        draw_text(
            display,
            f"CALIBRATION: HOLD BOTH HANDS AS {pose}",
            (15, 92),
            0.40,
            GOLD,
            2
        )

    # Display debug measurements when diagnostics are enabled.
    elif debug_enabled:

        draw_debug(
            display[:, :CAMERA_VIEW_WIDTH],
            root_diagnostics,
            quality_diagnostics,
            calibration_loaded
        )

    # Draw the right-side interface.
    draw_panel(
        display,
        active_chord,
        playing_chord,
        root_gesture,
        quality_count,
        hand_count,
        calibration_session,
        calibration_loaded,
        now
    )

    return display


# ============================================================
# 36. MAIN APPLICATION
# ============================================================

def run_application():
    """Initialize the camera, detector, audio and real-time processing loop."""

    print("=" * 48)
    print("VIRTUAL GUITAR - PERSONAL CALIBRATION EDITION")
    print("=" * 48)

    print(
        "Chord catalogue validated:",
        len(CHORDS),
        "chords"
    )

    if winsound is None:

        print(
            "WARNING: winsound is unavailable; audio playback is disabled."
        )

    # Generate the missing WAV files before starting the camera.
    create_chord_files()

    # Load calibration if this computer has used the application before.
    calibration_profiles = load_calibration()

    calibration_loaded = (
        calibration_profiles is not None
    )

    if calibration_loaded:

        print("Personal calibration loaded.")

    else:

        print(
            "No personal calibration found. Press C in the window to calibrate."
        )

    # --------------------------------------------------------
    # OPEN CAMERA
    # --------------------------------------------------------

    print("Opening camera...")

    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    # Try the default backend if DirectShow fails.
    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    if not camera.isOpened():

        raise RuntimeError(
            "Could not open camera. Check CAMERA_INDEX."
        )

    # Request the desired camera resolution.
    camera.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        CAMERA_WIDTH
    )

    camera.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        CAMERA_HEIGHT
    )

    # A small buffer reduces processing delay.
    camera.set(
        cv2.CAP_PROP_BUFFERSIZE,
        1
    )

    # Create the MediaPipe hand detector.
    detector = create_hand_detector()

    # --------------------------------------------------------
    # CREATE APPLICATION WINDOW
    # --------------------------------------------------------

    cv2.namedWindow(
        "VIRTUAL GUITAR",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "VIRTUAL GUITAR",
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    # --------------------------------------------------------
    # INITIALIZE HISTORY QUEUES
    # --------------------------------------------------------

    # Complete gesture history.
    root_history = deque(
        maxlen=GESTURE_HISTORY_SIZE
    )

    quality_history = deque(
        maxlen=GESTURE_HISTORY_SIZE
    )

    # Individual finger histories.
    root_finger_histories = create_finger_histories()

    quality_finger_histories = create_finger_histories()

    # Previous finger states support hysteresis.
    root_previous_states = {}
    quality_previous_states = {}

    # Fingertip movement trails.
    root_trails = create_trails()
    quality_trails = create_trails()

    # Calibration wizard state.
    calibration_session = create_calibration_session()

    # Current gesture states.
    root_gesture = None
    quality_count = None

    # Diagnostic measurements.
    root_diagnostics = None
    quality_diagnostics = None

    # Current selected chord and currently playing chord.
    active_chord = None
    playing_chord = None

    # Diagnostics begin disabled.
    debug_enabled = False

    # Video-mode MediaPipe timestamps must increase on every frame.
    last_timestamp = 0

    # Remember audio-release and last-two-hand times.
    last_valid_chord_time = 0.0
    last_two_hand_time = 0.0

    # --------------------------------------------------------
    # PRINT INSTRUCTIONS
    # --------------------------------------------------------

    print("\nLEFT / SCREEN-LEFT HAND:")
    print("Thumb closed + 0/1/2/3/4 fingers = C/D/E/F/G")
    print("Thumb open + 0/1 fingers = A/B")

    print("\nRIGHT / SCREEN-RIGHT HAND:")
    print("1 finger = Major | 2 = Minor | 3 = 7th")

    print("\nPress C to calibrate:")
    print("BOTH fists first, BOTH open palms second.")

    print("\nD = diagnostics")
    print("R = reset")
    print("SPACE = replay")
    print("Q/ESC = quit")

    # ========================================================
    # MAIN FRAME LOOP
    # ========================================================

    try:

        while True:

            # ------------------------------------------------
            # READ CAMERA FRAME
            # ------------------------------------------------

            success, frame = camera.read()

            if not success or frame is None:

                print("Could not read camera frame.")
                break

            # Mirror camera image.
            frame = cv2.flip(
                frame,
                1
            )

            # Monotonic time is used for elapsed-time measurements.
            now = time.monotonic()

            # MediaPipe requires timestamps in increasing milliseconds.
            timestamp = int(now * 1000)

            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp

            # Detect and sort hands.
            hands = detect_hands(
                detector,
                frame,
                timestamp
            )

            hand_count = len(hands)

            # ------------------------------------------------
            # CALIBRATION MODE
            # ------------------------------------------------

            if calibration_session["active"]:

                # Draw visible hands during calibration.
                for hand_index, hand in enumerate(hands[:2]):

                    role = (
                        "ROOT"
                        if hand_index == 0
                        else "QUALITY"
                    )

                    old_states = (
                        root_previous_states
                        if role == "ROOT"
                        else quality_previous_states
                    )

                    # Use the existing calibration profile for display if available.
                    role_profile = (
                        calibration_profiles[role]
                        if calibration_profiles is not None
                        else None
                    )

                    _, visible_states, _ = recognize_fingers(
                        hand[2],
                        old_states,
                        role_profile
                    )

                    trails = (
                        root_trails
                        if role == "ROOT"
                        else quality_trails
                    )

                    draw_hand_landmarks(
                        frame,
                        hand[1],
                        visible_states,
                        trails,
                        role
                    )

                # Collect the current calibration samples.
                completed_profile = update_calibration(
                    calibration_session,
                    hands,
                    now
                )

                # If calibration finished, switch to the newly learned profiles.
                if completed_profile is not None:

                    calibration_profiles = completed_profile
                    calibration_loaded = True

                    # Clear old gesture measurements so the new classifier
                    # begins with a clean history.
                    root_history.clear()
                    quality_history.clear()

                    root_finger_histories = create_finger_histories()
                    quality_finger_histories = create_finger_histories()

                    root_previous_states.clear()
                    quality_previous_states.clear()

                    root_gesture = None
                    quality_count = None

                    root_diagnostics = None
                    quality_diagnostics = None

            # ------------------------------------------------
            # NORMAL TWO-HAND RECOGNITION
            # ------------------------------------------------

            elif hand_count >= 2:

                last_two_hand_time = now

                # Left-most hand selects the root note.
                # Right-most hand selects chord quality.
                root_hand = hands[0]
                quality_hand = hands[1]

                # Use each hand's own calibration profile.
                root_profile = (
                    calibration_profiles["ROOT"]
                    if calibration_profiles is not None
                    else None
                )

                quality_profile = (
                    calibration_profiles["QUALITY"]
                    if calibration_profiles is not None
                    else None
                )

                # --------------------------------------------
                # ROOT HAND DETECTION
                # --------------------------------------------

                (
                    _,
                    root_raw_states,
                    root_diagnostics
                ) = recognize_fingers(
                    root_hand[2],
                    root_previous_states,
                    root_profile
                )

                # Smooth the five individual finger results.
                root_states = stabilize_fingers(
                    root_raw_states,
                    root_finger_histories
                )

                root_previous_states = root_states

                # Count index, middle, ring, and pinky.
                root_count = sum(
                    root_states[name]
                    for name in FINGER_LANDMARKS
                )

                # --------------------------------------------
                # QUALITY HAND DETECTION
                # --------------------------------------------

                (
                    _,
                    quality_raw_states,
                    quality_diagnostics
                ) = recognize_fingers(
                    quality_hand[2],
                    quality_previous_states,
                    quality_profile
                )

                quality_states = stabilize_fingers(
                    quality_raw_states,
                    quality_finger_histories
                )

                quality_previous_states = quality_states

                # Count open non-thumb fingers.
                quality_detected = sum(
                    quality_states[name]
                    for name in FINGER_LANDMARKS
                )

                # --------------------------------------------
                # DRAW BOTH HANDS
                # --------------------------------------------

                draw_hand_landmarks(
                    frame,
                    root_hand[1],
                    root_states,
                    root_trails,
                    "ROOT"
                )

                draw_hand_landmarks(
                    frame,
                    quality_hand[1],
                    quality_states,
                    quality_trails,
                    "QUALITY"
                )

                # --------------------------------------------
                # STORE COMPLETE GESTURES
                # --------------------------------------------

                # Root history stores both count and thumb state.
                root_history.append(
                    (
                        root_count,
                        root_states["THUMB"]
                    )
                )

                # Quality history stores the number of open fingers.
                quality_history.append(
                    quality_detected
                )

                # Get stable complete gestures.
                root_gesture = stable_value(
                    root_history
                )

                quality_count = stable_value(
                    quality_history
                )

                # --------------------------------------------
                # SELECT AND PLAY CHORD
                # --------------------------------------------

                selected_chord = select_chord(
                    root_gesture,
                    quality_count
                )

                if selected_chord is not None:

                    active_chord = selected_chord
                    last_valid_chord_time = now

                    # Do not restart the audio if the chord is unchanged.
                    if playing_chord != active_chord:

                        stop_chord()

                        play_chord(
                            active_chord
                        )

                        playing_chord = active_chord

            # ------------------------------------------------
            # ONE HAND OR NO HANDS
            # ------------------------------------------------

            else:

                # Draw and recognize a single hand for visual feedback.
                if hand_count == 1:

                    hand = hands[0]

                    # Select role based on which side of the image contains it.
                    if hand[0] < 0.5:

                        role = "ROOT"
                        old_states = root_previous_states

                        finger_histories = root_finger_histories

                        role_profile = (
                            calibration_profiles["ROOT"]
                            if calibration_profiles is not None
                            else None
                        )

                        trails = root_trails

                    else:

                        role = "QUALITY"
                        old_states = quality_previous_states

                        finger_histories = quality_finger_histories

                        role_profile = (
                            calibration_profiles["QUALITY"]
                            if calibration_profiles is not None
                            else None
                        )

                        trails = quality_trails

                    # Recognize the visible hand.
                    _, raw_states, diagnostics = recognize_fingers(
                        hand[2],
                        old_states,
                        role_profile
                    )

                    visible_states = stabilize_fingers(
                        raw_states,
                        finger_histories
                    )

                    # Update diagnostics and previous states for the matching role.
                    if role == "ROOT":

                        root_previous_states = visible_states
                        root_diagnostics = diagnostics

                    else:

                        quality_previous_states = visible_states
                        quality_diagnostics = diagnostics

                    # Draw the visible hand.
                    draw_hand_landmarks(
                        frame,
                        hand[1],
                        visible_states,
                        trails,
                        role
                    )

                # Clear old votes after both hands have been missing for a while.
                if now - last_two_hand_time > HAND_LOST_GRACE_SECONDS:

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
            # AUDIO RELEASE
            # ------------------------------------------------

            # Stop the sound after the gesture has disappeared for long enough.
            if (
                playing_chord is not None
                and now - last_valid_chord_time > RELEASE_GRACE_SECONDS
            ):

                stop_chord()
                playing_chord = None

            # ------------------------------------------------
            # DRAW USER INTERFACE
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
                quality_diagnostics,
                calibration_session,
                calibration_loaded,
                now
            )

            cv2.imshow(
                "VIRTUAL GUITAR",
                display
            )

            # Read keyboard input.
            key = cv2.waitKey(1) & 0xFF

            # ------------------------------------------------
            # KEYBOARD CONTROLS
            # ------------------------------------------------

            if key == ord("c"):

                # Press C during calibration to cancel it.
                if calibration_session["active"]:

                    cancel_calibration(
                        calibration_session
                    )

                    print("Calibration cancelled.")

                else:

                    # Stop playback before entering calibration.
                    stop_chord()

                    playing_chord = None
                    active_chord = None

                    start_calibration(
                        calibration_session,
                        time.monotonic()
                    )

            elif key == ord("d"):

                # Toggle debug measurements.
                debug_enabled = not debug_enabled

                print(
                    "Diagnostics:",
                    "ON" if debug_enabled else "OFF"
                )

            elif key == ord("r"):

                # Clear gesture histories and stop playback.
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

                last_valid_chord_time = 0.0

                stop_chord()

                print("Gesture and audio state reset.")

            elif (
                key == 32
                and active_chord is not None
                and not calibration_session["active"]
            ):

                # SPACE manually replays the currently selected chord.
                stop_chord()

                play_chord(
                    active_chord
                )

                playing_chord = active_chord
                last_valid_chord_time = now

            elif key in (ord("q"), 27):

                # Q or Escape exits the application.
                break

    finally:

        # Always stop audio and release camera/model/window resources.
        print("Closing Virtual Guitar...")

        stop_chord()

        camera.release()
        detector.close()

        cv2.destroyAllWindows()

        print("Application closed.")


# ============================================================
# 37. PROGRAM ENTRY POINT
# ============================================================

if __name__ == "__main__":

    try:

        # Start the complete application.
        run_application()

    except KeyboardInterrupt:

        # Treat Ctrl+C as a normal shutdown.
        print("\nApplication interrupted.")

        stop_chord()
        cv2.destroyAllWindows()

    except Exception as error:

        # Print the error and preserve the complete traceback.
        print("\nAPPLICATION ERROR:")
        print(error)

        stop_chord()
        cv2.destroyAllWindows()

        raise
