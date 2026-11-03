
"""Gesture-controlled virtual guitar application.

The program combines MediaPipe hand landmarks, geometric finger-state
recognition, temporal smoothing, chord lookup, generated WAV files, and an
OpenCV user interface.  The left-most detected hand supplies the root note;
the other hand supplies the chord quality.  A chord is played only after the
gesture has remained stable long enough to pass the voting thresholds.

The file is intentionally organized as a single executable module:

1. Configuration and visual constants define the application's fixed inputs.
2. Chord tables describe musical notes and guitar fingerings.
3. MediaPipe helpers acquire and interpret hand landmarks.
4. Drawing and audio helpers render feedback and play generated sounds.
5. ``run_application`` owns camera setup, the frame loop, and cleanup.
"""

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

# Camera and output dimensions are kept separate because the source camera
# stream is displayed inside only the left portion of the larger UI canvas.
CAMERA_INDEX = 0
CAMERA_WIDTH = 960
CAMERA_HEIGHT = 720

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_VIEW_WIDTH = 640

SAMPLE_RATE = 44100
LOOP_DURATION = 2.0

# MediaPipe is asked for exactly the two hands that have distinct roles in the
# interaction: root-note hand first, chord-quality hand second.
MAX_HANDS = 2

# The history constants control two different kinds of stability.  Finger
# histories smooth individual detections; gesture histories smooth the complete
# chord input.  REQUIRED_VOTES is shared by complete-gesture voting.
HISTORY_SIZE = 7
FINGER_HISTORY_SIZE = 5
REQUIRED_VOTES = 4

# These grace periods prevent short camera dropouts or finger releases from
# immediately clearing the current UI state or stopping the sound.
RELEASE_GRACE_SECONDS = 0.5
HAND_LOST_GRACE_SECONDS = 0.7

# All generated assets live beside this script so the program can be launched
# from any working directory without relying on the process's current folder.
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

# OpenCV uses BGR tuples rather than RGB tuples.  Keeping the palette in one
# place makes the visual meaning of UI states consistent across all panels.
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

# ROOTS and QUALITIES are the canonical iteration order used by chord creation
# and by the chord-library grid in the UI.
ROOTS = ["C", "D", "E", "F", "G", "A", "B"]
QUALITIES = ["MAJOR", "MINOR", "7TH"]

# A root gesture is represented as (thumb_open, non_thumb_finger_count).
# The closed-thumb range provides C-G; opening the thumb extends it to A-B.
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

# Semitone offsets locate chord tones in NOTE_NAMES.  Frequencies below use
# the corresponding root frequency and the equal-temperament ratio.
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

# A fret string lists the six guitar strings from low E to high e.  ``x`` is
# muted and ``0`` is open; other digits identify the fret number.
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

# Finger colors match the numbered finger legend in the right-hand panel.
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
    """Build the complete root/quality chord catalogue.

    Each chord entry contains the musical notes, synthesized frequencies,
    displayed fret shape, and per-string finger assignments.  The function
    also validates that every generated chord has a matching guitar fingering,
    so a missing data-table entry fails during startup instead of later in
    the camera loop.
    """

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

# MediaPipe's 21-point hand model uses indices 0-20.  Each tuple identifies
# the joints needed for one finger's geometry: MCP, PIP, DIP, and fingertip.
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

# These pairs describe the bones drawn by the camera overlay.  They are kept
# separate from FINGER_LANDMARKS because the palm also needs connecting edges.
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
    """Download the MediaPipe hand model when it is absent or incomplete.

    The size check prevents a truncated or empty file from being treated as a
    valid model.  Download failures are re-raised as ``RuntimeError`` with
    context for the user.
    """

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
    """Create and configure the MediaPipe video-mode hand detector.

    The detector is configured for the application's maximum hand count and
    confidence thresholds, then returned to the main loop for frame-by-frame
    inference.
    """

    # Make sure the model file exists before MediaPipe tries to open it.
    download_model()

    # HandLandmarkerOptions is the configuration object consumed by MediaPipe.
    options = mp.tasks.vision.HandLandmarkerOptions(

        # Tell MediaPipe where the downloaded neural-network asset is stored.
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),

        # VIDEO mode allows MediaPipe to use timestamps and track hands between
        # frames instead of treating every frame as an unrelated image.
        running_mode=mp.tasks.vision.RunningMode.VIDEO,

        # The application assigns roles to at most two detected hands.
        num_hands=MAX_HANDS,

        # These thresholds trade recall for stability: detections below them
        # are ignored as too uncertain.
        min_hand_detection_confidence=DETECTION_CONFIDENCE,

        min_hand_presence_confidence=PRESENCE_CONFIDENCE,

        min_tracking_confidence=TRACKING_CONFIDENCE,
    )

    # Build and return the detector that will be reused for every frame.
    return mp.tasks.vision.HandLandmarker.create_from_options(
        options
    )


# ============================================================
# 9. ANGLE AND SCORE FUNCTIONS
# ============================================================

def calculate_angle(a, b, c):
    """Return the angle at point ``b`` formed by points ``a`` and ``c``.

    Landmark coordinates are treated as three-dimensional vectors.  Degenerate
    vectors return zero, and the cosine is clipped before ``arccos`` to avoid
    floating-point drift outside the valid mathematical range.
    """

    # Both vectors start at the vertex point b and point toward the other
    # two points.  Their angle is the joint bend we want to measure.
    vector1 = a - b
    vector2 = c - b

    # The vector lengths are needed to normalize the dot product.
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)

    # A zero-length vector has no meaningful direction, so avoid division by
    # zero and return the safest neutral value.
    if norm1 < 1e-8 or norm2 < 1e-8:
        return 0.0

    # The dot-product formula gives cos(theta) for the angle between vectors.
    cosine = np.dot(vector1, vector2) / (norm1 * norm2)

    # Floating-point rounding can produce values slightly outside [-1, 1].
    cosine = np.clip(cosine, -1.0, 1.0)

    # Convert the inverse cosine from radians to degrees for readable thresholds.
    return float(
        np.degrees(np.arccos(cosine))
    )


def clamp_score(value):
    """Clamp a recognition score to the inclusive interval ``[0.0, 1.0]``."""

    # Recognition components are combined as weighted scores, so every
    # component must stay in the same normalized range.
    return float(
        np.clip(value, 0.0, 1.0)
    )


# ============================================================
# 10. STRICT FINGER RECOGNITION
# ============================================================

def recognize_fingers(world_landmarks, previous_states=None):
    """Classify each finger as open or closed from world-space landmarks.

    Non-thumb fingers use joint angles, extension ratios, and reach ratios.
    The previous state provides hysteresis: an already-open finger uses lower
    thresholds, reducing flicker when a gesture is held near a boundary.

    Returns:
        A tuple containing the number of open non-thumb fingers, a mapping of
        finger names to boolean states, and numeric diagnostics for the UI.
    """

    # The first frame has no previous state, so all fingers begin as closed.
    if previous_states is None:
        previous_states = {}

    # Convert MediaPipe landmark objects into a numeric array.  This makes
    # vector subtraction and distance calculations concise and fast.
    points = np.array(
        [
            [p.x, p.y, p.z]
            for p in world_landmarks
        ],
        dtype=np.float64
    )

    # states is the boolean result used by the application.
    states = {}
    # diagnostics retains the measurements shown by the optional debug panel.
    diagnostics = {}

    # Landmark 0 is the wrist, used as a stable reference for reach.
    wrist = points[0]

    # --------------------------------------------------------
    # INDEX, MIDDLE, RING AND PINKY
    # --------------------------------------------------------

    # Evaluate each non-thumb finger using the same geometric recipe.
    for name, indices in FINGER_LANDMARKS.items():

        # Unpack the four joints needed to measure this finger.
        mcp, pip, dip, tip = (
            points[index]
            for index in indices
        )

        # A straight finger has large angles at both bendable joints.
        pip_angle = calculate_angle(
            mcp, pip, dip
        )

        dip_angle = calculate_angle(
            pip, dip, tip
        )

        # Compare the direct MCP-to-tip distance with the first segment length.
        mcp_to_pip = np.linalg.norm(
            pip - mcp
        )

        mcp_to_tip = np.linalg.norm(
            tip - mcp
        )

        # Open fingers tend to reach farther from their MCP than folded ones.
        extension_ratio = (
            mcp_to_tip / max(mcp_to_pip, 1e-6)
        )

        # Reach is measured relative to the wrist so it is less dependent on
        # the absolute size or distance of the hand from the camera.
        mcp_to_wrist = np.linalg.norm(
            mcp - wrist
        )

        tip_to_wrist = np.linalg.norm(
            tip - wrist
        )

        reach_ratio = (
            tip_to_wrist / max(mcp_to_wrist, 1e-6)
        )

        # Convert each raw measurement into a 0-1 contribution.  The constants
        # are empirically chosen boundaries for this application's gestures.
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

        # Combine the four signals.  Joint angles are the strongest evidence,
        # while extension and reach provide supporting evidence.
        score = (
            0.36 * pip_score
            + 0.24 * dip_score
            + 0.25 * extension_score
            + 0.15 * reach_score
        )

        # Hysteresis uses different thresholds when leaving versus entering
        # the open state.  This prevents small landmark jitter from toggling a
        # finger repeatedly at one exact boundary.
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

        # A finger must pass all three hard checks, not just the weighted score.
        is_open = (
            score >= score_threshold
            and pip_angle >= pip_minimum
            and extension_ratio >= extension_minimum
        )

        # Store a real bool because NumPy boolean values can be awkward in UI
        # formatting and later dictionary comparisons.
        states[name] = bool(is_open)

        # Keep raw values available for threshold tuning and on-screen debug.
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

    # The thumb uses a separate geometry because it does not have the same
    # MCP/PIP/DIP layout as the four fingers above.
    thumb_angle = calculate_angle(
        points[2],
        points[3],
        points[4]
    )

    # Palm width provides a scale reference for thumb reach.
    palm_width = np.linalg.norm(
        points[5] - points[17]
    )

    # Measure how far the thumb tip is from the index MCP.
    thumb_distance = np.linalg.norm(
        points[4] - points[5]
    )

    # Normalize thumb reach by palm size so the threshold is scale-independent.
    thumb_ratio = (
        thumb_distance / max(palm_width, 1e-6)
    )

    # Apply thumb hysteresis just as for the other fingers.
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

    # Both the thumb bend and its distance from the palm must indicate open.
    thumb_open = (
        thumb_angle >= angle_threshold
        and thumb_ratio >= distance_threshold
    )

    # Store thumb state separately; it affects the root-note mapping but is not
    # included in the non-thumb root finger count.
    states["THUMB"] = bool(thumb_open)

    diagnostics["THUMB"] = {
        "pip": float(thumb_angle),
        "dip": 0.0,
        "ratio": float(thumb_ratio),
        "reach": 0.0,
        "score": 1.0 if thumb_open else 0.0,
        "open": bool(thumb_open),
    }

    # Count only index through pinky because root gestures reserve the thumb as
    # a separate binary switch between the C-G and A-B ranges.
    finger_count = sum(
        states[name]
        for name in FINGER_LANDMARKS
    )

    return finger_count, states, diagnostics


# ============================================================
# 11. TEMPORAL FINGER SMOOTHING
# ============================================================

def create_finger_histories():
    """Create one short boolean history queue for each recognized finger."""

    # A deque automatically discards the oldest sample when it reaches the
    # configured maximum length, keeping memory and response time bounded.
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
    """Use majority voting over recent frames to stabilize finger states.

    Ties intentionally fall back to the current raw value, allowing a state
    change to begin immediately instead of preserving an arbitrary older
    value.
    """

    # Build a new result rather than altering the raw recognition dictionary.
    stable_states = {}

    # Smooth each finger independently so one uncertain finger does not affect
    # the state of the other four.
    for name, raw_value in raw_states.items():

        # Normalize the incoming value to bool before storing it.
        histories[name].append(
            bool(raw_value)
        )

        # True behaves like 1 in Python, so summing gives the number of open
        # observations in the recent window.
        open_votes = sum(
            histories[name]
        )

        # Every remaining observation is a closed vote.
        closed_votes = (
            len(histories[name]) - open_votes
        )

        # A strict majority produces the stable result.
        if open_votes > closed_votes:

            stable_states[name] = True

        # The opposite majority produces a stable closed result.
        elif closed_votes > open_votes:

            stable_states[name] = False

        else:
            # A tie is possible during startup or a transition.  Preserve the
            # latest observation so the interface remains responsive.
            stable_states[name] = bool(raw_value)

    return stable_states


# ============================================================
# 12. DETECT HANDS
# ============================================================

def detect_hands(detector, frame, timestamp):
    """Run MediaPipe on one BGR frame and return hands sorted left-to-right.

    MediaPipe expects RGB input, while OpenCV captures BGR input.  Each result
    keeps its image-space landmarks for drawing and world-space landmarks for
    geometric recognition.  Sorting by the average image-space x coordinate
    makes hand roles deterministic for the rest of the application.
    """

    # OpenCV captures BGR pixels; MediaPipe expects RGB pixels.
    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    # Wrap the converted NumPy array in MediaPipe's image container.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb_frame
    )

    # Run inference with a monotonically increasing timestamp.
    result = detector.detect_for_video(
        mp_image,
        timestamp
    )

    # Store tuples of (horizontal center, image landmarks, world landmarks).
    hands = []

    # No detected landmarks means there is nothing to classify this frame.
    if not result.hand_landmarks:
        return hands

    # Image and world landmark lists use the same hand index.
    for index, image_landmarks in enumerate(
        result.hand_landmarks
    ):

        # World coordinates are used for geometry because they are less
        # affected by the hand's position in the camera image.
        world_landmarks = result.hand_world_landmarks[index]

        # Average x gives a simple left-to-right position for role assignment.
        center_x = float(
            np.mean([
                point.x
                for point in image_landmarks
            ])
        )

        # Keep image landmarks for drawing and world landmarks for recognition.
        hands.append(
            (
                center_x,
                image_landmarks,
                world_landmarks
            )
        )

    # The first hand becomes the root hand and the second becomes the quality
    # hand, making role selection deterministic from frame to frame.
    hands.sort(
        key=lambda hand: hand[0]
    )

    return hands


# ============================================================
# 13. FINGER TRACING
# ============================================================

def create_trails():
    """Create bounded fingertip paths used by the live hand overlay."""

    # Each deque stores recent fingertip pixel positions for one trace line.
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
    """Draw palm bones, colored fingers, fingertip trails, and labels.

    Open fingers use the success color and closed fingers use the danger color.
    The trail queues are mutated in place so the caller can reuse them across
    frames without allocating a new path for every image.
    """

    # Convert normalized landmark coordinates into this frame's pixels.
    height, width = frame.shape[:2]

    # First draw the neutral palm skeleton behind the colored finger segments.
    for start, end in HAND_CONNECTIONS:

        # Retrieve the two endpoints of the current bone.
        p1 = landmarks[start]
        p2 = landmarks[end]

        # MediaPipe coordinates are normalized from 0 to 1.
        x1 = int(p1.x * width)
        y1 = int(p1.y * height)

        x2 = int(p2.x * width)
        y2 = int(p2.y * height)

        # Draw one anti-aliased palm connection.
        cv2.line(
            frame,
            (x1, y1),
            (x2, y2),
            (110, 125, 145),
            2,
            cv2.LINE_AA
        )

    # Draw each finger using its current open/closed state.
    for finger_name, indices in TRACE_LANDMARKS.items():

        # Missing state is treated as closed for safe visual feedback.
        is_open = states.get(
            finger_name,
            False
        )

        # Green means extended; red means folded.
        color = (
            SUCCESS
            if is_open
            else DANGER
        )

        # Connect the finger's four landmarks one bone at a time.
        for i in range(len(indices) - 1):

            p1 = landmarks[indices[i]]
            p2 = landmarks[indices[i + 1]]

            x1 = int(p1.x * width)
            y1 = int(p1.y * height)

            x2 = int(p2.x * width)
            y2 = int(p2.y * height)

            # Use a thicker colored line so the state is visible over the video.
            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                4,
                cv2.LINE_AA
            )

        # The final landmark is the fingertip used for the trail and marker.
        tip = landmarks[indices[-1]]

        tip_position = (
            int(tip.x * width),
            int(tip.y * height)
        )

        # Save this tip position so motion remains visible over recent frames.
        trails[finger_name].append(
            tip_position
        )

        # Convert the deque to a list for OpenCV's polyline API.
        trail_points = list(
            trails[finger_name]
        )

        # A single point cannot form a line, so wait for at least two samples.
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

        # Draw a filled state-colored fingertip and a thin white outline.
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

        # Keep the label inside the image boundaries where possible.
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
    """Return the most-voted history value after enough votes accumulate."""

    # There is no meaningful winner before the first observation arrives.
    if not history:
        return None

    # Count how often every complete gesture occurred in the recent window.
    counts = Counter(history)

    # Counter returns the most frequent value and its vote count first.
    value, votes = counts.most_common(1)[0]

    # Require a minimum vote count so one transient frame cannot select a chord.
    if votes >= REQUIRED_VOTES:
        return value

    # The gesture is still undecided; the caller should keep the prior state.
    return None


# ============================================================
# 15. SELECT CHORD
# ============================================================

def select_chord(root_gesture, quality_count):
    """Translate stabilized root and quality gestures into a chord key.

    ``root_gesture`` stores the number of open non-thumb fingers and whether
    the thumb is open.  The quality hand uses its open-finger count directly.
    Invalid or incomplete combinations return ``None`` rather than selecting
    a partial chord.
    """

    # Both hands must have a stable value before a chord can be selected.
    if root_gesture is None or quality_count is None:
        return None

    # Unpack the root gesture into its two independent controls.
    root_count, thumb_open = root_gesture

    # Convert the root hand gesture into a note name.
    root = ROOT_GESTURES.get(
        (bool(thumb_open), root_count)
    )

    # Convert the quality hand's open-finger count into a chord quality.
    quality = QUALITY_GESTURES.get(
        quality_count
    )

    # Unknown counts are intentionally rejected rather than guessed.
    if root is None or quality is None:
        return None

    # Chord dictionary keys are the root name followed by the quality name.
    chord = root + quality

    # Confirm the generated key exists before handing it to audio/UI code.
    if chord in CHORDS:
        return chord

    # This is defensive: valid mappings should always produce a known chord.
    return None


# ============================================================
# 16. GENERATE CHORD AUDIO
# ============================================================

def generate_chord(frequencies, filename):
    """Synthesize a normalized looping chord and write it as a mono WAV file.

    Each supplied fundamental frequency receives four harmonics.  The result
    is normalized, attenuated, and given short fades at both ends to reduce
    clicks when Windows loops the file.
    """

    # Convert duration and sample rate into the number of audio samples.
    sample_count = int(
        SAMPLE_RATE * LOOP_DURATION
    )

    # Create one time value per sample for vectorized sine-wave generation.
    time_values = (
        np.arange(sample_count) / SAMPLE_RATE
    )

    # Start with silence; each note and its harmonics will be added to it.
    audio = np.zeros(
        sample_count,
        dtype=np.float64
    )

    # Add every chord tone to the same buffer.
    for frequency in frequencies:

        # The fundamental plus quieter harmonics produces a less sterile sound
        # than a single pure sine wave.
        note = (
            np.sin(2 * np.pi * frequency * time_values)
            + 0.30 * np.sin(2 * np.pi * frequency * 2 * time_values)
            + 0.12 * np.sin(2 * np.pi * frequency * 3 * time_values)
            + 0.05 * np.sin(2 * np.pi * frequency * 4 * time_values)
        )

        # Superpose this note on the other chord tones.
        audio += note

    # Find the loudest sample so the summed waveform can be normalized safely.
    maximum = np.max(
        np.abs(audio)
    )

    if maximum > 0:
        # Prevent clipping caused by adding multiple notes together.
        audio /= maximum

    # Leave headroom so the generated sound is not unnecessarily loud.
    audio *= 0.65

    # A short fade prevents an audible click at the loop boundaries.
    fade_size = int(
        SAMPLE_RATE * 0.01
    )

    if fade_size > 0:

        # Build a linear ramp from silence to full volume.
        fade = np.linspace(
            0,
            1,
            fade_size
        )

        # Fade in at the beginning and fade out in reverse at the end.
        audio[:fade_size] *= fade
        audio[-fade_size:] *= fade[::-1]

    # Convert floating-point amplitude [-1, 1] to signed 16-bit PCM samples.
    audio = (
        audio * 32767
    ).astype(np.int16)

    # Write a standard mono PCM WAV file understood by winsound.
    with wave.open(
        str(filename),
        "wb"
    ) as wav_file:

        # One channel means the same sound is used for both speakers.
        wav_file.setnchannels(1)
        # Two bytes per sample is 16-bit audio.
        wav_file.setsampwidth(2)
        # Store the sample rate used to construct time_values.
        wav_file.setframerate(SAMPLE_RATE)

        # Serialize the NumPy samples into the file payload.
        wav_file.writeframes(
            audio.tobytes()
        )


def create_chord_files():
    """Create missing WAV files for every chord in ``CHORDS``."""

    # Create the output folder if this is the first run.
    SOUND_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print("Preparing chord sounds...")

    # Generate each asset only once so later launches start quickly.
    for chord_name, chord_data in CHORDS.items():

        # Every chord uses its dictionary key as its WAV filename.
        filename = SOUND_DIR / f"{chord_name}.wav"

        if not filename.exists():

            # Generate only missing files; existing files are preserved.
            generate_chord(
                chord_data["frequencies"],
                filename
            )

    print("All 21 chord sounds are ready.")


# ============================================================
# 17. PLAY AND STOP AUDIO
# ============================================================

def stop_chord():
    """Stop the currently looping Windows sound, when audio is available."""

    # winsound is Windows-only; importing this file elsewhere should remain
    # possible even when that optional module is unavailable.
    if winsound is None:
        return

    try:

        # Passing None with SND_PURGE stops sounds started by winsound.
        winsound.PlaySound(
            None,
            winsound.SND_PURGE
        )

    except Exception:
        # Stopping is best-effort during shutdown or device changes.
        pass


def play_chord(chord_name):
    """Asynchronously loop the generated WAV file for ``chord_name``."""

    # Do nothing when the platform has no Windows playback API.
    if winsound is None:
        return

    # Resolve the chord name to the WAV generated during startup.
    filename = SOUND_DIR / f"{chord_name}.wav"

    # Missing assets should not crash the camera loop.
    if not filename.exists():
        return

    try:

        # Play asynchronously so the UI can continue processing frames.
        # LOOP keeps the chord sounding until stop_chord is called.
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
    """Render one anti-aliased text label using the application's font."""

    # Convert arbitrary messages to strings because OpenCV requires text.
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
    """Fit the camera image into the left display panel.

    The foreground image preserves its aspect ratio.  A stretched and blurred
    copy fills the panel behind it, preventing unused space from appearing as
    a plain black border when camera and display ratios differ.
    """

    # Read the source image dimensions without changing the frame.
    height, width = frame.shape[:2]

    # Choose the largest scale that fits both target dimensions.
    scale = min(
        CAMERA_VIEW_WIDTH / width,
        DISPLAY_HEIGHT / height
    )

    # Clamp dimensions to at least one pixel for unusual camera inputs.
    new_width = max(
        1,
        int(width * scale)
    )

    new_height = max(
        1,
        int(height * scale)
    )

    # Preserve the camera aspect ratio in the foreground image.
    resized = cv2.resize(
        frame,
        (new_width, new_height)
    )

    # Stretch a copy to fill the whole panel before blurring it.
    background = cv2.resize(
        frame,
        (CAMERA_VIEW_WIDTH, DISPLAY_HEIGHT)
    )

    # Blur the stretched copy so it acts as a subtle backdrop rather than a
    # visibly distorted duplicate of the camera image.
    background = cv2.GaussianBlur(
        background,
        (51, 51),
        0
    )

    # Copy prevents later writes from modifying the resized background object.
    canvas = background.copy()

    # Center the aspect-ratio-preserving foreground inside the panel.
    x = (CAMERA_VIEW_WIDTH - new_width) // 2
    y = (DISPLAY_HEIGHT - new_height) // 2

    # Replace the centered area with the sharp camera image.
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
    """Draw per-finger recognition measurements over the camera image.

    The diagnostic panel is intentionally derived from the latest raw
    measurements rather than the stabilized state, so threshold behavior can
    be inspected while tuning recognition.
    """

    # Put the diagnostic heading above the per-finger measurements.
    draw_text(
        frame,
        "FINGER DETECTION",
        (10, 105),
        0.48,
        GOLD,
        2
    )

    # y tracks the next free vertical text position.
    y = 132

    # Render the two hands in the same order used for chord selection.
    for hand_name, data in [
        ("ROOT HAND", root_data),
        ("QUALITY HAND", quality_data),
    ]:

        # No hands or a lost hand has no measurements to display.
        if data is None:
            continue

        # Identify which hand the following lines describe.
        draw_text(
            frame,
            hand_name,
            (10, y),
            0.40,
            MINT
        )

        y += 21

        # Each values dictionary contains the geometry captured in the current
        # frame, including the final open/closed decision.
        for name, values in data.items():

            # Convert the boolean state into a readable label and color.
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

            # Show the most useful tuning values:
            # A = joint angle, R = extension ratio, S = weighted score.
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
    """Draw string names, fret positions, and finger assignments for a chord."""

    # The guide is placed in the right half of the current-chord card.
    draw_text(
        panel,
        "FINGER GUIDE",
        (275, 137),
        0.48,
        MINT,
        2
    )

    # Six strings share a fixed horizontal grid.
    start_x = 295
    spacing = 46

    # Label each guitar string from low E to high e.
    for i, string_name in enumerate(STRING_NAMES):

        x = start_x + i * spacing

        # Labels in the first column explain the two rows below.
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

    # Show the fret string supplied by the chord data.
    for i, value in enumerate(chord_data["fret"]):

        x = start_x + i * spacing

        # Muted strings are red, open strings are muted gray, and fretted
        # strings use the normal text color.
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

    # Show which fretting-hand finger belongs on each string.
    for i, finger in enumerate(chord_data["fingers"]):

        x = start_x + i * spacing

        # None means muted, zero means open, and 1-4 identify fingers.
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
    """Render the right-side application panel.

    The panel contains the current chord, playback state, hand-role status,
    finger-guide data, and a complete 7-by-3 chord library.  It writes into
    the right slice of the supplied display image and therefore does not
    allocate a second full-size canvas.
    """

    # Slice the right side without copying it; drawing here updates display.
    panel = display[:, CAMERA_VIEW_WIDTH:]

    # Clear stale pixels from the previous frame.
    panel[:] = PANEL

    # HEADER
    # Draw the application title and subtitle first.
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

    # Separate the header from the current chord card.
    cv2.line(
        panel,
        (20, 88),
        (620, 88),
        BORDER,
        1
    )

    # CURRENT CHORD CARD
    # The card gives the user the selected chord and its fingering.
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

    # A selected chord gets detailed information; otherwise show instructions.
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

    # Playback status is separate from selection because release grace time
    # can leave a selected chord visible briefly after audio stops.
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
    # Show whether both role-defining hands are currently available.
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

    # Format the root hand's count and thumb switch for the user.
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

    # The quality hand is represented only by its non-thumb count.
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
    # Draw every valid root/quality combination as a selectable-looking cell.
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

    # Divide the remaining card width evenly among three quality columns.
    chord_width = (
        grid_right
        - grid_left
        - root_width
        - 3 * gap
    ) // 3

    # Each root occupies one row in the library.
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

        # Each row contains Major, Minor, and 7th cells.
        for column, quality in enumerate(QUALITIES):

            x1 = (
                grid_left
                + root_width
                + gap
                + column * (chord_width + gap)
            )

            x2 = x1 + chord_width

            chord_name = root + quality

            # Highlight only the currently active chord.
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
    # Explain keyboard controls and advertise the catalogue size.
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
    """Assemble the camera view, optional diagnostics, and control panel.

    This function is presentation-only: it reads the current application state
    and returns a fresh display image without changing gesture or audio state.
    """

    # Build the left camera panel before allocating the final UI canvas.
    camera_view = prepare_camera_view(frame)

    # Use an 8-bit, three-channel image because OpenCV drawing expects BGR data.
    display = np.zeros(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        dtype=np.uint8
    )

    # Place the camera panel on the left.
    display[:, :CAMERA_VIEW_WIDTH] = camera_view

    # Initialize the right panel background before draw_panel adds content.
    display[:, CAMERA_VIEW_WIDTH:] = PANEL

    # Add a small legend above the camera feed.
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

    # Diagnostics are optional because they occupy useful camera space.
    if debug_enabled:

        draw_debug(
            display[:, :CAMERA_VIEW_WIDTH],
            root_diagnostics,
            quality_diagnostics
        )

    # The right panel receives all current gesture and playback state.
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
    """Run the camera, gesture-recognition, audio, and display event loop.

    Startup prepares chord audio, opens the camera, creates the detector, and
    initializes all temporal state.  Each frame then detects hands, smooths
    gestures, selects and plays chords, updates the UI, and handles keyboard
    controls.  The ``finally`` block releases external resources even when the
    loop exits because of an error.
    """

    # Print a startup banner so the terminal clearly identifies the program.
    print("=" * 45)
    print("VIRTUAL GUITAR")
    print("=" * 45)

    print("Total chords:", len(CHORDS))

    # Warn early when the UI can run but sound playback cannot.
    if winsound is None:
        print("WARNING: Windows audio playback unavailable.")

    # Generate missing audio before entering the real-time loop.
    create_chord_files()

    print("Opening camera...")

    # Prefer DirectShow, which commonly gives better Windows camera behavior.
    camera = cv2.VideoCapture(
        CAMERA_INDEX,
        cv2.CAP_DSHOW
    )

    # Fall back to OpenCV's default backend if DirectShow did not open it.
    if not camera.isOpened():

        camera.release()

        camera = cv2.VideoCapture(
            CAMERA_INDEX
        )

    # Do not continue with an invalid camera object.
    if not camera.isOpened():

        raise RuntimeError(
            "Camera could not be opened. Check CAMERA_INDEX."
        )

    # Request the configured capture dimensions.  The camera may negotiate a
    # different supported size, so downstream code still reads frame.shape.
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

    # Create the detector only after camera startup has succeeded.
    detector = create_hand_detector()

    # Create a resizable window for the composed display.
    cv2.namedWindow(
        "VIRTUAL GUITAR",
        cv2.WINDOW_NORMAL
    )

    cv2.resizeWindow(
        "VIRTUAL GUITAR",
        DISPLAY_WIDTH,
        DISPLAY_HEIGHT
    )

    # Complete gestures are voted over recent frames.
    root_history = deque(
        maxlen=HISTORY_SIZE
    )

    quality_history = deque(
        maxlen=HISTORY_SIZE
    )

    # Individual fingers are smoothed before their states are counted.
    root_finger_histories = create_finger_histories()

    quality_finger_histories = create_finger_histories()

    # Previous states enable hysteresis in the geometric recognizer.
    root_previous_states = {}
    quality_previous_states = {}

    # Fingertip trails persist across frames for visual feedback.
    root_trails = create_trails()
    quality_trails = create_trails()

    # No stable gesture exists until enough frames agree.
    root_gesture = None
    quality_count = None

    root_diagnostics = None
    quality_diagnostics = None

    # Selection and playback are tracked separately because release grace time
    # can keep the selected chord visible after sound stops.
    active_chord = None
    playing_chord = None

    # The D key toggles raw recognition diagnostics.
    debug_enabled = False

    # MediaPipe VIDEO mode requires strictly increasing timestamps.
    last_timestamp = 0

    # Monotonic times drive the release and hand-loss grace periods.
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

        # Keep processing frames until the camera fails or Q is pressed.
        while True:

            # Read the newest frame from the camera.
            success, frame = camera.read()

            # A failed read cannot be passed safely to OpenCV or MediaPipe.
            if not success or frame is None:

                print("Could not read camera frame.")
                break

            # Mirror the camera so movement matches the user's perspective.
            frame = cv2.flip(
                frame,
                1
            )

            # Monotonic time is safe for measuring elapsed durations.
            current_time = time.monotonic()

            # MediaPipe expects video timestamps in milliseconds.
            timestamp = int(
                current_time * 1000
            )

            # Ensure two quick frames never reuse the same timestamp.
            if timestamp <= last_timestamp:
                timestamp = last_timestamp + 1

            last_timestamp = timestamp

            # Detect hands and sort them from left to right.
            hands = detect_hands(
                detector,
                frame,
                timestamp
            )

            # The UI displays this even when there are fewer than two hands.
            hand_count = len(hands)

            # ------------------------------------------------
            # TWO HANDS DETECTED
            # ------------------------------------------------

            # Chord recognition requires both the root and quality hands.
            if hand_count >= 2:

                # Remember the latest time when both hands were available.
                last_two_hand_time = current_time

                # detect_hands sorted these roles consistently.
                root_hand = hands[0]
                quality_hand = hands[1]

                # Image landmarks are used for drawing colored bones.
                root_image = root_hand[1]
                # World landmarks are used for scale-independent geometry.
                root_world = root_hand[2]

                # Repeat the same split for the quality hand.
                quality_image = quality_hand[1]
                quality_world = quality_hand[2]

                # ROOT HAND: RAW RECOGNITION.
                # Classify this frame before applying temporal smoothing.
                (
                    _,
                    root_raw_states,
                    root_diagnostics
                ) = recognize_fingers(
                    root_world,
                    root_previous_states
                )

                # ROOT HAND: TEMPORAL SMOOTHING.
                # Majority voting suppresses one-frame landmark noise.
                root_states = stabilize_fingers(
                    root_raw_states,
                    root_finger_histories
                )

                # Feed the stabilized result into the next frame's hysteresis.
                root_previous_states = root_states

                # Count only index through pinky; the thumb is a separate root
                # selector for A/B versus C-G.
                root_count = sum(
                    root_states[name]
                    for name in FINGER_LANDMARKS
                )

                # QUALITY HAND: RAW RECOGNITION.
                # Reuse the same classifier for the second hand.
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

                # Preserve the stabilized result for the next recognition call.
                quality_previous_states = quality_states

                # Quality uses the non-thumb count directly.
                quality_detected = sum(
                    quality_states[name]
                    for name in FINGER_LANDMARKS
                )

                # DRAW FINGER TRACES.
                # Both overlays are drawn onto the camera frame in place.
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
                # Store complete inputs separately from individual finger data.
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
                # A value is returned only after enough recent votes agree.
                root_gesture = stable_value(
                    root_history
                )

                quality_count = stable_value(
                    quality_history
                )

                # SELECT CHORD.
                # Combine root and quality into a catalogue key.
                selected_chord = select_chord(
                    root_gesture,
                    quality_count
                )

                # Invalid or incomplete inputs leave the last valid selection
                # unchanged.
                if selected_chord is not None:

                    # Show the newest valid chord immediately.
                    active_chord = selected_chord

                    # Refresh the audio release timer while the input is valid.
                    last_valid_chord_time = current_time

                    # Do not restart the same looping sound every camera frame.
                    if playing_chord != active_chord:

                        stop_chord()

                        play_chord(
                            active_chord
                        )

                        playing_chord = active_chord

            # ------------------------------------------------
            # HANDS TEMPORARILY LOST
            # ------------------------------------------------

            # A brief detection gap is tolerated; a longer gap resets state.
            elif (
                current_time - last_two_hand_time
                > HAND_LOST_GRACE_SECONDS
            ):

                # Discard stale complete-gesture votes.
                root_history.clear()
                quality_history.clear()

                # Discard stale hysteresis state because the next hands may be
                # different people or orientations.
                root_previous_states.clear()
                quality_previous_states.clear()

                # Start new smoothing windows after prolonged hand loss.
                root_finger_histories = create_finger_histories()

                quality_finger_histories = create_finger_histories()

                # Clear trails so old fingertips do not remain on screen.
                root_trails = create_trails()
                quality_trails = create_trails()

                root_gesture = None
                quality_count = None

                root_diagnostics = None
                quality_diagnostics = None

            # ------------------------------------------------
            # STOP AUDIO AFTER RELEASE
            # Keep sound alive briefly in case the detector misses one frame.
            # ------------------------------------------------

            if (
                playing_chord is not None
                and current_time - last_valid_chord_time
                > RELEASE_GRACE_SECONDS
            ):

                # Stop only after the configured release grace interval.
                stop_chord()

                playing_chord = None

            # ------------------------------------------------
            # DISPLAY
            # Compose and show the current camera/UI state.
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

            # OpenCV presents the composed BGR image in the named window.
            cv2.imshow(
                "VIRTUAL GUITAR",
                display
            )

            # ------------------------------------------------
            # KEYBOARD CONTROLS
            # waitKey refreshes the window and returns the pressed key code.
            # ------------------------------------------------

            key = cv2.waitKey(1) & 0xFF

            # Q exits the loop and proceeds to cleanup.
            if key == ord("q"):

                break

            # D toggles the detailed raw-finger diagnostics.
            elif key == ord("d"):

                debug_enabled = not debug_enabled

                print(
                    "Diagnostics:",
                    "ON" if debug_enabled else "OFF"
                )

            # R clears current gesture, chord, and playback state.
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

            # Space manually replays the currently selected chord.
            elif key == 32:

                if active_chord is not None:

                    stop_chord()

                    play_chord(
                        active_chord
                    )

                    playing_chord = active_chord

                    last_valid_chord_time = current_time

    finally:

        # Cleanup runs for normal exit and exceptions alike.
        print("Closing Virtual Guitar...")

        # Stop any asynchronously looping sound first.
        stop_chord()

        # Release native camera and MediaPipe resources.
        camera.release()
        detector.close()

        # Remove the GUI window created by OpenCV.
        cv2.destroyAllWindows()

        print("Application closed.")


# ============================================================
# 25. START PROGRAM
# ============================================================

if __name__ == "__main__":

    # This guard prevents the camera from opening when the module is imported
    # for testing or for reuse of its chord/recognition helpers.
    try:

        # Start the complete interactive application.
        run_application()

    except KeyboardInterrupt:

        # Treat Ctrl+C as a normal user-initiated shutdown.
        print("\nApplication interrupted.")

        # Ensure no sound or window survives the interruption.
        stop_chord()
        cv2.destroyAllWindows()

    except Exception as error:

        # Print a short human-readable error before re-raising it for debugging.
        print("\nAPPLICATION ERROR:")
        print(error)

        # Perform best-effort UI/audio cleanup on unexpected failures.
        stop_chord()
        cv2.destroyAllWindows()

        # Re-raise so the original traceback and exit status are preserved.
        raise
