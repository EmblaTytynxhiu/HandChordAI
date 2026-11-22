"""UPGRADE v3.1: automatic tests for HandChordAI's logic.

WHY TESTS?
----------
When you change the code months from now (and have forgotten how it
works - that's normal!), these tests check that the important parts
still behave correctly. If a test fails, you broke something.

HOW TO RUN THEM (from the HandChordAI folder):
    .\\.venv\\Scripts\\python.exe -m unittest discover tests

You should see something like "Ran 14 tests ... OK".
A failing test prints which check failed and what the values were.

WHAT IS TESTED?
---------------
Only the "pure" parts that don't need a webcam: the chord table,
gesture matching, vote counting, the smoothing filter and the finger
measurements. The camera and window can't be tested automatically.

Uses Python's built-in 'unittest' module - nothing extra to install.
"""

import sys
import unittest
from pathlib import Path

import numpy as np

# The tests live in tests/, but HandChordAI.py is one folder up.
# Add that folder to Python's search path so "import HandChordAI" works.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import HandChordAI as app  # noqa: E402  (import after the path fix on purpose)


class ChordTableTests(unittest.TestCase):
    """The 21 chords built at startup."""

    def test_there_are_21_chords(self):
        # 7 roots x 3 qualities.
        self.assertEqual(len(app.CHORDS), 21)

    def test_e_minor_has_the_right_notes(self):
        chord = app.CHORDS["EMINOR"]
        self.assertEqual(chord["display"], "Em")
        self.assertEqual(chord["notes"], ["E", "G", "B"])

    def test_c7_has_four_notes(self):
        # A 7th chord = major chord + the 7th -> 4 notes.
        self.assertEqual(app.CHORDS["C7TH"]["notes"], ["C", "E", "G", "A#"])

    def test_a4_is_440_hz(self):
        # MIDI note 69 is A4, the standard tuning note.
        self.assertAlmostEqual(app.midi_to_frequency(69), 440.0)

    def test_muted_strings_have_no_frequency(self):
        # D major = "xx0232": the first two strings are muted (x).
        freqs = app.CHORDS["DMAJOR"]["string_frequencies"]
        self.assertIsNone(freqs[0])
        self.assertIsNone(freqs[1])
        self.assertIsNotNone(freqs[2])


def probs_from_pattern(pattern, sure=0.95):
    """Helper: turn a (T, I, M, R, P) pattern of 1/0 into finger
    probabilities, as if the classifier were very sure."""
    return {finger: (sure if up else 1 - sure)
            for finger, up in zip(app.FINGER_NAMES, pattern)}


class GestureMatchingTests(unittest.TestCase):
    """match_gesture(): finger probabilities -> gesture name."""

    def test_every_root_gesture_is_recognised(self):
        # Each template, fed back in, must be recognised as itself.
        for name, pattern in app.ROOT_TEMPLATES.items():
            with self.subTest(root=name):   # subTest = report each root separately
                best, _, _ = app.match_gesture(probs_from_pattern(pattern),
                                               app.ROOT_TEMPLATES)
                self.assertEqual(best, name)

    def test_quality_ignores_the_thumb(self):
        # Minor = index + middle. Thumb up or down must not matter.
        for thumb in (0, 1):
            probs = probs_from_pattern((thumb, 1, 1, 0, 0))
            best, _, _ = app.match_gesture(probs, app.QUALITY_TEMPLATES)
            self.assertEqual(best, "MINOR")

    def test_unclear_hand_gives_no_gesture(self):
        # Every finger at 50% = "no idea" -> must return None, not a guess.
        probs = {finger: 0.5 for finger in app.FINGER_NAMES}
        best, _, _ = app.match_gesture(probs, app.ROOT_TEMPLATES)
        self.assertIsNone(best)


class VotingTests(unittest.TestCase):
    """stable_value(): the majority vote over recent frames."""

    def test_enough_votes_wins(self):
        history = ["E", "E", "E", "E", "D", None]
        self.assertEqual(app.stable_value(history), "E")

    def test_too_few_votes_gives_none(self):
        # Only 3 "E" votes, but REQUIRED_GESTURE_VOTES is 4.
        history = ["E", "E", "E", "D", "D", None]
        self.assertIsNone(app.stable_value(history))

    def test_empty_history(self):
        self.assertIsNone(app.stable_value([]))


class OneEuroFilterTests(unittest.TestCase):
    """The landmark smoothing filter."""

    def test_first_value_passes_straight_through(self):
        f = app.OneEuroFilter(min_cutoff=1.0, beta=0.0)
        self.assertTrue(np.allclose(f(np.array([5.0]), 0.0), [5.0]))

    def test_jump_is_smoothed(self):
        # A sudden jump from 0 to 1 should NOT fully arrive in one frame.
        f = app.OneEuroFilter(min_cutoff=1.0, beta=0.0)
        f(np.array([0.0]), 0.0)
        out = f(np.array([1.0]), 1 / 30)   # one frame later at 30 FPS
        self.assertGreater(out[0], 0.0)
        self.assertLess(out[0], 1.0)


class FeatureTests(unittest.TestCase):
    """extract_features(): 21 points -> finger measurements."""

    def make_flat_hand(self):
        """A fake, perfectly flat open hand: every finger points straight
        'up' (+y) from its knuckle. Units are roughly metres."""
        points = np.zeros((21, 3))
        knuckle_x = {"INDEX": -0.03, "MIDDLE": -0.01, "RING": 0.01, "PINKY": 0.03}
        for finger, x in knuckle_x.items():
            mcp, pip, dip, tip = app.FINGER_LANDMARKS[finger]
            for step, index in enumerate((mcp, pip, dip, tip)):
                points[index] = [x, 0.08 + 0.025 * step, 0.0]
        # Thumb sticking out to the side (away from the pinky).
        for step, index in enumerate(app.FINGER_LANDMARKS["THUMB"]):
            points[index] = [-0.04 - 0.02 * step, 0.02 + 0.01 * step, 0.0]
        return points

    def test_straight_fingers_have_almost_no_bend(self):
        # The comment in extract_features() says "about 0-40 straight".
        # (Not exactly 0: the knuckles sit to the side of the wrist, so the
        # wrist->knuckle->finger line has a small angle at the big knuckle.)
        features = app.extract_features(self.make_flat_hand())
        for finger in app.NON_THUMB:
            flex = features[finger][0]
            self.assertLess(flex, 40, f"{finger} flex was {flex:.1f}")

    def test_fallback_says_flat_hand_is_open(self):
        # The built-in (uncalibrated) rules should call every finger "up".
        probs = app.fallback_probabilities(app.extract_features(self.make_flat_hand()))
        for finger in app.FINGER_NAMES:
            self.assertGreater(probs[finger], 0.5, f"{finger} p={probs[finger]:.2f}")


class EvaluationTests(unittest.TestCase):
    """UPGRADE v3.2: the evaluation mode and its analysis script."""

    def test_trial_order_has_every_chord_and_no_repeats_in_a_row(self):
        import random
        # Try many seeds - a bug might only show up for some orders.
        for seed in range(200):
            order = app.make_trial_order(3, random.Random(seed))
            # 3 repetitions x 21 chords, each chord exactly 3 times.
            self.assertEqual(len(order), 63)
            self.assertEqual(set(order), set(app.CHORDS))
            # The same chord twice in a row would break the latency measurement.
            for a, b in zip(order, order[1:]):
                self.assertNotEqual(a, b, f"seed {seed}")

    def make_rows(self, stable_from, frame_from):
        """Helper: fake frames of one trial (target E minor) at 30 FPS.
        The per-frame guess becomes right at 'frame_from' seconds,
        the stable (voted) guess at 'stable_from' seconds."""
        rows = []
        for i in range(150):                 # 5 seconds x 30 frames
            t = i / 30
            rows.append({
                "participant": "P00", "session": "s", "trial": 0,
                "target": "EMINOR", "target_root": "E", "target_quality": "MINOR",
                "live_classifier": "fallback", "t_trial": t,
                "phase": "settle" if t < 2 else "record",
                "root_visible": True, "quality_visible": True,
                "root_frame": "E" if t >= frame_from else "D",
                "quality_frame": "MINOR",
                "root_stable": "E" if t >= stable_from else "NONE",
                "quality_stable": "MINOR",
            })
        return rows

    def test_latency_splits_human_and_system_time(self):
        import analyze_evaluation as analysis
        rows = self.make_rows(stable_from=1.0, frame_from=0.8)
        result = analysis.trial_latencies(rows)[0]
        self.assertTrue(result["success"])
        self.assertAlmostEqual(result["t_stable"], 1.0, delta=1 / 30)
        self.assertAlmostEqual(result["t_onset"], 0.8, delta=1 / 30)
        self.assertAlmostEqual(result["decision_delay"], 0.2, delta=2 / 30)

    def test_never_recognised_trial_is_a_failure(self):
        import analyze_evaluation as analysis
        rows = self.make_rows(stable_from=99, frame_from=99)
        self.assertFalse(analysis.trial_latencies(rows)[0]["success"])

    def test_accuracy_counts_only_record_phase(self):
        import analyze_evaluation as analysis
        # Stable is right from 2.5 s; the record phase is 2-5 s, so
        # 0.5 s of its 3 s is wrong -> about 83% accuracy.
        rows = analysis.scored_frames(self.make_rows(stable_from=2.5, frame_from=0))
        acc, rejection, n = analysis.accuracy(rows, "chord", "stable")
        self.assertEqual(n, 90)
        self.assertAlmostEqual(acc, 75 / 90)
        # The wrong frames were "NONE" -> counted as rejections, not errors.
        self.assertAlmostEqual(rejection, 15 / 90)


if __name__ == "__main__":
    unittest.main()
