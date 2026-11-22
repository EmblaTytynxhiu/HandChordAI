"""UPGRADE v3.2: turn evaluation recordings into thesis results.

WHAT IT DOES
------------
Reads the CSV files that the evaluation mode (E key in HandChordAI.py)
saved in the evaluation/ folder and computes:

  1. ACCURACY - how often the recognised gesture equals the target,
     for four "methods" (see METHODS below), split by condition
     (calibrated vs fallback live pipeline) and by hand (root / quality).
  2. CONFUSION MATRICES - which gesture was recognised when the user
     made which gesture. Shows exactly WHICH gestures get mixed up.
  3. LATENCY - per trial: how long until the correct chord appeared,
     and how much of that time is the system (voting) vs the human.

Results are printed AND saved in the results/ folder:
  results/accuracy.csv          the accuracy table
  results/trials.csv            one row per trial with its timings
  results/confusion_*.png       confusion matrix pictures (needs matplotlib)
  results/latency.png           latency histogram       (needs matplotlib)

HOW TO RUN (from the HandChordAI folder)
----------------------------------------
    .\\.venv\\Scripts\\python.exe analyze_evaluation.py
        -> uses every CSV in evaluation/
    .\\.venv\\Scripts\\python.exe analyze_evaluation.py evaluation\\P01_20261122_143005.csv
        -> uses only the file(s) you name
    .\\.venv\\Scripts\\python.exe analyze_evaluation.py --method frame
        -> draw the confusion matrices for another method (default: stable)

THE FOUR METHODS (columns in the CSV)
-------------------------------------
  stable      the FULL pipeline incl. voting = what the user actually sees
  frame       the pipeline's per-frame guess (smoothing + hysteresis, no voting)
  fallback    the built-in rules on a single frame, nothing else
  calibrated  the calibrated model on a single frame, nothing else
Comparing "fallback" vs "calibrated" answers "does calibration help?".
Comparing "frame" vs "stable" shows what the voting step adds.

HOW ACCURACY IS COUNTED
-----------------------
Only frames in the "record" phase (the user had time to change hands)
where BOTH hands were visible. "NONE" (the system said "unclear") counts
as wrong, and is ALSO reported separately as the "rejection rate", because
"unclear" is a much less bad mistake than a wrong chord.
"""

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Import the main program to reuse its lists of roots and qualities, so
# this script can never disagree with the program about the gesture names.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import HandChordAI as app  # noqa: E402

BASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = BASE_DIR / "results"

# The four ways of predicting a gesture that the CSV contains.
METHODS = ["stable", "frame", "fallback", "calibrated"]
# The two hands. "root" -> columns like root_stable, target is target_root.
HANDS = {"root": app.ROOTS, "quality": app.QUALITIES}


# ------------------------------------------------------------
# Loading
# ------------------------------------------------------------

def find_csv_files(paths):
    """Turn the command-line arguments into a list of CSV files.
    No arguments -> every CSV in evaluation/. A folder -> every CSV in it."""
    if not paths:
        paths = [str(app.EVAL_DIR)]
    files = []
    for name in paths:
        path = Path(name)
        if path.is_dir():
            files += sorted(path.glob("*.csv"))
        elif path.exists():
            files.append(path)
        else:
            print(f"Not found, skipped: {path}")
    return files


def load_rows(files):
    """Read all CSV files into one list of dictionaries (one per frame).
    Numbers are converted from text so they can be compared."""
    rows = []
    for path in files:
        with open(path, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row["trial"] = int(row["trial"])
                row["t_trial"] = float(row["t_trial"])
                row["root_visible"] = row["root_visible"] == "1"
                row["quality_visible"] = row["quality_visible"] == "1"
                rows.append(row)
    return rows


# ------------------------------------------------------------
# 1. Accuracy
# ------------------------------------------------------------

def scored_frames(rows):
    """The frames accuracy is measured on: record phase, both hands visible."""
    return [r for r in rows
            if r["phase"] == "record" and r["root_visible"] and r["quality_visible"]]


def accuracy(rows, hand, method):
    """Accuracy of one method for one hand ("root"/"quality"/"chord").

    "chord" = BOTH hands right in the same frame (what matters musically).
    Returns (accuracy, rejection_rate, frame_count). Frames where the
    method has no value (e.g. "calibrated" without a calibration) are
    left out. Returns (None, None, 0) if nothing could be scored."""
    hands = ["root", "quality"] if hand == "chord" else [hand]
    correct = rejected = total = 0
    for row in rows:
        predictions = [row[f"{h}_{method}"] for h in hands]
        if any(p == "" for p in predictions):
            continue                       # not available for this frame
        total += 1
        if all(row[f"{h}_{method}"] == row[f"target_{h}"] for h in hands):
            correct += 1
        elif any(p == "NONE" for p in predictions):
            rejected += 1                  # "unclear" rather than a wrong chord
    if total == 0:
        return None, None, 0
    return correct / total, rejected / total, total


def accuracy_table(rows):
    """Accuracy for every condition x method x hand. Returns a list of dicts."""
    table = []
    # Group frames by the live condition (calibrated / fallback pipeline).
    conditions = defaultdict(list)
    for row in scored_frames(rows):
        conditions[row["live_classifier"]].append(row)

    for condition, frames in sorted(conditions.items()):
        for method in METHODS:
            for hand in ("root", "quality", "chord"):
                acc, rej, n = accuracy(frames, hand, method)
                if n == 0:
                    continue
                table.append({"condition": condition, "method": method, "hand": hand,
                              "accuracy": round(acc, 4), "rejection": round(rej, 4),
                              "frames": n})
    return table


def per_participant_table(rows):
    """Chord accuracy of the full pipeline (stable) per participant & condition.
    Useful for showing spread between people, not only the average."""
    groups = defaultdict(list)
    for row in scored_frames(rows):
        groups[(row["participant"], row["live_classifier"])].append(row)
    table = []
    for (participant, condition), frames in sorted(groups.items()):
        acc, rej, n = accuracy(frames, "chord", "stable")
        if n:
            table.append({"participant": participant, "condition": condition,
                          "accuracy": round(acc, 4), "rejection": round(rej, 4),
                          "frames": n})
    return table


# ------------------------------------------------------------
# 2. Confusion matrices
# ------------------------------------------------------------

def confusion_matrix(rows, hand, method):
    """Count how often target X was recognised as Y.

    Returns (matrix, row_labels, column_labels).
    matrix[i][j] = frames where the target was row_labels[i] and the
    prediction was column_labels[j]. The last column is "NONE" (unclear)."""
    labels = HANDS[hand]
    columns = labels + ["NONE"]
    matrix = np.zeros((len(labels), len(columns)), dtype=int)
    for row in scored_frames(rows):
        predicted = row[f"{hand}_{method}"]
        if predicted == "":
            continue
        matrix[labels.index(row[f"target_{hand}"]), columns.index(predicted)] += 1
    return matrix, labels, columns


def print_confusion(matrix, labels, columns, title):
    """Print a confusion matrix as a text table, values in % of each row."""
    print(f"\n{title}  (rows = target, columns = recognised, % of row)")
    print("        " + "".join(f"{c:>8}" for c in columns))
    for label, counts in zip(labels, matrix):
        total = counts.sum()
        cells = "".join(f"{100 * c / total:8.1f}" if total else f"{'-':>8}" for c in counts)
        print(f"{label:>8}{cells}   (n={total})")


# ------------------------------------------------------------
# 3. Latency
# ------------------------------------------------------------

def trial_latencies(rows):
    """One result per trial, describing how fast the chord was recognised.

    For every trial (in time order):
      t_stable = seconds from the trial start until the STABLE (voted)
                 chord first equalled the target. This is what a user
                 experiences: "I was told the chord -> it sounded".
      t_onset  = start of the unbroken run of correct PER-FRAME guesses
                 that led to that moment = roughly when the user's hands
                 were in the right pose.
      decision_delay = t_stable - t_onset = time added by the system's
                 voting (the human part is excluded).
    If the chord never became stable within the trial, success = False.

    The same chord never appears twice in a row (make_trial_order makes
    sure), so a correct chord can't be "left over" from the trial before.
    """
    trials = defaultdict(list)
    for row in rows:
        # One trial = same file/session + same trial number.
        trials[(row["participant"], row["session"], row["trial"])].append(row)

    results = []
    for (participant, session, trial), frames in sorted(trials.items()):
        frames.sort(key=lambda r: r["t_trial"])
        target_root = frames[0]["target_root"]
        target_quality = frames[0]["target_quality"]

        def stable_ok(r):
            return r["root_stable"] == target_root and r["quality_stable"] == target_quality

        def frame_ok(r):
            return r["root_frame"] == target_root and r["quality_frame"] == target_quality

        result = {"participant": participant, "session": session, "trial": trial,
                  "target": frames[0]["target"],
                  "condition": frames[0]["live_classifier"],
                  "success": False, "t_stable": None, "t_onset": None,
                  "decision_delay": None}

        # First frame where the stable chord is right.
        first = next((i for i, r in enumerate(frames) if stable_ok(r)), None)
        if first is not None:
            result["success"] = True
            result["t_stable"] = frames[first]["t_trial"]
            # Walk backwards while the per-frame guess was also right.
            if frame_ok(frames[first]):
                start = first
                while start > 0 and frame_ok(frames[start - 1]):
                    start -= 1
                result["t_onset"] = frames[start]["t_trial"]
                result["decision_delay"] = result["t_stable"] - result["t_onset"]
        results.append(result)
    return results


def summarise_latencies(results):
    """Print success rate and median / spread of the timings per condition."""
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r["condition"]].append(r)

    for condition, trials in sorted(by_condition.items()):
        ok = [r for r in trials if r["success"]]
        print(f"\n[{condition}] trials: {len(trials)}, chord reached: {len(ok)} "
              f"({100 * len(ok) / len(trials):.1f}%)")
        for key, text in (("t_stable", "time to correct chord"),
                          ("decision_delay", "system decision delay")):
            values = np.array([r[key] for r in ok if r[key] is not None])
            if len(values):
                # Median + interquartile range: robust against a few very slow trials.
                q1, median, q3 = np.percentile(values, [25, 50, 75])
                print(f"  {text:<24} median {median * 1000:6.0f} ms   "
                      f"IQR {q1 * 1000:.0f}-{q3 * 1000:.0f} ms   (n={len(values)})")


# ------------------------------------------------------------
# Saving
# ------------------------------------------------------------

def write_csv(path, table):
    """Save a list of dicts as a CSV (opens in Excel)."""
    if not table:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def save_plots(rows, results, method):
    """Confusion-matrix pictures and a latency histogram. Skipped (with a
    message) if matplotlib isn't installed - all numbers are printed anyway."""
    try:
        import matplotlib
        matplotlib.use("Agg")        # draw to files only, no window
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed -> no plots. Install it with:"
              "\n    python -m pip install matplotlib")
        return

    for hand in HANDS:
        matrix, labels, columns = confusion_matrix(rows, hand, method)
        if matrix.sum() == 0:
            continue
        # Row-normalise: each row sums to 100% (fair even if some targets
        # have more frames than others).
        totals = matrix.sum(axis=1, keepdims=True)
        percent = np.divide(100 * matrix, totals, out=np.zeros(matrix.shape), where=totals > 0)

        fig, ax = plt.subplots(figsize=(1 + 0.8 * len(columns), 1 + 0.7 * len(labels)))
        image = ax.imshow(percent, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(len(columns)), columns)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Recognised")
        ax.set_ylabel("Target")
        ax.set_title(f"{hand.title()} hand - {method}")
        for i in range(len(labels)):
            for j in range(len(columns)):
                if matrix[i, j]:
                    ax.text(j, i, f"{percent[i, j]:.0f}", ha="center", va="center",
                            color="white" if percent[i, j] > 50 else "black", fontsize=8)
        fig.colorbar(image, ax=ax, label="% of target frames")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / f"confusion_{hand}_{method}.png", dpi=200)
        plt.close(fig)

    values = [r["t_stable"] for r in results if r["t_stable"] is not None]
    if values:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.hist(np.array(values) * 1000, bins=20, color="#4a78b5")
        ax.set_xlabel("Time from target shown to correct chord (ms)")
        ax.set_ylabel("Trials")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "latency.png", dpi=200)
        plt.close(fig)
    print(f"\nPlots saved in {RESULTS_DIR}")


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Analyse HandChordAI evaluation recordings.")
    parser.add_argument("files", nargs="*",
                        help="CSV files or folders (default: the evaluation folder)")
    parser.add_argument("--method", choices=METHODS, default="stable",
                        help="which method the confusion-matrix plots show")
    args = parser.parse_args()

    files = find_csv_files(args.files)
    if not files:
        print("No evaluation files found. Record one first: start HandChordAI.py "
              "and press E.")
        return 1
    rows = load_rows(files)
    print(f"Loaded {len(rows)} frames from {len(files)} file(s).")

    scored = scored_frames(rows)
    print(f"{len(scored)} frames scored (record phase, both hands visible).")
    if rows:
        # Frames where a hand was missing are not scored - say how many,
        # because a lot of them would mean a tracking problem.
        record = [r for r in rows if r["phase"] == "record"]
        if record:
            print(f"Both hands visible in {100 * len(scored) / len(record):.1f}% "
                  f"of record-phase frames.")

    RESULTS_DIR.mkdir(exist_ok=True)

    # ---- 1. Accuracy ----
    table = accuracy_table(rows)
    print("\nACCURACY  (rejection = share of frames answered 'unclear')")
    print(f"{'condition':<12}{'method':<12}{'hand':<9}{'accuracy':>9}{'reject':>8}{'frames':>8}")
    for t in table:
        print(f"{t['condition']:<12}{t['method']:<12}{t['hand']:<9}"
              f"{100 * t['accuracy']:8.1f}%{100 * t['rejection']:7.1f}%{t['frames']:8}")
    write_csv(RESULTS_DIR / "accuracy.csv", table)

    participants = per_participant_table(rows)
    if len(participants) > 1:
        print("\nPER PARTICIPANT (full pipeline, chord accuracy)")
        for p in participants:
            print(f"  {p['participant']:<8}{p['condition']:<12}{100 * p['accuracy']:6.1f}%"
                  f"  (n={p['frames']})")
        write_csv(RESULTS_DIR / "participants.csv", participants)

    # ---- 2. Confusion matrices ----
    for hand in HANDS:
        matrix, labels, columns = confusion_matrix(rows, hand, args.method)
        if matrix.sum():
            print_confusion(matrix, labels, columns, f"CONFUSION {hand.upper()} ({args.method})")

    # ---- 3. Latency ----
    results = trial_latencies(rows)
    print("\nLATENCY")
    summarise_latencies(results)
    write_csv(RESULTS_DIR / "trials.csv", results)

    save_plots(rows, results, args.method)
    return 0


if __name__ == "__main__":
    sys.exit(main())
