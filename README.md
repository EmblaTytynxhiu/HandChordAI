# HandChordAI

HandChordAI is a Windows desktop virtual guitar that turns two-hand gestures
into guitar chords. It uses a webcam to track hand landmarks, interprets
finger poses, and loops a guitar sound when a stable chord gesture is
recognized.

The application is implemented in [HandChordAI.py](./HandChordAI.py) with
OpenCV, MediaPipe, NumPy, and Python's built-in audio/file modules.

## Features

- Real-time webcam hand tracking for up to two hands.
- Left-hand gesture selection for seven roots: C, D, E, F, G, A, or B.
- Right-hand gesture selection for major, minor, or dominant seventh quality.
- 21 supported chords: seven roots multiplied by three qualities.
- Pre-generated strummed-style WAV audio stored in `sounds_v3`, with missing
  files generated automatically at startup.
- A six-pose personal calibration wizard that improves finger recognition for
  each hand and saves its profiles in `finger_calibration.json`.
- One-Euro filtering and temporal smoothing for steadier hand and finger
  tracking.
- On-screen studio interface showing detected hands, finger states, the active
  chord, notes, guitar fingering, and FPS.
- An in-app gesture guide (`G`) and optional diagnostic overlay (`D`).
- Gesture stabilization using a rolling history and majority voting to reduce
  accidental chord changes.
- Chord playback stops shortly after the hands leave the frame or both gestures
  are released.
- Automatic download of the MediaPipe hand-landmarker model on first launch.

## Gesture controls

The application assigns hands by their position on the mirrored camera image:

| Hand | Extended fingers | Selection |
| --- | ---: | --- |
| Left / root | Fist, thumb tucked | C |
| Left / root | Index up | D |
| Left / root | Index and middle up | E |
| Left / root | Index, middle, and ring up | F |
| Left / root | All four fingers up | G |
| Left / root | Thumb out only | A |
| Left / root | Thumb and index up | B |
| Right / quality | Index up | Major |
| Right / quality | Index and middle up | Minor |
| Right / quality | Index, middle, and ring up | Dominant 7th |

Hold both gestures steadily until the interface recognizes them. The selected
chord starts looping when it becomes active and restarts only when the chord
changes.

The thumb on the right/quality hand is ignored. Press `C` to start or cancel
the calibration wizard. Calibration is saved beside the script and is loaded
automatically on the next run.

## Supported chords

| Root | Major | Minor | 7th |
| --- | --- | --- | --- |
| C | C (`x32010`) | Cm (`x35543`) | C7 (`x32310`) |
| D | D (`xx0232`) | Dm (`xx0231`) | D7 (`xx0212`) |
| E | E (`022100`) | Em (`022000`) | E7 (`020100`) |
| F | F (`133211`) | Fm (`133111`) | F7 (`131211`) |
| G | G (`320003`) | Gm (`355333`) | G7 (`320001`) |
| A | A (`x02220`) | Am (`x02210`) | A7 (`x02020`) |
| B | B (`x24442`) | Bm (`x24432`) | B7 (`x21202`) |

The displayed fret strings use standard guitar chord notation, where `x`
indicates a muted string and `0` indicates an open string.

## Requirements

- Windows, because playback uses the `winsound` module.
- Python 3.10 or newer recommended.
- A working webcam.
- Internet access on the first run so the MediaPipe model can be downloaded.
- Speakers or headphones.

## Installation

Create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run the script through the environment's
interpreter directly:

```powershell
.\.venv\Scripts\python.exe HandChordAI.py
```

## Run

```powershell
python HandChordAI.py
```

Command-line options:

| Option | Effect |
| --- | --- |
| `--camera N` | Use webcam number `N` (default `0`) |
| `--mute` | Start with sound off (press `M` to turn it on) |
| `--help` | List all options |

At startup the program opens the webcam and loads `finger_calibration.json` if
it exists. It creates any missing chord WAV files in `sounds_v3` and downloads
`hand_landmarker.task` beside the script if it is not already present. The
audio files, calibration profile, and downloaded model are local runtime assets
and do not need to be created manually.

## Keyboard controls

| Key | Action |
| --- | --- |
| `C` | Start/cancel the calibration wizard |
| `G` | Show/hide the gesture guide |
| `D` | Toggle diagnostic information |
| `R` | Reset gesture recognition history and the active chord |
| `Space` | Replay the currently selected chord |
| `M` | Mute / unmute the sound |
| `S` | Save a screenshot to the `screenshots` folder |
| `Q`, `Esc`, or the window's X | Quit the application |

## Tests

The non-camera logic (chord table, gesture matching, voting, filtering,
finger features) has automatic tests that use only built-in Python:

```powershell
.\.venv\Scripts\python.exe -m unittest discover tests
```

## How it works

1. OpenCV captures and mirrors a webcam frame.
2. MediaPipe's Hand Landmarker detects up to two hands and produces image and
   world landmarks.
3. Hands are sorted by horizontal position: the screen-left hand controls the
   root and the screen-right hand controls the quality.
4. Landmark positions are smoothed with One-Euro filters. Finger measurements
   use landmark angles and relative distances, with probability smoothing and
   hysteresis to prevent flickering.
5. Recent gesture votes are accumulated. A template is accepted only after
   enough matching frames in the short history.
6. Calibration can replace the built-in finger templates with per-hand models
   generated from six recorded poses.
7. The root and quality are combined into one of 21 chord keys.
8. The matching WAV file from `sounds_v3` is looped asynchronously with Windows
   audio playback while the OpenCV interface continues rendering.

## Troubleshooting

- **The camera does not open:** check camera permissions, close other camera
  applications, and try another camera with `python HandChordAI.py --camera 1`.
- **The hand model cannot be downloaded:** check the internet connection and
  try again. The model URL and local path are defined near the top of
  `HandChordAI.py`.
- **Gestures are unstable:** improve lighting, keep both hands 40–70 cm from
  the camera without overlapping, and hold each pose still. Press `C` and
  complete the calibration wizard for more accurate personal recognition.
- **No sound is heard:** check Windows output volume and make sure the WAV
  files exist in the `sounds_v3` directory. Missing files are regenerated
  automatically at startup.
- **Only one hand is detected:** move both hands farther apart and ensure
  fingers are visible without heavy occlusion.

## Project status

This is a self-contained prototype for experimenting with gesture-controlled
musical interaction. Configuration values such as camera resolution,
recognition confidence, chord duration, and stabilization thresholds are
defined directly in `HandChordAI.py`.

## License

No license has been specified yet.
