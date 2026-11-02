# HandChordAI

HandChordAI is a Windows desktop virtual guitar that turns two-hand gestures
into guitar chords. It uses a webcam to track hand landmarks, interprets the
number of extended fingers on each hand, and plays a synthesized chord when a
stable gesture is recognized.

The application is implemented in [HandChordAI.py](./HandChordAI.py) with
OpenCV, MediaPipe, NumPy, and Python's built-in audio/file modules.

## Features

- Real-time webcam hand tracking for up to two hands.
- Left-hand gesture selection for the chord root: E, C, G, D, or A.
- Right-hand gesture selection for chord quality: major, minor, or dominant
  seventh.
- 15 supported chords: five roots multiplied by three qualities.
- Procedurally generated, strummed-style WAV audio with harmonics and
  attack/decay envelopes.
- On-screen studio interface showing detected hands, finger states, the active
  chord, notes, guitar fingering, and FPS.
- Gesture stabilization using a rolling history and majority voting to reduce
  accidental chord changes.
- Optional diagnostic overlays for finger angles, ratios, scores, and
  recognition state.
- Automatic download of the MediaPipe hand-landmarker model on first launch.

## Gesture controls

The application assigns hands by their position on the mirrored camera image:

| Hand | Extended fingers | Selection |
| --- | ---: | --- |
| Left | 0 | E root |
| Left | 1 | C root |
| Left | 2 | G root |
| Left | 3 | D root |
| Left | 4 | A root |
| Right | 1 | Major |
| Right | 2 | Minor |
| Right | 3 | Dominant 7th |

Hold both gestures steadily until the interface recognizes them. The selected
chord is played once when it becomes active. A short cooldown prevents repeated
playback while the same gesture remains in view.

## Supported chords

| Root | Major | Minor | 7th |
| --- | --- | --- | --- |
| C | C major (`x32010`) | C minor (`x35543`) | C7 (`x32310`) |
| G | G major (`320003`) | G minor (`355333`) | G7 (`320001`) |
| D | D major (`xx0232`) | D minor (`xx0231`) | D7 (`xx0212`) |
| A | A major (`x02220`) | A minor (`x02210`) | A7 (`x02020`) |
| E | E major (`022100`) | E minor (`022000`) | E7 (`022130`) |

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
python -m pip install opencv-python numpy mediapipe
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

At startup the program generates the chord WAV files in the `sounds` folder,
opens the webcam, and downloads `hand_landmarker.task` beside the script if it
is not already present. The generated audio and downloaded model are local
runtime assets and do not need to be created manually.

## Keyboard controls

| Key | Action |
| --- | --- |
| `Q` | Quit the application |
| `D` | Toggle diagnostic information |
| `R` | Reset gesture recognition history and the active chord |
| `Space` | Replay the currently selected chord |

## How it works

1. OpenCV captures and mirrors a webcam frame.
2. MediaPipe's Hand Landmarker detects up to two hands and produces image and
   world landmarks.
3. Hands are sorted by horizontal position: the screen-left hand controls the
   root and the screen-right hand controls the quality.
4. Finger extension is estimated from landmark angles, relative distances, and
   hysteresis thresholds that help prevent flickering between states.
5. Recent finger counts are stored in short histories. A count is accepted only
   after enough matching votes have accumulated.
6. The root and quality are combined into a chord key.
7. NumPy synthesizes each chord as several harmonically related sine waves with
   staggered starts, an attack/decay envelope, normalization, and a short fade
   out.
8. The WAV file is played asynchronously with Windows audio playback while the
   OpenCV interface continues rendering.

## Troubleshooting

- **The camera does not open:** check camera permissions, close other camera
  applications, and verify that the configured `CAMERA_INDEX` is correct.
- **The hand model cannot be downloaded:** check the internet connection and
  try again. The model URL and local path are defined near the top of
  `HandChordAI.py`.
- **Gestures are unstable:** improve lighting, keep both hands inside the
  camera view, and hold the gesture still for a moment. The recognition
  thresholds and vote history are configurable constants in the script.
- **No sound is heard:** check Windows output volume and make sure the WAV
  files are being generated in the `sounds` directory.
- **Only one hand is detected:** move both hands farther apart and ensure
  fingers are visible without heavy occlusion.

## Project status

This is a self-contained prototype for experimenting with gesture-controlled
musical interaction. Configuration values such as camera resolution,
recognition confidence, chord duration, and stabilization thresholds are
defined directly in `HandChordAI.py`.

## License

No license has been specified yet.
