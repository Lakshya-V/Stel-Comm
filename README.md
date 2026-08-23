# MediaPipe Hand Landmarker

This project uses the MediaPipe Hand Landmarker task with OpenCV to detect hand landmarks from a webcam.

## Model File

The program requires the `hand_landmarker.task` model file in the project root, beside `detect.py`.

The model is downloaded separately from the official [MediaPipe Hand Landmarker documentation](https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker).

## Install

```bash
pip install -r requirements.txt
```

## Run

```bash
python detect.py
```

Press `q` to stop the webcam window.
