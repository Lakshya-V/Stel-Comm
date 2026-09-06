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

gesture_mesh_project/
│
├── models/
│   └── hand_landmarker.task       # Google's MediaPipe model binary
│
├── src/
│   ├── __init__.py
│   ├── gesture_detector.py        # MediaPipe pipeline & gesture recognition logic
│   ├── network_mesh.py            # NetworkX graph representation & routing rules
│   └── socket_comm.py             # Socket server/client for inter-laptop messaging
│
├── sender_app.py                  # MAIN program for Laptop A (Sender)
├── receiver_app.py                # MAIN program for Laptop B (Receiver)
│
├── requirements.txt               # Dependencies list
└── README.md                      # Setup and usage instructions
