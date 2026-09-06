import cv2
import torch
import numpy as np
import mediapipe as mp
import time
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from model import TacticalGestureMLP

class LiveGestureDetector:
    def __init__(self, model_path='models/tactical_mlp.pth', mp_task_path='models/hand_landmarker.task', stability_threshold=5):
        self.stability_threshold = stability_threshold
        self.gesture_counter = 0
        self.last_detected_gesture = "NONE"
        self.confirmed_gesture = "NONE"

        # 1. Load PyTorch Model Checkpoint
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        self.id_to_label = checkpoint['id_to_label']
        num_classes = checkpoint['num_classes']

        self.mlp_model = TacticalGestureMLP(num_classes=num_classes)
        self.mlp_model.load_state_dict(checkpoint['model_state_dict'])
        self.mlp_model.eval()

        # 2. Setup MediaPipe Landmarker
        base_options = python.BaseOptions(model_asset_path=mp_task_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = vision.HandLandmarker.create_from_options(options)

    def process_frame(self, frame):
        h, w, _ = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        frame_timestamp_ms = int(time.time() * 1000)

        results = self.landmarker.detect_for_video(mp_image, frame_timestamp_ms)

        raw_gesture = "NO_HAND"

        if results.hand_landmarks:
            hand_landmarks = results.hand_landmarks[0]
            
            # Normalize Landmarks (Wrist Origin)
            wrist = hand_landmarks[0]
            normalized = []
            for lm in hand_landmarks:
                normalized.extend([lm.x - wrist.x, lm.y - wrist.y, lm.z - wrist.z])

            # Model Inference
            input_tensor = torch.tensor([normalized], dtype=torch.float32)
            with torch.no_grad():
                outputs = self.mlp_model(input_tensor)
                _, pred = torch.max(outputs, 1)
                raw_gesture = self.id_to_label[pred.item()]

            # Draw Hand Overlays
            mp_hands = mp.tasks.vision.HandLandmarksConnections
            mp_drawing = mp.tasks.vision.drawing_utils
            mp_drawing_styles = mp.tasks.vision.drawing_styles
            
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Confidence Buffer Check (Hold gesture for N frames)
        if raw_gesture == self.last_detected_gesture and raw_gesture not in ["NO_HAND", "NONE"]:
            self.gesture_counter += 1
        else:
            self.gesture_counter = 0
            self.last_detected_gesture = raw_gesture

        trigger_event = False
        if self.gesture_counter >= self.stability_threshold:
            if self.confirmed_gesture != raw_gesture:
                self.confirmed_gesture = raw_gesture
                trigger_event = True  # Flag to send network message once verified

        return frame, raw_gesture, self.confirmed_gesture, trigger_event

    def close(self):
        self.landmarker.close()