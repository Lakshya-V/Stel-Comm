import cv2
import time
from src.gesture_detector import LiveGestureDetector
from src.socket_comm import NetworkNode

# --- CONFIGURATION ---
RECEIVER_IP = "192.168.1.100"  # CHANGE THIS TO LAPTOP B's LOCAL IP ADDRESS!
RECEIVER_PORT = 5005
NODE_ID = "Tactical_Unit_Alpha"

detector = LiveGestureDetector(stability_threshold=6)
node = NetworkNode(node_id=NODE_ID)

cap = cv2.VideoCapture(0)

print(f"\n[SENDER RUNNING] Transmitting tactical commands to {RECEIVER_IP}:{RECEIVER_PORT}")
print("Press 'q' to Quit.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    frame, raw, confirmed, trigger = detector.process_frame(frame)

    # Trigger action if gesture passes stability buffer
    if trigger:
        print(f"[TACTICAL ALERT] Confirmed Gesture: '{confirmed}' -> Sending Network Packet...")
        node.send_command(RECEIVER_IP, RECEIVER_PORT, command=confirmed)

    # UI Overlay
    cv2.putText(frame, f"Raw: {raw}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"Confirmed Command: {confirmed}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Laptop A - Tactical Camera Node", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
detector.close()
node.stop()
cv2.destroyAllWindows()