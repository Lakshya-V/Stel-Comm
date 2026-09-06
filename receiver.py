import cv2
import numpy as np
import time
from src.socket_comm import NetworkNode

# Global state tracker
latest_command = "WAITING_FOR_DATA"
last_sender = "N/A"
last_received_time = 0

def handle_incoming_packet(payload, addr):
    global latest_command, last_sender, last_received_time
    latest_command = payload.get("command", "UNKNOWN")
    last_sender = payload.get("source", "UNKNOWN")
    last_received_time = time.time()
    
    print(f"\n[PACKET RECEIVED] From {addr[0]}")
    print(f" Source Node: {last_sender}")
    print(f" Command:     {latest_command}")
    print(f" Timestamp:   {payload.get('timestamp')}")

node = NetworkNode(node_id="Base_Station_Beta", listen_port=5005)
node.start_listening(handle_incoming_packet)

print("\n[RECEIVER RUNNING] Listening for incoming tactical transmissions on port 5005...")
print("Press 'q' to Quit.\n")

# Create a simple visual dashboard window
while True:
    # Build 500x800 RGB Canvas
    canvas = np.zeros((500, 800, 3), dtype=np.uint8)
    
    # Alert Styling based on Command
    if latest_command == "FIST":
        bg_color = (0, 0, 180)  # Red Alert
        status_msg = "TACTICAL HOLD / EMERGENCY ALERT"
    elif latest_command == "OPEN_PALM":
        bg_color = (0, 180, 0)  # Green Normal
        status_msg = "SYSTEM NORMAL / ALL CLEAR"
    elif latest_command == "PEACE":
        bg_color = (180, 180, 0) # Cyan Recon Mode
        status_msg = "RECON MODE ACTIVATED"
    else:
        bg_color = (40, 40, 40) # Idle Gray
        status_msg = "AWAITING TRANSMISSION..."

    cv2.rectangle(canvas, (20, 20), (780, 480), bg_color, -1)
    
    # Dashboard Text
    cv2.putText(canvas, "BASE STATION MESH COMMAND CENTER", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(canvas, f"Source Node: {last_sender}", (40, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(canvas, f"Received Command: {latest_command}", (40, 200), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
    cv2.putText(canvas, f"Status: {status_msg}", (40, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    if last_received_time > 0:
        elapsed = time.time() - last_received_time
        cv2.putText(canvas, f"Last packet: {elapsed:.1f}s ago", (40, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

    cv2.imshow("Laptop B - Base Receiver Dashboard", canvas)
    
    if cv2.waitKey(100) & 0xFF == ord('q'):
        break

node.stop()
cv2.destroyAllWindows()