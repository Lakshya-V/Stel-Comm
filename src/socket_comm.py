import socket
import json
import threading
import time


class NetworkNode:
    def __init__(self, node_id, listen_port=5005):
        self.node_id = node_id
        self.listen_port = listen_port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(('0.0.0.0', self.listen_port))
        self.is_running = True

    def start_listening(self, callback_fn):
        def listen_loop():
            while self.is_running:
                try:
                    data, addr = self.sock.recvfrom(1024)
                    payload = json.loads(data.decode('utf-8'))
                    callback_fn(payload, addr)
                except Exception as e:
                    if not self.is_running:
                        break

        thread = threading.Thread(target=listen_loop, daemon=True)
        thread.start()

    def send_command(self, target_ip, target_port, command, destination_node="Base_Node"):
        payload = {
            "source": self.node_id,
            "destination": destination_node,
            "command": command,
            "timestamp": time.time()
        }
        message = json.dumps(payload).encode('utf-8')
        self.sock.sendto(message, (target_ip, target_port))

    def stop(self):
        self.is_running = False
        self.sock.close()