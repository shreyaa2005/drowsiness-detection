from flask import Flask
import threading

import mainV2

app = Flask(__name__)
server_started = False

@app.route("/<address>")
def start(address):
    global server_started
    print(f"Received ESP32 IP: {address}")
    mainV2.esp32_ip = address
    
    if not server_started:
        server_started = True
        thread = threading.Thread(target=mainV2.startServer, daemon=True)
        thread.start()
    
    return "Connected successfully"

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)