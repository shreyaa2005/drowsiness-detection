import socket
import numpy as np
import cv2
import time
import struct
from micro_model2 import getPrediction

UDP_HOST = '0.0.0.0'
UDP_PORT = 8080  
ESP32_FEEDBACK_PORT = 8081 

FRAME_START_MARKER = 0xAA
FRAME_END_MARKER = 0xFF

frame_count = 0
start_time = 0
esp32_ip = ""

def simulate_ai_model(frame):
    global frame_count,esp32_ip
    res = getPrediction(frame)
    if esp32_ip is None:
        print("Cannot send result: ESP32 IP address unknown")
        return res
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)        
        
        packet = bytearray(5)
        
        packet[0] = frame_count & 0xFF
        packet[1] = (frame_count >> 8) & 0xFF
        packet[2] = (frame_count >> 16) & 0xFF
        packet[3] = (frame_count >> 24) & 0xFF
        
        packet[4] = res

        sock.sendto(packet, (esp32_ip, ESP32_FEEDBACK_PORT))
        
        
    except Exception as e:
        print(f"Error sending AI result: {e}")
    return res


def display(frame_number, frame):
    global start_time
    fps = 0
    if frame is not None:
        frame_count += 1
        current_time = time.time()
        elapsed_time = current_time - start_time
            
        if elapsed_time >= 1.0:
            fps = frame_count / elapsed_time
            frame_count = 0
            start_time = current_time
            
        display_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            
        ai_result = simulate_ai_model(frame)
            
        h, w = frame.shape
        center_y, center_x = h // 2, w // 2
        center_h, center_w = h // 4, w // 4
        cv2.rectangle(display_frame, 
                     (center_x-center_w//2, center_y-center_h//2),
                     (center_x+center_w//2, center_y+center_h//2),
                     (0, 255, 0) if ai_result == 1 else (0, 0, 255), 2)
            
        cv2.putText(display_frame, f"Frame: {frame_number}", (10, 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"FPS: {fps:.1f}", (10, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(display_frame, f"AI Result: {ai_result}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if ai_result == 1 else (0, 0, 255), 1)
        cv2.putText(display_frame, "Buzzer: " + ("ON" if ai_result == 1 else "OFF"), 
                   (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                   (0, 255, 0) if ai_result == 1 else (0, 0, 255), 1)
            
        cv2.imshow('ESP32 Camera AI Processing', display_frame)

def receiveStream():
    global start_time
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_HOST, UDP_PORT))
    sock.settimeout(1.0)

    current_frame_buffer = bytearray()
    current_frame_size = 0
    current_frame_width = 0
    current_frame_height = 0
    current_frame_counter = 0
    frame_in_progress = False
    start_time = time.time()
    while True:
        try:
            data, _ = sock.recvfrom(65535)

            if not data:
                continue

            if len(data) >= 13 and data[0] == FRAME_START_MARKER:
                width = struct.unpack("<H", data[1:3])[0]        
                height = struct.unpack("<H", data[3:5])[0]       
                total_size = struct.unpack("<I", data[5:9])[0]
                frame_counter = struct.unpack("<I", data[9:13])[0]  
                
                current_frame_buffer = bytearray()
                current_frame_size = total_size
                current_frame_width = width
                current_frame_height = height
                current_frame_counter = frame_counter
                frame_in_progress = True
                
                #if frame_counter %30 == 2: 
                print(f"New frame #{frame_counter}: {width}x{height}, {total_size} bytes")
                continue
                
            if len(data) == 1 and data[0] == FRAME_END_MARKER:
                if frame_in_progress and len(current_frame_buffer) == current_frame_size:
                    try:
                        frame_array = np.frombuffer(current_frame_buffer, dtype=np.uint8)
                        
                        frame = frame_array.reshape((current_frame_height, current_frame_width))
                        display(current_frame_counter, frame)
                        
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        
                frame_in_progress = False
                continue
                
            if frame_in_progress:
                current_frame_buffer.extend(data)

        except socket.timeout:
            continue
        except Exception as e:
            print(f"Error receiving data: {e}")
            time.sleep(0.1)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    print("Exiting....")
    cv2.destroyAllWindows()

def startServer():
    print("Received Address: ",esp32_ip)
    time.sleep(5)
    print("ESP32 Camera AI System Starting")
    print("===============================")
    print("This system receives grayscale images from an ESP32")
    print("Processes them with an AI model, and sends results back")
    print("Press 'q' on the display window to exit")
    print("===============================")

    receiveStream()