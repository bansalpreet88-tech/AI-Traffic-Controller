from flask import Flask, render_template, Response, jsonify
import cv2
from ultralytics import YOLO
import time

app = Flask(__name__)
model = YOLO("models/yolov8n.pt")
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer size to minimize latency

VEHICLE_CLASSES = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

traffic_signal = "red"
signal_timer = 15
last_signal_change = time.time()

def detect_vehicles(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(rgb_frame)[0]  # Fix: Extract results properly
    vehicles = []
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id in VEHICLE_CLASSES:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            vehicles.append((class_id, (x1, y1, x2, y2)))
    return vehicles

def process_frame():
    global traffic_signal, signal_timer, last_signal_change

    while True:
        cap.grab()  # Grab the latest frame to reduce latency
        ret, frame = cap.read()
        if not ret:
            break

        vehicles = detect_vehicles(frame)
        vehicle_count = len(vehicles)

        for class_id, bbox in vehicles:
            x1, y1, x2, y2 = bbox
            label = VEHICLE_CLASSES[class_id]
            color = (0, 255, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Traffic signal control logic
        current_time = time.time()
        elapsed_time = current_time - last_signal_change

        if elapsed_time >= signal_timer:
            if traffic_signal == "red":
                traffic_signal = "green" if vehicle_count >= 10 else "yellow" if vehicle_count >= 5 else "red"
                signal_timer = 15 if vehicle_count >= 10 else 10 if vehicle_count >= 5 else 15
            elif traffic_signal == "green":
                traffic_signal = "yellow"
                signal_timer = 4
            elif traffic_signal == "yellow":
                traffic_signal = "red"
                signal_timer = 15
            last_signal_change = current_time

        cv2.putText(frame, f"Signal: {traffic_signal}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, f"Vehicle Count: {vehicle_count}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])  # Optimize JPEG encoding for speed
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(process_frame(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/traffic_status')
def traffic_status():
    return jsonify({"traffic_light": traffic_signal, "vehicle_count": len(detect_vehicles(cap.read()[1]))})

if __name__ == "__main__":
    try:
        app.run(debug=True, threaded=True)  # Enable threading for better performance
    finally:
        cap.release()
        cv2.destroyAllWindows()