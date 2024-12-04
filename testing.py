from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from threading import Thread, Lock

# Load models
faceModel = "model/model/facenet/opencv_face_detector_uint8.pb"
faceProto = "model/model/facenet/opencv_face_detector.pbtxt"
ageModel = "model/model/age/age_net.caffemodel"
ageProto = "model/model/age/age_deploy.prototxt"
genderModel = "model/model/gender/gender_net.caffemodel"
genderProto = "model/model/gender/gender_deploy.prototxt"
emotionModel = "model/model/emotion/emotion-ferplus-8.onnx"

faceNet = cv2.dnn.readNetFromTensorflow(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)
emotionNet = cv2.dnn.readNetFromONNX(emotionModel)

# Flask app
app = Flask(__name__)

# Shared video stream
video_lock = Lock()
video_stream = cv2.VideoCapture(0)

def detect_and_stream():
    while True:
        with video_lock:
            ret_val, img = video_stream.read()
        if not ret_val:
            break

        h, w = img.shape[:2]
        blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
        faceNet.setInput(blob)
        detections = faceNet.forward()

        detected = False
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.7:
                detected = True
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(img, (startX, startY), (endX, endY), (255, 103, 7), 3)
                label = "Person Detected"
                cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if not detected:
            cv2.putText(img, "No Person Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        _, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/detect_person', methods=['GET'])
def detect_person():
    """API Endpoint to detect if a person is present."""
    with video_lock:
        ret_val, img = video_stream.read()
    if not ret_val:
        return jsonify({"error": "Could not access camera"}), 500

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            detected = True
            break

    return jsonify({"person_detected": detected})

if __name__ == "__main__":
    try:
        app.run(debug=True)
    finally:
        with video_lock:
            video_stream.release()
