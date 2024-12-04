from flask import Flask, jsonify, Response
from flask_cors import CORS
import cv2
import numpy as np

app = Flask(__name__)
CORS(app)

# Load face detection model
faceModel = "model/model/facenet/opencv_face_detector_uint8.pb"
faceProto = "model/model/facenet/opencv_face_detector.pbtxt"
faceNet = cv2.dnn.readNetFromTensorflow(faceModel, faceProto)

@app.route('/detect', methods=['GET'])
def detect_person():
    video_read = cv2.VideoCapture(0)  # Access the webcam
    ret_val, img = video_read.read()

    if not ret_val:
        return jsonify({"error": "Unable to access webcam"}), 500

    h, w = img.shape[:2]
    blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
    faceNet.setInput(blob)
    detections = faceNet.forward()

    person_detected = False
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.7:
            person_detected = True
            break

    video_read.release()
    return jsonify({"person_detected": person_detected})


if __name__ == '__main__':
    app.run(debug=True)
