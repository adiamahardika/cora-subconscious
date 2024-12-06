from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from flask_socketio import SocketIO, emit
from flask_cors import CORS

# Load models (same as your provided code)
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

def genderAge(image):

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    genderList=['Male','Female']
    blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)  
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]
    return gender

def crop_with_padding(image, box, padding):
    # Extract the coordinates from the box
    startX, startY, endX, endY = box

    # Apply padding to the box
    startX -= padding
    startY -= padding
    endX += padding
    endY += padding

    # Ensure the coordinates are within the image boundaries
    startX = max(0, startX)
    startY = max(0, startY)
    endX = min(endX, image.shape[1])
    endY = min(endY, image.shape[0])

    # Crop the image using the adjusted box coordinates
    cropped_image = image[startY:endY, startX:endX]

    return cropped_image

def draw_rounded_rectangle(image, top_left, bottom_right, color, thickness, radius, dash=False):
    x1, y1 = top_left
    x2, y2 = bottom_right
    
    if thickness == cv2.FILLED:
        # Draw filled rectangle in the middle
        cv2.rectangle(image, (x1 + radius, y1), (x2 - radius, y2), color, cv2.FILLED)
        cv2.rectangle(image, (x1, y1 + radius), (x2, y2 - radius), color, cv2.FILLED)
        
        # Draw four filled circles at the corners
        cv2.circle(image, (x1 + radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(image, (x2 - radius, y1 + radius), radius, color, cv2.FILLED)
        cv2.circle(image, (x1 + radius, y2 - radius), radius, color, cv2.FILLED)
        cv2.circle(image, (x2 - radius, y2 - radius), radius, color, cv2.FILLED)
    else:
        # Draw the four corner arcs
        cv2.ellipse(image, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
        cv2.ellipse(image, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
        cv2.ellipse(image, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)
            
        cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
        cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
        cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
        cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


def draw_label(image, bbox, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, font_thickness=1):
    startX, startY, endX, endY = bbox

    # Split label text into lines
    lines = label.split('\n')

    # Determine the size of the text
    text_height = cv2.getTextSize('A', font, font_scale, font_thickness)[0][1]
    text_widths = [cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines]
    max_text_width = max(text_widths)

    # Calculate padding around the text
    padding_x = 10
    padding_y = 10

    # Calculate the position for the label background
    x = startX
    y = startY - (len(lines) * (text_height + padding_y)) - 10  # Position above the bounding box
    w = x + max_text_width + (2 * padding_x)
    h = y + len(lines) * (text_height + padding_y)

    # Ensure the rectangle does not go out of the image boundaries
    y = max(0, y)
    h = min(image.shape[0], h)

    # Draw the rounded rectangle background
    draw_rounded_rectangle(image, (x, y), (w, h), (7, 186, 255), cv2.FILLED, 10)

    # Draw each line of the label text
    for i, line in enumerate(lines):
        text_x = x + padding_x
        text_y = y + padding_y + i * (text_height + padding_y)
        cv2.putText(image, line, (text_x, text_y), font, font_scale, (30, 30, 30), font_thickness)

# Flask app
app = Flask(__name__)
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode = 'threading')

is_streaming = False

def detect_and_stream():
    global is_streaming
    
    if is_streaming:
        print("Stream already active, skipping...")
        return
    is_streaming = True

    try:

        video_read = cv2.VideoCapture(0)
        previous_boxes = []

        while True:
            ret_val, img = video_read.read()
            if not ret_val:
                break

            h, w = img.shape[:2]
            blob = cv2.dnn.blobFromImage(img, 1.0, (300, 300), (104.0, 177.0, 123.0), swapRB=False, crop=False)
            faceNet.setInput(blob)
            detections = faceNet.forward()

            detected = False
            
            current_boxes = []
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.7:
                    detected = True
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    current_boxes.append((startX, startY, endX, endY))

                    is_new_detection = True
                    for prev_box in previous_boxes:
                        iou = calculate_iou(prev_box, (startX, startY, endX, endY))
                        if iou > 0.5:
                            is_new_detection = False
                            break
                    
                    if is_new_detection:
                        cropped_image = crop_with_padding(img, box.astype(int), 20)
                        gender = genderAge(cropped_image)
                        #send data to front end
                        socketio.emit('detection', {'gender': gender})  # Emit gender if detected
                        print(f"Emitting gender: {gender}")

                    draw_rounded_rectangle(img, (startX, startY), (endX, endY), (255, 103, 7), 4, radius=10, dash=True)
                    label = f"Person Detected: {detected}\nGender: {gender}"
                    draw_label(img, (startX, startY, endX, endY), label)

            previous_boxes = current_boxes
            
            if not current_boxes:
                cv2.putText(img, "No Person Detected", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            # Encode the frame for streaming
            _, buffer = cv2.imencode('.jpg', img)
            frame = buffer.tobytes()
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    finally:
        is_streaming = False
        video_read.release()

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    # Calculate the intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate the union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = box1_area + box2_area - intersection

    # Return IoU
    if union == 0:
        return 0
    return intersection / union

@app.route('/')
def index():
    return render_template('frontend.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_and_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

@socketio.on('connect')
def handle_connect():
    print("Client connected")

@socketio.on('disconnect')
def handle_disconnect():
    print("Client disconnected")

if __name__ == "__main__":
    socketio.run(app, debug=True)
