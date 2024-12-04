import cv2
import numpy as np

# faceModel="model/facenet/yolov8n-face.onnx"
faceModel="model/model/facenet/opencv_face_detector_uint8.pb"
faceProto="model/model/facenet/opencv_face_detector.pbtxt"

ageModel="model/model/age/age_net.caffemodel"
ageProto="model/model/age/age_deploy.prototxt"

genderModel="model/model/gender/gender_net.caffemodel"
genderProto="model/model/gender/gender_deploy.prototxt"

emotionModel="model/model/emotion/emotion-ferplus-8.onnx"

attrModel="model/model/attr/single_path_resnet_celeba.caffemodel"
attrProto="model/model/attr/celeba.prototxt"

# faceNet=cv2.dnn.readNetFromONNX(faceModel)
# faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
# faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
faceNet=cv2.dnn.readNetFromTensorflow(faceModel, faceProto)
#Load age detection model
ageNet=cv2.dnn.readNet(ageModel,ageProto)
#Load gender detection model
genderNet=cv2.dnn.readNet(genderModel,genderProto)
#Load emotion detection model
emotionNet=cv2.dnn.readNetFromONNX(emotionModel)

#Load attr detection model
attrNet=cv2.dnn.readNet(attrModel,attrProto)


""" Detects age and gender """
def genderAge(image):

    MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
    ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList=['Male','Female']
    emotionList = ['neutral', 'happiness', 'surprise', 'sadness','anger', 'disgust', 'fear']
    attrList = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive',
        'Bags_Under_Eyes', 'Bald', 'Bangs', 'Big_Lips', 'Big_Nose',
        'Black_Hair', 'Blond_Hair', 'Blurry', 'Brown_Hair', 'Bushy_Eyebrows',
        'Chubby', 'Double_Chin', 'Eyeglasses', 'Goatee', 'Gray_Hair',
        'Heavy_Makeup', 'High_Cheekbones', 'Male', 'Mouth_Slightly_Open',
        'Mustache', 'Narrow_Eyes', 'No_Beard', 'Oval_Face', 'Pale_Skin',
        'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks', 'Sideburns',
        'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
        'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace',
        'Wearing_Necktie', 'Young']
    
    blob = cv2.dnn.blobFromImage(image, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
    # Print the shape and type of the image


    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Resize the image to 64x64 pixels
    blob2 = cv2.dnn.blobFromImage(gray_image, 1.0, (64, 64), MODEL_MEAN_VALUES, swapRB=False)
    # Since the model expects 1 channel, ensure blob shape is correct
    blob2 = blob2.reshape(1, 1, 64, 64)


    # Predict the gender
    genderNet.setInput(blob)
    genderPreds=genderNet.forward()
    gender=genderList[genderPreds[0].argmax()]

    # Predict the age
    ageNet.setInput(blob)
    agePreds=ageNet.forward()
    age=ageList[agePreds[0].argmax()]

    # Predict the emotion
    emotionNet.setInput(blob2)
    emotionPreds=emotionNet.forward()
    emotion=emotionList[emotionPreds[0].argmax()]
    
    # Predict the attribute
    attrNet.setInput(blob)
    attrPreds=attrNet.forward()
    topAttr  = np.where(attrPreds[0] > 0.5)[0]
    topAttrLabel = [attrList[i] for i in topAttr]
    

    # Return
    return gender,age,emotion, topAttrLabel

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

def draw_dashed_line(image, pt1, pt2, color, thickness, dash_length=5):
    dist = np.linalg.norm(np.array(pt2) - np.array(pt1))
    num_dashes = int(dist // (2 * dash_length))
    if num_dashes == 0:
        cv2.line(image, pt1, pt2, color, thickness)
        return
    
    dash_x = (pt2[0] - pt1[0]) / (2 * num_dashes)
    dash_y = (pt2[1] - pt1[1]) / (2 * num_dashes)
    
    for i in range(num_dashes):
        start_pt = (int(pt1[0] + (2 * i * dash_x)), int(pt1[1] + (2 * i * dash_y)))
        end_pt = (int(pt1[0] + ((2 * i + 1) * dash_x)), int(pt1[1] + ((2 * i + 1) * dash_y)))
        cv2.line(image, start_pt, end_pt, color, thickness)

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
        
        if dash:
            draw_dashed_line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness, 15)
            draw_dashed_line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness, 15)
            draw_dashed_line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness, 15)
            draw_dashed_line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness, 15)
        else:
            # Draw the four connecting lines
            cv2.line(image, (x1 + radius, y1), (x2 - radius, y1), color, thickness)
            cv2.line(image, (x1 + radius, y2), (x2 - radius, y2), color, thickness)
            cv2.line(image, (x1, y1 + radius), (x1, y2 - radius), color, thickness)
            cv2.line(image, (x2, y1 + radius), (x2, y2 - radius), color, thickness)


        

def draw_label(image, bbox, label, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.3, font_thickness=1):
    startX,startY,endX,endY = bbox
    # Determine the size of the text
    lines = label.split('\n')
    
    text_height = cv2.getTextSize('A', font, font_scale, font_thickness)[0][1]
    text_widths = [cv2.getTextSize(line, font, font_scale, font_thickness)[0][0] for line in lines]
    max_text_width = max(text_widths)
    
    # Calculate the position for the label background
    x = startX + (endX - startX) + 20
    y = startY
    w = endX + max_text_width + 35
    h = startY

    
    for i, line in enumerate(lines):
        h += text_height + 7

    # Draw the rounded rectangle background
    draw_rounded_rectangle(image, (x, y), (w, h), (7, 186, 255), cv2.FILLED, 10)
    
    
    # Draw each line of the label text
    for i, line in enumerate(lines):
        cv2.putText(image, line, 
                    (x + 10, y + 20 + (i * (text_height + 7))), 
                    font, font_scale, (30, 30, 30), font_thickness)

video_read = cv2.VideoCapture(0)

while True:
    ret_val, img = video_read.read()

    if not ret_val:
        break

    h, w = img.shape[:2]

    blob=cv2.dnn.blobFromImage(img,1.0,(300,300),(104.0,177.0,123.0),swapRB=False,crop=False)

    faceNet.setInput(blob)

    detections=faceNet.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence>0.7:
            box=detections[0,0,i,3:7]*np.array([w,h,w,h])
            (startX,startY,endX,endY)=box.astype("int")

            cropped_image = crop_with_padding(img, box.astype(int), 20)
            gender, age, emotion, attrs = genderAge(cropped_image)

            draw_rounded_rectangle(img, (startX, startY), (endX, endY), (255, 103, 7), 4, radius=10, dash=True)

            label = f"Gender: {gender}\nAge: {age}\nEmotion: {emotion}\nAttribute:\n"

            for i, attr in enumerate(attrs):
                label += f"- {attr}\n"

            draw_label(img, box.astype(int), label)

    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break  # q to quit   

cv2.destroyAllWindows()
