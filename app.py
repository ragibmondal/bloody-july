import streamlit as st
import cv2
import numpy as np

# Load cascade files
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'
mouth_cascade_path = 'haarcascade_mcs_mouth.xml'

# Load cascade classifiers
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# Function to convert image to red cloth covered mouth and eye
def convert_image(image):
    # Load the image
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Detect eyes
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            # Draw rectangle around eyes
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)

        # Detect mouth
        mouth = mouth_cascade.detectMultiScale(roi_gray)
        for (mx, my, mw, mh) in mouth:
            # Draw rectangle around mouth
            cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), 1)

        # Draw rectangle around face
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        # Cover eyes and mouth with red rectangle
        cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), -1)
        cv2.rectangle(roi_color, (mx, my), (mx + mw, my + mh), (0, 0, 255), -1)

    # Return the converted image
    return cv2.imencode('.jpg', img)[1].tobytes()

# Create Streamlit app
st.title("Red Cloth Covered Mouth and Eye Converter")

# Upload image file
image_file = st.file_uploader("Upload your profile picture", type=['jpg', 'png', 'jpeg'])

# Convert image if file is uploaded
if image_file is not None:
    converted_image = convert_image(image_file.getvalue())
    st.image(converted_image, caption="Converted Image")
