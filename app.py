import streamlit as st
import cv2
import numpy as np

# Function to convert image to red cloth covered mouth and eye
def convert_image(image):
    # Load the image
    img = cv2.imdecode(np.frombuffer(image, np.uint8), cv2.IMREAD_COLOR)

    # Convert image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    # Loop through detected faces
    for (x, y, w, h) in faces:
        # Draw red rectangle around mouth
        mouth_x = x + w // 4
        mouth_y = y + h // 2
        mouth_w = w - w // 2
        mouth_h = h // 4
        cv2.rectangle(img, (mouth_x, mouth_y), (mouth_x + mouth_w, mouth_y + mouth_h), (0, 0, 255), -1)

        # Draw red rectangle around eyes
        eye_x = x + w // 4
        eye_y = y + h // 4
        eye_w = w - w // 2
        eye_h = h // 4
        cv2.rectangle(img, (eye_x, eye_y), (eye_x + eye_w, eye_y + eye_h), (0, 0, 255), -1)

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
