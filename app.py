import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Load the Haar cascades for face, eyes, and mouth detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_mcs_mouth.xml')

# Function to detect and cover eyes and mouth
def cover_eyes_and_mouth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3, 11)
        
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), -1)

        for (mx, my, mw, mh) in mouth:
            my = int(my - 0.15*mh)
            cv2.rectangle(roi_color, (mx, my), (mx+mw, my+mh), (0, 0, 255), -1)

    return image

st.title("Eye and Mouth Covering App")
st.write("Upload a photo, and the app will cover the eyes and mouth with red rectangles.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    image_np = np.array(image)

    st.image(image, caption='Uploaded Image', use_column_width=True)

    processed_image = cover_eyes_and_mouth(image_np)

    st.image(processed_image, caption='Processed Image', use_column_width=True)
