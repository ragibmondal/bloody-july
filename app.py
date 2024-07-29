import cv2
import streamlit as st
from PIL import Image
import numpy as np
import os

# Define the paths to the Haar cascade XML files
face_cascade_path = 'haarcascade_frontalface_default.xml'
eye_cascade_path = 'haarcascade_eye.xml'
mouth_cascade_path = 'haarcascade_mcs_mouth.xml'

# Verify if the Haar cascade files exist
if not (os.path.exists(face_cascade_path) and os.path.exists(eye_cascade_path) and os.path.exists(mouth_cascade_path)):
    st.error("Haar cascade files are missing. Ensure they are in the project directory.")
    st.stop()

# Load the Haar cascades
face_cascade = cv2.CascadeClassifier(face_cascade_path)
eye_cascade = cv2.CascadeClassifier(eye_cascade_path)
mouth_cascade = cv2.CascadeClassifier(mouth_cascade_path)

# Function to detect and cover eyes and mouth
def cover_eyes_and_mouth(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)
        mouth = mouth_cascade.detectMultiScale(roi_gray, 1.3, 11)
        
        # Cover eyes with a red rectangle
        for (ex, ey, ew, eh) in eyes:
            top_left = (x + ex, y + ey)
            bottom_right = (x + ex + ew, y + ey + eh)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)
        
        # Cover mouth with a red rectangle
        for (mx, my, mw, mh) in mouth:
            my = int(my - 0.15*mh)  # Adjust position to better cover the mouth
            top_left = (x + mx, y + my)
            bottom_right = (x + mx + mw, y + my + mh)
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)

    return image

st.title("Eye and Mouth Covering App")
st.write("Upload a photo, and the app will cover the eyes and mouth with red rectangles.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        processed_image = cover_eyes_and_mouth(image_np)

        st.image(processed_image, caption='Processed Image', use_column_width=True)
    except cv2.error as e:
        st.error("An error occurred while processing the image.")
        st.error(f"Error details: {str(e)}")
    except Exception as e:
        st.error("An unexpected error occurred.")
        st.error(f"Error details: {str(e)}")
