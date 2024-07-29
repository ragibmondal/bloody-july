import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def apply_red_strips_effect(image):
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert back to BGR for consistent color space
    image = cv2.cvtColor(gray_image, cv2.COLOR_GRAY2BGR)
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Detect faces
    results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Define regions for eyes and mouth
            eye_y = y + int(h * 0.2)
            eye_h = int(h * 0.15)
            mouth_y = y + int(h * 0.7)
            mouth_h = int(h * 0.15)
            
            # Create red strips
            image[eye_y:eye_y+eye_h, x:x+w] = [0, 0, 255]  # Red color
            image[mouth_y:mouth_y+mouth_h, x:x+w] = [0, 0, 255]  # Red color
    
    return image

def main():
    st.title("Red Strips Effect on Face")
    st.write("Upload an image to apply red strips over the eyes and mouth.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        processed_image = apply_red_strips_effect(image_np)
        
        st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
