import streamlit as st
import cv2
import numpy as np
from PIL import Image
import mediapipe as mp

def apply_red_cloth_effect(image):
    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Initialize MediaPipe Face Detection
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)
    
    # Detect faces
    results = face_detection.process(image_rgb)
    
    if results.detections:
        for detection in results.detections:
            bboxC = detection.location_data.relative_bounding_box
            ih, iw, _ = image.shape
            x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)
            
            # Define regions for eyes and mouth
            eye_y = y + int(h * 0.3)
            eye_h = int(h * 0.2)
            mouth_y = y + int(h * 0.7)
            mouth_h = int(h * 0.2)
            
            # Create red cloth mask
            red_cloth = np.zeros(image.shape, dtype=np.uint8)
            red_cloth[:] = (0, 0, 255)  # Red color
            
            # Apply red cloth to eyes and mouth regions
            image[eye_y:eye_y+eye_h, x:x+w] = cv2.addWeighted(image[eye_y:eye_y+eye_h, x:x+w], 0.5, red_cloth[eye_y:eye_y+eye_h, x:x+w], 0.5, 0)
            image[mouth_y:mouth_y+mouth_h, x:x+w] = cv2.addWeighted(image[mouth_y:mouth_y+mouth_h, x:x+w], 0.5, red_cloth[mouth_y:mouth_y+mouth_h, x:x+w], 0.5, 0)
    
    return image

def main():
    st.title("Red Cloth Effect on Face")
    st.write("Upload an image to apply a red cloth effect on the eyes and mouth.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        processed_image = apply_red_cloth_effect(image_np)
        
        st.image(processed_image, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
