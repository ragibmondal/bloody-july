import streamlit as st
import cv2
import numpy as np
from PIL import Image
import dlib
import io

# Load the face detector and landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def create_eye_mask(image, landmarks, color=(0, 0, 255), thickness=30):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    
    # Get eye landmarks
    left_eye = landmarks[36:42]
    right_eye = landmarks[42:48]
    
    # Calculate eye centers
    left_center = np.mean(left_eye, axis=0).astype(int)
    right_center = np.mean(right_eye, axis=0).astype(int)
    
    # Draw line connecting eyes
    cv2.line(mask, tuple(left_center), tuple(right_center), 255, thickness)
    
    # Expand mask
    kernel = np.ones((thickness, thickness), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=1)
    
    # Create colored mask
    colored_mask = np.zeros(image.shape, dtype=np.uint8)
    colored_mask[mask > 0] = color
    
    return colored_mask

def process_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    
    for face in faces:
        landmarks = predictor(gray, face)
        landmarks = np.array([(p.x, p.y) for p in landmarks.parts()])
        
        eye_mask = create_eye_mask(image, landmarks)
        
        # Blend the mask with the original image
        alpha = 0.7
        image = cv2.addWeighted(image, 1, eye_mask, alpha, 0)
    
    return image

def main():
    st.title("Advanced Face Feature Cover App")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = load_image(uploaded_file)
        
        st.write("Original Image")
        st.image(image, channels="BGR")
        
        if st.button('Process Image'):
            result = process_image(image)
            
            st.write("Processed Image")
            st.image(result, channels="BGR")
            
            # Option to download the processed image
            result_img = Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
            buf = io.BytesIO()
            result_img.save(buf, format="PNG")
            byte_im = buf.getvalue()
            
            st.download_button(
                label="Download processed image",
                data=byte_im,
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == '__main__':
    main()
