import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def add_red_cloth(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Define smaller regions for eyes and mouth
        eye_region_y = y + int(h * 0.2)  # Start eyes lower
        eye_region_h = int(h * 0.2)  # Smaller eye region
        mouth_region_y = y + int(h * 0.65)  # Start mouth higher
        mouth_region_h = int(h * 0.2)  # Smaller mouth region
        
        # Create and apply red masks
        eye_mask = np.zeros((eye_region_h, w, 3), dtype=np.uint8)
        cv2.rectangle(eye_mask, (0, 0), (w, eye_region_h), (0, 0, 255), -1)
        image[eye_region_y:eye_region_y+eye_region_h, x:x+w] = cv2.addWeighted(
            image[eye_region_y:eye_region_y+eye_region_h, x:x+w], 0.7, eye_mask, 0.3, 0)
        
        mouth_mask = np.zeros((mouth_region_h, w, 3), dtype=np.uint8)
        cv2.rectangle(mouth_mask, (0, 0), (w, mouth_region_h), (0, 0, 255), -1)
        image[mouth_region_y:mouth_region_y+mouth_region_h, x:x+w] = cv2.addWeighted(
            image[mouth_region_y:mouth_region_y+mouth_region_h, x:x+w], 0.7, mouth_mask, 0.3, 0)
    
    return image

def process_image(image):
    # Add red cloth effect
    final_image = add_red_cloth(image)
    return final_image

def main():
    st.title("Refined Image Processing App")
    st.write("Upload an image to add smaller red cloth over eyes and mouth.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)

        # Process the image
        processed_image = process_image(image.copy())

        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with col2:
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

        # Provide download option for the processed image
        is_success, buffer = cv2.imencode(".png", processed_image)
        if is_success:
            btn = st.download_button(
                label="Download processed image",
                data=buffer.tobytes(),
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
