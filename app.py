import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def apply_cloth_effect(image, mask, color=(0, 0, 255), alpha=0.7):
    cloth = np.zeros(image.shape, dtype=np.uint8)
    cloth[:] = color
    
    # Add texture to the cloth
    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    cloth = cv2.add(cloth, noise)
    
    # Apply the mask
    cloth_masked = cv2.bitwise_and(cloth, cloth, mask=mask)
    
    # Blend the cloth with the original image
    result = cv2.addWeighted(image, 1 - alpha, cloth_masked, alpha, 0)
    
    # Add shadow effect
    shadow = cv2.GaussianBlur(mask, (5, 5), 0)
    shadow = cv2.merge([shadow, shadow, shadow])
    result = cv2.subtract(result, shadow * 0.5)
    
    return result

def add_red_cloth(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Create masks for eyes and mouth
        eye_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        mouth_mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Eye region
        eye_y = y + int(h * 0.2)
        eye_h = int(h * 0.15)
        cv2.rectangle(eye_mask, (x, eye_y), (x + w, eye_y + eye_h), 255, -1)
        
        # Mouth region
        mouth_y = y + int(h * 0.65)
        mouth_h = int(h * 0.2)
        cv2.rectangle(mouth_mask, (x, mouth_y), (x + w, mouth_y + mouth_h), 255, -1)
        
        # Apply cloth effect
        image = apply_cloth_effect(image, eye_mask)
        image = apply_cloth_effect(image, mouth_mask)
    
    return image

def process_image(image):
    return add_red_cloth(image)

def main():
    st.title("Advanced Image Processing App")
    st.write("Upload an image to add realistic red cloth over eyes and mouth.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
        processed_image = process_image(image.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        with col2:
            st.subheader("Processed Image")
            st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))

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
