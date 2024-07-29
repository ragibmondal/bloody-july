import streamlit as st
import cv2
import numpy as np
from PIL import Image

def apply_cloth_effect(image, face_features):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in face_features:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    cloth = np.full(image.shape, (0, 0, 255), dtype=np.uint8)  # Red cloth
    cloth_area = cv2.bitwise_and(cloth, cloth, mask=mask)
    
    # Add some texture to the cloth
    noise = np.random.randint(0, 50, (image.shape[0], image.shape[1], 3)).astype(np.uint8)
    cloth_area = cv2.add(cloth_area, noise)
    
    # Apply Gaussian blur to simulate cloth texture
    cloth_area = cv2.GaussianBlur(cloth_area, (5, 5), 0)
    
    # Create a seamless blend
    result = cv2.seamlessClone(cloth_area, image, mask, (image.shape[1]//2, image.shape[0]//2), cv2.MIXED_CLONE)
    return result

def process_image(image):
    try:
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face, eyes, and mouth
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            raise ValueError("No faces detected in the image.")
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            
            eyes = eye_cascade.detectMultiScale(roi_gray)
            mouth = mouth_cascade.detectMultiScale(roi_gray, 1.7, 11)
            
            if len(eyes) == 0 and len(mouth) == 0:
                raise ValueError("No eyes or mouth detected in the face.")
            
            face_features = []
            for (ex, ey, ew, eh) in eyes:
                face_features.append((x+ex, y+ey, ew, eh))
            
            for (mx, my, mw, mh) in mouth:
                face_features.append((x+mx, y+my, mw, mh))
            
            image = apply_cloth_effect(image, face_features)
        
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        return None

def main():
    st.title("Eye and Mouth Cover App")
    st.write("Upload a portrait photo to cover eyes and mouth with a red cloth.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Apply Red Cloth Effect"):
            result = process_image(image)
            if result is not None:
                st.image(result, caption="Processed Image", use_column_width=True)

if __name__ == "__main__":
    main()
