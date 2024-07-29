import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def apply_cloth_effect(image, eyes):
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    for (x, y, w, h) in eyes:
        cv2.rectangle(mask, (x, y), (x+w, y+h), 255, -1)
    
    cloth = np.full(image.shape, (0, 0, 255), dtype=np.uint8)  # Red cloth
    cloth_area = cv2.bitwise_and(cloth, cloth, mask=mask)
    
    # Add some texture to the cloth (fixed)
    noise = np.random.randint(0, 50, (image.shape[0], image.shape[1], 3)).astype(np.uint8)
    cloth_area = cv2.add(cloth_area, noise)
    
    result = cv2.seamlessClone(cloth_area, image, mask, (image.shape[1]//2, image.shape[0]//2), cv2.NORMAL_CLONE)
    return result

def create_flag_background():
    flag = np.zeros((300, 500, 3), dtype=np.uint8)
    flag[:100] = [255, 0, 0]  # Red
    flag[100:200] = [0, 255, 0]  # Green
    cv2.circle(flag, (250, 150), 50, (255, 255, 255), -1)  # White circle
    return flag

def process_image(image):
    try:
        # Convert PIL Image to OpenCV format
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect face and eyes
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        
        if len(faces) == 0:
            raise ValueError("No faces detected in the image.")
        
        for (x, y, w, h) in faces:
            roi_gray = gray[y:y+h, x:x+w]
            roi_color = image[y:y+h, x:x+w]
            eyes = eye_cascade.detectMultiScale(roi_gray)
            
            if len(eyes) > 0:
                image = apply_cloth_effect(image, eyes)
            else:
                raise ValueError("No eyes detected in the face.")
        
        # Create flag background
        flag = create_flag_background()
        
        # Resize image to fit on flag
        height, width = flag.shape[:2]
        image = cv2.resize(image, (width // 2, height), interpolation=cv2.INTER_AREA)
        
        # Overlay image on flag
        x_offset = width // 4
        y_offset = 0
        flag[y_offset:y_offset+image.shape[0], x_offset:x_offset+image.shape[1]] = image
        
        return cv2.cvtColor(flag, cv2.COLOR_BGR2RGB)
    except Exception as e:
        st.error(f"An error occurred while processing the image: {str(e)}")
        return None

def main():
    st.title("Flag Portrait Creator")
    st.write("Upload a portrait photo to create an artistic flag portrait.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        if st.button("Create Flag Portrait"):
            result = process_image(image)
            if result is not None:
                st.image(result, caption="Flag Portrait", use_column_width=True)

if __name__ == "__main__":
    main()
