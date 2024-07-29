import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io

def load_image(image_file):
    img = Image.open(image_file)
    return np.array(img)

def cover_features(image, features, color=(0, 0, 255), alpha=0.5):
    overlay = image.copy()
    for (x, y, w, h) in features:
        cv2.rectangle(overlay, (x, y), (x+w, y+h), color, -1)
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def process_image(image):
    # Convert to grayscale for detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Load pre-trained classifiers
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    mouth_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
    
    # Detect eyes and mouth
    eyes = eye_cascade.detectMultiScale(gray, 1.3, 5)
    mouth = mouth_cascade.detectMultiScale(gray, 1.8, 20)
    
    # Cover detected features
    image = cover_features(image, eyes)
    image = cover_features(image, mouth)
    
    return image

def main():
    st.title("Face Feature Cover App")
    
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
