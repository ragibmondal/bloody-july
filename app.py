import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile

def load_face_cascade():
    face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(face_cascade_path)
    if face_cascade.empty():
        st.error("Error loading Haar cascade file.")
        return None
    return face_cascade

def add_red_band(image):
    face_cascade = load_face_cascade()
    if face_cascade is None:
        return image
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Convert grayscale to BGR for colored overlay
    image_gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        eye_y = y + int(h * 0.25)  # Adjust this value to position the band
        band_height = int(h * 0.1)  # Adjust for desired band thickness
        
        # Draw red band
        cv2.rectangle(image_gray, (x, eye_y), (x + w, eye_y + band_height), (0, 0, 255), -1)
    
    return image_gray

def process_image(image):
    final_image = add_red_band(image)
    return final_image

def main():
    st.title("Image Processing App")
    st.write("Upload an image to add a red band across the eyes.")
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        try:
            original_image = np.array(Image.open(uploaded_file).convert('RGB'))
            image_for_processing = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            
            processed_image = process_image(image_for_processing)
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, channels="RGB")
            with col2:
                st.subheader("Processed Image")
                st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            
            processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                processed_pil_image.save(tmpfile.name, format="PNG")
                st.download_button(
                    label="Download processed image",
                    data=open(tmpfile.name, 'rb').read(),
                    file_name="processed_image.png",
                    mime="image/png"
                )
        
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
