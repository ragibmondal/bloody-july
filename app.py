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

def add_red_cloth(image):
    face_cascade = load_face_cascade()
    if face_cascade is None:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        eye_region_height = int(h / 4)
        mouth_region_height = int(h / 6)
        mouth_region_width = int(w * 0.5)
        
        eye_y_start = y + int(h / 5)
        eye_y_end = eye_y_start + eye_region_height
        image[eye_y_start:eye_y_end, x:x+w] = [0, 0, 255]
        
        mouth_y_start = y + int(2 * h / 3)
        mouth_y_end = mouth_y_start + mouth_region_height
        mouth_x_start = x + int(w / 4)
        mouth_x_end = mouth_x_start + mouth_region_width
        image[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = [0, 0, 255]
    
    return image

def process_image(image):
    final_image = add_red_cloth(image)
    return final_image

def main():
    st.title("Image Processing App")
    st.write("Upload an image to add red cloth over eyes and mouth.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        try:
            original_image = np.array(Image.open(uploaded_file).convert('RGB'))
            image_for_processing = original_image.copy()
            image_for_processing = cv2.cvtColor(image_for_processing, cv2.COLOR_RGB2BGR)

            processed_image = process_image(image_for_processing)

            original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
            processed_image = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Image")
                st.image(original_image, channels="BGR")
            with col2:
                st.subheader("Processed Image")
                st.image(processed_image, channels="RGB")

            processed_pil_image = Image.fromarray(processed_image)
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
                processed_pil_image.save(tmpfile.name)
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
