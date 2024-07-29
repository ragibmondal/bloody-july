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

def add_cloth(image, color, intensity, eye_height, mouth_height, mouth_width):
    face_cascade = load_face_cascade()
    if face_cascade is None:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, intensity, 5)
    
    for (x, y, w, h) in faces:
        eye_region_height = int(h * eye_height)
        mouth_region_height = int(h * mouth_height)
        mouth_region_width = int(w * mouth_width)
        
        eye_y_start = y + int(h / 5)
        eye_y_end = eye_y_start + eye_region_height
        image[eye_y_start:eye_y_end, x:x+w] = color
        
        mouth_y_start = y + int(2 * h / 3)
        mouth_y_end = mouth_y_start + mouth_region_height
        mouth_x_start = x + int(w / 4)
        mouth_x_end = mouth_x_start + mouth_region_width
        image[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = color
    
    return image

def process_image(image, color, intensity, eye_height, mouth_height, mouth_width):
    final_image = add_cloth(image, color, intensity, eye_height, mouth_height, mouth_width)
    return final_image

def main():
    st.title("Enhanced Image Processing App")
    st.write("Upload an image and customize the processing.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    cloth_color = st.color_picker("Choose cloth color", "#FF0000")
    cloth_color_rgb = tuple(int(cloth_color[i:i+2], 16) for i in (1, 3, 5))
    intensity = st.slider("Face detection intensity", 1.1, 2.0, 1.3, 0.1)
    eye_height = st.slider("Eye region height as a fraction of face height", 0.1, 0.5, 0.25, 0.05)
    mouth_height = st.slider("Mouth region height as a fraction of face height", 0.1, 0.5, 0.16, 0.05)
    mouth_width = st.slider("Mouth region width as a fraction of face width", 0.2, 1.0, 0.5, 0.1)
    save_format = st.selectbox("Save image as", ["JPEG", "PNG"])

    if uploaded_file is not None:
        try:
            original_image = np.array(Image.open(uploaded_file).convert('RGB'))
            image_for_processing = original_image.copy()
            image_for_processing = cv2.cvtColor(image_for_processing, cv2.COLOR_RGB2BGR)

            processed_image = process_image(image_for_processing, cloth_color_rgb, intensity, eye_height, mouth_height, mouth_width)
            
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
            with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{save_format.lower()}') as tmpfile:
                processed_pil_image.save(tmpfile.name, format=save_format)
                st.download_button(
                    label=f"Download processed image as {save_format}",
                    data=open(tmpfile.name, 'rb').read(),
                    file_name=f"processed_image.{save_format.lower()}",
                    mime=f"image/{save_format.lower()}"
                )
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
