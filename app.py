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

def load_cloth_texture():
    # Load a red cloth texture image
    cloth_texture = cv2.imread('red_cloth_texture.jpg')  # You need to provide this texture image
    if cloth_texture is None:
        st.error("Error loading cloth texture.")
        return None
    return cloth_texture

def apply_cloth_texture(face_image, cloth_texture, region):
    (x, y, w, h) = region
    cloth_resized = cv2.resize(cloth_texture, (w, h))
    cloth_mask = cloth_resized[:, :, 2] > 0  # Assuming the cloth is red, use red channel for mask
    face_image[y:y+h, x:x+w][cloth_mask] = cloth_resized[cloth_mask]
    return face_image

def add_realistic_cloth(image):
    face_cascade = load_face_cascade()
    cloth_texture = load_cloth_texture()
    if face_cascade is None or cloth_texture is None:
        return image
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        eye_region = (x, y + int(h / 5), w, int(h / 4))
        mouth_region = (x + int(w / 4), y + int(2 * h / 3), int(w / 2), int(h / 6))

        image = apply_cloth_texture(image, cloth_texture, eye_region)
        image = apply_cloth_texture(image, cloth_texture, mouth_region)
    
    return image

def process_image(image):
    final_image = add_realistic_cloth(image)
    return final_image

def main():
    st.title("Image Processing App")
    st.write("Upload an image to add realistic red cloth over eyes and mouth.")

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
