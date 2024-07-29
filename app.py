import streamlit as st
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import tempfile

def enhance_image(image):
    # Convert to PIL Image
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Enhance brightness and contrast
    enhancer = ImageEnhance.Brightness(pil_image)
    pil_image = enhancer.enhance(1.2)
    enhancer = ImageEnhance.Contrast(pil_image)
    pil_image = enhancer.enhance(1.2)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

def add_red_cloth(image):
    # Load the pre-trained face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        # Define regions for eyes and mouth with reduced size
        eye_region_height = int(h / 4)
        mouth_region_height = int(h / 6)
        mouth_region_width = int(w * 0.5)  # Reduced width of the mouth region
        
        # Cover eyes
        eye_y_start = y + int(h / 5)
        eye_y_end = eye_y_start + eye_region_height
        image[eye_y_start:eye_y_end, x:x+w] = [0, 0, 255]
        
        # Cover mouth
        mouth_y_start = y + int(2 * h / 3)
        mouth_y_end = mouth_y_start + mouth_region_height
        mouth_x_start = x + int((w - mouth_region_width) / 2)
        mouth_x_end = mouth_x_start + mouth_region_width
        image[mouth_y_start:mouth_y_end, mouth_x_start:mouth_x_end] = [0, 0, 255]
    
    return image

def process_image(image):
    # Enhance image quality
    enhanced_image = enhance_image(image)
    
    # Add red cloth effect
    final_image = add_red_cloth(enhanced_image)
    
    return final_image

def main():
    st.title("Image Processing App")
    st.write("Upload an image to add red cloth over eyes and mouth, and enhance quality.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = np.array(Image.open(uploaded_file))

        # Process the image
        processed_image = process_image(image)

        # Display original and processed images side by side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image, channels="BGR")
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image, channels="BGR")

        # Provide download option for the processed image
        processed_pil_image = Image.fromarray(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB))
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as tmpfile:
            processed_pil_image.save(tmpfile.name)
            st.download_button(
                label="Download processed image",
                data=open(tmpfile.name, 'rb').read(),
                file_name="processed_image.png",
                mime="image/png"
            )

if __name__ == "__main__":
    main()
