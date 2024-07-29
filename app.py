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
        # Define regions for eyes and mouth
        eye_region = image[y:y+int(h/2), x:x+w]
        mouth_region = image[y+int(2*h/3):y+h, x:x+w]
        
        # Create red masks
        eye_mask = np.zeros(eye_region.shape[:2], dtype=np.uint8)
        cv2.rectangle(eye_mask, (0, 0), (w, int(h/2)), (255, 255, 255), -1)
        mouth_mask = np.zeros(mouth_region.shape[:2], dtype=np.uint8)
        cv2.rectangle(mouth_mask, (0, 0), (w, int(h/3)), (255, 255, 255), -1)
        
        # Apply red color
        eye_region[eye_mask != 0] = [0, 0, 255]
        mouth_region[mouth_mask != 0] = [0, 0, 255]
    
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
