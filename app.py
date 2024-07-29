import streamlit as st
from PIL import Image, ImageDraw, ImageFilter
import numpy as np
import io

def apply_cloth_effect(img, mask, color=(255, 0, 0), alpha=0.7):
    # Create a new image for the cloth
    cloth = Image.new('RGB', img.size, color)
    
    # Add some texture to the cloth
    cloth_array = np.array(cloth)
    noise = np.random.randint(0, 50, cloth_array.shape, dtype=np.uint8)
    cloth = Image.fromarray(cv2.add(cloth_array, noise))
    
    # Apply the mask
    cloth.putalpha(Image.fromarray(mask).convert('L'))
    
    # Blend the cloth with the original image
    result = Image.blend(img, cloth, alpha)
    
    # Add shadow effect
    shadow = Image.fromarray(mask).convert('L').filter(ImageFilter.GaussianBlur(5))
    shadow = Image.merge('RGB', [shadow, shadow, shadow])
    result = Image.blend(result, shadow.convert('RGB'), 0.3)
    
    return result

def add_red_cloth(img):
    draw = ImageDraw.Draw(img)
    width, height = img.size
    
    # Eye region
    eye_y = int(height * 0.2)
    eye_h = int(height * 0.15)
    eye_mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(eye_mask).rectangle([0, eye_y, width, eye_y + eye_h], fill=255)
    
    # Mouth region
    mouth_y = int(height * 0.65)
    mouth_h = int(height * 0.2)
    mouth_mask = Image.new('L', img.size, 0)
    ImageDraw.Draw(mouth_mask).rectangle([0, mouth_y, width, mouth_y + mouth_h], fill=255)
    
    # Apply cloth effect
    img = apply_cloth_effect(img, np.array(eye_mask))
    img = apply_cloth_effect(img, np.array(mouth_mask))
    
    return img

def process_image(image):
    return add_red_cloth(image)

def main():
    st.title("Deployment-Friendly Image Processing App")
    st.write("Upload an image to add realistic red cloth over eyes and mouth.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        processed_image = process_image(image.copy())

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(image)
        with col2:
            st.subheader("Processed Image")
            st.image(processed_image)

        # Provide download option for the processed image
        buf = io.BytesIO()
        processed_image.save(buf, format="PNG")
        btn = st.download_button(
            label="Download processed image",
            data=buf.getvalue(),
            file_name="processed_image.png",
            mime="image/png"
        )

if __name__ == "__main__":
    main()
