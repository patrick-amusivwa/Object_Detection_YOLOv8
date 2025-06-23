import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import os
import uuid
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---- Page Config ---- #
st.set_page_config(page_title="🛣️ Lane Detection", layout="centered")

# ---- Constants ---- #
IMG_HEIGHT, IMG_WIDTH = 256, 256
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ---- Optional: Disable GPU ---- #
try:
    tf.config.set_visible_devices([], 'GPU')
except:
    pass

# ---- Load Model ---- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_for_lane.keras", compile=False)

model = load_model()

# ---- Preprocessing & Prediction ---- #
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

def predict_mask(image_path):
    image = preprocess_image(image_path)
    pred_mask = model.predict(image)[0]
    pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255
    pred_mask = np.squeeze(pred_mask)

    # Save predicted mask
    mask_img = Image.fromarray(pred_mask).convert("L").resize((IMG_WIDTH, IMG_HEIGHT))
    output_path = os.path.join(UPLOAD_FOLDER, f"mask_{uuid.uuid4().hex}.png")
    mask_img.save(output_path)
    return output_path

# ---- Sidebar ---- #
with st.sidebar:
    st.title("🚗 About the App")
    st.markdown("""
    This application performs **lane detection** on road images using a trained **Cascaded Unet model**.

    ---
    ### 📋 How to Use:
    1. Upload a road image (JPG or PNG).
    2. The model will predict lane markings.
    3. View both the **original image** and the **predicted lane mask** side-by-side.

    ---
    ### 🧠 Behind the Scenes:
    - Input size: 256×256
    - Model: Custom-trained Cascaded U-Net Architecture for Lane Detection
    - Output: Binary mask indicating detected lane pixels
    """)

# ---- Main Title ---- #
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🛣️ Lane Detection Using Cascaded UNET Architecture</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a road image to visualize the lane mask using deep learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- File Upload ---- #
uploaded_file = st.file_uploader("📤 Upload a Road Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Save original uploaded file
        temp_filename = f"{uuid.uuid4().hex}.png"
        img_path = os.path.join(UPLOAD_FOLDER, temp_filename)
        with open(img_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("🔍 Detecting lanes..."):
            mask_path = predict_mask(img_path)

        # Resize both images to the same size for display
        original_img = Image.open(img_path).resize((IMG_WIDTH, IMG_HEIGHT))
        mask_img = Image.open(mask_path).resize((IMG_WIDTH, IMG_HEIGHT))

        # ---- Display Images Side by Side ---- #
        col1, col2 = st.columns(2)
        with col1:
            st.image(original_img, caption="📷 Original Image", use_column_width=True)
        with col2:
            st.image(mask_img, caption="🛣️ Predicted Lane ", use_column_width=True)

        st.markdown("---")
        st.success("✅ Lane detection completed successfully!")

    except Exception as e:
        st.error(f"⚠️ An error occurred: {str(e)}")
