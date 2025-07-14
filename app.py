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
st.set_page_config(page_title="Lane Detection UNET ", layout="centered",page_icon="🛣️")

# ---- Constants ---- #
IMG_HEIGHT, IMG_WIDTH = 256, 256
UPLOADED_FOLDER = 'static/uploaded'
PREDICTED_FOLDER = 'static/predicted'
SAMPLE_IMAGE_PATH = 'upload_image/road.webp'

os.makedirs(UPLOADED_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

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
    output_path = os.path.join(PREDICTED_FOLDER, f"mask_{uuid.uuid4().hex}.png")
    mask_img.save(output_path)
    return output_path

# ---- Sidebar ---- #
with st.sidebar:
    st.title("🚗 Lane Detection App")

    st.markdown("""
    This is a **deep learning-based lane detection application** built using a custom-trained **Cascaded U-Net model**.
    Upload a road image to see the predicted lane markings.
    """)

    with st.expander("📌 How It Works"):
        st.markdown("""
        1. Upload a road scene image (JPG or PNG), or use the sample image.
        2. The model processes the image to detect lane boundaries.
        3. You'll see the **original** and **predicted lane mask** side by side.
        """)

    with st.expander("🧠 Technical Overview"):
        st.markdown("""
        - **Input Size**: 256 × 256 pixels  
        - **Model**: Cascaded U-Net (deep learning-based)  
        - **Output**: Binary segmentation mask (lane pixel regions)  
        """)

    with st.expander("✅ Features"):
        st.markdown("""
        - Fast and accurate lane detection  
        - Clean and interactive interface  
        - Visual result comparison  
        - Ideal for self-driving & road safety research  
        """)

    st.markdown("---")
    st.markdown("👨‍💻 **Created by Akshwin T**")
    st.markdown("📧 akshwint.2003@gmail.com")

# ---- Main Title ---- #
st.markdown("<h1 style='text-align: center; color: #2C3E50;'>🛣️ Lane Detection Using Cascaded UNET Architecture</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Upload a road image or use a sample to visualize the lane mask using deep learning.</p>", unsafe_allow_html=True)
st.markdown("---")

# ---- Upload Section ---- #
uploaded_file = st.file_uploader("📤 Upload a Road Image", type=["jpg", "jpeg", "png"])
use_sample_image = st.button("🖼️ Use Sample Road Image")

# ---- Determine Source ---- #
image_source_path = None
source_label = ""

if uploaded_file and not use_sample_image:
    # Save uploaded image
    temp_filename = f"{uuid.uuid4().hex}.png"
    image_source_path = os.path.join(UPLOADED_FOLDER, temp_filename)
    with open(image_source_path, "wb") as f:
        f.write(uploaded_file.read())
    source_label = "📷 Uploaded Image"

elif use_sample_image and not uploaded_file:
    image_source_path = SAMPLE_IMAGE_PATH
    source_label = "🖼️ Sample Road Image"

elif uploaded_file and use_sample_image:
    st.warning("⚠️ Please either upload an image or click the sample button — not both.")
elif not uploaded_file and not use_sample_image:
    st.info("📥 Upload an image or click the 'Use Sample Road Image' button to begin.")

# ---- Run Prediction if an image is available ---- #
if image_source_path and os.path.exists(image_source_path):
    try:
        with st.spinner("🔍 Detecting lanes..."):
            predicted_mask_path = predict_mask(image_source_path)

        # Resize for display
        original_img = Image.open(image_source_path).resize((IMG_WIDTH, IMG_HEIGHT))
        mask_img = Image.open(predicted_mask_path).resize((IMG_WIDTH, IMG_HEIGHT))

        # Centered side-by-side columns (left and right with spacing)
        col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
        with col2:
            st.image(original_img, caption=source_label)
        with col3:
            st.image(mask_img, caption="🛣️ Predicted Lane")

        st.markdown("---")
        st.success("✅ Lane detection completed successfully!")

    except Exception as e:
        st.error(f"⚠️ An error occurred during prediction:\n\n{str(e)}")