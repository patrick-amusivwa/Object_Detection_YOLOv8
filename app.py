import os
import uuid

import cv2
import matplotlib
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

matplotlib.use("Agg")

# ---- Page Config ---- #
st.set_page_config(page_title="Lane Detection UNET", layout="centered", page_icon="🛣️")

# ---- Constants ---- #
MODEL_INPUT_H, MODEL_INPUT_W = 256, 256
UPLOADED_FOLDER = "static/uploaded"
PREDICTED_FOLDER = "static/predicted"
SAMPLE_IMAGE_PATH = "upload_image/road.webp"

os.makedirs(UPLOADED_FOLDER, exist_ok=True)
os.makedirs(PREDICTED_FOLDER, exist_ok=True)

# ---- Optional: Disable GPU ---- #
try:
    tf.config.set_visible_devices([], "GPU")
except Exception:
    pass


# ---- Load Model ---- #
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_for_lane.keras", compile=False)


model = load_model()


# ---- YOLO Model ---- #

DETECT_CLASSES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO

    return YOLO("yolov8n.pt")  # auto-downloads ~6 MB on first run


def draw_detections(bgr_frame, conf_threshold=0.40):
    """Draw YOLO bounding boxes for vehicles/persons on a BGR frame."""
    yolo = load_yolo()
    results = yolo(bgr_frame, verbose=False, conf=conf_threshold)[0]
    for box in results.boxes:
        cls_id = int(box.cls[0])
        if cls_id not in DETECT_CLASSES:
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        label = f"{DETECT_CLASSES[cls_id]} {float(box.conf[0]):.2f}"
        cv2.rectangle(bgr_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
        cv2.putText(
            bgr_frame,
            label,
            (x1, max(y1 - 8, 12)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 165, 255),
            2,
        )
    return bgr_frame


# ---- Core Prediction Helpers ---- #


def predict_mask_for_frame(rgb_array, road_start_ratio=0.0):
    """
    Given an RGB numpy array of any size, run the model and return a
    grayscale mask (0/255) at the original frame dimensions.

    road_start_ratio: fraction of frame height to skip from the top before
    running inference (e.g. 0.40 skips the top 40% sky/sign area).
    The mask for that skipped region is always zero.
    """
    orig_h, orig_w = rgb_array.shape[:2]

    # Crop to the road region of interest
    crop_top = int(orig_h * road_start_ratio)
    roi = rgb_array[crop_top:, :, :]  # bottom portion where road lives
    roi_h, roi_w = roi.shape[:2]

    # Resize ROI to model input — model only sees the road area
    img_resized = cv2.resize(
        roi, (MODEL_INPUT_W, MODEL_INPUT_H), interpolation=cv2.INTER_LINEAR
    )
    tensor = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

    pred = model.predict(tensor, verbose=0)[0]
    mask_256 = (pred > 0.5).astype(np.uint8) * 255
    mask_256 = np.squeeze(mask_256)  # (256, 256)

    # Resize mask back to the ROI dimensions
    mask_roi = cv2.resize(mask_256, (roi_w, roi_h), interpolation=cv2.INTER_NEAREST)

    # Embed ROI mask into a blank full-frame mask
    full_mask = np.zeros((orig_h, orig_w), dtype=np.uint8)
    full_mask[crop_top:, :] = mask_roi
    return full_mask  # shape (orig_h, orig_w)


def overlay_mask_on_frame(bgr_frame, mask, color=(0, 255, 80), alpha=0.45):
    """
    Blend a coloured lane mask onto a BGR frame.
    Returns a BGR frame at the same resolution.
    """
    colored = np.zeros_like(bgr_frame)
    colored[mask > 0] = color
    blended = cv2.addWeighted(bgr_frame, 1.0, colored, alpha, 0)
    return blended


# ---- Image Prediction ---- #


def predict_image(image_path, road_start_ratio=0.0):
    """
    Load an image, predict the lane mask, and return:
      - original PIL Image (original size)
      - mask PIL Image (original size, greyscale)
    """
    img_pil = Image.open(image_path).convert("RGB")
    rgb = np.array(img_pil)
    mask = predict_mask_for_frame(rgb, road_start_ratio=road_start_ratio)
    mask_pil = Image.fromarray(mask, mode="L")
    return img_pil, mask_pil


# ---- Video Processing ---- #


def process_video(video_path, road_start_ratio=0.0, test_seconds=0):
    """
    Process a video file frame by frame. Returns path to output MP4 with
    lane mask overlaid at the original video resolution (portrait-safe).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(fps * test_seconds) if test_seconds > 0 else 0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(PREDICTED_FOLDER, f"lane_{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    progress_bar = st.progress(0, text="Processing frames…")
    frame_idx = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mask = predict_mask_for_frame(rgb_frame, road_start_ratio=road_start_ratio)
        result = overlay_mask_on_frame(bgr_frame, mask)
        writer.write(result)

        frame_idx += 1
        pct = frame_idx / total_frames if total_frames > 0 else 0
        progress_bar.progress(
            min(pct, 1.0), text=f"Processing frame {frame_idx} / {total_frames}"
        )
        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()
    progress_bar.empty()
    return out_path


def process_video_advanced(video_path, road_start_ratio=0.0, test_seconds=0, conf=0.40):
    """
    Process video with lane mask (green) + YOLO object detection (orange boxes).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    max_frames = int(fps * test_seconds) if test_seconds > 0 else 0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = os.path.join(PREDICTED_FOLDER, f"advanced_{uuid.uuid4().hex}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (orig_w, orig_h))

    progress_bar = st.progress(0, text="Processing frames (lanes + objects)…")
    frame_idx = 0

    while True:
        ret, bgr_frame = cap.read()
        if not ret:
            break

        # 1 — Lane mask overlay (green)
        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        mask = predict_mask_for_frame(rgb_frame, road_start_ratio=road_start_ratio)
        result = overlay_mask_on_frame(bgr_frame, mask)

        # 2 — YOLO object detection boxes (orange) on top
        result = draw_detections(result, conf_threshold=conf)
        writer.write(result)

        frame_idx += 1
        pct = frame_idx / total_frames if total_frames > 0 else 0
        progress_bar.progress(
            min(pct, 1.0), text=f"Processing frame {frame_idx} / {total_frames}"
        )
        if max_frames > 0 and frame_idx >= max_frames:
            break

    cap.release()
    writer.release()
    progress_bar.empty()
    return out_path


# ---- Sidebar ---- #
with st.sidebar:
    st.title("🚗 Lane Detection App")
    st.markdown("""
    Deep learning-based lane detection using a custom **Cascaded U-Net** model.
    Upload a road image **or portrait/landscape video** to see predicted lane markings.
    """)

    with st.expander("📌 How It Works"):
        st.markdown("""
        1. Choose the **Image** or **Video** tab.
        2. Upload your file (or use the sample image).
        3. The model detects lane boundaries on each frame.
        4. Results are shown with the mask overlaid at the **original resolution** — portrait videos are fully supported.
        """)

    with st.expander("🧠 Technical Overview"):
        st.markdown("""
        - **Model input**: 256 × 256 px (internal resize only)
        - **Output**: mask scaled back to original frame size
        - **Model**: Cascaded U-Net (binary segmentation)
        - **Video**: frame-by-frame inference, original resolution preserved
        """)

    with st.expander("✅ Features"):
        st.markdown("""
        - Portrait & landscape video support
        - Aspect-ratio-correct mask overlay
        - Downloadable processed video
        - Fast and interactive interface
        """)

    st.markdown("---")
    st.markdown("👨‍💻 **Created by Akshwin T**")
    st.markdown("📧 akshwint.2003@gmail.com")

# ---- Main Title ---- #
st.markdown(
    "<h1 style='text-align:center;color:#2C3E50;'>🛣️ Lane Detection — Cascaded UNET</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='text-align:center;'>Upload a road image or video (portrait/landscape) to visualise lane masks.</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ---- Tabs ---- #
tab_image, tab_simple, tab_advanced = st.tabs(["🖼️ Image", "🛣️ Simple", "🚗 Advanced"])

# ======================================================
# IMAGE TAB
# ======================================================
with tab_image:
    uploaded_file = st.file_uploader(
        "📤 Upload a Road Image", type=["jpg", "jpeg", "png", "webp"], key="img_upload"
    )
    use_sample = st.button("🖼️ Use Sample Road Image")

    img_road_start = (
        st.slider(
            "🔽 Skip top of frame (road start %)",
            min_value=0,
            max_value=70,
            value=0,
            step=5,
            help="For portrait images with sky/signs at the top, slide this up so the model focuses only on the road area.",
            key="img_road_start",
        )
        / 100.0
    )

    image_source_path = None
    source_label = ""

    if uploaded_file and not use_sample:
        temp_name = f"{uuid.uuid4().hex}.png"
        image_source_path = os.path.join(UPLOADED_FOLDER, temp_name)
        with open(image_source_path, "wb") as f:
            f.write(uploaded_file.read())
        source_label = "📷 Uploaded Image"

    elif use_sample and not uploaded_file:
        image_source_path = SAMPLE_IMAGE_PATH
        source_label = "🖼️ Sample Road Image"

    elif uploaded_file and use_sample:
        st.warning(
            "⚠️ Please either upload an image or click the sample button — not both."
        )

    elif not uploaded_file and not use_sample:
        st.info("📥 Upload an image or click 'Use Sample Road Image' to begin.")

    if image_source_path and os.path.exists(image_source_path):
        try:
            with st.spinner("🔍 Detecting lanes…"):
                orig_pil, mask_pil = predict_image(
                    image_source_path, road_start_ratio=img_road_start
                )

            col1, col2, col3, col4 = st.columns([1, 2, 2, 1])
            with col2:
                st.image(orig_pil, caption=source_label, use_container_width=True)
            with col3:
                st.image(
                    mask_pil, caption="🛣️ Predicted Lane Mask", use_container_width=True
                )

            st.markdown("---")
            st.success("✅ Lane detection completed successfully!")

        except Exception as e:
            st.error(f"⚠️ Error during prediction:\n\n{str(e)}")

# ======================================================
# SIMPLE TAB — Lane detection only
# ======================================================
with tab_simple:
    st.markdown(
        "Upload a **portrait or landscape** road video. Each frame is processed individually and the lane mask is overlaid at the original resolution."
    )

    video_file = st.file_uploader(
        "📤 Upload a Road Video", type=["mp4", "mov", "avi", "mkv"], key="vid_upload"
    )

    test_seconds = st.slider(
        "⏱️ Seconds to process (0 = full video)",
        min_value=0,
        max_value=60,
        value=6,
        step=1,
        help="6 s is enough to verify detection is working. Set to 0 to process the whole video.",
        key="test_seconds",
    )

    vid_road_start = (
        st.slider(
            "🔽 Skip top of frame (road start %)",
            min_value=0,
            max_value=70,
            value=40,
            step=5,
            help="Portrait videos typically have sky/signs in the top ~40 %. Slide this to where the road surface begins so the model focuses there.",
            key="vid_road_start",
        )
        / 100.0
    )

    # Visual guide line
    if video_file is not None and vid_road_start > 0:
        st.caption(
            f"📌 Model will ignore the top {int(vid_road_start * 100)} % of each frame and run inference on the road area below."
        )

    if video_file is not None:
        # Save upload to disk so OpenCV can read it
        vid_in_path = os.path.join(
            UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{video_file.name}"
        )
        with open(vid_in_path, "wb") as f:
            f.write(video_file.read())

        st.video(vid_in_path)
        st.markdown(f"**Uploaded:** `{video_file.name}`")

        if st.button("🚀 Process Video"):
            try:
                with st.spinner("⚙️ Running lane detection on every frame…"):
                    out_video_path = process_video(
                        vid_in_path,
                        road_start_ratio=vid_road_start,
                        test_seconds=test_seconds,
                    )

                st.success("✅ Video processed successfully!")

                # Preview first processed frame
                cap_preview = cv2.VideoCapture(out_video_path)
                ret, preview_bgr = cap_preview.read()
                cap_preview.release()
                if ret:
                    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
                    st.image(
                        preview_rgb,
                        caption="Preview — first processed frame",
                        use_container_width=True,
                    )

                # Download button
                with open(out_video_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Processed Video",
                        data=vf,
                        file_name="lane_detection_output.mp4",
                        mime="video/mp4",
                    )

            except Exception as e:
                st.error(f"⚠️ Error during video processing:\n\n{str(e)}")
    else:
        st.info("📥 Upload a video file to begin.")

# ======================================================
# ADVANCED TAB — Lane detection + YOLO object detection
# ======================================================
with tab_advanced:
    st.markdown(
        "🚗 **Lane detection** (green mask) + **object detection** (orange boxes) combined.  "
        "Detects cars, trucks, buses, motorcycles, bicycles and persons."
    )

    adv_video_file = st.file_uploader(
        "📤 Upload a Road Video",
        type=["mp4", "mov", "avi", "mkv"],
        key="adv_vid_upload",
    )

    adv_test_seconds = st.slider(
        "⏱️ Seconds to process (0 = full video)",
        min_value=0,
        max_value=60,
        value=6,
        step=1,
        key="adv_test_seconds",
    )

    adv_road_start = (
        st.slider(
            "🔽 Skip top of frame (road start %)",
            min_value=0,
            max_value=70,
            value=40,
            step=5,
            help="Skip sky/signs at the top so lane detection focuses on the road.",
            key="adv_road_start",
        )
        / 100.0
    )

    adv_conf = (
        st.slider(
            "🎯 Object detection confidence threshold",
            min_value=10,
            max_value=90,
            value=40,
            step=5,
            help="Lower = more detections (may include false positives). Higher = only confident detections.",
            key="adv_conf",
        )
        / 100.0
    )

    if adv_video_file is not None:
        adv_in_path = os.path.join(
            UPLOADED_FOLDER, f"{uuid.uuid4().hex}_{adv_video_file.name}"
        )
        with open(adv_in_path, "wb") as f:
            f.write(adv_video_file.read())

        st.video(adv_in_path)
        st.markdown(f"**Uploaded:** `{adv_video_file.name}`")

        if st.button("🚀 Process Video (Advanced)"):
            try:
                with st.spinner("⚙️ Running lane + object detection on every frame…"):
                    adv_out_path = process_video_advanced(
                        adv_in_path,
                        road_start_ratio=adv_road_start,
                        test_seconds=adv_test_seconds,
                        conf=adv_conf,
                    )

                st.success("✅ Advanced detection completed!")

                cap_prev = cv2.VideoCapture(adv_out_path)
                ret, prev_bgr = cap_prev.read()
                cap_prev.release()
                if ret:
                    st.image(
                        cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2RGB),
                        caption="Preview — first processed frame (green = lanes, orange = objects)",
                        use_container_width=True,
                    )

                with open(adv_out_path, "rb") as vf:
                    st.download_button(
                        label="⬇️ Download Advanced Video",
                        data=vf,
                        file_name="advanced_detection_output.mp4",
                        mime="video/mp4",
                    )

            except Exception as e:
                st.error(f"⚠️ Error during advanced processing:\n\n{str(e)}")
    else:
        st.info("📥 Upload a video file to begin.")
