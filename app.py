import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import math
from collections import defaultdict

st.set_page_config(page_title="Egyptian Coin Detector", layout="wide")
st.title("Egyptian Coin Detector 💰")
st.write("Detect coins from an uploaded image or your camera and track total per session.")

# Load YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")  # Replace with your trained model

model = load_model()

# Session state for centroids and totals
if "coin_centroids" not in st.session_state:
    st.session_state.coin_centroids = []  # (cx, cy, label)
if "total_counts" not in st.session_state:
    st.session_state.total_counts = defaultdict(int)

# Distance threshold to prevent double-counting
DIST_THRESHOLD = 30

# Map labels to coin values
coin_values = {
    "Gneh": 1.0,
    "NosGneh": 0.5
}

def centroid(box):
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    return (x1 + x2) / 2, (y1 + y2) / 2

def is_new_coin(cx, cy, label):
    for existing_cx, existing_cy, existing_label in st.session_state.coin_centroids:
        if existing_label == label:
            dist = math.hypot(existing_cx - cx, existing_cy - cy)
            if dist < DIST_THRESHOLD:
                return False
    return True

# Input: file upload or camera
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
camera_image = st.camera_input("Or take a photo")

image = None
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
elif camera_image is not None:
    image = Image.open(camera_image).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_container_width=True)

    try:
        results = model(image, conf=0.25)[0]
    except Exception as e:
        st.error(f"Model inference failed: {e}")
        results = None

    if results and len(results.boxes) > 0:
        annotated_frame = results.plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        frame_counts = defaultdict(int)
        cropped_coins = []

        for box in results.boxes:
            cls_id = int(box.cls.item())
            label = model.names[cls_id]
            cx, cy = centroid(box)

            # Crop coin
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            coin_crop = image.crop((x1, y1, x2, y2))
            cropped_coins.append((label, coin_crop))

            # Update session totals if this coin is new
            if is_new_coin(cx, cy, label):
                st.session_state.coin_centroids.append((cx, cy, label))
                st.session_state.total_counts[label] += 1

            frame_counts[label] += 1

        # Display annotated frame
        st.image(annotated_frame, caption="Detection Result", use_container_width=True)

        # Display cropped coins
        st.subheader("🪙 Cropped Coins")
        cols = st.columns(5)
        for idx, (label, coin_img) in enumerate(cropped_coins):
            cols[idx % 5].image(coin_img, caption=label, use_column_width=True)

        # Display per-frame counts and session totals
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Frame Counts")
            for label, count in frame_counts.items():
                st.write(f"{label}: {count}")
        with col2:
            st.subheader("Session Totals (avoiding double-counting)")
            total_value = 0
            for label, count in st.session_state.total_counts.items():
                st.write(f"{label}: {count}")
                total_value += coin_values.get(label, 0) * count
            st.subheader(f"💵 Total Amount: {total_value} EGP")

    else:
        st.info("No coins detected in this image.")
else:
    st.info("Upload an image or take a photo to start detection.")
