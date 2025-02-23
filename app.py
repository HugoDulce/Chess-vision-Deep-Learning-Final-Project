import streamlit as st
import cv2
import numpy as np
from PIL import Image
import chess_utils  # Your chess processing script
import models  # Your model loading script

# 🏆 Streamlit Page Config
st.set_page_config(page_title="Chess Vision App", layout="wide")

# 🏆 Title & Description
st.title("♟️ Chess Vision AI")
st.write("Upload a chessboard image to detect the board and pieces using AI.")

# 🚀 Cache Model Loading to Avoid Reloading
@st.cache_resource
def load_models():
    return models.load_models()

# Load YOLO models
board_model, piece_model = load_models()

# 📤 File Upload
uploaded_file = st.file_uploader("📤 Upload a chessboard image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # 🖼️ Convert uploaded image to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format

    # 🎨 Display uploaded image
    st.image(image, caption="📷 Uploaded Image", use_column_width=True, channels="BGR")

    try:
        # 🔄 Run YOLO models directly since `process_image()` does not exist
        st.write("🔄 Detecting chessboard...")
        board_detections = board_model.predict(image)

        st.write("🔄 Detecting chess pieces...")
        piece_detections = piece_model.predict(image)

        # ✅ Debugging Information
        st.write(f"📝 Board Detections: {board_detections}")
        st.write(f"📝 Piece Detections: {piece_detections}")

        # 🎯 Draw Detected Chessboard and Pieces
        processed_image = chess_utils.draw_detections(image, board_detections, piece_detections)

        # 🖼️ Show Processed Image with Detections
        st.image(processed_image, caption="🧩 Detected Chessboard & Pieces", use_column_width=True, channels="BGR")

    except AttributeError as e:
        st.error("⚠️ Error: Check `chess_utils.py` and `models.py` for missing functions.")
        st.write(f"🔍 Debug Info: {e}")

    except Exception as e:
        st.error("⚠️ Unexpected Error. Check logs for details.")
        st.write(f"🔍 Debug Info: {e}")
