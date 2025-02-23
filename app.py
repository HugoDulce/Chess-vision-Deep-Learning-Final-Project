import streamlit as st
import cv2
import numpy as np
from PIL import Image
import chess_utils  # Import chess processing functions
import models  # Import YOLO models

# Title
st.title("Chess Vision App")
st.write("Upload a chessboard image to detect the board and pieces.")

@st.cache_resource  # Cache the models for efficiency
def load_models():
    return models.load_models()

# Load models once
board_model, piece_model = load_models()

# Upload Image
uploaded_file = st.file_uploader("Upload a chessboard image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True, channels="BGR")

    try:
        # Process image using chess_utils
        st.write("Processing image...")
        board_detections, piece_detections = chess_utils.process_image(image, board_model, piece_model)

        # Debugging prints
        st.write(f"Board Detections: {board_detections}")
        st.write(f"Piece Detections: {piece_detections}")

        # Draw results
        processed_image = chess_utils.draw_detections(image, board_detections, piece_detections)

        # Display processed image
        st.image(processed_image, caption="Detected Chessboard & Pieces", use_column_width=True, channels="BGR")

    except AttributeError as e:
        st.error(f"⚠️ AttributeError: {e}")
        st.write("Check if `chess_utils.process_image` exists and is correctly implemented.")
