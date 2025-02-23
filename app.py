import streamlit as st
import chess_utils  # Import your chess detection functions
import models       # Import your model
import cv2
import numpy as np
from PIL import Image

# Load models once (outside of the `if uploaded_file` block)
st.title("Chess Vision App")
st.write("Upload a chessboard image to detect the board and pieces.")

@st.cache_resource  # Cache the models to avoid reloading them each time
def load_models():
    return models.load_models()

board_model, piece_model = load_models()

uploaded_file = st.file_uploader("Upload a chessboard image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to an OpenCV image
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert PIL Image to OpenCV format

    # Display uploaded image
    st.image(image, caption="Uploaded Image", use_column_width=True, channels="BGR")

    # Process the image using functions in chess_utils.py
    board_detections, piece_detections = chess_utils.process_image(image, board_model, piece_model)

    # Draw results
    processed_image = chess_utils.draw_detections(image, board_detections, piece_detections)

    # Display processed image
    st.image(processed_image, caption="Detected Chessboard & Pieces", use_column_width=True, channels="BGR")
