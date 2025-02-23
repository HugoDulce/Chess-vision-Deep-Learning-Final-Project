import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import chess_utils  # Ensure this is correctly imported

# Load YOLO models
board_model, piece_model = chess_utils.load_models()

# Streamlit App Title
st.title("Chess Board & Piece Detection")

# Upload Image
uploaded_file = st.file_uploader("Upload a Chessboard Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV image format
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to NumPy array for OpenCV processing

    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Detect the Chessboard
    st.write("🔄 Detecting chessboard...")
    board_results = board_model(image)

    # Detect Chess Pieces
    st.write("🔄 Detecting chess pieces...")
    piece_results = piece_model(image)

    # Debugging Output
    st.write("📝 Board Detections:", board_results)
    st.write("📝 Piece Detections:", piece_results)

    # Extracting Chessboard Corners
    if len(board_results) > 0 and board_results[0].boxes is not None:
        board_detections = board_results[0].boxes
        crossings = [(int(x), int(y)) for box in board_detections for x, y, _, _ in box.xywh.cpu().numpy()]

        if len(crossings) > 0:
            # Generate a 7x7 grid of intersections
            grid = chess_utils.complete_grid(crossings, image.shape)

            # Draw the grid on the image
            image_with_grid, horizontal_lines, vertical_lines = chess_utils.draw_infinite_grid(image, grid)

            # Display the result
            st.image(image_with_grid, caption="Detected Chessboard Grid", use_column_width=True)
        else:
            st.error("⚠️ Chessboard detected, but no valid crossings found.")
    else:
        st.error("⚠️ No chessboard detected. Try another image.")

    # Processing Piece Detections
    if len(piece_results) > 0 and piece_results[0].boxes is not None:
        piece_detections = piece_results[0].boxes
        if len(piece_detections) > 0:
            st.success("✅ Chess pieces detected!")
        else:
            st.error("⚠️ No chess pieces detected.")
    else:
        st.error("⚠️ No chess pieces detected.")

    st.write("✅ Processing Complete!")

