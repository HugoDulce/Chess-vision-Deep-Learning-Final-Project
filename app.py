import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import chess_utils  # Ensure this file is in the same directory

# Load YOLO models
st.write("🔄 Loading YOLO models...")
try:
    board_model, piece_model = chess_utils.load_models()
    st.success("✅ Models loaded successfully!")
except FileNotFoundError as e:
    st.error(f"⚠️ Model file missing: {e}")
    st.stop()

# Streamlit App Title
st.title("♟️ Chess Board & Piece Detection")

# Upload Image (ONLY ONCE)
uploaded_file = st.file_uploader("📂 Upload a Chessboard Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert to OpenCV image format
    image = Image.open(uploaded_file)
    image = np.array(image)  # Convert to NumPy array for OpenCV processing

    # Ensure 3-channel RGB format
    if image.ndim == 2:  # Grayscale image
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:  # RGBA image
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    
    st.image(image, caption="📷 Uploaded Image", use_column_width=True)

    # Detect the Chessboard
    st.write("🔄 Detecting chessboard...")
    try:
        board_results = board_model.predict(image)
    except Exception as e:
        st.error(f"⚠️ Error detecting chessboard: {e}")
        st.stop()

    # Detect Chess Pieces
    st.write("🔄 Detecting chess pieces...")
    try:
        piece_results = piece_model.predict(image)
    except Exception as e:
        st.error(f"⚠️ Error detecting chess pieces: {e}")
        st.stop()

    # Debugging Output
    st.write("📝 Board Detections:", board_results)
    st.write("📝 Piece Detections:", piece_results)

    # Extracting Chessboard Corners
    board_detections = board_results[0].boxes if board_results else None
    piece_detections = piece_results[0].boxes if piece_results else None

    if not board_detections or len(board_detections) == 0:
        st.error("⚠️ No chessboard detected. Try another image.")
    else:
        crossings = [(int(box.xywh[0][0]), int(box.xywh[0][1])) for box in board_detections]

        if len(crossings) < 10:
            st.warning("⚠️ Low number of board crossings detected. Grid estimation may be inaccurate.")
        
        # Generate a 7x7 grid of intersections
        grid = chess_utils.complete_grid(crossings, image.shape)

        # Draw the grid on the image
        image_with_grid, horizontal_lines, vertical_lines = chess_utils.draw_infinite_grid(image, grid)

        # Display the result
        st.image(image_with_grid, caption="🟩 Detected Chessboard Grid", use_column_width=True)

    if not piece_detections or len(piece_detections) == 0:
        st.error("⚠️ No chess pieces detected.")
    else:
        st.success("✅ Chess pieces detected!")

        # Convert to DataFrame for chessboard representation
        board_df = chess_utils.create_chessboard_dataframe(piece_detections, horizontal_lines, vertical_lines)
        board_df = chess_utils.reorient_board(board_df)  # Ensure proper orientation

        # Convert to FEN notation
        fen_string = chess_utils.df_to_fen(board_df)
        st.write("📜 FEN Representation:")
        st.code(fen_string, language="text")

    st.write("✅ Processing Complete!")
