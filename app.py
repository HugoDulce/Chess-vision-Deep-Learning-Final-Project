import streamlit as st
import chess_utils  # Import your chess detection functions
import models       # Import your model

st.title("Chess Vision App")
st.write("Upload a chessboard image to detect the board and pieces.")

uploaded_file = st.file_uploader("Upload a chessboard image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Load models from your script
    board_model, piece_model = models.load_models()

    # Process the image using functions in chess_utils.py
    board_detections, piece_detections = chess_utils.process_image(uploaded_file, board_model, piece_model)

    # Draw results
    processed_image = chess_utils.draw_detections(uploaded_file, board_detections, piece_detections)
    st.image(processed_image, caption="Detected Chessboard & Pieces", use_column_width=True)
