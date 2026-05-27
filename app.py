import streamlit as st
import cv2
import numpy as np

from tensorflow.keras.models import load_model

# Load trained model
model = load_model("models/deepfake_model.keras")

# Image size
IMG_SIZE = 224

# App title
st.title("AI Deepfake Detection System")

st.write("Upload an image to check whether it is REAL or FAKE.")

# File uploader
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

# Check if file uploaded
if uploaded_file is not None:

    # Read image bytes
    file_bytes = np.asarray(
        bytearray(uploaded_file.read()),
        dtype=np.uint8
    )

    # Decode image
    image = cv2.imdecode(file_bytes, 1)

    # Display uploaded image
    st.image(
        image,
        channels="BGR",
        caption="Uploaded Image"
    )

    # Resize image
    image_resized = cv2.resize(image, (224, 224))

    # Normalize image
    image_normalized = image_resized / 255.0

    # Convert to array
    image_array = np.array(image_normalized)

    # Expand dimensions
    image_array = np.expand_dims(image_array, axis=0)

    # Predict
    prediction = model.predict(image_array)

    probability = prediction[0][0]

    st.write(f"Prediction Score: {probability:.4f}")

    # Classification
    if probability >= 0.5:
        st.error("Result: FAKE IMAGE")
    else:
        st.success("Result: REAL IMAGE")