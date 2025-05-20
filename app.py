import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import tensorflow as tf
from PIL import Image

# Wide layout for full-screen columns
st.set_page_config(layout="wide")

# Load model once
@st.cache_resource
def load_denoising_model(path="saved_model.keras"):
    return load_model(path)

model = load_denoising_model()

# Add noise function
def add_noise(image, noise_factor=0.1):
    image = image / 255.0  # Normalize
    gaussian = noise_factor * np.random.normal(loc=0.0, scale=1.0, size=image.shape)
    poisson = noise_factor * np.random.poisson(lam=0.1, size=image.shape) / 255.0
    noisy = image + gaussian + poisson
    return np.clip(noisy, 0.0, 1.0)

# Denoise using model
def denoise_image(noisy_img):
    input_img = np.expand_dims(noisy_img, axis=0)
    prediction = model.predict(input_img)[0]
    return np.clip(prediction, 0.0, 1.0)

# Streamlit UI
st.title("🧼 Image Denoising using Trained Deep CNN")

uploaded_file = st.file_uploader("📤 Upload a clean image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Load and preprocess image
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize((256, 256))
    img_array = np.array(img).astype(np.float32)

    # Add noise
    noisy = add_noise(img_array)

    # Denoise
    denoised = denoise_image(noisy)

    # Layout with 3 wide columns
    col1, col2, col3 = st.columns([1, 1, 1])  # All equal width

    with col1:
        st.subheader("Original")
        st.image(img_array.astype(np.uint8), use_column_width=True)

    with col2:
        st.subheader("Noisy")
        st.image((noisy * 255).astype(np.uint8), use_column_width=True)

    with col3:
        st.subheader("Denoised")
        st.image((denoised * 255).astype(np.uint8), use_column_width=True)
