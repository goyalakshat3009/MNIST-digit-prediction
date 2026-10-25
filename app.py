import os

import numpy as np
import streamlit as st
from PIL import Image

from predictor import load_weights, predict, preprocess

st.set_page_config(page_title="MNIST Digit Classifier", page_icon="🔢")
st.title("🔢 Handwritten Digit Classifier")
st.write("Upload a picture of a single handwritten digit (0-9) and the neural network will recognise it.")


@st.cache_resource
def get_weights():
    return load_weights(os.path.join(os.path.dirname(__file__), "weights.npz"))


weights = get_weights()

file = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg"])

if file is not None:
    img = Image.open(file)
    x = preprocess(img)
    probs = predict(x, weights)[0]
    label = int(np.argmax(probs))

    col1, col2 = st.columns(2)
    with col1:
        st.image(img, caption="Your image", use_container_width=True)
    with col2:
        st.image(x[0], caption="What the model sees (28x28)", clamp=True, width=140)
        st.metric("Predicted digit", label, f"{probs[label] * 100:.1f}% confident")

    st.bar_chart({"probability": {str(i): float(p) for i, p in enumerate(probs)}})
else:
    st.info("Tip: best results with a white digit on a black background, or a dark digit on white paper.")
