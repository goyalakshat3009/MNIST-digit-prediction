"""Lightweight MNIST inference using only NumPy + Pillow (no TensorFlow needed)."""
import numpy as np
from PIL import Image, ImageOps


def load_weights(path="weights.npz"):
    return np.load(path)


def preprocess(img: Image.Image) -> np.ndarray:
    """Convert any uploaded image to a (1, 28, 28) float array like MNIST."""
    img = ImageOps.exif_transpose(img).convert("L")  # grayscale
    arr = np.asarray(img.resize((28, 28), Image.LANCZOS), dtype=np.float32)
    # MNIST digits are white on black; invert dark-on-light images
    if arr.mean() > 127:
        arr = 255.0 - arr
    return (arr / 255.0).reshape(1, 28, 28)


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


def predict(x: np.ndarray, w) -> np.ndarray:
    """Forward pass of the 784-50-50-10 network. Returns 10 probabilities."""
    x = x.reshape(x.shape[0], -1)
    h1 = np.maximum(0, x @ w["W1"] + w["b1"])
    h2 = np.maximum(0, h1 @ w["W2"] + w["b2"])
    logits = h2 @ w["W3"] + w["b3"]
    return softmax(logits)
