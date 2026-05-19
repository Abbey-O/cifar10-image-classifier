import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf

st.set_page_config(page_title="CIFAR-10 Classifier", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model_transfer.keras")

@st.cache_data
def load_classes():
    with open("artifacts/classes.txt", "r") as f:
        return [line.strip() for line in f.readlines()]

model = load_model()
class_names = load_classes()

st.title("CIFAR-10 Image Classifier (TensorFlow/Keras)")
st.write("Upload an image. The app will resize it to 32×32 and predict a CIFAR-10 class.")

uploaded = st.file_uploader("Choose an image (jpg/png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess: resize to CIFAR-10 size
    img_resized = image.resize((32, 32))
    x = np.array(img_resized, dtype=np.float32)
    x = np.expand_dims(x, axis=0)  # (1, 32, 32, 3)

    # Predict
    probs = model.predict(x, verbose=0)[0]
    top_idx = int(np.argmax(probs))
    top_class = class_names[top_idx]
    top_prob = float(probs[top_idx])

    st.subheader(f"Prediction: **{top_class}** ({top_prob*100:.2f}%)")

    # Show probability table
    st.write("Class probabilities:")
    for name, p in sorted(zip(class_names, probs), key=lambda t: t[1], reverse=True):
        st.write(f"- {name}: {p*100:.2f}%")
else:
    st.info("Upload a file to get a prediction.")
