import streamlit as st
import numpy as np
import json
import os
from PIL import Image
import tensorflow as tf

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Wildlife AI Classifier",
    page_icon="🔥",
    layout="centered"
)

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #ff7e5f, #ff512f, #f9d423);
    color: white;
}

/* Title */
.title {
    text-align: center;
    font-size: 46px;
    font-weight: 700;
    margin-top: 10px;
}

/* Subtitle */
.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 25px;
    color: #fff3e0;
}

/* Glass Card */
.card {
    background: rgba(255, 255, 255, 0.15);
    padding: 25px;
    border-radius: 18px;
    backdrop-filter: blur(12px);
    text-align: center;
    margin-top: 20px;
    box-shadow: 0px 6px 25px rgba(0,0,0,0.25);
}

/* Prediction text */
.prediction {
    font-size: 30px;
    font-weight: bold;
}

/* Footer */
.footer {
    text-align: center;
    font-size: 14px;
    margin-top: 40px;
    color: #ffe0b2;
}

</style>
""", unsafe_allow_html=True)

# ------------------ TITLE ------------------
st.markdown('<div class="title">🔥 Wildlife Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">AI-powered animal recognition system</div>', unsafe_allow_html=True)

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_model():
    if os.path.exists("mobilenet_best.keras"):
        return tf.keras.models.load_model("mobilenet_best.keras")
    elif os.path.exists("model.h5"):
        return tf.keras.models.load_model("model.h5")
    else:
        st.error("❌ Model file not found!")
        return None

model = load_model()

# ------------------ LOAD LABELS ------------------
if os.path.exists("class_labels_s.json"):
    with open("class_labels_s.json", "r") as f:
        class_labels_s = json.load(f)

    idx_to_class = {v: k for k, v in class_labels_s.items()}
else:
    st.error("❌ class_labels_s.json not found!")
    st.stop()

# ------------------ FILE UPLOAD ------------------
uploaded_file = st.file_uploader("📂 Upload Image", type=["jpg", "png", "jpeg"])

# ------------------ PREDICT FUNCTION ------------------
def predict(img):
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return model.predict(img_array)

# ------------------ MAIN ------------------
if uploaded_file is not None and model is not None:

    img = Image.open(uploaded_file).convert("RGB")

    st.image(img, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Analyzing Image..."):
        pred = predict(img)

    # Top prediction
    top_idx = np.argmax(pred)
    confidence = np.max(pred)

    # ------------------ RESULT CARD ------------------
    st.markdown(f"""
    <div class="card">
        <div class="prediction">🐾 {idx_to_class[top_idx]}</div>
        <br>
        Confidence: {confidence*100:.2f}%
    </div>
    """, unsafe_allow_html=True)

    # ------------------ CONFIDENCE BAR ------------------
    st.progress(float(confidence))

    # ------------------ TOP 3 ------------------
    st.markdown("### 🔝 Top 3 Predictions")

    top3_idx = pred[0].argsort()[-3:][::-1]

    for i in top3_idx:
        st.write(f"**{idx_to_class[i]}**")
        st.progress(float(pred[0][i]))

# ------------------ FOOTER ------------------
st.markdown('<div class="footer">Made with ❤️ using Streamlit</div>', unsafe_allow_html=True)