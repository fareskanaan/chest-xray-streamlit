
# app.py

import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# إعدادات عامة
st.set_page_config(page_title="Chest X-ray Classifier", layout="centered")
st.title("🫁 Chest X-ray Classifier (VGG16 Model)")
st.markdown("Upload a chest X-ray image and the model will predict its class.")

# تحميل النموذج
@st.cache_resource
import tempfile
import requests

@st.cache_resource
def load_vgg_model():
    # Google Drive file ID
    file_id = "1zMEAPzM2QsUP_aTuJ6jBbO8yNpemlq4-"
    url = f"https://drive.google.com/uc?id={file_id}"

    # تحميل النموذج مؤقتًا
    response = requests.get(url)
    tmp_file = tempfile.NamedTemporaryFile(delete=False)
    tmp_file.write(response.content)
    tmp_file.flush()

    model = load_model(tmp_file.name)
    return model


model = load_vgg_model()

# إعداد الفئات (يفترض ترتيبها كما في التدريب)
class_labels = ['Bacterial', 'Normal', 'Viral']

# رفع صورة
uploaded_file = st.file_uploader("Upload Chest X-ray Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(600, 600))
    img_array = image.img_to_array(img) / 255.0
    img_batch = np.expand_dims(img_array, axis=0)

    # عرض الصورة
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # التنبؤ
    preds = model.predict(img_batch)
    pred_class = class_labels[np.argmax(preds)]
    confidence = np.max(preds) * 100

    # النتيجة
    st.markdown("### 🧠 Prediction Result")
    st.write(f"**Predicted Class:** {pred_class}")
    st.write(f"**Confidence:** {confidence:.2f}%")

    # رسم الاحتمالات
    st.markdown("### 📊 Prediction Probabilities")
    fig, ax = plt.subplots()
    ax.bar(class_labels, preds[0], color='skyblue')
    ax.set_ylabel("Probability")
    ax.set_ylim([0, 1])
    st.pyplot(fig)


