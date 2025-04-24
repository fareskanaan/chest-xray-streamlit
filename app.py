import os
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import gdown

# ---------- إعدادات الصفحة ----------
st.set_page_config(page_title="Chest X-ray Diagnosis", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🫁 Chest X-ray Classifier (VGG16)</h1>", unsafe_allow_html=True)
st.markdown("<hr style='border: 2px solid #4CAF50;'>", unsafe_allow_html=True)

# ---------- تحميل النموذج ----------
model_path = "vgg16_stream.h5"
file_id = "1zMEAPzM2QsUP_aTuJ6jBbO8yNpemlq4-"
url = f"https://drive.google.com/uc?id={file_id}"

if not os.path.exists(model_path):
    gdown.download(url, model_path, quiet=False)

model = load_model(model_path, compile=False)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# اسم طبقة Grad-CAM
last_conv_layer_name = 'block5_conv3'

# الفئات
class_names = ['Normal', 'Pneumonia-Bacterial', 'Viral Pneumonia']

# ---------- واجهة رفع الصورة ----------
st.subheader("Upload a Chest X-ray Image")
uploaded_file = st.file_uploader("Choose a chest X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # معالجة الصورة
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
        image = cv2.resize(image, (224, 224))
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        image = clahe.apply(image)
        image_3ch = cv2.merge([image, image, image])
        image_norm = image_3ch.astype('float32') / 255.0
        img_array = np.expand_dims(image_norm, axis=0)

        # التنبؤ
        preds = model.predict(img_array)
        class_idx = int(np.argmax(preds[0]))
        predicted_class = class_names[class_idx]
        confidence = np.max(preds[0]) * 100

        st.success(f"✅ **Predicted Class:** {predicted_class} ({confidence:.2f}%)")

        st.image(image_3ch, caption="Input Chest X-ray Image", use_column_width=True)

        # Grad-CAM
        grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, class_idx]

        grads = tape.gradient(loss, conv_outputs)[0]
        conv_outputs = conv_outputs[0]
        weights = tf.reduce_mean(grads, axis=(0, 1))
        cam = np.zeros(conv_outputs.shape[:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * conv_outputs[:, :, i]

        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))  # ✅ حذف .numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        heatmap = (cam * 255).astype("uint8")
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(image_3ch, 0.6, heatmap, 0.4, 0)

        st.markdown("---")
        st.subheader("Grad-CAM Heatmap")
        st.image(overlay, caption="Model Explanation", use_column_width=True)

    except Exception as e:
        st.error(f"❌ Failed to process the image.\n\n**{str(e)}**")
