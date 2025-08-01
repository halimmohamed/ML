import streamlit as st
import pandas as pd
import numpy as np
import joblib
import io
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from PIL import Image
import base64
import os

# إعداد الصفحة العامة
st.set_page_config(page_title="Intrusion Detection App",
                   layout="centered",
                   initial_sidebar_state="collapsed")

# تحميل الخلفية
@st.cache_data
def get_base64_of_bg(file_path):
    with open(file_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg(file_path):
    bg_ext = os.path.splitext(file_path)[-1][1:]
    bg_base64 = get_base64_of_bg(file_path)
    page_bg_img = f'''
    <style>
    [data-testid="stApp"] {{
        background: url("data:image/{bg_ext};base64,{bg_base64}") no-repeat center center fixed;
        background-size: cover;
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

set_bg("background.avif")

# إعداد الثيمات
THEME = st.session_state.get("theme", "dark")
LANG = st.session_state.get("lang", "en")

def toggle_theme():
    st.session_state["theme"] = "light" if THEME == "dark" else "dark"

def toggle_lang():
    st.session_state["lang"] = "ar" if LANG == "en" else "en"

# الشريط العلوي
with st.container():
    cols = st.columns([6, 1, 1])
    with cols[1]:
        if st.button("🌙" if THEME == "dark" else "☀️", use_container_width=True):
            toggle_theme()
            st.experimental_rerun()
    with cols[2]:
        if st.button("🇬🇧" if LANG == "en" else "🇸🇦", use_container_width=True):
            toggle_lang()
            st.experimental_rerun()

# ألوان حسب الثيم
box_color = "rgba(255, 255, 255, 0.85)" if THEME == "light" else "rgba(50, 50, 50, 0.8)"
text_color = "black" if THEME == "light" else "white"

st.markdown(f'''
    <div style="background-color:{box_color}; padding: 2rem; border-radius: 20px; color:{text_color};">
''', unsafe_allow_html=True)

st.markdown(f"## {'Intrusion Detection System' if LANG == 'en' else 'نظام كشف التسلل'}")

# تحميل الموديلات
uploaded_model = st.file_uploader("🔍 Upload Final Model (.PKI)", type=["pki"])
uploaded_scaler = st.file_uploader("⚖️ Upload Scaler (.PKI)", type=["pki"])
uploaded_binary = st.file_uploader("🔁 Upload Binary LabelEncoder (.PKI)", type=["pki"])
uploaded_attack = st.file_uploader("🎯 Upload AttackType LabelEncoder (.PKI)", type=["pki"])

if uploaded_model and uploaded_scaler and uploaded_binary and uploaded_attack:
    final_model = joblib.load(uploaded_model)
    scaler = joblib.load(uploaded_scaler)
    le_binary = joblib.load(uploaded_binary)
    le_attack_type = joblib.load(uploaded_attack)

    st.success("✅ Models Loaded Successfully")

    option = st.radio("Select Input Method", ["Upload Parquet File", "Manual Entry", "Train New Model"], horizontal=True)

    if option == "Upload Parquet File":
        parquet_file = st.file_uploader("📂 Upload Parquet File", type=["parquet"])
        if parquet_file:
            df = pd.read_parquet(parquet_file)
            st.subheader("📊 Data Summary")
            st.write(f"Shape: {df.shape}")
            st.write(f"Columns: {list(df.columns)}")
            st.dataframe(df.head())

            # المعالجة والتنبؤ
            X = df.drop(columns=['label', 'attack_cat'], errors='ignore')
            X_scaled = scaler.transform(X)
            binary_preds = le_binary.inverse_transform(final_model.predict(X_scaled))
            attack_preds = le_attack_type.inverse_transform(final_model.predict(X_scaled))

            df['Prediction'] = attack_preds
            df['Binary'] = binary_preds

            st.subheader("📥 Predictions")
            st.dataframe(df[['Prediction', 'Binary']])

            # تحميل النتائج
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("💾 Download CSV", csv, "results.csv", "text/csv")

            # رسومات
            st.subheader("📈 Visualization")
            count_data = df['Prediction'].value_counts()
            fig1, ax1 = plt.subplots()
            count_data.plot(kind='bar', ax=ax1)
            st.pyplot(fig1)

            fig2, ax2 = plt.subplots()
            df['Binary'].value_counts().plot(kind='pie', autopct='%1.1f%%', ax=ax2)
            st.pyplot(fig2)

    elif option == "Manual Entry":
        st.subheader("✍️ Enter Data Manually")
        manual_input = {}
        cols = st.columns(3)
        sample_features = ['dur', 'sbytes', 'dbytes']
        for i, feature in enumerate(sample_features):
            manual_input[feature] = cols[i % 3].number_input(feature, value=0.0)

        if st.button("Predict", type="primary"):
            X = pd.DataFrame([manual_input])
            X_scaled = scaler.transform(X)
            binary = le_binary.inverse_transform(final_model.predict(X_scaled))[0]
            attack = le_attack_type.inverse_transform(final_model.predict(X_scaled))[0]
            st.success(f"Prediction: {attack} | Binary: {binary}")

    elif option == "Train New Model":
        st.warning("⚠️ Feature under development")
else:
    st.info("📌 Please upload all required models above to start.")

st.markdown("</div>", unsafe_allow_html=True)