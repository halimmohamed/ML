import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import os
import time
import logging
import requests
from sklearn.preprocessing import StandardScaler, LabelEncoder
from io import BytesIO

# Setup logging
logging.basicConfig(filename="app.log", level=logging.INFO)

# Streamlit page configuration
st.set_page_config(page_title="UNSW-NB15 Intrusion Detection", layout="wide")

# Language support
lang = st.sidebar.selectbox("Language / اللغة", ["English", "Arabic"])

# Language dictionary
texts = {
    "English": {
        "title": "🛡️ UNSW-NB15 Intrusion Detection System",
        "mode": "Mode",
        "upload_file": "Upload File",
        "manual_input": "Manual Input",
        "train_model": "Train Model",
        "upload_test": "Upload Test Parquet File",
        "upload_train": "Upload Training Parquet File",
        "prediction_results": "Prediction Results",
        "binary_distribution": "Binary Prediction Distribution",
        "attack_distribution": "Attack Type Distribution",
        "normal": "✅ Normal traffic detected.",
        "attack": "⚠️ Warning: Attack(s) detected!",
        "model_not_loaded": "Model not loaded. Please train the model first.",
        "invalid_file": "Invalid parquet file. Please upload a valid .parquet file.",
        "manual_input_label": "Enter",
        "predict": "Predict",
        "attack_type": "Attack Type",
        "train_success": "Model saved to",
        "train_error": "Error training model:",
        "file_error": "Error processing file:",
        "exploration": "Data Exploration",
        "dataset_preview": "Dataset Preview:",
        "class_distribution": "Class Distribution:",
        "download_report": "Download CSV Report",
        "theme_color": "Theme Accent Color",
        "monitoring": "Enable Real-Time Monitoring",
        "dashboard": "Live Dashboard",
        "total_attacks": "Total Attacks Detected",
        "model_url": "Enter Model URL (optional)",
        "model_loaded": "Pre-trained model loaded successfully!"
    },
    "Arabic": {
        "title": "🛡️ نظام كشف التسلل UNSW-NB15",
        "mode": "الوضع",
        "upload_file": "رفع ملف",
        "manual_input": "إدخال يدوي",
        "train_model": "تدريب النموذج",
        "upload_test": "ارفع ملف الاختبار (.parquet)",
        "upload_train": "ارفع ملف التدريب (.parquet)",
        "prediction_results": "نتائج التنبؤ",
        "binary_distribution": "توزيع التنبؤ الثنائي",
        "attack_distribution": "توزيع نوع الهجوم",
        "normal": "✅ تم اكتشاف حركة مرور عادية.",
        "attack": "⚠️ تحذير: تم اكتشاف هجوم!",
        "model_not_loaded": "لم يتم تحميل النموذج. من فضلك درب النموذج أولاً.",
        "invalid_file": "ملف غير صالح. من فضلك ارفع ملف parquet صالح.",
        "manual_input_label": "ادخل",
        "predict": "تنبوء",
        "attack_type": "نوع الهجوم",
        "train_success": "تم حفظ النموذج في",
        "train_error": "خطأ أثناء تدريب النموذج:",
        "file_error": "حدث خطأ أثناء معالجة الملف:",
        "exploration": "استكشاف البيانات",
        "dataset_preview": "عرض أولي للبيانات:",
        "class_distribution": "توزيع الفئات:",
        "download_report": "تحميل تقرير CSV",
        "theme_color": "لون الثيم",
        "monitoring": "تمكين المراقبة اللحظية",
        "dashboard": "لوحة التحكم الحية",
        "total_attacks": "إجمالي الهجمات المكتشفة",
        "model_url": "رابط تحميل النموذج (اختياري)",
        "model_loaded": "تم تحميل النموذج المُدرَّب بنجاح!"
    }
}

T = texts[lang]

# Theme customization
accent_color = st.sidebar.selectbox(T["theme_color"], ["Gold", "Silver", "Red"])
color_map = {"Gold": "#E8D96D", "Silver": "#C0C0C0", "Red": "#FF4040"}

st.markdown(f"""
    <style>
    body {{ background-color: #1C2526; color: {color_map[accent_color]}; }}
    .stApp {{ border: 2px solid {color_map[accent_color]}; }}
    </style>
""", unsafe_allow_html=True)

# Function to load model
@st.cache_resource
def load_model(model_path):
    try:
        models = joblib.load(model_path)
        return models['svm'], models['rf'], models['scaler']
    except Exception as e:
        st.error(f"{T['file_error']} {e}")
        return None, None, None

# Function to preprocess row
def preprocess_row(row, scaler, feature_columns):
    row_df = pd.DataFrame([row], columns=feature_columns)
    row_df = row_df.select_dtypes(include=[np.number])
    row_scaled = scaler.transform(row_df)
    return row_scaled

# Main app
st.title(T["title"])
st.sidebar.header(T["mode"])
mode = st.sidebar.radio("", [T["upload_file"], T["manual_input"], T["train_model"]])
model_url = st.sidebar.text_input(T["model_url"])

model_path = "model/final_model.pkl"
svm, rf, scaler = None, None, None

if model_url:
    try:
        response = requests.get(model_url)
        with open("temp_model.pkl", "wb") as f:
            f.write(response.content)
        svm, rf, scaler = load_model("temp_model.pkl")
        st.sidebar.success(T["model_loaded"])
    except Exception as e:
        st.sidebar.error(f"{T['file_error']} {e}")
elif os.path.exists(model_path):
    svm, rf, scaler = load_model(model_path)
    st.sidebar.success(T["model_loaded"])

if mode == T["upload_file"]:
    test_file = st.sidebar.file_uploader(T["upload_test"], type=["parquet"])
    if test_file:
        test_path = f"temp_test_{test_file.name}"
        with open(test_path, "wb") as f:
            f.write(test_file.read())
        try:
            test_df = pd.read_parquet(test_path)
            st.subheader(T["exploration"])
            st.write(T["dataset_preview"])
            st.dataframe(test_df.head())
            if "BinaryLabel" in test_df.columns:
                st.write(T["class_distribution"])
                fig = px.histogram(test_df, x="BinaryLabel", color="BinaryLabel")
                st.plotly_chart(fig)
            features = test_df.select_dtypes(include=[np.number])
            if svm and rf and scaler:
                X_test = scaler.transform(features)
                y_pred_bin = svm.predict(X_test)
                attack_indices = np.where(y_pred_bin == 1)[0]
                y_pred_att = rf.predict(X_test[attack_indices]) if attack_indices.size > 0 else []
                results_df = pd.DataFrame({
                    "Binary Prediction": ["Attack" if pred == 1 else "Normal" for pred in y_pred_bin]
                })
                st.subheader(T["prediction_results"])
                st.dataframe(results_df)
                st.plotly_chart(px.pie(values=pd.Series(y_pred_bin).value_counts(),
                                       names=["Normal", "Attack"],
                                       title=T["binary_distribution"]))
                if attack_indices.size > 0:
                    attack_counts = pd.Series(y_pred_att).value_counts()
                    fig2 = px.bar(x=attack_counts.index, y=attack_counts.values,
                                  labels={"x": T["attack_type"], "y": "Count"},
                                  title=T["attack_distribution"])
                    st.plotly_chart(fig2)
                if 1 in y_pred_bin:
                    st.error(T["attack"])
                else:
                    st.success(T["normal"])
                results_df.to_csv("report.csv")
                with open("report.csv", "rb") as f:
                    st.download_button(T["download_report"], f, file_name="report.csv")
            else:
                st.error(T["model_not_loaded"])
            os.remove(test_path)
        except Exception as e:
            st.error(f"{T['file_error']} {e}")

if mode == T["manual_input"]:
    feature_columns = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes']
    row_input = {}
    for col in feature_columns:
        row_input[col] = st.sidebar.number_input(f"{T['manual_input_label']} {col}", value=0.0)
    if st.sidebar.button(T["predict"]):
        if svm and rf and scaler:
            row = [row_input[col] for col in feature_columns]
            X_row = preprocess_row(row, scaler, feature_columns)
            pred_bin = svm.predict(X_row)[0]
            pred_label = "Attack" if pred_bin == 1 else "Normal"
            st.write(f"**{T['prediction_results']}**: {pred_label}")
            if pred_bin == 1:
                pred_att = rf.predict(X_row)[0]
                st.write(f"**{T['attack_type']}**: {pred_att}")
                st.error(T["attack"])
            else:
                st.success(T["normal"])
        else:
            st.error(T["model_not_loaded"])

if mode == T["train_model"]:
    train_file = st.sidebar.file_uploader(T["upload_train"], type=["parquet"])
    test_file = st.sidebar.file_uploader(T["upload_test"], type=["parquet"])
    if train_file and test_file:
        train_path = f"temp_train_{train_file.name}"
        test_path = f"temp_test_{test_file.name}"
        with open(train_path, "wb") as f:
            f.write(train_file.read())
        with open(test_path, "wb") as f:
            f.write(test_file.read())
        try:
            train_df = pd.read_parquet(train_path)
            test_df = pd.read_parquet(test_path)
            train_df['BinaryLabel'] = train_df['attack_cat'].apply(lambda x: 'Normal' if str(x).lower() == 'normal' else 'Attack')
            test_df['BinaryLabel'] = test_df['attack_cat'].apply(lambda x: 'Normal' if str(x).lower() == 'normal' else 'Attack')
            le_binary = LabelEncoder()
            train_df['binary_label'] = le_binary.fit_transform(train_df['BinaryLabel'])
            test_df['binary_label'] = le_binary.transform(test_df['BinaryLabel'])
            le_attack_type = LabelEncoder()
            train_df['attack_type_label'] = le_attack_type.fit_transform(train_df['attack_cat'].astype(str))
            test_df['attack_type_label'] = le_attack_type.transform(test_df['attack_cat'].astype(str))
            features = train_df.select_dtypes(include=[np.number]).drop(['attack_type_label', 'binary_label'], axis=1)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(features)
            y_train_bin = train_df['binary_label']
            y_train_att = train_df['attack_type_label']
            from sklearn.svm import SVC
            from sklearn.ensemble import RandomForestClassifier
            svm = SVC(kernel='rbf', random_state=42)
            svm.fit(X_train, y_train_bin)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_train, y_train_att)
            os.makedirs("model", exist_ok=True)
            joblib.dump({'svm': svm, 'rf': rf, 'scaler': scaler}, model_path, compress=3)
            st.success(f"{T['train_success']} {model_path}")
            from sklearn.metrics import classification_report, accuracy_score
            X_test = scaler.transform(test_df[features.columns])
            y_test_bin = test_df['binary_label']
            y_pred_bin = svm.predict(X_test)
            report = classification_report(y_test_bin, y_pred_bin, output_dict=True)
            accuracy = accuracy_score(y_test_bin, y_pred_bin)
            st.write(f"Accuracy: {accuracy:.4f}")
            st.json(report)
            fig = px.bar(x=list(report.keys())[:-3], y=[report[label]["f1-score"] for label in report.keys() if label not in ["accuracy", "macro avg", "weighted avg"]], labels={"x": "Class", "y": "F1-Score"}, title="SVM F1-Score per Class")
            st.plotly_chart(fig)
            os.remove(train_path)
            os.remove(test_path)
        except Exception as e:
            st.error(f"{T['train_error']} {e}")

# Real-time Monitoring (example simulation)
if st.sidebar.checkbox(T["monitoring"]):
    st.subheader(T["dashboard"])
    attack_count = np.random.randint(0, 100)  # simulate
    col1, col2 = st.columns(2)
    with col1:
        st.metric(T["total_attacks"], attack_count)
    with col2:
        fig = px.pie(values=[attack_count, 200 - attack_count], names=["Attack", "Normal"])
        st.plotly_chart(fig)
