# 🔐 Cyber Attack Detection with Machine Learning

This project uses trained machine learning models (SVM & Random Forest) to detect and classify network attacks using the UNSW-NB15 dataset.

## 👨‍💻 Features
- Upload `.parquet` test files or enter custom input manually.
- Detects if traffic is normal or an attack.
- If an attack, it classifies the type (e.g. DoS, Reconnaissance, etc.).
- Displays prediction summary and charts.

## 📦 Models Used
- `SVM` for binary classification (Normal vs Attack)
- `Random Forest` for attack type classification

## 🚀 Deployed using Streamlit

## 🔗 Dataset
[UNSW-NB15 Kaggle Dataset](https://www.kaggle.com/datasets/dhoogla/unswnb15)

---

> Make sure to upload the following model files with the app:
- `final_model.pkl`
- `scaler.pkl`
- `binary_encoder.pkl`
- `attack_encoder.pkl`

