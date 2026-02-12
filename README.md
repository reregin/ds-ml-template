# 🛡️ Data Science Project Template

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-App-ff4b4b.svg)
![Status](https://img.shields.io/badge/Status-Development-green.svg)
![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)

A production-ready, modular structure for Data Science and Machine Learning projects. Designed to separate **experimentation** (notebooks) from **engineering** (src), ensuring reproducibility and scalability from Day 1.

---

## 📂 Project Structure

This project follows a strict separation of concerns.

```text
├── data/
│   ├── raw/                  # Immutable original data (do not edit)
│   ├── processed/            # Cleaned data used for modeling
│   └── external/             # Third-party data/references
│
├── notebooks/                              # Experimental Laboratory
│   ├── 01_eda_and_discovery.ipynb          # Discovery & Analysis
│   ├── 02_cleaning_and_features.ipynb      # Cleaning & Feature Engineering
│   ├── 03_training.ipynb                   # Model Selection & Hyperparameter Tuning
│   └── 04_inference.ipynb                  # Pipeline Verification
│
├── src/                      # Production Codebase
│   ├── config.py             # Global Control Center (Paths, Params)
│   ├── data_loader.py        # Robust Data Ingestion
│   ├── preprocessing.py      # Reusable Cleaning Logic
│   ├── train.py              # Model Training Pipeline
│   └── inference.py          # Prediction Engine (Singleton)
│
├── models/                   # Serialized Models (.pkl, .pth)
├── app/
│   └── main.py               # User Interface (Streamlit/FastAPI)
└── requirements.txt          # Dependencies