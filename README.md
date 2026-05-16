# Disease Prediction System from Medical Data

## Project Overview

An end-to-end machine learning project that predicts diseases from structured patient medical data. Includes a Jupyter Notebook for exploration and training, a training script, and a Streamlit web application for real-time predictions.

## Features

- Synthetic realistic medical dataset generation (`disease_prediction_dataset.csv`) if missing
- Data preprocessing and feature engineering
- Trains multiple classifiers: Logistic Regression, SVM, Random Forest, XGBoost
- Model evaluation with metrics and visualizations
- SHAP explainability for feature impacts
- Saves `best_model.pkl`, `scaler.pkl`, and `encoder.pkl`
- Streamlit app (`app.py`) with modern medical UI

## Tech Stack

- Python, pandas, numpy
- scikit-learn, xgboost
- SHAP, joblib
- Streamlit for web app

## Installation

1. Create and activate a Python environment (recommended).
2. Install requirements:

```bash
pip install -r requirements.txt
```

## Run Notebook

Open `notebook.ipynb` in Jupyter and run cells top-to-bottom.

## Train & Save Model (script)

```bash
python train_model.py
```

## Run Streamlit App

```bash
streamlit run app.py
```

## Notes for Demo

- The notebook installs missing packages automatically if needed.
- The dataset is generated automatically on first run.

## Screenshots

- Add your screenshots of the app here for the README.
