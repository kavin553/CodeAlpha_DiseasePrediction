"""train_model.py
Generates dataset (if missing), trains multiple classifiers, evaluates them,
and saves the best model, scaler, and encoder.
"""
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import roc_auc_score
import joblib


DATA_PATH = "disease_prediction_dataset.csv"


def generate_dataset(path=DATA_PATH, n=1500, random_state=42):
    np.random.seed(random_state)
    ages = np.random.randint(18, 90, size=n)
    gender = np.random.choice(["Male", "Female"], size=n, p=[0.48, 0.52])
    bp = np.random.normal(120, 18, size=n).round(0)
    hr = np.random.normal(75, 10, size=n).round(0)
    cholesterol = np.random.normal(200, 40, size=n).round(0)
    blood_sugar = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    bmi = np.clip(np.random.normal(26, 5, size=n), 15, 45).round(1)
    smoking = np.random.choice(["Never", "Former", "Current"], size=n, p=[0.6, 0.2, 0.2])
    alcohol = np.random.choice(["None", "Moderate", "High"], size=n, p=[0.5, 0.4, 0.1])
    activity = np.random.choice(["Low", "Medium", "High"], size=n, p=[0.4, 0.4, 0.2])
    family = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    chest_pain = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    fatigue = np.random.choice([0, 1], size=n, p=[0.7, 0.3])
    fever = np.random.choice([0, 1], size=n, p=[0.9, 0.1])
    cough = np.random.choice([0, 1], size=n, p=[0.85, 0.15])
    breath = np.random.choice([0, 1], size=n, p=[0.88, 0.12])
    headache = np.random.choice([0, 1], size=n, p=[0.8, 0.2])
    nausea = np.random.choice([0, 1], size=n, p=[0.88, 0.12])
    glucose = np.clip(np.random.normal(110, 30, size=n), 60, 400).round(1)
    insulin = np.clip(np.random.normal(15, 10, size=n), 2, 300).round(1)
    symptom_severity = np.random.randint(0, 10, size=n)

    # Risk score heuristics for disease label assignment
    score_hd = (bp > 140).astype(int) + (cholesterol > 240).astype(int) + (smoking == "Current").astype(int) + family
    score_db = (glucose > 126).astype(int) + (bmi > 30).astype(int) + (insulin > 25).astype(int) + (blood_sugar == 1).astype(int)
    score_bc = (gender == "Female").astype(int) * ((ages > 40).astype(int)) + family

    labels = []
    for i in range(n):
        if score_db[i] >= 2:
            labels.append("Diabetes")
        elif score_hd[i] >= 2:
            labels.append("Heart Disease")
        elif score_bc[i] >= 2 and np.random.rand() < 0.15:
            labels.append("Breast Cancer")
        else:
            labels.append("Healthy")

    df = pd.DataFrame({
        "Age": ages,
        "Gender": gender,
        "Blood Pressure": bp,
        "Heart Rate": hr,
        "Cholesterol Level": cholesterol,
        "Blood Sugar": blood_sugar,
        "BMI": bmi,
        "Smoking Habit": smoking,
        "Alcohol Consumption": alcohol,
        "Physical Activity": activity,
        "Family History": family,
        "Chest Pain": chest_pain,
        "Fatigue": fatigue,
        "Fever": fever,
        "Cough": cough,
        "Breathing Difficulty": breath,
        "Headache": headache,
        "Nausea": nausea,
        "Glucose Level": glucose,
        "Insulin Level": insulin,
        "Symptom Severity": symptom_severity,
        "Disease Label": labels,
    })

    # Introduce a small amount of missingness
    for col in ["Cholesterol Level", "BMI", "Glucose Level"]:
        df.loc[df.sample(frac=0.02, random_state=random_state).index, col] = np.nan

    df.to_csv(path, index=False)
    print(f"Generated dataset and saved to {path}")
    return df


def load_or_generate(path=DATA_PATH):
    if os.path.exists(path):
        print(f"Loading existing dataset from {path}")
        return pd.read_csv(path)
    else:
        return generate_dataset(path)


def preprocess(df):
    df = df.copy()
    # Basic missing handling
    df["Cholesterol Level"].fillna(df["Cholesterol Level"].median(), inplace=True)
    df["BMI"].fillna(df["BMI"].median(), inplace=True)
    df["Glucose Level"].fillna(df["Glucose Level"].median(), inplace=True)

    # Categorical columns
    cat_cols = ["Gender", "Smoking Habit", "Alcohol Consumption", "Physical Activity"]
    num_cols = [c for c in df.columns if c not in cat_cols + ["Disease Label"]]

    # Label encode target
    le = LabelEncoder()
    df["target"] = le.fit_transform(df["Disease Label"])

    # Column transformer for encoding and scaling
    preprocessor = ColumnTransformer(
        transformers=[
            ("onehot", OneHotEncoder(handle_unknown="ignore"), cat_cols),
            ("scale", StandardScaler(), num_cols),
        ]
    )

    X = df.drop(columns=["Disease Label", "target"])
    y = df["target"].values
    return X, y, preprocessor, le


def train_and_evaluate():
    df = load_or_generate()
    X, y, preprocessor, le = preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(probability=True),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    }

    best_score = 0
    best_model = None

    # Fit preprocessor
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train_proc, y_train)
        preds = model.predict(X_test_proc)
        acc = accuracy_score(y_test, preds)
        print(name, "Accuracy:", acc)
        if acc > best_score:
            best_score = acc
            best_model = model

    print("Best model:", best_model)

    # Save artifacts
    os.makedirs("models", exist_ok=True)
    joblib.dump(best_model, "models/best_model.pkl")
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    joblib.dump(le, "models/label_encoder.pkl")
    print("Saved best_model.pkl, preprocessor.pkl, label_encoder.pkl in models/")


if __name__ == "__main__":
    train_and_evaluate()
