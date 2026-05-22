"""
Pipeline de preprocesamiento para el dataset de EcoSwap.

Transforma las features categóricas y numéricas en tensores listos
para alimentar la RNA. Los encoders y scaler se guardan en disco
para reutilizarlos en inferencia sin re-entrenar.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

SAVED_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "saved")

CATEGORICAL_COLS = ["product_category", "product_condition"]
NUMERIC_COLS = [
    "days_published",
    "product_estimated_value",
    "offer_estimated_value",
    "value_ratio",
    "interactions_30d",
    "user_success_history",
    "user_rating_avg",
    "distance_km",
]
TARGET_COL = "exchange_successful"


def fit_and_transform(df: pd.DataFrame):
    """
    Ajusta encoders y scaler sobre el DataFrame completo y transforma.
    Guarda los objetos entrenados en models/saved/.

    Returns:
        X (np.ndarray), y (np.ndarray), encoders (dict), scaler
    """
    os.makedirs(SAVED_DIR, exist_ok=True)

    encoders = {}
    X_parts = []

    for col in CATEGORICAL_COLS:
        le = LabelEncoder()
        encoded = le.fit_transform(df[col].astype(str)).reshape(-1, 1)
        encoders[col] = le
        X_parts.append(encoded)
        joblib.dump(le, os.path.join(SAVED_DIR, f"encoder_{col}.pkl"))

    scaler = StandardScaler()
    numeric_data = scaler.fit_transform(df[NUMERIC_COLS].values)
    joblib.dump(scaler, os.path.join(SAVED_DIR, "scaler.pkl"))

    X = np.hstack(X_parts + [numeric_data]).astype(np.float32)
    y = df[TARGET_COL].astype(int).values

    return X, y, encoders, scaler


def transform_features(features: dict) -> np.ndarray:
    """
    Transforma un diccionario de features usando los encoders/scaler guardados.
    Usado en inferencia (API Flask).

    Args:
        features: dict con las 10 features de entrada

    Returns:
        np.ndarray de shape (1, 10)
    """
    encoded_parts = []

    for col in CATEGORICAL_COLS:
        le: LabelEncoder = joblib.load(os.path.join(SAVED_DIR, f"encoder_{col}.pkl"))
        val = str(features[col])
        if val not in le.classes_:
            # Si la categoría es desconocida, usar la más frecuente (índice 0)
            encoded = np.array([[0]])
        else:
            encoded = le.transform([val]).reshape(-1, 1)
        encoded_parts.append(encoded)

    scaler: StandardScaler = joblib.load(os.path.join(SAVED_DIR, "scaler.pkl"))
    numeric_vals = np.array([[features[col] for col in NUMERIC_COLS]], dtype=np.float32)
    scaled_numeric = scaler.transform(numeric_vals)

    X = np.hstack(encoded_parts + [scaled_numeric]).astype(np.float32)
    return X


def split_data(X: np.ndarray, y: np.ndarray, val_size=0.15, test_size=0.15, random_state=42):
    """Divide el dataset en train / validación / test."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=(val_size + test_size), random_state=random_state, stratify=y
    )
    relative_test = test_size / (val_size + test_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=relative_test, random_state=random_state, stratify=y_temp
    )
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, X_val, X_test, y_train, y_val, y_test
