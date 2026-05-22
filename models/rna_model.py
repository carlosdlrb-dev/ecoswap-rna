"""
Arquitectura de la Red Neuronal Artificial (RNA) para predecir
si un intercambio en EcoSwap será exitoso.

Arquitectura:
    Input(10) → Dense(64, ReLU) → Dropout(0.3)
              → Dense(32, ReLU) → Dropout(0.2)
              → Dense(1, Sigmoid)

Tarea: Clasificación binaria (exchange_successful: True/False)
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SAVED_DIR = os.path.join(os.path.dirname(__file__), "saved")
MODEL_PATH = os.path.join(SAVED_DIR, "rna_ecoswap.keras")


def build_model(input_dim: int = 10) -> keras.Model:
    """Construye y compila la RNA."""
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.2),
        layers.Dense(1, activation="sigmoid"),
    ], name="rna_ecoswap")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.AUC(name="auc")],
    )
    return model


def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 100,
    batch_size: int = 16,
) -> keras.callbacks.History:
    """Entrena la RNA con early stopping."""
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=15,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1,
    )
    return history


def save_model(model: keras.Model, path: str = None):
    os.makedirs(SAVED_DIR, exist_ok=True)
    save_path = path or MODEL_PATH
    model.save(save_path)
    print(f"Modelo guardado en: {save_path}")


def load_model(path: str = None) -> keras.Model:
    load_path = path or MODEL_PATH
    if not os.path.exists(load_path):
        raise FileNotFoundError(
            f"Modelo no encontrado en {load_path}. Ejecuta train.py primero."
        )
    return keras.models.load_model(load_path)


def predict(model: keras.Model, X: np.ndarray, threshold: float = 0.5):
    """
    Retorna (etiqueta_bool, probabilidad) para cada fila de X.
    """
    probs = model.predict(X, verbose=0).flatten()
    labels = (probs >= threshold).tolist()
    return labels, probs.tolist()
