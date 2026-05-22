"""
Evaluación detallada del modelo entrenado.

Genera:
  - Reporte de clasificación (precision, recall, F1)
  - Matriz de confusión
  - Curva ROC con AUC
  - Curva de aprendizaje (loss y accuracy por época)

Uso:
    python evaluate.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)

from data.load_excel import load_dataset
from data.preprocess import fit_and_transform, split_data
from models.rna_model import build_model, train_model, load_model, predict

SAVED_DIR = os.path.join("models", "saved")


def plot_confusion_matrix(y_true, y_pred, save_path):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["No exitoso", "Exitoso"],
        yticklabels=["No exitoso", "Exitoso"],
    )
    plt.title("Matriz de Confusión - RNA EcoSwap")
    plt.ylabel("Real")
    plt.xlabel("Predicho")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Matriz de confusión guardada: {save_path}")


def plot_roc_curve(y_true, y_probs, save_path):
    fpr, tpr, _ = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Tasa de Falsos Positivos")
    plt.ylabel("Tasa de Verdaderos Positivos")
    plt.title("Curva ROC - RNA EcoSwap")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curva ROC guardada: {save_path}")


def plot_training_history(history, save_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history["loss"], label="Train")
    ax1.plot(history.history["val_loss"], label="Validación")
    ax1.set_title("Pérdida (Loss)")
    ax1.set_xlabel("Época")
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train")
    ax2.plot(history.history["val_accuracy"], label="Validación")
    ax2.set_title("Exactitud (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.legend()

    plt.suptitle("Curvas de Aprendizaje - RNA EcoSwap")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Curvas de aprendizaje guardadas: {save_path}")


def main():
    print("=" * 50)
    print("  RNA EcoSwap - Evaluación")
    print("=" * 50)

    os.makedirs(SAVED_DIR, exist_ok=True)

    df = load_dataset()
    X, y, _, _ = fit_and_transform(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Re-entrenar para obtener history (o cargar si ya existe)
    model_path = os.path.join(SAVED_DIR, "rna_ecoswap.keras")
    if os.path.exists(model_path):
        print("Cargando modelo guardado...")
        model = load_model()
    else:
        print("Modelo no encontrado. Entrenando desde cero...")
        model = build_model(input_dim=X_train.shape[1])
        history = train_model(model, X_train, y_train, X_val, y_val)
        plot_training_history(history, os.path.join(SAVED_DIR, "curvas_aprendizaje.png"))

    # Predicciones en test
    y_pred_labels, y_probs = predict(model, X_test)
    y_pred = [int(l) for l in y_pred_labels]

    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred, target_names=["No exitoso", "Exitoso"]))

    plot_confusion_matrix(y_test, y_pred, os.path.join(SAVED_DIR, "confusion_matrix.png"))
    plot_roc_curve(y_test, y_probs, os.path.join(SAVED_DIR, "roc_curve.png"))

    print("\nEvaluación completada. Gráficas guardadas en models/saved/")


if __name__ == "__main__":
    main()
