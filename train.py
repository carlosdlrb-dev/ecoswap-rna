"""
Script principal de entrenamiento de la RNA.

Uso:
    python train.py
"""

from data.load_excel import load_dataset
from data.preprocess import fit_and_transform, split_data
from models.rna_model import build_model, train_model, save_model


def main():
    print("=" * 50)
    print("  RNA EcoSwap - Entrenamiento")
    print("=" * 50)

    # 1. Cargar datos
    df = load_dataset()

    # 2. Preprocesar y dividir
    X, y, encoders, scaler = fit_and_transform(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # 3. Construir y entrenar
    model = build_model(input_dim=X_train.shape[1])
    model.summary()

    print("\nEntrenando la RNA...")
    history = train_model(model, X_train, y_train, X_val, y_val)

    # 4. Evaluar en test set
    print("\nEvaluación en conjunto de prueba:")
    loss, accuracy, auc = model.evaluate(X_test, y_test, verbose=0)
    print(f"  Loss:     {loss:.4f}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
    print(f"  AUC-ROC:  {auc:.4f}")

    # 5. Guardar modelo
    save_model(model)
    print("\nEntrenamiento completado exitosamente.")
    print("Ahora puedes ejecutar:")
    print("  python evaluate.py    -> metricas detalladas y graficas")
    print("  python api/app.py     -> servidor Flask para predicciones")


if __name__ == "__main__":
    main()
