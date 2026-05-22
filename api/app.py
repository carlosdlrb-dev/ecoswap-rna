"""
API Flask para servir predicciones de la RNA EcoSwap.

Endpoints:
  GET  /health                  → Estado del servidor y modelo
  POST /predict/features        → Predicción con features explícitas
  POST /predict/from-db         → Predicción consultando MySQL por ID de productos

Cómo llamar desde Spring Boot:
    POST http://localhost:5000/predict/features
    Content-Type: application/json
    Body: { "product_category": "tecnologia", "product_condition": "usado_bueno", ... }

Uso:
    python api/app.py
"""

import os
import sys

# Permite importar desde la raíz del proyecto
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from data.preprocess import transform_features
from models.rna_model import load_model, predict as rna_predict

load_dotenv()

app = Flask(__name__)
CORS(app)

# Cargar modelo al iniciar el servidor
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_model()
    return _model


REQUIRED_FEATURES = [
    "product_category",
    "product_condition",
    "days_published",
    "product_estimated_value",
    "offer_estimated_value",
    "value_ratio",
    "interactions_30d",
    "user_success_history",
    "user_rating_avg",
    "distance_km",
]


@app.route("/health", methods=["GET"])
def health():
    try:
        get_model()
        model_status = "loaded"
    except Exception as e:
        model_status = f"error: {str(e)}"

    return jsonify({
        "status": "ok",
        "model": model_status,
        "version": "1.0.0",
    })


@app.route("/predict/features", methods=["POST"])
def predict_from_features():
    """
    Recibe las 10 features explícitas y retorna la predicción.

    Request body:
    {
        "product_category": "tecnologia",
        "product_condition": "usado_bueno",
        "days_published": 15,
        "product_estimated_value": 500000,
        "offer_estimated_value": 480000,
        "value_ratio": 0.96,
        "interactions_30d": 12,
        "user_success_history": 5,
        "user_rating_avg": 4.2,
        "distance_km": 8.5
    }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Se requiere un cuerpo JSON"}), 400

    missing = [f for f in REQUIRED_FEATURES if f not in data]
    if missing:
        return jsonify({"error": f"Faltan campos: {missing}"}), 400

    try:
        X = transform_features(data)
        model = get_model()
        labels, probs = rna_predict(model, X)

        return jsonify({
            "exchange_successful": bool(labels[0]),
            "probability": round(float(probs[0]), 4),
            "confidence": "alta" if abs(probs[0] - 0.5) > 0.3 else "media" if abs(probs[0] - 0.5) > 0.15 else "baja",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/from-db", methods=["POST"])
def predict_from_db():
    """
    Recibe IDs de los dos productos a intercambiar,
    consulta MySQL para calcular las features y retorna la predicción.

    Request body:
    {
        "product_from_id": 101,
        "product_to_id": 205
    }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Se requiere un cuerpo JSON"}), 400

    product_from_id = data.get("product_from_id")
    product_to_id = data.get("product_to_id")

    if product_from_id is None or product_to_id is None:
        return jsonify({"error": "Se requieren product_from_id y product_to_id"}), 400

    try:
        from data.fetch_mysql import get_product_features_by_id
        features = get_product_features_by_id(int(product_from_id), int(product_to_id))

        X = transform_features(features)
        model = get_model()
        labels, probs = rna_predict(model, X)

        return jsonify({
            "product_from_id": product_from_id,
            "product_to_id": product_to_id,
            "exchange_successful": bool(labels[0]),
            "probability": round(float(probs[0]), 4),
            "confidence": "alta" if abs(probs[0] - 0.5) > 0.3 else "media" if abs(probs[0] - 0.5) > 0.15 else "baja",
            "features_used": features,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict/simple", methods=["POST"])
def predict_simple():
    """
    Endpoint simplificado para Flutter.
    Solo recibe category, condition y days_published.
    Rellena defaults para las demás features del modelo.

    Request body:
    {
        "product_category": "tecnologia",
        "product_condition": "como_nuevo",
        "days_published": 23
    }
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "Se requiere un cuerpo JSON"}), 400

    required = ["product_category", "product_condition", "days_published"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Faltan campos: {missing}"}), 400

    # Valores por defecto basados en las medianas del dataset de entrenamiento
    features = {
        "product_category": data["product_category"],
        "product_condition": data["product_condition"],
        "days_published": int(data["days_published"]),
        "product_estimated_value": 350000,
        "offer_estimated_value": 320000,
        "value_ratio": 0.91,
        "interactions_30d": 30,
        "user_success_history": 25,
        "user_rating_avg": 3.2,
        "distance_km": 18.0,
    }

    try:
        X = transform_features(features)
        model = get_model()
        labels, probs = rna_predict(model, X)

        return jsonify({
            "exchange_successful": bool(labels[0]),
            "probability": round(float(probs[0]), 4),
            "confidence": "alta" if abs(probs[0] - 0.5) > 0.3 else "media" if abs(probs[0] - 0.5) > 0.15 else "baja",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", 5050))
    debug = os.getenv("FLASK_DEBUG", "True").lower() == "true"

    print(f"Iniciando servidor Flask en http://localhost:{port}")
    print("Endpoints disponibles:")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  POST http://localhost:{port}/predict/features")
    print(f"  POST http://localhost:{port}/predict/from-db")
    print(f"  POST http://localhost:{port}/predict/simple")

    app.run(host="0.0.0.0", port=port, debug=debug)
