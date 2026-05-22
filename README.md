# ecoswap-rna

Red Neuronal Artificial (RNA) en Python para predecir si un intercambio de productos en la plataforma **EcoSwap** será exitoso. El modelo está integrado con el proyecto Spring Boot de EcoSwap a través de una API Flask.

---

## Tabla de contenidos

1. [Descripción del problema](#descripción-del-problema)
2. [Dataset](#dataset)
3. [Arquitectura de la RNA](#arquitectura-de-la-rna)
4. [Estructura del proyecto](#estructura-del-proyecto)
5. [Instalación y configuración](#instalación-y-configuración)
6. [Uso](#uso)
7. [API Flask](#api-flask)
8. [Integración con Spring Boot](#integración-con-spring-boot)
9. [Métricas y evaluación](#métricas-y-evaluación)

---

## Descripción del problema

EcoSwap es una plataforma de intercambio sostenible de productos. Cuando dos usuarios quieren intercambiar productos, la plataforma debe predecir si ese intercambio tiene probabilidades de completarse exitosamente.

**Tarea:** Clasificación binaria  
**Target:** `exchange_successful` → `True` (exitoso) / `False` (no exitoso)

---

## Dataset

### Fuente

Archivo Excel: `plantilla dataset_diccionario_corregido.xlsx`  
Hoja: **"Plantilla Dataset"**  
Registros: **100 filas**

### Features (10 variables de entrada)

| Feature | Tipo | Descripción |
|---|---|---|
| `product_category` | Categórico | Categoría del producto: `tecnologia`, `hogar`, `ropa`, `deportes`, `libros`, `otros` |
| `product_condition` | Categórico | Estado del producto: `nuevo`, `como_nuevo`, `usado_bueno`, `usado_regular` |
| `days_published` | Numérico | Días que lleva publicado el producto |
| `product_estimated_value` | Numérico | Valor estimado del producto ofertado (COP) |
| `offer_estimated_value` | Numérico | Valor estimado del producto ofrecido a cambio (COP) |
| `value_ratio` | Numérico | Relación de valores: `offer_value / product_value` |
| `interactions_30d` | Numérico | Número de intercambios solicitados en los últimos 30 días |
| `user_success_history` | Numérico | Historial de intercambios exitosos del usuario |
| `user_rating_avg` | Numérico | Calificación promedio del usuario (1.0 - 5.0) |
| `distance_km` | Numérico | Distancia entre los usuarios en kilómetros |

### Variable objetivo

| Variable | Tipo | Descripción |
|---|---|---|
| `exchange_successful` | Booleano | `True` si el intercambio se completó, `False` si fue cancelado/rechazado |

---

## Arquitectura de la RNA

```
Input Layer (10 neuronas)
     ↓
Dense Layer (64 neuronas, activación ReLU)
     ↓
Dropout (30%)
     ↓
Dense Layer (32 neuronas, activación ReLU)
     ↓
Dropout (20%)
     ↓
Output Layer (1 neurona, activación Sigmoid)
```

**Configuración de entrenamiento:**
- Optimizador: Adam (lr=0.001)
- Función de pérdida: Binary Crossentropy
- Métricas: Accuracy, AUC-ROC
- Early Stopping: patience=15 épocas
- Reducción de LR: factor=0.5, patience=7
- División de datos: 70% train / 15% validación / 15% test

---

## Estructura del proyecto

```
ecoswap-rna/
├── data/
│   ├── __init__.py
│   ├── load_excel.py       # Carga el dataset desde el archivo Excel
│   ├── fetch_mysql.py      # Extrae features desde la BD MySQL de EcoSwap
│   └── preprocess.py       # Encoders, scaler y split de datos
├── models/
│   ├── __init__.py
│   ├── rna_model.py        # Arquitectura, entrenamiento y predicción
│   └── saved/              # Archivos generados al entrenar
│       ├── rna_ecoswap.keras           # Modelo entrenado
│       ├── encoder_product_category.pkl
│       ├── encoder_product_condition.pkl
│       ├── scaler.pkl
│       ├── confusion_matrix.png
│       ├── roc_curve.png
│       └── curvas_aprendizaje.png
├── api/
│   ├── __init__.py
│   └── app.py              # Servidor Flask con endpoints de predicción
├── train.py                # Entrenar el modelo
├── evaluate.py             # Evaluar y generar gráficas
├── requirements.txt
├── .env.example            # Plantilla de variables de entorno
└── README.md
```

---

## Instalación y configuración

### Prerrequisitos

- Python 3.10+
- pip

### 1. Crear entorno virtual

```bash
cd C:\Users\POWER\Documents\Dev\Python\ecoswap-rna

python -m venv venv
venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Configurar variables de entorno

```bash
# Copiar plantilla
copy .env.example .env
```

Editar `.env` con los valores correctos:

```env
EXCEL_PATH=c:\Users\POWER\Documents\DESARROLLO DE SOFTWARE\SEMESTRE 8\MINERIA DE DATOS\sf\plantilla dataset_diccionario_corregido.xlsx

DB_HOST=localhost
DB_PORT=3306
DB_NAME=ecoswap
DB_USER=root
DB_PASSWORD=Carlos1009

FLASK_PORT=5000
FLASK_DEBUG=True
```

> **Nota:** La conexión a MySQL solo es necesaria para el endpoint `/predict/from-db`. El entrenamiento usa únicamente el archivo Excel.

---

## Uso

### Paso 1: Entrenar el modelo

```bash
# Con el entorno virtual activo
python train.py
```

Salida esperada:
```
==================================================
  RNA EcoSwap - Entrenamiento
==================================================
Dataset cargado: 100 registros, 11 columnas
Distribución target: {True: 58, False: 42}
Train: 70 | Val: 15 | Test: 15
...
Evaluación en conjunto de prueba:
  Loss:     0.4532
  Accuracy: 0.8000 (80.0%)
  AUC-ROC:  0.8647
Modelo guardado en: models/saved/rna_ecoswap.keras
```

### Paso 2: Evaluar con gráficas

```bash
python evaluate.py
```

Genera en `models/saved/`:
- `confusion_matrix.png` — Matriz de confusión
- `roc_curve.png` — Curva ROC con AUC
- `curvas_aprendizaje.png` — Loss y accuracy por época

### Paso 3: Iniciar el servidor Flask

```bash
python api/app.py
```

El servidor queda disponible en `http://localhost:5050`.

---

## API Flask

### `GET /health`

Verifica que el servidor y el modelo están funcionando.

**Respuesta:**
```json
{
    "status": "ok",
    "model": "loaded",
    "version": "1.0.0"
}
```

---

### `POST /predict/features`

Predicción enviando las features directamente.

**Request:**
```json
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
```

**Respuesta:**
```json
{
    "exchange_successful": true,
    "probability": 0.8247,
    "confidence": "alta"
}
```

**Ejemplo con curl:**
```bash
curl -X POST http://localhost:5050/predict/features \
  -H "Content-Type: application/json" \
  -d "{\"product_category\":\"tecnologia\",\"product_condition\":\"usado_bueno\",\"days_published\":15,\"product_estimated_value\":500000,\"offer_estimated_value\":480000,\"value_ratio\":0.96,\"interactions_30d\":12,\"user_success_history\":5,\"user_rating_avg\":4.2,\"distance_km\":8.5}"
```

---

### `POST /predict/from-db`

Predicción consultando los productos directamente en la base de datos MySQL de EcoSwap.

**Request:**
```json
{
    "product_from_id": 101,
    "product_to_id": 205
}
```

**Respuesta:**
```json
{
    "product_from_id": 101,
    "product_to_id": 205,
    "exchange_successful": true,
    "probability": 0.7512,
    "confidence": "alta",
    "features_used": {
        "product_category": "hogar",
        "product_condition": "como_nuevo",
        "days_published": 23,
        ...
    }
}
```

> **Requisito:** El proyecto EcoSwap Spring Boot debe estar corriendo y la BD MySQL accesible.

---


## Métricas y evaluación

Las métricas se calculan sobre el conjunto de test (15% del dataset):

| Métrica | Descripción |
|---|---|
| **Accuracy** | Porcentaje de predicciones correctas |
| **AUC-ROC** | Área bajo la curva ROC (1.0 = perfecto) |
| **Precision** | De los intercambios predichos como exitosos, cuántos lo fueron realmente |
| **Recall** | De todos los intercambios exitosos reales, cuántos detectó el modelo |
| **F1-Score** | Media armónica entre Precision y Recall |

Las gráficas se guardan automáticamente en `models/saved/` al ejecutar `evaluate.py`.

---

## Tecnologías utilizadas

| Tecnología | Versión | Uso |
|---|---|---|
| Python | 3.10+ | Lenguaje principal |
| TensorFlow/Keras | 2.13+ | Construcción y entrenamiento de la RNA |
| scikit-learn | 1.3+ | Preprocesamiento y métricas |
| Flask | 3.0+ | API REST para servir predicciones |
| pandas | 2.0+ | Manipulación de datos |
| SQLAlchemy + PyMySQL | Latest | Conexión a MySQL |
| openpyxl | 3.1+ | Lectura del archivo Excel |
| matplotlib / seaborn | Latest | Visualizaciones |

---

## Contexto académico

- **Institución:** Fundación Universitaria Tecnológico Comfenalco
- **Asignatura:** Minería de Datos II
- **Docente:** Walter Alberto Espriella Castellar
- **Proyecto base:** EcoSwap — Plataforma de intercambio sostenible de productos
