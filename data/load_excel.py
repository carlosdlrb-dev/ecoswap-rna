"""
Carga y parsea el dataset desde la hoja "Plantilla Dataset" del archivo Excel.

Estructura del archivo:
- Hoja: "Plantilla Dataset"
- Fila 48 (0-indexed): nombres de columnas
- Filas 49-148: datos reales (100 registros)
- Columnas en posiciones no consecutivas: 0,1,4,5,9,13,14,15,16,17,18
"""

import os
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

EXCEL_PATH = os.getenv(
    "EXCEL_PATH",
    r"c:\Users\POWER\Documents\DESARROLLO DE SOFTWARE\SEMESTRE 8\MINERIA DE DATOS\sf\plantilla dataset_diccionario_corregido.xlsx"
)

SHEET_NAME = "Plantilla Dataset"

# Posiciones de columnas en el Excel (no consecutivas)
COL_INDICES = [0, 1, 4, 5, 9, 13, 14, 15, 16, 17, 18]
COL_NAMES = [
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
    "exchange_successful",
]

DATA_START_ROW = 49  # Fila donde comienzan los datos (0-indexed)


def load_dataset(excel_path: str = None) -> pd.DataFrame:
    """
    Carga el dataset desde el archivo Excel y retorna un DataFrame limpio.

    Returns:
        DataFrame con 100 filas y 11 columnas, listo para preprocesar.
    """
    path = excel_path or EXCEL_PATH

    if not os.path.exists(path):
        raise FileNotFoundError(f"Archivo Excel no encontrado: {path}")

    raw = pd.read_excel(path, sheet_name=SHEET_NAME, header=None)

    data = raw.iloc[DATA_START_ROW:, COL_INDICES].copy()
    data.columns = COL_NAMES
    data = data.dropna(how="all").reset_index(drop=True)

    # Convertir target a booleano
    data["exchange_successful"] = data["exchange_successful"].map(
        {"True": True, "False": False, True: True, False: False}
    )

    # Asegurar tipos numéricos
    numeric_cols = [
        "days_published", "product_estimated_value", "offer_estimated_value",
        "value_ratio", "interactions_30d", "user_success_history",
        "user_rating_avg", "distance_km",
    ]
    for col in numeric_cols:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna().reset_index(drop=True)

    print(f"Dataset cargado: {len(data)} registros, {len(data.columns)} columnas")
    print(f"Distribución target: {data['exchange_successful'].value_counts().to_dict()}")
    return data


if __name__ == "__main__":
    df = load_dataset()
    print(df.head())
    print(df.dtypes)
