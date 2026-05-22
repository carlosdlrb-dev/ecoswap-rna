"""
Extrae datos de intercambios desde la base de datos MySQL de EcoSwap
y los transforma en el mismo formato de features que el dataset Excel.

Requiere que el proyecto Spring Boot esté corriendo y la BD accesible.
"""

import math
import os
import pandas as pd
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()

DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "3306")
DB_NAME = os.getenv("DB_NAME", "ecoswap")
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "Carlos1009")

DATABASE_URL = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"


def get_engine():
    return create_engine(DATABASE_URL)


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2 +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def fetch_exchange_features(engine=None) -> pd.DataFrame:
    """
    Extrae y calcula las features de intercambios desde MySQL.

    Consulta: productos + usuarios + exchanges completados/rechazados.
    Retorna DataFrame con el mismo formato que load_excel.load_dataset().
    """
    if engine is None:
        engine = get_engine()

    query = text("""
        SELECT
            p_from.category                             AS product_category,
            p_from.condition_product                    AS product_condition,
            DATEDIFF(NOW(), p_from.release_date)        AS days_published,
            0                                           AS product_estimated_value,
            0                                           AS offer_estimated_value,
            1.0                                         AS value_ratio,
            (
                SELECT COUNT(*) FROM exchanges e2
                WHERE (e2.product_from_id = p_from.id OR e2.product_to_id = p_from.id)
                  AND e2.exchange_requested_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            )                                           AS interactions_30d,
            (
                SELECT COUNT(*) FROM exchanges e3
                WHERE (
                    e3.product_from_id IN (SELECT id FROM products WHERE id_user = u_from.id)
                    OR e3.product_to_id IN (SELECT id FROM products WHERE id_user = u_from.id)
                )
                AND e3.status = 'COMPLETED'
            )                                           AS user_success_history,
            COALESCE(
                (SELECT AVG(ur.score) FROM user_ratings ur WHERE ur.reviewed_id = u_from.id),
                3.5
            )                                           AS user_rating_avg,
            (6371 * ACOS(
                GREATEST(-1, LEAST(1,
                    COS(RADIANS(u_from.latitude)) * COS(RADIANS(u_to.latitude)) *
                    COS(RADIANS(u_to.longitude) - RADIANS(u_from.longitude)) +
                    SIN(RADIANS(u_from.latitude)) * SIN(RADIANS(u_to.latitude))
                ))
            ))                                          AS distance_km,
            CASE WHEN e.status = 'COMPLETED' THEN 1 ELSE 0 END AS exchange_successful
        FROM exchanges e
        JOIN products p_from ON e.product_from_id = p_from.id
        JOIN users u_from ON p_from.id_user = u_from.id
        JOIN products p_to ON e.product_to_id = p_to.id
        JOIN users u_to ON p_to.id_user = u_to.id
        WHERE e.status IN ('COMPLETED', 'CANCELLED', 'REJECTED')
    """)

    with engine.connect() as conn:
        df = pd.read_sql(query, conn)

    print(f"Registros extraídos de MySQL: {len(df)}")
    return df


def get_product_features_by_id(product_from_id: int, product_to_id: int, engine=None) -> dict:
    """
    Obtiene las features calculadas para un par de productos específicos.
    Usado por el endpoint Flask /predict/from-db.
    """
    if engine is None:
        engine = get_engine()

    query = text("""
        SELECT
            p.category          AS product_category,
            p.condition_product AS product_condition,
            DATEDIFF(NOW(), p.release_date) AS days_published,
            0                   AS product_estimated_value,
            0                   AS offer_estimated_value,
            1.0                 AS value_ratio,
            (
                SELECT COUNT(*) FROM exchanges e2
                WHERE (e2.product_from_id = p.id OR e2.product_to_id = p.id)
                  AND e2.exchange_requested_at >= DATE_SUB(NOW(), INTERVAL 30 DAY)
            )                   AS interactions_30d,
            (
                SELECT COUNT(*) FROM exchanges e3
                WHERE (
                    e3.product_from_id IN (SELECT id FROM products WHERE id_user = p.id_user)
                    OR e3.product_to_id IN (SELECT id FROM products WHERE id_user = p.id_user)
                )
                AND e3.status = 'COMPLETED'
            )                   AS user_success_history,
            COALESCE(
                (SELECT AVG(ur.score) FROM user_ratings ur WHERE ur.reviewed_id = p.id_user),
                3.5
            )                   AS user_rating_avg,
            u.latitude          AS user_latitude,
            u.longitude         AS user_longitude
        FROM products p
        JOIN users u ON p.id_user = u.id
        WHERE p.id = :product_id
    """)

    with engine.connect() as conn:
        row_from = conn.execute(query, {"product_id": product_from_id}).mappings().fetchone()
        row_to = conn.execute(query, {"product_id": product_to_id}).mappings().fetchone()

    if row_from is None:
        raise ValueError(f"Producto {product_from_id} no encontrado")
    if row_to is None:
        raise ValueError(f"Producto {product_to_id} no encontrado")

    lat1 = row_from["user_latitude"] or 6.2442
    lon1 = row_from["user_longitude"] or -75.5812
    lat2 = row_to["user_latitude"] or 6.2442
    lon2 = row_to["user_longitude"] or -75.5812
    distance = haversine_km(lat1, lon1, lat2, lon2)

    return {
        "product_category": row_from["product_category"],
        "product_condition": row_from["product_condition"],
        "days_published": int(row_from["days_published"] or 0),
        "product_estimated_value": 0,
        "offer_estimated_value": 0,
        "value_ratio": 1.0,
        "interactions_30d": int(row_from["interactions_30d"] or 0),
        "user_success_history": int(row_from["user_success_history"] or 0),
        "user_rating_avg": float(row_from["user_rating_avg"] or 3.5),
        "distance_km": round(distance, 2),
    }


if __name__ == "__main__":
    try:
        df = fetch_exchange_features()
        print(df.head())
    except Exception as e:
        print(f"Error conectando a MySQL: {e}")
        print("Asegúrate de que el proyecto Spring Boot esté corriendo y la BD accesible.")
