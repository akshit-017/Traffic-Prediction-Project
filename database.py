"""
database.py — MySQL Telemetry Pipeline
========================================

Connects to a local MySQL server via pymysql and logs every route
query into the `route_telemetry` table.

Resilience: Every database operation is wrapped in try/except.
If MySQL is unreachable, the routing API continues to work — only
telemetry logging is silently skipped.
"""

import json
import os
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

# ── Connection parameters ────────────────────────────────────────────
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_USER = os.environ.get("DB_USER", "root")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_NAME = os.environ.get("DB_NAME", "traffic_db")


def _get_connection():
    """Return a new pymysql connection.  Returns None if MySQL is down."""
    try:
        import pymysql
        conn = pymysql.connect(
            host=DB_HOST,
            user=DB_USER,
            password=DB_PASSWORD,
            database=DB_NAME,
            charset="utf8mb4",
            cursorclass=pymysql.cursors.DictCursor,
            connect_timeout=3,
        )
        return conn
    except Exception as exc:
        print(f"[DB] MySQL connection failed (non-fatal): {exc}")
        return None


def init_db():
    """
    Create the route_telemetry table if it doesn't exist.
    Silently skips if MySQL is unreachable.
    """
    conn = _get_connection()
    if conn is None:
        print("[DB] Skipping table creation — MySQL unavailable.")
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS route_telemetry (
                    id                  INT AUTO_INCREMENT PRIMARY KEY,
                    source_node         VARCHAR(100),
                    destination_node    VARCHAR(100),
                    routing_mode        VARCHAR(20),
                    total_distance_km   FLOAT,
                    travel_time_min     INT,
                    estimated_cost_inr  INT,
                    path_taken          TEXT,
                    fallback_triggered  BOOLEAN DEFAULT FALSE,
                    query_timestamp     DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
        conn.commit()
        print("[DB] route_telemetry table ready.")
    except Exception as exc:
        print(f"[DB] Table creation failed (non-fatal): {exc}")
    finally:
        conn.close()


def log_telemetry(
    source_node: str,
    destination_node: str,
    routing_mode: str,
    total_distance_km: float,
    travel_time_min: int,
    estimated_cost_inr: int,
    path_taken: list,
    fallback_triggered: bool = False,
):
    """
    Insert a route telemetry record.

    Entire operation is wrapped in try/except — if MySQL is stopped,
    this function returns silently without raising.
    """
    conn = _get_connection()
    if conn is None:
        return

    try:
        with conn.cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO route_telemetry
                    (source_node, destination_node, routing_mode,
                     total_distance_km, travel_time_min, estimated_cost_inr,
                     path_taken, fallback_triggered, query_timestamp)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    source_node,
                    destination_node,
                    routing_mode,
                    round(total_distance_km, 1),
                    travel_time_min,
                    estimated_cost_inr,
                    json.dumps(path_taken),
                    fallback_triggered,
                    datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S"),
                ),
            )
        conn.commit()
    except Exception as exc:
        print(f"[DB] Telemetry insert failed (non-fatal): {exc}")
    finally:
        conn.close()
