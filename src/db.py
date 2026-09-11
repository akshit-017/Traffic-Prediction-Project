"""
Database module — manages the SQLite database for storing anonymous
route search history.

Tables
------
search_history
    id              INTEGER PRIMARY KEY
    source          TEXT        — source node ID (e.g. 'A')
    destination     TEXT        — destination node ID (e.g. 'G')
    source_name     TEXT        — human-readable source name
    dest_name       TEXT        — human-readable destination name
    optimal_route   TEXT        — optimal path as JSON array
    travel_time     REAL        — effective travel time in minutes
    ml_model_used   INTEGER     — 1 if ML model was active, 0 otherwise
    created_at      TEXT        — ISO-8601 timestamp (IST)
"""

import os
import sqlite3
import json
from datetime import datetime, timedelta, timezone

IST = timezone(timedelta(hours=5, minutes=30))

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_PROJECT_ROOT, 'traffic_history.db')

# Node ID → human-readable name mapping
NODE_NAMES = {
    'A': 'MG Road',
    'B': 'Indiranagar',
    'C': 'Koramangala',
    'D': 'HSR Layout',
    'E': 'Jayanagar',
    'F': 'JP Nagar',
    'G': 'Electronic City',
}


def _get_connection():
    """Return a new SQLite connection with row factory enabled."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create the search_history table if it doesn't already exist."""
    conn = _get_connection()
    try:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS search_history (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                source          TEXT    NOT NULL,
                destination     TEXT    NOT NULL,
                source_name     TEXT    NOT NULL,
                dest_name       TEXT    NOT NULL,
                optimal_route   TEXT    NOT NULL,
                travel_time     REAL    NOT NULL,
                ml_model_used   INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT    NOT NULL
            )
        ''')
        conn.commit()
    finally:
        conn.close()


def save_search(source: str, destination: str, optimal_route: list,
                travel_time: float, ml_model_used: bool):
    """
    Persist a single route search to the database.

    Parameters
    ----------
    source : str          Node ID (e.g. 'A')
    destination : str     Node ID (e.g. 'G')
    optimal_route : list  List of node IDs forming the optimal path
    travel_time : float   Effective travel time in minutes
    ml_model_used : bool  Whether the ML model powered the prediction
    """
    conn = _get_connection()
    try:
        conn.execute(
            '''INSERT INTO search_history
               (source, destination, source_name, dest_name,
                optimal_route, travel_time, ml_model_used, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)''',
            (
                source,
                destination,
                NODE_NAMES.get(source, source),
                NODE_NAMES.get(destination, destination),
                json.dumps(optimal_route),
                round(travel_time, 1),
                1 if ml_model_used else 0,
                datetime.now(IST).isoformat(),
            )
        )
        conn.commit()
    finally:
        conn.close()


def get_recent_searches(limit: int = 20) -> list[dict]:
    """
    Return the most recent searches, newest first.

    Parameters
    ----------
    limit : int  Maximum number of rows to return (default 20).

    Returns
    -------
    list[dict]  Each dict contains: id, source, destination, source_name,
                dest_name, optimal_route (list), travel_time, ml_model_used,
                created_at.
    """
    conn = _get_connection()
    try:
        rows = conn.execute(
            '''SELECT * FROM search_history
               ORDER BY id DESC
               LIMIT ?''',
            (limit,)
        ).fetchall()

        results = []
        for row in rows:
            entry = dict(row)
            entry['optimal_route'] = json.loads(entry['optimal_route'])
            entry['ml_model_used'] = bool(entry['ml_model_used'])
            results.append(entry)
        return results
    finally:
        conn.close()


def get_search_count() -> int:
    """Return the total number of searches stored."""
    conn = _get_connection()
    try:
        count = conn.execute(
            'SELECT COUNT(*) FROM search_history'
        ).fetchone()[0]
        return count
    finally:
        conn.close()


def clear_history():
    """Delete all search history records."""
    conn = _get_connection()
    try:
        conn.execute('DELETE FROM search_history')
        conn.commit()
    finally:
        conn.close()
