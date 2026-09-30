"""
app.py - Hybrid Routing Engine (Flask)
=======================================

Enterprise-grade traffic routing combining:
  - Live TomTom Traffic Flow telemetry
  - Historical ML predictions (best_traffic_model.pkl)
  - Dijkstra pathfinding on a 30-node Bengaluru road graph
  - MySQL telemetry logging (resilient)
  - Per-edge congestion color prediction (/api/predict_route)
"""

import json
import math
import os
import random
import time as _time
import uuid
from datetime import datetime, timedelta, timezone

import networkx as nx
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

from bengaluru_graph import load_graph
from database import init_db, log_telemetry

# ── Constants ────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_traffic_model.pkl")

# ── Realistic Bengaluru speed profiles (km/h) ───────────────────────
# Based on Bengaluru Traffic Police / Google Maps data
SPEED_PROFILES = {
    # hour_range: (weekday_speed, weekend_speed)
    "night":      (35, 40),   # 22:00 – 06:00
    "early_am":   (30, 38),   # 06:00 – 08:00
    "morning_pk": (14, 28),   # 08:00 – 11:30  (heavy congestion)
    "midday":     (22, 30),   # 11:30 – 16:00
    "evening_pk": (12, 25),   # 17:00 – 20:30  (worst congestion)
    "late_eve":   (25, 32),   # 20:30 – 22:00
    "pre_pk":     (18, 28),   # 16:00 – 17:00
}

# ── Congestion color thresholds ──────────────────────────────────────
GREEN_SPEED_THRESHOLD  = 25  # km/h — above this = Optimal / Clear
YELLOW_SPEED_THRESHOLD = 15  # km/h — above this but ≤ 25 = Moderate
# Below 15 km/h = Red / Heavy Traffic


def _get_speed_for_time(hour: float, is_weekend: bool) -> float:
    """Return realistic average speed (km/h) for a given hour."""
    idx = 1 if is_weekend else 0
    if hour < 6:
        return SPEED_PROFILES["night"][idx]
    elif hour < 8:
        return SPEED_PROFILES["early_am"][idx]
    elif hour < 11.5:
        return SPEED_PROFILES["morning_pk"][idx]
    elif hour < 16:
        return SPEED_PROFILES["midday"][idx]
    elif hour < 17:
        return SPEED_PROFILES["pre_pk"][idx]
    elif hour < 20.5:
        return SPEED_PROFILES["evening_pk"][idx]
    elif hour < 22:
        return SPEED_PROFILES["late_eve"][idx]
    else:
        return SPEED_PROFILES["night"][idx]


# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Fleet Routing State (in-memory) ──────────────────────────────────
# Tracks how many active vehicles are traversing each edge
edge_counters = {}   # {(node_u, node_v): int}
# Tracks per-session data: assigned edges & last telemetry timestamp
active_sessions = {} # {session_id: {'assigned_edges': [(u,v), ...], 'last_seen': float}}

# ── Load Graph ───────────────────────────────────────────────────────
GRAPH = load_graph()
print(f"[GRAPH] Loaded: {GRAPH.number_of_nodes()} nodes, {GRAPH.number_of_edges()} edges")

# ── Load ML Model ────────────────────────────────────────────────────
ML_MODEL = None
ML_MODEL_NAME = None

try:
    import joblib
    if os.path.exists(MODEL_PATH):
        ML_MODEL = joblib.load(MODEL_PATH)
        # Extract model name from pipeline
        if hasattr(ML_MODEL, "named_steps"):
            reg = ML_MODEL.named_steps.get("regressor")
            if reg is not None:
                ML_MODEL_NAME = type(reg).__name__
            else:
                ML_MODEL_NAME = "Unknown"
        else:
            ML_MODEL_NAME = type(ML_MODEL).__name__
        print(f"[ML] Model loaded: {ML_MODEL_NAME}")
    else:
        print(f"[ML] No model found at {MODEL_PATH}. Run train_models.py first.")
except Exception as exc:
    print(f"[ML] Failed to load model: {exc}")

# ── Detect model feature schema ─────────────────────────────────────
# The model may have been trained on Uber Movement data (numeric only)
# or on the original Bengaluru CSV (with categorical features).
# We inspect the preprocessor's column transformer to find out.
_MODEL_FEATURE_NAMES = None
try:
    if ML_MODEL and hasattr(ML_MODEL, "named_steps"):
        pre = ML_MODEL.named_steps.get("preprocessor")
        if pre is not None and hasattr(pre, "transformers"):
            names = []
            for name, transformer, cols in pre.transformers:
                if isinstance(cols, list):
                    names.extend(cols)
            _MODEL_FEATURE_NAMES = names
            print(f"[ML] Feature columns: {_MODEL_FEATURE_NAMES}")
except Exception:
    pass

# ── Init Database ────────────────────────────────────────────────────
init_db()


# =====================================================================
# HELPER FUNCTIONS
# =====================================================================

def _get_ist_datetime(departure_time: str = None) -> datetime:
    """Parse departure_time string or return current IST."""
    if departure_time:
        try:
            dt = datetime.fromisoformat(departure_time)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=IST)
            return dt
        except (ValueError, TypeError):
            pass
    return datetime.now(IST)


def _is_peak_hour(hour: float) -> int:
    """Return 1 if hour falls within morning (08-11:30) or evening (17-20:30) peak."""
    return int((8 <= hour < 11.5) or (17 <= hour < 20.5))


def _congestion_color(speed_kmh: float) -> str:
    """Return Green / Yellow / Red hex color based on predicted speed."""
    if speed_kmh > GREEN_SPEED_THRESHOLD:
        return "#4A9B6B"   # Muted green — Clear / Optimal
    elif speed_kmh > YELLOW_SPEED_THRESHOLD:
        return "#C4963E"   # Muted amber — Moderate
    else:
        return "#B84C3E"   # Muted red — Heavy Traffic


def _congestion_label(speed_kmh: float) -> str:
    """Return human-readable congestion label."""
    if speed_kmh > GREEN_SPEED_THRESHOLD:
        return "Clear"
    elif speed_kmh > YELLOW_SPEED_THRESHOLD:
        return "Moderate"
    else:
        return "Heavy"


def _query_tomtom_speed(lat: float, lng: float) -> tuple[float, float] | None:
    """
    Query TomTom Traffic Flow API for current speed at a coordinate.
    Returns (speed_kmh, free_flow_speed_kmh) or None on failure.
    """
    if not TOMTOM_API_KEY:
        return None

    url = (
        f"https://api.tomtom.com/traffic/services/4/flowSegmentData"
        f"/absolute/10/json"
        f"?point={lat},{lng}"
        f"&key={TOMTOM_API_KEY}"
    )
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        data = resp.json()
        flow = data.get("flowSegmentData", {})
        speed = flow.get("currentSpeed")
        free_flow = flow.get("freeFlowSpeed", speed)
        if speed and speed > 0:
            return (float(speed), float(free_flow) if free_flow else float(speed))
    except Exception:
        pass
    return None


def _assign_live_weights(graph: nx.Graph) -> tuple[bool, str]:
    """
    Assign edge weights using TomTom live traffic data.
    Returns (fallback_triggered, mode_label).
    """
    fallback = False
    now = datetime.now(IST)
    hour = now.hour + now.minute / 60.0
    is_weekend = now.weekday() >= 5

    for u, v, data in graph.edges(data=True):
        dist_km = data.get("distance_km", 1.0)

        # Use midpoint of edge for TomTom query
        u_data = graph.nodes[u]
        v_data = graph.nodes[v]
        mid_lat = (u_data["lat"] + v_data["lat"]) / 2
        mid_lng = (u_data["lng"] + v_data["lng"]) / 2

        result = _query_tomtom_speed(mid_lat, mid_lng)

        if result is not None:
            speed, free_flow = result
            weight = (dist_km / speed) * 60.0
            congestion = free_flow / speed if speed > 0 else 1.0
        else:
            # Fallback to realistic speed profile
            fallback = True
            speed = _get_speed_for_time(hour, is_weekend)
            # Deterministic per-edge variation (±15%) for realism
            edge_hash = hash((u, v)) % 10000 / 10000.0  # 0.0 – 1.0
            variation = 0.85 + edge_hash * 0.30  # 0.85 – 1.15
            effective_speed = speed * variation
            weight = (dist_km / effective_speed) * 60.0
            # Estimate congestion from speed vs free-flow
            free_flow_speed = _get_speed_for_time(3, False)  # 3 AM = free flow
            congestion = free_flow_speed / effective_speed

        data["weight"] = round(weight, 2)
        data["congestion_multiplier"] = round(max(1.0, min(3.0, congestion)), 2)

    mode_label = "Live Telemetry Feed"
    if fallback and ML_MODEL is not None:
        mode_label = "ML Fallback Active"
    elif fallback:
        mode_label = "Speed Profile Estimate"

    return fallback, mode_label


def _predict_edge_travel_time(u: str, v: str, dist_km: float,
                               hour_of_day: float, day_of_week: int,
                               is_peak: int, base_speed: float) -> float:
    """
    Predict travel time (minutes) for a single edge using the realistic
    Bengaluru speed profile, optionally adjusted by an ML congestion factor.

    The base_speed already reflects time-of-day congestion (e.g. 12 km/h
    during evening peak). The ML model, if available, provides an additional
    congestion multiplier to fine-tune per-edge.
    """
    import pandas as pd

    # Start with the realistic speed-profile estimate
    effective_speed = max(5.0, base_speed)

    # If ML model is loaded, use it to derive a congestion adjustment
    if ML_MODEL is not None:
        try:
            if _MODEL_FEATURE_NAMES is not None:
                features = {}
                for col in _MODEL_FEATURE_NAMES:
                    if col == "hour_of_day":
                        features[col] = [hour_of_day]
                    elif col == "day_of_week":
                        features[col] = [day_of_week]
                    elif col == "is_peak_hour":
                        features[col] = [is_peak]
                    elif col == "sourceid":
                        features[col] = [hash(u) % 10000]
                    elif col == "dstid":
                        features[col] = [hash(v) % 10000]
                    elif col in ("standard_deviation_travel_time",
                                 "geometric_standard_deviation_travel_time"):
                        features[col] = [0.0]
                    elif col == "geometric_mean_travel_time":
                        features[col] = [(dist_km / max(base_speed, 5)) * 3600]
                    else:
                        features[col] = [0.0]
                df = pd.DataFrame(features)
            else:
                df = pd.DataFrame({
                    "hour_of_day": [hour_of_day],
                    "day_of_week": [day_of_week],
                    "is_peak_hour": [is_peak],
                    "sourceid": [hash(u) % 10000],
                    "dstid": [hash(v) % 10000],
                })

            prediction = float(ML_MODEL.predict(df)[0])

            # The model predicts travel_time_min (mean_travel_time / 60).
            # Use it to derive a congestion multiplier relative to free-flow.
            # Free-flow baseline: ~35 km/h → 5 km in ~8.6 min.
            free_flow_estimate = (dist_km / 35.0) * 60.0  # minutes at free flow
            if free_flow_estimate > 0 and prediction > 0:
                ml_congestion = prediction / max(free_flow_estimate, 0.5)
                ml_congestion = max(0.8, min(3.0, ml_congestion))
                # Blend: 70% speed-profile + 30% ML adjustment
                effective_speed = effective_speed / (0.7 + 0.3 * ml_congestion)
                effective_speed = max(5.0, effective_speed)

        except Exception as e:
            print(f"[ML] Prediction failed for {u}->{v}: {e}")

    # time = distance / speed, converted to minutes
    travel_time_min = (dist_km / effective_speed) * 60.0
    return max(0.5, travel_time_min)


def _assign_predictive_weights(graph: nx.Graph, departure: datetime) -> tuple[bool, str]:
    """
    Assign edge weights using the ML model for congestion prediction
    combined with time-aware speed profiles for realistic travel times.
    Returns (fallback_triggered, mode_label).
    """
    hour_of_day = departure.hour + departure.minute / 60.0
    day_of_week = departure.weekday()
    is_weekend = day_of_week >= 5
    is_peak = _is_peak_hour(hour_of_day)
    base_speed = _get_speed_for_time(hour_of_day, is_weekend)
    free_flow_speed = _get_speed_for_time(3, False)  # 3 AM reference

    if ML_MODEL is None:
        # No ML model — use speed profile heuristic
        for u, v, data in graph.edges(data=True):
            dist_km = data.get("distance_km", 1.0)
            weight = (dist_km / base_speed) * 60.0
            congestion = free_flow_speed / base_speed
            data["weight"] = round(weight, 2)
            data["congestion_multiplier"] = round(max(1.0, min(3.0, congestion)), 2)
        return True, "Heuristic Fallback (No Model)"

    import pandas as pd

    fallback_triggered = False

    for u, v, data in graph.edges(data=True):
        dist_km = data.get("distance_km", 1.0)

        try:
            # Build feature row matching the Uber Movement schema
            # the model was trained on (8 numeric features)
            features = {
                "hour_of_day": [hour_of_day],
                "day_of_week": [day_of_week],
                "is_peak_hour": [is_peak],
                "sourceid": [hash(u) % 10000],
                "dstid": [hash(v) % 10000],
                "standard_deviation_travel_time": [0.0],
                "geometric_mean_travel_time": [(dist_km / max(base_speed, 5)) * 3600],
                "geometric_standard_deviation_travel_time": [0.0],
            }

            df = pd.DataFrame(features)
            predicted_time_min = float(ML_MODEL.predict(df)[0])

            # The model predicts travel_time_min directly.
            # Use it to derive a congestion multiplier relative to free-flow.
            free_flow_estimate = (dist_km / 35.0) * 60.0  # minutes at free flow
            if free_flow_estimate > 0 and predicted_time_min > 0:
                ml_congestion = predicted_time_min / max(free_flow_estimate, 0.5)
                ml_congestion = max(0.8, min(3.0, ml_congestion))
            else:
                ml_congestion = 1.0

            # Compute weight using time-aware speed adjusted by ML congestion
            effective_speed = base_speed / ml_congestion
            effective_speed = max(5.0, effective_speed)  # minimum 5 km/h
            weight = (dist_km / effective_speed) * 60.0

            # Deterministic per-edge variation for realism
            edge_hash = hash((u, v)) % 10000 / 10000.0
            variation = 0.92 + edge_hash * 0.16  # 0.92 – 1.08
            weight *= variation

            congestion = ml_congestion

        except Exception as e:
            # Fallback for this edge: pure speed profile
            print(f"[ML] _assign_predictive_weights failed for {u}->{v}: {e}")
            fallback_triggered = True
            weight = (dist_km / base_speed) * 60.0
            congestion = free_flow_speed / base_speed

        data["weight"] = round(max(0.5, weight), 2)
        data["congestion_multiplier"] = round(max(1.0, min(3.0, congestion)), 2)

    model_name = ML_MODEL_NAME or "ML"
    if fallback_triggered:
        return True, f"{model_name} (Partial Fallback)"
    return False, f"{model_name} Historical Model"


def _compute_route(graph: nx.Graph, source: str, destination: str):
    """
    Run Dijkstra and return (path, total_distance_km, travel_time_min).
    Returns (None, 0, 0) if no path exists.
    """
    try:
        path = nx.dijkstra_path(graph, source, destination, weight="weight")
        travel_time = nx.dijkstra_path_length(graph, source, destination, weight="weight")
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None, 0, 0

    # Sum physical distances along path
    total_dist = 0.0
    for i in range(len(path) - 1):
        edge_data = graph.edges[path[i], path[i + 1]]
        total_dist += edge_data.get("distance_km", 0)

    return path, round(total_dist, 1), round(travel_time)


# =====================================================================
# ROUTES
# =====================================================================

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/sw.js")
def service_worker():
    """Serve the Service Worker from root scope so it can control all requests."""
    return send_from_directory(
        os.path.join(app.static_folder),
        "sw.js",
        mimetype="application/javascript",
    )


@app.route("/api/nodes", methods=["GET"])
def get_nodes():
    """Return all graph nodes for dynamic dropdown population."""
    nodes = []
    for node_id, attrs in GRAPH.nodes(data=True):
        nodes.append({
            "id": node_id,
            "name": attrs.get("name", node_id),
            "lat": attrs.get("lat", 0),
            "lng": attrs.get("lng", 0),
        })
    # Sort alphabetically
    nodes.sort(key=lambda n: n["name"])
    return jsonify({"nodes": nodes})


@app.route("/api/predict_route", methods=["POST"])
def predict_route():
    """
    Per-edge ML congestion prediction endpoint.

    Accepts:
        { "source_id": "Silk Board", "destination_id": "Whitefield",
          "departure_time": "2026-09-11T08:30" }

    Returns segments with [[lat1,lon1],[lat2,lon2]], color (hex), and
    per-segment distance/time, plus totals for the dashboard.
    """
    payload = request.json or {}
    source = payload.get("source_id", "").strip()
    destination = payload.get("destination_id", "").strip()
    departure_time = payload.get("departure_time")
    mode = payload.get("mode", "predictive").strip().lower()

    if not source or not destination:
        return jsonify({"error": "source_id and destination_id are required."}), 400
    if source not in GRAPH.nodes:
        return jsonify({"error": f"Unknown source node: {source}"}), 400
    if destination not in GRAPH.nodes:
        return jsonify({"error": f"Unknown destination node: {destination}"}), 400
    if source == destination:
        return jsonify({"error": "Source and destination must differ."}), 400

    # Parse departure time
    departure = _get_ist_datetime(departure_time)
    hour_of_day = departure.hour + departure.minute / 60.0
    day_of_week = departure.weekday()
    is_weekend = day_of_week >= 5
    is_peak = _is_peak_hour(hour_of_day)
    base_speed = _get_speed_for_time(hour_of_day, is_weekend)

    # Work on a copy so concurrent requests don't clash
    G = GRAPH.copy()

    mode_label = "TomTom Live Traffic" if mode == "live" else (ML_MODEL_NAME or "Heuristic")
    fallback_triggered = False

    # ── Assign Weights based on Mode ─────────────────────────────
    for u, v, data in G.edges(data=True):
        dist_km = data.get("distance_km", 1.0)
        predicted_time = -1.0
        
        # Try TomTom first if mode is live
        if mode == "live":
            u_data = G.nodes[u]
            v_data = G.nodes[v]
            mid_lat = (u_data["lat"] + v_data["lat"]) / 2
            mid_lng = (u_data["lng"] + v_data["lng"]) / 2
            result = _query_tomtom_speed(mid_lat, mid_lng)
            
            if result is not None:
                speed, _ = result
                predicted_time = (dist_km / speed) * 60.0
            else:
                fallback_triggered = True

        # Fallback to ML / Heuristic
        if predicted_time < 0:
            predicted_time = _predict_edge_travel_time(
                u, v, dist_km, hour_of_day, day_of_week, is_peak, base_speed
            )
            # Deterministic per-edge variation for realism in predictive mode
            edge_hash = hash((u, v)) % 10000 / 10000.0
            variation = 0.90 + edge_hash * 0.20  # 0.90 – 1.10
            predicted_time *= variation

        # ── Fleet congestion penalty ─────────────────────────────
        fleet_count = edge_counters.get((u, v), 0) + edge_counters.get((v, u), 0)
        if fleet_count > 0:
            predicted_time = predicted_time * (1.15 ** fleet_count)

        data["weight"] = round(max(0.3, predicted_time), 2)

    if fallback_triggered and mode == "live":
        mode_label = f"Live Fallback to {ML_MODEL_NAME or 'Heuristic'}"

    # ── Dijkstra shortest path ───────────────────────────────────
    path, total_distance_km, total_time_min = _compute_route(G, source, destination)

    if path is None:
        return jsonify({"error": "No valid route found between these points."}), 404

    # ── Build per-edge segments with congestion colors ───────────
    segments = []
    total_time_recalc = 0.0
    total_dist_recalc = 0.0

    for i in range(len(path) - 1):
        u, v = path[i], path[i + 1]
        u_data = G.nodes[u]
        v_data = G.nodes[v]
        edge_data = G.edges[u, v]

        dist_km = edge_data.get("distance_km", 1.0)
        edge_time_min = edge_data.get("weight", 1.0)

        # Predicted speed = distance / time (convert min to hours)
        if edge_time_min > 0:
            predicted_speed_kmh = dist_km / (edge_time_min / 60.0)
        else:
            predicted_speed_kmh = 40.0

        color = _congestion_color(predicted_speed_kmh)
        label = _congestion_label(predicted_speed_kmh)

        total_time_recalc += edge_time_min
        total_dist_recalc += dist_km

        segments.append({
            "from": u,
            "to": v,
            "coordinates": [
                [u_data["lat"], u_data["lng"]],
                [v_data["lat"], v_data["lng"]],
            ],
            "road_coords": edge_data.get("road_coords", None),

            "color": color,
            "congestion": label,
            "distance_km": round(dist_km, 2),
            "time_min": round(edge_time_min, 1),
            "speed_kmh": round(predicted_speed_kmh, 1),
        })

    # ── Cost estimation ──────────────────────────────────────────
    total_time_final = round(total_time_recalc)
    total_dist_final = round(total_dist_recalc, 1)
    estimated_cost_inr = round((total_dist_final * 10) + (total_time_final * 2))

    # ── Log telemetry (resilient) ────────────────────────────────
    try:
        log_telemetry(
            source_node=source,
            destination_node=destination,
            routing_mode="predict_route",
            total_distance_km=total_dist_final,
            travel_time_min=total_time_final,
            estimated_cost_inr=estimated_cost_inr,
            path_taken=path,
            fallback_triggered=(ML_MODEL is None),
        )
    except Exception:
        pass  # never crash the API

    return jsonify({
        "path": path,
        "segments": segments,
        "total_distance_km": total_dist_final,
        "travel_time_min": total_time_final,
        "estimated_cost_inr": estimated_cost_inr,
        "model_name": mode_label,  # Use mode_label instead of hardcoded ML_MODEL_NAME
        "departure": departure.isoformat(),
        "is_peak": bool(is_peak),
    })


@app.route("/api/route", methods=["POST"])
def get_route():
    """
    Hybrid routing endpoint (existing).

    Accepts:
        { "source": "...", "destination": "...",
          "mode": "live" | "predictive",
          "departure_time": "2026-09-11T08:30" }
    """
    data = request.json or {}
    source = data.get("source", "").strip()
    destination = data.get("destination", "").strip()
    mode = data.get("mode", "live").strip().lower()
    departure_time = data.get("departure_time")

    if not source or not destination:
        return jsonify({"error": "Source and destination required."}), 400

    if source not in GRAPH.nodes:
        return jsonify({"error": f"Unknown source node: {source}"}), 400
    if destination not in GRAPH.nodes:
        return jsonify({"error": f"Unknown destination node: {destination}"}), 400
    if source == destination:
        return jsonify({"error": "Source and destination must differ."}), 400

    # Work on a copy so concurrent requests don't clash
    G = GRAPH.copy()

    # ── Assign weights based on mode ──────────────────────────────
    fallback_triggered = False
    mode_label = ""

    if mode == "live":
        fallback_triggered, mode_label = _assign_live_weights(G)
        # If all TomTom calls failed, fall back to ML predictive
        if fallback_triggered and ML_MODEL is not None:
            departure = _get_ist_datetime(departure_time)
            _, mode_label = _assign_predictive_weights(G, departure)
            mode_label = "ML Fallback Active"
    else:
        departure = _get_ist_datetime(departure_time)
        fallback_triggered, mode_label = _assign_predictive_weights(G, departure)

    # ── Fleet congestion penalty (anti-herding) ───────────────────
    for u, v, edata in G.edges(data=True):
        fleet_count = edge_counters.get((u, v), 0) + edge_counters.get((v, u), 0)
        if fleet_count > 0:
            edata["weight"] = round(edata["weight"] * (1.15 ** fleet_count), 2)

    # ── Dijkstra pathfinding ──────────────────────────────────────
    path, total_distance_km, travel_time_min = _compute_route(G, source, destination)

    if path is None:
        return jsonify({"error": "No valid route found between these points."}), 404

    # ── Cost estimation ───────────────────────────────────────────
    estimated_cost_inr = round((total_distance_km * 10) + (travel_time_min * 2))

    # ── Build graph data for frontend ─────────────────────────────
    graph_nodes = []
    for node_id in G.nodes:
        attrs = G.nodes[node_id]
        graph_nodes.append({
            "id": node_id,
            "name": attrs.get("name", node_id),
            "lat": attrs.get("lat", 0),
            "lng": attrs.get("lng", 0),
        })

    graph_edges = []
    for u, v, edata in G.edges(data=True):
        graph_edges.append({
            "source": u,
            "target": v,
            "distance_km": edata.get("distance_km", 0),
            "weight": edata.get("weight", 0),
            "congestion_multiplier": edata.get("congestion_multiplier", 1.0),
        })

    # ── Log telemetry (resilient) ─────────────────────────────────
    try:
        log_telemetry(
            source_node=source,
            destination_node=destination,
            routing_mode=mode,
            total_distance_km=total_distance_km,
            travel_time_min=travel_time_min,
            estimated_cost_inr=estimated_cost_inr,
            path_taken=path,
            fallback_triggered=fallback_triggered,
        )
    except Exception:
        pass  # never crash the API

    # ── Response ──────────────────────────────────────────────────
    return jsonify({
        "path": path,
        "total_distance_km": total_distance_km,
        "travel_time_min": travel_time_min,
        "estimated_cost_inr": estimated_cost_inr,
        "mode": mode,
        "mode_label": mode_label,
        "fallback_triggered": fallback_triggered,
        "model_name": ML_MODEL_NAME,
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
    })


# =====================================================================
# FLEET ROUTING ENDPOINTS
# =====================================================================

def _haversine(lat1, lng1, lat2, lng2):
    """Return distance in meters between two lat/lng points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


@app.route("/api/start_trip", methods=["POST"])
def start_trip():
    """
    Register a vehicle on the fleet routing system.

    Accepts:
        { "session_id": "abc-123", "path": ["NodeA", "NodeB", "NodeC"] }

    Increments edge_counters for every edge on the path and stores
    the session in active_sessions.  Multiple concurrent sessions are
    fully supported — this only touches the counters for THIS session.
    """
    payload = request.json or {}
    session_id = payload.get("session_id", "").strip()
    path = payload.get("path", [])

    if not session_id or len(path) < 2:
        return jsonify({"error": "session_id and a path with ≥2 nodes are required."}), 400

    print(f"[START_TRIP] Called: session={session_id!r}, path_nodes={len(path)}, "
          f"currently_active_sessions={len(active_sessions)}, "
          f"current_edge_total={sum(edge_counters.values())}")

    # ── Release this session's OLD edges before re-registering ────
    # This handles the case where the same session_id re-routes mid-trip.
    # IMPORTANT: we only decrement edges belonging to THIS session —
    # other sessions' counters are completely untouched.
    released_count = 0
    if session_id in active_sessions:
        old_edges = active_sessions[session_id].get("assigned_edges", [])
        print(f"[START_TRIP] Re-registering existing session={session_id!r}: "
              f"releasing {len(old_edges)} old edges first")
        for edge in old_edges:
            key = tuple(edge)
            edge_counters[key] = max(0, edge_counters.get(key, 0) - 1)
            if edge_counters[key] == 0:
                edge_counters.pop(key, None)
            released_count += 1

    # ── Register new edges and increment counters ─────────────────
    edges = []
    for i in range(len(path) - 1):
        edge = (path[i], path[i + 1])
        edges.append(edge)
        edge_counters[edge] = edge_counters.get(edge, 0) + 1

    active_sessions[session_id] = {
        "assigned_edges": edges,
        "last_seen": _time.time(),
        "source": path[0],
        "destination": path[-1],
        "registered_at": _time.time(),
    }

    print(f"[START_TRIP] Registered: session={session_id!r}, "
          f"new_edges={len(edges)}, released_old_edges={released_count}, "
          f"total_active_sessions={len(active_sessions)}, "
          f"total_edge_slots={sum(edge_counters.values())}")

    return jsonify({
        "status": "registered",
        "session_id": session_id,
        "edges_tracked": len(edges),
        "total_active_sessions": len(active_sessions),
    })


@app.route("/api/end_trip", methods=["POST"])
def end_trip():
    """
    Explicitly release a session's edge slots from fleet state.

    Accepts:
        { "session_id": "abc-123", "reason": "completed" | "rerouted" | "cancelled" }

    Call this when:
      - Navigation completes (car reaches destination)
      - The user selects a new route (rerouted)
      - The user manually stops navigation (cancelled)

    This is safe to call multiple times for the same session_id;
    if the session is already gone it returns {"status": "already_gone"}.
    Other active sessions are NEVER affected.
    """
    payload = request.json or {}
    session_id = payload.get("session_id", "").strip()
    reason = payload.get("reason", "unknown")

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400

    session = active_sessions.get(session_id)
    if not session:
        print(f"[END_TRIP] Session {session_id!r} not found (already gone or never started). "
              f"reason={reason!r}")
        return jsonify({"status": "already_gone", "session_id": session_id})

    # Release only this session's edges
    edges = session.get("assigned_edges", [])
    for edge in edges:
        key = tuple(edge)
        edge_counters[key] = max(0, edge_counters.get(key, 0) - 1)
        if edge_counters[key] == 0:
            edge_counters.pop(key, None)

    active_sessions.pop(session_id, None)

    print(f"[END_TRIP] Released: session={session_id!r}, reason={reason!r}, "
          f"edges_released={len(edges)}, "
          f"remaining_active_sessions={len(active_sessions)}, "
          f"remaining_edge_slots={sum(edge_counters.values())}")

    return jsonify({
        "status": "released",
        "session_id": session_id,
        "edges_released": len(edges),
        "reason": reason,
        "remaining_active_sessions": len(active_sessions),
    })


@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    """
    Receive a telemetry ping from a driving client.

    Accepts:
        { "session_id": "abc-123", "lat": 12.97, "lng": 77.59 }

    Compares the current position to the assigned route edges.
    Returns {"status": "on_track"} or {"status": "deviated"}.
    On deviation the session's edges are decremented from edge_counters.
    """
    payload = request.json or {}
    session_id = payload.get("session_id", "").strip()
    lat = payload.get("lat")
    lng = payload.get("lng")

    # ── Debug: log every incoming ping to Render stdout ──────────
    print(f"[TELEMETRY] Ping received: session={session_id!r}, "
          f"lat={lat}, lng={lng}, "
          f"active_sessions={len(active_sessions)}, "
          f"tracked_edges={len(edge_counters)}")

    if not session_id or lat is None or lng is None:
        print(f"[TELEMETRY] ERROR: Missing required fields "
              f"(session_id={session_id!r}, lat={lat}, lng={lng})")
        return jsonify({"error": "session_id, lat, and lng are required."}), 400

    session = active_sessions.get(session_id)
    if not session:
        # This means /api/start_trip was never called for this session,
        # or the session was already evicted (deviation / stale cleanup).
        print(f"[TELEMETRY] WARNING: No active session for id={session_id!r}. "
              f"Known sessions: {list(active_sessions.keys())}")
        return jsonify({"status": "no_session"}), 404

    # Update last seen
    session["last_seen"] = _time.time()

    # Check proximity to any assigned edge (within 500 m of either endpoint)
    DEVIATION_THRESHOLD_M = 500
    on_route = False
    closest_dist_m = float("inf")

    for edge in session["assigned_edges"]:
        u_id, v_id = edge
        u_node = GRAPH.nodes.get(u_id, {})
        v_node = GRAPH.nodes.get(v_id, {})

        u_lat, u_lng = u_node.get("lat", 0), u_node.get("lng", 0)
        v_lat, v_lng = v_node.get("lat", 0), v_node.get("lng", 0)

        # Check distance to either endpoint of the edge
        dist_u = _haversine(lat, lng, u_lat, u_lng)
        dist_v = _haversine(lat, lng, v_lat, v_lng)

        # Also check distance to the midpoint
        mid_lat, mid_lng = (u_lat + v_lat) / 2, (u_lng + v_lng) / 2
        dist_mid = _haversine(lat, lng, mid_lat, mid_lng)

        edge_min = min(dist_u, dist_v, dist_mid)
        closest_dist_m = min(closest_dist_m, edge_min)

        if edge_min <= DEVIATION_THRESHOLD_M:
            on_route = True
            break

    if on_route:
        print(f"[TELEMETRY] ON_TRACK: session={session_id!r}, "
              f"pos=({lat:.4f},{lng:.4f}), closest_edge_dist={closest_dist_m:.0f}m, "
              f"active_sessions={len(active_sessions)}, "
              f"edge_counters={sum(edge_counters.values())}")
        return jsonify({"status": "on_track"})

    # ── Deviated: decrement counters and remove session ──────────
    for edge in session["assigned_edges"]:
        key = tuple(edge)
        edge_counters[key] = max(0, edge_counters.get(key, 0) - 1)
        if edge_counters[key] == 0:
            edge_counters.pop(key, None)

    active_sessions.pop(session_id, None)

    print(f"[TELEMETRY] DEVIATED: session={session_id!r}, "
          f"pos=({lat:.4f},{lng:.4f}), closest_edge_dist={closest_dist_m:.0f}m, "
          f"edges_released={len(session['assigned_edges'])}, "
          f"remaining_active_sessions={len(active_sessions)}")

    return jsonify({"status": "deviated"})


@app.route("/api/fleet_status", methods=["GET"])
def fleet_status():
    """
    Debug endpoint: return current fleet state.
    Shows active sessions, edge load counters, and total tracked vehicles.
    """
    # Clean up stale sessions (idle > 10 minutes)
    STALE_THRESHOLD = 600  # seconds
    now = _time.time()
    stale_ids = [
        sid for sid, sdata in active_sessions.items()
        if now - sdata.get("last_seen", 0) > STALE_THRESHOLD
    ]
    for sid in stale_ids:
        for edge in active_sessions[sid].get("assigned_edges", []):
            key = tuple(edge)
            edge_counters[key] = max(0, edge_counters.get(key, 0) - 1)
            if edge_counters[key] == 0:
                edge_counters.pop(key, None)
        active_sessions.pop(sid, None)
        print(f"[FLEET] Stale session cleaned: {sid}")

    active_count = len(active_sessions)
    edge_total   = sum(edge_counters.values())

    print(f"[FLEET_STATUS] active_sessions={active_count}, "
          f"tracked_edges={len(edge_counters)}, "
          f"total_vehicle_edge_slots={edge_total}, "
          f"stale_cleaned={len(stale_ids)}")

    return jsonify({
        # Primary keys (used internally)
        "active_sessions": active_count,
        "tracked_edges": len(edge_counters),
        "total_vehicle_edge_slots": edge_total,
        # Alias keys — match what the frontend / monitoring dashboards expect
        "active_vehicles": active_count,
        "edge_counters": edge_total,
        "sessions": {
            sid: {
                "edges": len(sdata.get("assigned_edges", [])),
                "idle_seconds": round(now - sdata.get("last_seen", now)),
            }
            for sid, sdata in active_sessions.items()
        },
        "edge_loads": {
            f"{u}->{v}": count
            for (u, v), count in edge_counters.items()
        },
        "stale_cleaned": len(stale_ids),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
