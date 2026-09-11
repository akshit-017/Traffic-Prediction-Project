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
import os
import random
from datetime import datetime, timedelta, timezone

import networkx as nx
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request

from bengaluru_graph import load_graph
from database import init_db, log_telemetry

# ── Constants ────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY", "s7siMcrVMhjeY4ljVp3xFcIoVSacpr9t")
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
    return int((8 <= hour <= 11.5) or (17 <= hour <= 20.5))


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


def _query_tomtom_speed(lat: float, lng: float) -> float | None:
    """
    Query TomTom Traffic Flow API for current speed at a coordinate.
    Returns speed in km/h or None on failure.
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
            return float(speed), float(free_flow) if free_flow else float(speed)
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
            # Add per-edge variation (±15%) for realism
            variation = random.uniform(0.85, 1.15)
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

    for u, v, data in graph.edges(data=True):
        dist_km = data.get("distance_km", 1.0)

        try:
            # Build a feature row matching the training schema
            features = {
                "hour_of_day": [hour_of_day],
                "day_of_week": [day_of_week],
                "is_peak_hour": [is_peak],
                "area name": [u],
                "road/intersection name": [f"{u}-{v}"],
                "weather conditions": ["Clear"],
                "average speed": [base_speed],
                "traffic volume": [30000.0 * (1.4 if is_peak else 0.8)],
                "congestion level": [70.0 if is_peak else 35.0],
                "road capacity utilization": [80.0 if is_peak else 45.0],
            }

            df = pd.DataFrame(features)
            predicted_tti_scaled = ML_MODEL.predict(df)[0]

            # The model predicts travel_time_index * 10.
            # Convert back to TTI (congestion multiplier).
            predicted_tti = predicted_tti_scaled / 10.0
            predicted_tti = max(0.8, min(2.8, predicted_tti))

            # Compute weight using time-aware speed adjusted by ML congestion
            effective_speed = base_speed / predicted_tti
            effective_speed = max(5.0, effective_speed)  # minimum 5 km/h
            weight = (dist_km / effective_speed) * 60.0

            # Add small per-edge variation for realism
            variation = random.uniform(0.92, 1.08)
            weight *= variation

            congestion = predicted_tti

        except Exception:
            # Fallback for this edge: pure speed profile
            weight = (dist_km / base_speed) * 60.0
            congestion = free_flow_speed / base_speed

        data["weight"] = round(max(0.5, weight), 2)
        data["congestion_multiplier"] = round(max(1.0, min(3.0, congestion)), 2)

    model_name = ML_MODEL_NAME or "ML"
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
            # Add small per-edge randomness for realism in predictive mode
            variation = random.uniform(0.90, 1.10)
            predicted_time *= variation

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


if __name__ == "__main__":
    app.run(debug=True, port=5000)
