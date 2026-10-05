"""
app.py - Hybrid Routing Engine (Flask)
=======================================

Enterprise-grade traffic routing combining:
  - Live TomTom Traffic Flow telemetry
  - Historical ML predictions (best_traffic_model.pkl)
  - Dijkstra pathfinding on a 30-node Bengaluru road graph
  - MySQL telemetry logging (resilient)
  - Per-edge congestion color prediction (/api/predict_route)
  - Fleet anti-herding via exponential edge penalties (1.15 ** fleet_count)
"""

import json
import math
import os
import random
import threading
import time as _time
import uuid
from datetime import datetime, timedelta, timezone

import networkx as nx
import numpy as np
import requests
from flask import Flask, jsonify, render_template, request, send_from_directory

from bengaluru_graph import load_graph
from database import init_db, log_telemetry
from route_optimization import (
    compute_weighted_graph,
    find_optimal_route,
    is_peak_hour,
)

# ── Constants ────────────────────────────────────────────────────────
IST = timezone(timedelta(hours=5, minutes=30))
TOMTOM_API_KEY = os.environ.get("TOMTOM_API_KEY")
MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
MODEL_PATH = os.path.join(MODELS_DIR, "best_traffic_model.pkl")

# ── Master Toggle: Live API vs. Pure ML Prediction (Spec §1) ────────
# When False: bypasses TomTom entirely, defaults to the trained
# HistGradientBoostingRegressor for all edge weight predictions.
# When True: TomTom live speeds are attempted first, ML is the fallback.
# To enforce pure ML prediction mode, set this to False explicitly.
USE_LIVE_API = bool(TOMTOM_API_KEY)

# ── Congestion color thresholds ──────────────────────────────────────
GREEN_SPEED_THRESHOLD  = 25  # km/h — above this = Optimal / Clear
YELLOW_SPEED_THRESHOLD = 15  # km/h — above this but ≤ 25 = Moderate
# Below 15 km/h = Red / Heavy Traffic


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


# ── Flask App ────────────────────────────────────────────────────────
app = Flask(__name__)

# ── Fleet Routing State (in-memory) ──────────────────────────────────
# Tracks how many active vehicles are traversing each edge
edge_counters = {}   # {(node_u, node_v): int}
# Tracks per-session data: assigned edges & last telemetry timestamp
active_sessions = {} # {session_id: {'assigned_edges': [(u,v), ...], 'last_seen': float}}
# Thread lock: Flask on Render uses threads — this prevents concurrent
# requests from corrupting edge_counters / active_sessions.
_fleet_lock = threading.Lock()

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


def _haversine(lat1, lng1, lat2, lng2):
    """Return distance in meters between two lat/lng points."""
    R = 6_371_000  # Earth radius in meters
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lng2 - lng1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _point_to_segment_dist(lat, lng, lat1, lng1, lat2, lng2):
    """
    Calculate shortest distance in meters from point (lat, lng) to line segment 
    (lat1, lng1)-(lat2, lng2) using an equirectangular projection approximation.
    """
    R = 6371000 # meters
    lat, lng, lat1, lng1, lat2, lng2 = map(math.radians, [lat, lng, lat1, lng1, lat2, lng2])
    cos_lat = math.cos((lat1 + lat2) / 2.0)
    
    x, y = lng * cos_lat, lat
    x1, y1 = lng1 * cos_lat, lat1
    x2, y2 = lng2 * cos_lat, lat2
    
    dx, dy = x2 - x1, y2 - y1
    length_sq = dx*dx + dy*dy
    
    if length_sq == 0:
        return _haversine(math.degrees(lat), math.degrees(lng), math.degrees(lat1), math.degrees(lng1))
        
    t = max(0.0, min(1.0, ((x - x1) * dx + (y - y1) * dy) / length_sq))
    return math.hypot(x - (x1 + t * dx), y - (y1 + t * dy)) * R


def _update_session_edges(session_id: str, new_edges: list = None):
    """
    Unified helper to update a session's edge counters. Thread-safe.
    Pass new_edges (list of edge tuples) to register, or None to end trip.
    """
    with _fleet_lock:
        session = active_sessions.get(session_id, {})
        old_edges = session.get("assigned_edges", [])
        
        for edge in old_edges:
            key = tuple(edge)
            edge_counters[key] = max(0, edge_counters.get(key, 0) - 1)
            if edge_counters[key] == 0:
                edge_counters.pop(key, None)
                
        if not new_edges:
            active_sessions.pop(session_id, None)
            return len(old_edges)
            
        for edge in new_edges:
            key = tuple(edge)
            edge_counters[key] = edge_counters.get(key, 0) + 1
            
        if session_id not in active_sessions:
            active_sessions[session_id] = {"registered_at": _time.time()}
            
        active_sessions[session_id]["assigned_edges"] = new_edges
        active_sessions[session_id]["last_seen"] = _time.time()
        return len(new_edges)


# ── Background Stale Session Cleanup ─────────────────────────────────
def _cleanup_stale_sessions_worker():
    """Background daemon to clear out ghost users over 600s old."""
    while True:
        _time.sleep(60)
        with _fleet_lock:
            now = _time.time()
            stale_ids = [
                sid for sid, sdata in active_sessions.items()
                if now - sdata.get("last_seen", now) > 600
            ]
        for sid in stale_ids:
            _update_session_edges(sid, None)
            print(f"[FLEET] Auto-cleaned ghost session: {sid}")

threading.Thread(target=_cleanup_stale_sessions_worker, daemon=True).start()


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

    # Work on a copy so concurrent requests don't clash
    G = GRAPH.copy()

    # ── THREAD-SAFE: Snapshot edge_counters under lock (Spec §2) ─────
    # This prevents data races when multiple Flask threads compute
    # routes concurrently while navigation sessions modify counters.
    with _fleet_lock:
        counter_snapshot = dict(edge_counters)

    # ── Determine if live API should be used (Spec §1) ───────────────
    # USE_LIVE_API is the master toggle. mode="live" from the frontend
    # is only honored when the toggle is True. When False, all routing
    # defaults to the trained HistGradientBoostingRegressor model.
    use_live = USE_LIVE_API and (mode == "live")

    # ── Unified weight computation pipeline ──────────────────────────
    # compute_weighted_graph() handles the entire pipeline in one loop:
    #   1. ML predicted_time for each edge (Historic Core)
    #   2. Optional TomTom live override (if use_live=True)
    #   3. Fleet fusion: final_weight = predicted_time * (1.15 ** fleet_count)
    #   4. Assignment of final_weight to edge['weight']
    mode_label, fallback_triggered = compute_weighted_graph(
        graph=G,
        departure=departure,
        ml_model=ML_MODEL,
        model_feature_names=_MODEL_FEATURE_NAMES,
        edge_counter_snapshot=counter_snapshot,
        use_live_api=use_live,
        tomtom_api_key=TOMTOM_API_KEY,
    )

    # ── Dijkstra shortest path (Spec §3) ─────────────────────────────
    # nx.dijkstra_path(G, source, target, weight='weight') strictly uses
    # the final_weight computed above to determine the optimal route.
    # The optimal path array is returned for the /api/start_trip lock-in.
    path, total_distance_km, total_time_min = find_optimal_route(
        G, source, destination
    )

    if path is None:
        return jsonify({"error": "No valid route found between these points."}), 404

    # ── Build per-edge segments with congestion colors ───────────────
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
            fallback_triggered=fallback_triggered,
        )
    except Exception as e:
        print(f"[MySQL FALLBACK] Database offline. Route returned to user. Error: {e}")

    hour_of_day = departure.hour + departure.minute / 60.0

    return jsonify({
        "path": path,
        "segments": segments,
        "total_distance_km": total_dist_final,
        "travel_time_min": total_time_final,
        "estimated_cost_inr": estimated_cost_inr,
        "model_name": mode_label,
        "departure": departure.isoformat(),
        "is_peak": bool(is_peak_hour(hour_of_day)),
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
    departure = _get_ist_datetime(departure_time)

    # ── THREAD-SAFE: Snapshot edge_counters under lock (Spec §2) ─────
    with _fleet_lock:
        counter_snapshot = dict(edge_counters)

    # ── Determine if live API should be used (Spec §1) ───────────────
    use_live = USE_LIVE_API and (mode == "live")

    # ── Unified weight computation + fleet fusion ────────────────────
    mode_label, fallback_triggered = compute_weighted_graph(
        graph=G,
        departure=departure,
        ml_model=ML_MODEL,
        model_feature_names=_MODEL_FEATURE_NAMES,
        edge_counter_snapshot=counter_snapshot,
        use_live_api=use_live,
        tomtom_api_key=TOMTOM_API_KEY,
    )

    # ── Dijkstra pathfinding (Spec §3) ───────────────────────────────
    path, total_distance_km, travel_time_min = find_optimal_route(
        G, source, destination
    )

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
    except Exception as e:
        print(f"[MySQL FALLBACK] Database offline. Route returned to user. Error: {e}")

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
# FLEET ROUTING ENDPOINTS (Spec §4 + §5)
# =====================================================================

@app.route("/api/start_trip", methods=["POST"])
def start_trip():
    """
    Start Trip (Spec §4):
    - Server generates a UUID session_id
    - Acquires the lock
    - Appends the session_id to active_sessions
    - Strictly increments edge_counters only for the edges in the chosen path
    - Stores destination coordinates for backend proximity check
    """
    payload = request.json or {}
    path = payload.get("path", [])

    if len(path) < 2:
        return jsonify({"error": "A path with >=2 nodes is required."}), 400

    # ── Server generates the UUID session_id (Spec §4) ────────────
    # The client does NOT send session_id — it receives the server-generated
    # one in the response and uses it for all subsequent telemetry/end_trip calls.
    session_id = f"ses-{uuid.uuid4()}"

    edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)]

    # Acquire lock, increment edge_counters for chosen path, register session
    edges_tracked = _update_session_edges(session_id, edges)

    # ── Store destination coordinates for telemetry proximity check (Spec §5f) ──
    # The /api/telemetry endpoint will use these to detect arrival (< 50m).
    dest_node = path[-1]
    dest_data = GRAPH.nodes.get(dest_node, {})
    with _fleet_lock:
        active_sessions[session_id]["dest_lat"] = dest_data.get("lat")
        active_sessions[session_id]["dest_lng"] = dest_data.get("lng")
        active_sessions[session_id]["path"] = path

    print(f"[START_TRIP] Registered: session={session_id!r}, edges_tracked={edges_tracked}")
    return jsonify({
        "status": "registered",
        "session_id": session_id,   # Client must use this for telemetry & end_trip
        "edges_tracked": edges_tracked,
    })


@app.route("/api/end_trip", methods=["POST"])
def end_trip():
    """
    End Trip (Spec §4):
    - Acquires the lock
    - Decrements edge_counters for this session_id's path
    - Deletes the session
    - Zero global resets allowed — only per-session cleanup
    """
    payload = request.json or {}
    session_id = payload.get("session_id", "").strip()
    reason = payload.get("reason", "unknown")

    if not session_id:
        return jsonify({"error": "session_id is required."}), 400

    with _fleet_lock:
        if session_id not in active_sessions:
            return jsonify({"status": "already_gone", "session_id": session_id})

    edges_released = _update_session_edges(session_id, None)
    
    print(f"[END_TRIP] Released: session={session_id!r}, reason={reason!r}, edges_released={edges_released}")
    return jsonify({
        "status": "released",
        "session_id": session_id,
        "edges_released": edges_released,
        "reason": reason,
    })


@app.route("/api/telemetry", methods=["POST"])
def telemetry():
    """
    Telemetry endpoint (Spec §5d-5f):

    Backend Snapping (Cross-Track Math):
      Calculates the perpendicular equirectangular distance from the
      raw GPS ping to the mathematical line segment of the active route.

    Stop Conditions:
      - If distance to final destination < 50 meters:
          Auto-terminate, release session, return {"status": "arrived"}
      - If cross-track deviation > 100 meters:
          Release session, return {"status": "deviated"}
      - Otherwise:
          Return {"status": "on_track"}
    """
    payload = request.json or {}
    session_id = payload.get("session_id", "").strip()
    lat = payload.get("lat")
    lng = payload.get("lng")

    if not session_id or lat is None or lng is None:
        return jsonify({"error": "session_id, lat, and lng are required."}), 400

    # 1. Threading Deadlock fix: Copy assigned_edges and release lock immediately
    with _fleet_lock:
        session = active_sessions.get(session_id)
        if not session:
            return jsonify({"status": "no_session"}), 404
        edges = list(session.get("assigned_edges", []))
        dest_lat = session.get("dest_lat")
        dest_lng = session.get("dest_lng")
        session["last_seen"] = _time.time()

    # ── Stop Condition: Destination Reached (< 50 meters) (Spec §5f) ──
    # Check BEFORE deviation so arriving drivers aren't flagged as deviated
    # when they're simply at the destination but off the last edge segment.
    if dest_lat is not None and dest_lng is not None:
        dist_to_dest = _haversine(lat, lng, dest_lat, dest_lng)
        if dist_to_dest < 50:
            # Auto-terminate: release edge slots and clean up session
            _update_session_edges(session_id, None)
            print(f"[TELEMETRY] ARRIVED: session={session_id!r}, dist={dist_to_dest:.0f}m")
            return jsonify({
                "status": "arrived",
                "distance_to_dest_m": round(dist_to_dest),
            })

    # ── Stop Condition: Cross-Track Deviation (> 100 meters) (Spec §5d-5e) ──
    DEVIATION_THRESHOLD_M = 100
    on_route = False

    # 2. Edge Snapping fix: math done OUTSIDE the lock to prevent blocking
    for edge in edges:
        u_id, v_id = edge
        edge_data = GRAPH.get_edge_data(u_id, v_id, default={})
        road_coords = edge_data.get("road_coords")

        if road_coords and len(road_coords) >= 2:
            # Check distance against actual road polyline segments
            for i in range(len(road_coords) - 1):
                p1 = road_coords[i]
                p2 = road_coords[i + 1]
                dist = _point_to_segment_dist(lat, lng, p1[0], p1[1], p2[0], p2[1])
                if dist <= DEVIATION_THRESHOLD_M:
                    on_route = True
                    break
            if on_route:
                break
        else:
            # Fallback to straight line between nodes
            u_node = GRAPH.nodes.get(u_id, {})
            v_node = GRAPH.nodes.get(v_id, {})
            u_lat, u_lng = u_node.get("lat", 0), u_node.get("lng", 0)
            v_lat, v_lng = v_node.get("lat", 0), v_node.get("lng", 0)

            dist = _point_to_segment_dist(lat, lng, u_lat, u_lng, v_lat, v_lng)
            if dist <= DEVIATION_THRESHOLD_M:
                on_route = True
                break

    if on_route:
        return jsonify({"status": "on_track"})

    # Deviation > 100m: release edge slots and return deviated status
    _update_session_edges(session_id, None)
    print(f"[TELEMETRY] DEVIATED: session={session_id!r}")
    return jsonify({"status": "deviated"})


@app.route("/api/fleet_status", methods=["GET"])
def fleet_status():
    with _fleet_lock:
        STALE_THRESHOLD = 600
        now = _time.time()
        stale_ids = [
            sid for sid, sdata in active_sessions.items()
            if now - sdata.get("last_seen", 0) > STALE_THRESHOLD
        ]
        
    for sid in stale_ids:
        _update_session_edges(sid, None)
        print(f"[FLEET] Stale session cleaned: {sid}")

    with _fleet_lock:
        active_count = len(active_sessions)
        edge_total   = sum(edge_counters.values())
        sessions_copy = {
            sid: {
                "edges": len(sdata.get("assigned_edges", [])),
                "idle_seconds": round(_time.time() - sdata.get("last_seen", _time.time())),
            }
            for sid, sdata in active_sessions.items()
        }
        edge_loads_copy = {
            f"{u}->{v}": count
            for (u, v), count in edge_counters.items()
        }

    return jsonify({
        "active_sessions": active_count,
        "tracked_edges": len(edge_loads_copy),
        "total_vehicle_edge_slots": edge_total,
        "active_vehicles": active_count,
        "edge_counters": edge_total,
        "sessions": sessions_copy,
        "edge_loads": edge_loads_copy,
        "stale_cleaned": len(stale_ids),
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
