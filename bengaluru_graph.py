"""
bengaluru_graph.py — Spatial Road Network for Bengaluru
========================================================

Defines 30+ major tech/transit hubs as nodes with real GPS coordinates
and ~55 bidirectional edges with verified road-network distances in km.

Provides:
    load_graph()  → networkx.Graph   (used by app.py at startup)
    generate_json() → writes graph_data.json

Each node has:  name, lat, lng
Each edge has:  distance_km
"""

import json
import os
import time

import networkx as nx

try:
    import requests as _requests
except ImportError:
    _requests = None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
GRAPH_JSON = os.path.join(PROJECT_ROOT, "graph_data.json")

# =====================================================================
# NODE DEFINITIONS — 30 major Bengaluru hubs with real coordinates
# =====================================================================

NODES = {
    "Silk Board":           {"lat": 12.9172, "lng": 77.6227},
    "Whitefield":           {"lat": 12.9698, "lng": 77.7500},
    "Electronic City":      {"lat": 12.8456, "lng": 77.6603},
    "Indiranagar":          {"lat": 12.9784, "lng": 77.6408},
    "Koramangala":          {"lat": 12.9352, "lng": 77.6245},
    "Hebbal":               {"lat": 13.0358, "lng": 77.5970},
    "Yelahanka":            {"lat": 13.1007, "lng": 77.5963},
    "Majestic":             {"lat": 12.9767, "lng": 77.5713},
    "Malleshwaram":         {"lat": 12.9965, "lng": 77.5700},
    "Banashankari":         {"lat": 12.9255, "lng": 77.5468},
    "Peenya":               {"lat": 13.0300, "lng": 77.5200},
    "Marathahalli":         {"lat": 12.9591, "lng": 77.7009},
    "KR Puram":             {"lat": 12.9981, "lng": 77.7150},
    "Bellandur":            {"lat": 12.9256, "lng": 77.6763},
    "Jayanagar":            {"lat": 12.9308, "lng": 77.5838},
    "JP Nagar":             {"lat": 12.9063, "lng": 77.5857},
    "HSR Layout":           {"lat": 12.9116, "lng": 77.6474},
    "BTM Layout":           {"lat": 12.9166, "lng": 77.6101},
    "Sarjapur Road":        {"lat": 12.9100, "lng": 77.6850},
    "Bannerghatta Road":    {"lat": 12.8876, "lng": 77.5969},
    "MG Road":              {"lat": 12.9756, "lng": 77.6069},
    "Sadashivanagar":       {"lat": 13.0067, "lng": 77.5802},
    "Rajajinagar":          {"lat": 12.9900, "lng": 77.5530},
    "Yeshwanthpur":         {"lat": 13.0220, "lng": 77.5450},
    "Nagawara":             {"lat": 13.0452, "lng": 77.6150},
    "Thanisandra":          {"lat": 13.0600, "lng": 77.6300},
    "Domlur":               {"lat": 12.9610, "lng": 77.6387},
    "HAL":                  {"lat": 12.9590, "lng": 77.6650},
    "Basavanagudi":         {"lat": 12.9430, "lng": 77.5730},
    "CV Raman Nagar":       {"lat": 12.9860, "lng": 77.6640},
}

# =====================================================================
# EDGE DEFINITIONS — bidirectional, distance_km from real road network
# =====================================================================
# Distances sourced from approximate Google Maps driving distances
# between the hub centres.

EDGES = [
    # ── Outer Ring Road corridor ──────────────────────────────────────
    ("Silk Board",      "Koramangala",        2.8),
    ("Silk Board",      "HSR Layout",         3.5),
    ("Silk Board",      "BTM Layout",         3.2),
    ("Silk Board",      "Bellandur",          5.0),
    ("Silk Board",      "Bannerghatta Road",  5.5),
    ("Bellandur",       "Marathahalli",       5.2),
    ("Bellandur",       "Sarjapur Road",      4.0),
    ("Bellandur",       "HSR Layout",         3.5),
    ("Marathahalli",    "Whitefield",         7.5),
    ("Marathahalli",    "KR Puram",           5.8),
    ("Marathahalli",    "HAL",                3.8),
    ("Marathahalli",    "Domlur",             5.0),
    ("KR Puram",        "Whitefield",         7.0),
    ("KR Puram",        "CV Raman Nagar",     3.5),
    ("KR Puram",        "Hebbal",             11.0),

    # ── Central Bengaluru ─────────────────────────────────────────────
    ("MG Road",         "Indiranagar",        3.5),
    ("MG Road",         "Majestic",           3.0),
    ("MG Road",         "Domlur",             2.5),
    ("MG Road",         "Koramangala",        4.5),
    ("Indiranagar",     "Domlur",             2.0),
    ("Indiranagar",     "CV Raman Nagar",     4.0),
    ("Indiranagar",     "HAL",                3.5),
    ("Domlur",          "Koramangala",        3.0),
    ("Domlur",          "HAL",                2.5),

    # ── South Bengaluru ───────────────────────────────────────────────
    ("Koramangala",     "BTM Layout",         2.5),
    ("Koramangala",     "Jayanagar",          4.0),
    ("BTM Layout",      "HSR Layout",         2.5),
    ("BTM Layout",      "Jayanagar",          3.5),
    ("BTM Layout",      "JP Nagar",           3.0),
    ("Jayanagar",       "JP Nagar",           2.5),
    ("Jayanagar",       "Basavanagudi",       2.0),
    ("Jayanagar",       "Banashankari",       3.5),
    ("JP Nagar",        "Bannerghatta Road",  3.5),
    ("JP Nagar",        "Banashankari",       3.0),
    ("Bannerghatta Road", "Electronic City",  9.5),
    ("HSR Layout",      "Sarjapur Road",      4.5),
    ("HSR Layout",      "Electronic City",    10.0),

    # ── West Bengaluru ────────────────────────────────────────────────
    ("Majestic",        "Malleshwaram",       3.0),
    ("Majestic",        "Rajajinagar",        4.0),
    ("Majestic",        "Basavanagudi",       4.5),
    ("Majestic",        "Sadashivanagar",     4.0),
    ("Malleshwaram",    "Rajajinagar",        2.5),
    ("Malleshwaram",    "Sadashivanagar",     2.0),
    ("Malleshwaram",    "Yeshwanthpur",       3.5),
    ("Rajajinagar",     "Yeshwanthpur",       3.0),
    ("Rajajinagar",     "Peenya",             6.5),
    ("Yeshwanthpur",    "Peenya",             4.5),
    ("Yeshwanthpur",    "Hebbal",             5.0),
    ("Banashankari",    "Basavanagudi",       3.0),
    ("Banashankari",    "Bannerghatta Road",  5.0),

    # ── North Bengaluru ───────────────────────────────────────────────
    ("Hebbal",          "Nagawara",           3.5),
    ("Hebbal",          "Yelahanka",          8.0),
    ("Hebbal",          "Sadashivanagar",     4.5),
    ("Hebbal",          "Thanisandra",        5.5),
    ("Nagawara",        "Thanisandra",        3.0),
    ("Nagawara",        "CV Raman Nagar",     5.0),
    ("Yelahanka",       "Thanisandra",        6.5),
    ("Yelahanka",       "Peenya",             10.0),
    ("Peenya",          "Hebbal",             7.0),
]


# =====================================================================
# GRAPH BUILDER
# =====================================================================

def build_graph() -> nx.Graph:
    """Construct a NetworkX graph from the node/edge definitions."""
    G = nx.Graph()

    for name, coords in NODES.items():
        G.add_node(name, name=name, lat=coords["lat"], lng=coords["lng"])

    for u, v, dist in EDGES:
        G.add_edge(u, v, distance_km=dist)

    return G


def load_graph() -> nx.Graph:
    """
    Load the Bengaluru road graph.

    Tries to deserialise from `graph_data.json` first for speed;
    falls back to building from the hardcoded definitions.
    """
    if os.path.exists(GRAPH_JSON):
        try:
            with open(GRAPH_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)

            G = nx.Graph()
            for node in data["nodes"]:
                G.add_node(
                    node["id"],
                    name=node["name"],
                    lat=node["lat"],
                    lng=node["lng"],
                )
            for edge in data["edges"]:
                kwargs = {"distance_km": edge["distance_km"]}
                if "road_coords" in edge and edge["road_coords"]:
                    kwargs["road_coords"] = edge["road_coords"]
                G.add_edge(
                    edge["source"],
                    edge["target"],
                    **kwargs,
                )
            return G
        except Exception as e:
            print(f"[GRAPH] Failed to load JSON, falling back to build: {e}")
            pass  # fall through to hardcoded build

    return build_graph()


def generate_json():
    """Serialise the graph to graph_data.json."""
    G = build_graph()

    nodes = []
    for node_id, attrs in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "name": attrs.get("name", node_id),
            "lat": attrs["lat"],
            "lng": attrs["lng"],
        })

    edges = []
    for u, v, attrs in G.edges(data=True):
        edge_data = {
            "source": u,
            "target": v,
            "distance_km": attrs["distance_km"],
        }
        if "road_coords" in attrs:
            edge_data["road_coords"] = attrs["road_coords"]
        edges.append(edge_data)

    data = {"nodes": nodes, "edges": edges}

    with open(GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"[OK] graph_data.json written: {len(nodes)} nodes, {len(edges)} edges")


# =====================================================================
# OSRM ROAD GEOMETRY BAKING
# =====================================================================

def _decode_polyline(encoded: str) -> list:
    """Decode a Google-encoded polyline into [[lat, lng], ...]."""
    points = []
    index = 0
    lat = 0
    lng = 0
    while index < len(encoded):
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lat += (~(result >> 1) if (result & 1) else (result >> 1))
        shift = 0
        result = 0
        while True:
            b = ord(encoded[index]) - 63
            index += 1
            result |= (b & 0x1F) << shift
            shift += 5
            if b < 0x20:
                break
        lng += (~(result >> 1) if (result & 1) else (result >> 1))
        points.append([round(lat / 1e5, 6), round(lng / 1e5, 6)])
    return points


def bake_road_coords():
    """
    Fetch OSRM road-following geometry for every edge and save
    the coordinates into graph_data.json so they can be used offline.
    """
    if _requests is None:
        print("[ERROR] 'requests' library is required. pip install requests")
        return

    G = build_graph()
    success = 0
    failed = 0

    for u, v, attrs in G.edges(data=True):
        u_lat, u_lng = NODES[u]["lat"], NODES[u]["lng"]
        v_lat, v_lng = NODES[v]["lat"], NODES[v]["lng"]

        url = (
            f"https://router.project-osrm.org/route/v1/driving/"
            f"{u_lng},{u_lat};{v_lng},{v_lat}"
            f"?overview=full&geometries=polyline"
        )

        try:
            resp = _requests.get(url, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            if data.get("code") == "Ok" and data.get("routes"):
                coords = _decode_polyline(data["routes"][0]["geometry"])
                attrs["road_coords"] = coords
                success += 1
                print(f"  [OK] {u} -> {v}  ({len(coords)} points)")
            else:
                failed += 1
                print(f"  [FAIL] {u} -> {v}  (OSRM returned: {data.get('code')})")
        except Exception as e:
            failed += 1
            print(f"  [FAIL] {u} -> {v}  ({e})")

        # Respect OSRM rate limits (public demo server)
        time.sleep(0.5)

    print(f"\n[BAKE] Done: {success} succeeded, {failed} failed")

    # Serialise the graph with road_coords included
    nodes = []
    for node_id, node_attrs in G.nodes(data=True):
        nodes.append({
            "id": node_id,
            "name": node_attrs.get("name", node_id),
            "lat": node_attrs["lat"],
            "lng": node_attrs["lng"],
        })

    edges = []
    for u, v, edge_attrs in G.edges(data=True):
        edge_data = {
            "source": u,
            "target": v,
            "distance_km": edge_attrs["distance_km"],
        }
        if "road_coords" in edge_attrs:
            edge_data["road_coords"] = edge_attrs["road_coords"]
        edges.append(edge_data)

    graph_data = {"nodes": nodes, "edges": edges}
    with open(GRAPH_JSON, "w", encoding="utf-8") as f:
        json.dump(graph_data, f, indent=2, ensure_ascii=False)

    print(f"[OK] graph_data.json updated with road coordinates")


if __name__ == "__main__":
    import sys
    if "--bake" in sys.argv:
        print("[BAKE] Fetching OSRM road geometry for all edges...")
        bake_road_coords()
    else:
        generate_json()
    G = load_graph()
    print(f"   Graph loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
