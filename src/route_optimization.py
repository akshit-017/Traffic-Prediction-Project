import math
from datetime import datetime, timedelta, timezone

import networkx as nx

from src.predictor import TrafficPredictor


# IST is UTC+5:30 — defined once using the stdlib so we avoid extra dependencies
IST = timezone(timedelta(hours=5, minutes=30))

# Average urban driving speed in Bengaluru (km/h) used to convert distance → time.
# 33 km/h is a well-established free-flow average for Bengaluru arterial roads.
BASE_SPEED_KMH = 33.0

# ──────────────────────────────────────────────────────────────────────
# Mapping: graph edge → (dataset area, dataset road)
# For edges where no exact area exists in the dataset (HSR Layout,
# JP Nagar), the nearest area's road is used as a spatial proxy.
# ──────────────────────────────────────────────────────────────────────
EDGE_ROAD_MAP = {
    ('A', 'B'): ('M.G. Road', 'Trinity Circle'),
    ('A', 'C'): ('M.G. Road', 'Anil Kumble Circle'),
    ('B', 'C'): ('Indiranagar', '100 Feet Road'),
    ('B', 'D'): ('Indiranagar', 'CMH Road'),
    ('C', 'E'): ('Koramangala', 'Sony World Junction'),
    ('D', 'E'): ('Koramangala', 'Sarjapur Road'),        # HSR→Jayanagar proxy
    ('D', 'F'): ('Electronic City', 'Silk Board Junction'),  # HSR→JP Nagar proxy
    ('E', 'F'): ('Jayanagar', 'South End Circle'),
    ('E', 'G'): ('Jayanagar', 'Jayanagar 4th Block'),
    ('F', 'G'): ('Electronic City', 'Hosur Road'),
}


def _haversine_km(lat1, lng1, lat2, lng2):
    """Return the great-circle distance in **kilometres** between two points."""
    R = 6371.0  # Earth radius in km
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1))
         * math.cos(math.radians(lat2))
         * math.sin(dlng / 2) ** 2)
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def _road_distance_km(lat1, lng1, lat2, lng2, waypoints):
    """
    Compute the approximate road distance by summing haversine segments
    through all waypoints (instead of a straight-line shortcut).
    A small 1.05× detour factor accounts for minor turns and lane
    changes that the sparse waypoints don't capture.  The bulk of the
    road curvature is already modelled by the waypoints themselves.
    """
    DETOUR_FACTOR = 1.05
    points = [[lat1, lng1]] + (waypoints or []) + [[lat2, lng2]]
    total = 0.0
    for i in range(len(points) - 1):
        total += _haversine_km(points[i][0], points[i][1],
                               points[i + 1][0], points[i + 1][1])
    return total * DETOUR_FACTOR


class RouteOptimizer:
    def __init__(self):
        """
        Initializes the RouteOptimizer with a graph and the ML predictor.
        """
        self.graph = nx.Graph()
        self.predictor = TrafficPredictor()
        self._build_sample_graph()

    def _build_sample_graph(self):
        """
        Builds a sample road network graph for demonstration purposes.
        Nodes represent intersections, edges represent road segments.
        ``base_weight`` is the **free-flow travel time in minutes** derived
        from the real haversine distance between nodes and a typical
        Bengaluru urban speed.
        Each edge includes 'waypoints' — intermediate [lat, lng] coordinates
        that trace the approximate physical road path in Bengaluru.
        """
        # Define nodes with Bengaluru GPS coordinates (lat, lng)
        nodes = {
            'A': {'name': 'MG Road', 'lat': 12.973, 'lng': 77.611},
            'B': {'name': 'Indiranagar', 'lat': 12.978, 'lng': 77.638},
            'C': {'name': 'Koramangala', 'lat': 12.935, 'lng': 77.624},
            'D': {'name': 'HSR Layout', 'lat': 12.912, 'lng': 77.644},
            'E': {'name': 'Jayanagar', 'lat': 12.929, 'lng': 77.580},
            'F': {'name': 'JP Nagar', 'lat': 12.906, 'lng': 77.585},
            'G': {'name': 'Electronic City', 'lat': 12.845, 'lng': 77.660}
        }

        for node_id, attrs in nodes.items():
            self.graph.add_node(node_id, **attrs)

        # Intermediate waypoints tracing approximate real road curves.
        # Each list contains [lat, lng] pairs between the two endpoint nodes.
        edge_waypoints = {
            # A→B: MG Road east along Old Airport Rd / 100 Feet Rd to Indiranagar
            ('A', 'B'): [
                [12.974, 77.615], [12.975, 77.620], [12.977, 77.625],
                [12.978, 77.630], [12.978, 77.635]
            ],
            # A→C: MG Road south via Residency Rd, Hosur Rd to Koramangala
            ('A', 'C'): [
                [12.970, 77.612], [12.965, 77.614], [12.958, 77.616],
                [12.950, 77.618], [12.945, 77.620], [12.940, 77.622]
            ],
            # B→C: Indiranagar south along Inner Ring Rd to Koramangala
            ('B', 'C'): [
                [12.975, 77.637], [12.970, 77.636], [12.963, 77.634],
                [12.955, 77.632], [12.948, 77.630], [12.940, 77.627]
            ],
            # B→D: Indiranagar south-east via HAL / Outer Ring Rd to HSR Layout
            ('B', 'D'): [
                [12.975, 77.640], [12.968, 77.641], [12.958, 77.642],
                [12.945, 77.643], [12.935, 77.643], [12.925, 77.644],
                [12.918, 77.644]
            ],
            # C→E: Koramangala west via Hosur Rd / South End Circle to Jayanagar
            ('C', 'E'): [
                [12.934, 77.618], [12.933, 77.612], [12.932, 77.605],
                [12.931, 77.598], [12.930, 77.590], [12.929, 77.585]
            ],
            # D→E: HSR Layout west via BTM Layout / Bannerghatta Rd to Jayanagar
            ('D', 'E'): [
                [12.915, 77.638], [12.918, 77.630], [12.920, 77.620],
                [12.922, 77.610], [12.925, 77.600], [12.927, 77.590]
            ],
            # D→F: HSR Layout south-west via BTM Layout to JP Nagar
            ('D', 'F'): [
                [12.912, 77.638], [12.911, 77.630], [12.910, 77.620],
                [12.909, 77.610], [12.908, 77.600], [12.907, 77.590]
            ],
            # E→F: Jayanagar south along main road to JP Nagar
            ('E', 'F'): [
                [12.926, 77.580], [12.922, 77.581], [12.918, 77.582],
                [12.914, 77.583], [12.910, 77.584]
            ],
            # E→G: Jayanagar south-east via Bannerghatta / Hosur Rd to Electronic City
            ('E', 'G'): [
                [12.925, 77.585], [12.918, 77.595], [12.910, 77.608],
                [12.900, 77.620], [12.890, 77.632], [12.875, 77.645],
                [12.860, 77.652], [12.850, 77.657]
            ],
            # F→G: JP Nagar south via Bannerghatta Rd to Electronic City
            ('F', 'G'): [
                [12.900, 77.588], [12.893, 77.595], [12.885, 77.608],
                [12.875, 77.622], [12.865, 77.638], [12.855, 77.650]
            ],
        }

        # Edge pairs (waypoints looked up from the dict above)
        edge_pairs = [
            ('A', 'B'), ('A', 'C'),
            ('B', 'C'), ('B', 'D'),
            ('C', 'E'),
            ('D', 'E'), ('D', 'F'),
            ('E', 'F'), ('E', 'G'),
            ('F', 'G'),
        ]

        for u, v in edge_pairs:
            waypoints = edge_waypoints.get((u, v), [])
            u_data = nodes[u]
            v_data = nodes[v]

            # Real road distance in km (through waypoints + detour factor)
            dist_km = _road_distance_km(
                u_data['lat'], u_data['lng'],
                v_data['lat'], v_data['lng'],
                waypoints,
            )
            # Free-flow travel time in minutes
            base_minutes = round((dist_km / BASE_SPEED_KMH) * 60.0, 1)

            self.graph.add_edge(
                u, v,
                distance_km=round(dist_km, 2),
                base_weight=base_minutes,
                congestion_multiplier=1.0,
                predicted_volume=0.0,
                effective_weight=base_minutes,
                waypoints=waypoints,
            )

    # ------------------------------------------------------------------
    # ML-powered congestion prediction
    # ------------------------------------------------------------------
    def generate_dynamic_predictions(self):
        """
        Generates congestion multipliers for every edge using the trained
        Random Forest model (via ``TrafficPredictor``).

        Each edge is mapped to a real (area, road) pair from the Bengaluru
        traffic dataset (see ``EDGE_ROAD_MAP``).  The predictor returns a
        traffic volume prediction → converted to a multiplier (1.0–2.5×).

        If the ML model isn't loaded, the predictor automatically falls
        back to a time-based heuristic so the app still works.

        Returns a dict mapping (u, v) -> {multiplier, predicted_volume}.
        """
        predictions = {}
        for u, v, _data in self.graph.edges(data=True):
            area, road = EDGE_ROAD_MAP.get((u, v),
                                           EDGE_ROAD_MAP.get((v, u),
                                                             ('Indiranagar', '100 Feet Road')))

            predicted_vol = self.predictor.predict_volume(area, road)
            multiplier = self.predictor.volume_to_multiplier(predicted_vol, area)

            predictions[(u, v)] = {
                'multiplier': round(multiplier, 4),
                'predicted_volume': round(predicted_vol, 0),
            }

        return predictions

    def update_edge_weights(self, predictions_dict):
        """
        Updates the congestion weights of edges based on predictions.

        :param predictions_dict: Dictionary mapping edge tuple (u, v) to
            either a dict with 'multiplier' and 'predicted_volume' keys,
            or a plain float (treated as a multiplier for backwards compat).
        """
        for (u, v), value in predictions_dict.items():
            if isinstance(value, dict):
                multiplier = value['multiplier']
                pred_vol = value.get('predicted_volume', 0.0)
            else:
                multiplier = float(value)
                pred_vol = 0.0

            multiplier = max(1.0, multiplier)   # never faster than free-flow

            if self.graph.has_edge(u, v):
                self.graph[u][v]['congestion_multiplier'] = multiplier
                self.graph[u][v]['predicted_volume'] = pred_vol
                base = self.graph[u][v]['base_weight']
                self.graph[u][v]['effective_weight'] = round(base * multiplier, 1)
            elif self.graph.has_edge(v, u):  # undirected graph handles reverse lookup
                self.graph[v][u]['congestion_multiplier'] = multiplier
                self.graph[v][u]['predicted_volume'] = pred_vol
                base = self.graph[v][u]['base_weight']
                self.graph[v][u]['effective_weight'] = round(base * multiplier, 1)

    def find_best_route(self, source, destination):
        """
        Finds the most efficient route using Dijkstra's algorithm based on effective_weight.
        
        :param source: The starting node (e.g., 'A')
        :param destination: The ending node (e.g., 'G')
        :return: Tuple of (shortest_path_nodes, total_effective_weight)
        """
        try:
            # Dijkstra's algorithm using networkx
            shortest_path = nx.shortest_path(self.graph, source=source, target=destination, weight='effective_weight')
            path_weight = nx.shortest_path_length(self.graph, source=source, target=destination, weight='effective_weight')
            return shortest_path, round(path_weight, 1)
        except nx.NetworkXNoPath:
            return None, float('inf')
        except nx.NodeNotFound as e:
            print(f"Error: {e}")
            return None, float('inf')

    def get_graph_data(self):
        """
        Exports the graph data (nodes, edges, weights) as a dictionary for the frontend.
        Includes waypoints for road-aligned polyline rendering.
        """
        nodes = []
        for node, data in self.graph.nodes(data=True):
            nodes.append({
                'id': node,
                'name': data.get('name', node),
                'lat': data.get('lat', 0),
                'lng': data.get('lng', 0)
            })

        edges = []
        for u, v, data in self.graph.edges(data=True):
            edges.append({
                'source': u,
                'target': v,
                'distance_km': data.get('distance_km', 0),
                'base_weight': data.get('base_weight', 0),
                'congestion_multiplier': data.get('congestion_multiplier', 1.0),
                'predicted_volume': data.get('predicted_volume', 0.0),
                'effective_weight': data.get('effective_weight', 0),
                'waypoints': data.get('waypoints', [])
            })

        return {
            'nodes': nodes,
            'edges': edges,
            'ml_model_loaded': self.predictor.is_loaded,
        }
