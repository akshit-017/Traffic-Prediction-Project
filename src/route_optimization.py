import math
from datetime import datetime, timedelta, timezone

import networkx as nx


# IST is UTC+5:30 — defined once using the stdlib so we avoid extra dependencies
IST = timezone(timedelta(hours=5, minutes=30))


class RouteOptimizer:
    def __init__(self):
        """
        Initializes the RouteOptimizer with a graph.
        """
        self.graph = nx.Graph()
        self._build_sample_graph()

    def _build_sample_graph(self):
        """
        Builds a sample road network graph for demonstration purposes.
        Nodes represent intersections, edges represent road segments.
        Default weight (distance/base travel time) is assigned, but congestion
        will be added dynamically later.
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

        # Add edges (u, v, base_weight) with waypoints
        edges = [
            ('A', 'B', 5), ('A', 'C', 10),
            ('B', 'C', 2), ('B', 'D', 8),
            ('C', 'E', 4),
            ('D', 'E', 6), ('D', 'F', 3),
            ('E', 'F', 2), ('E', 'G', 7),
            ('F', 'G', 4)
        ]

        for u, v, w in edges:
            waypoints = edge_waypoints.get((u, v), [])
            self.graph.add_edge(
                u, v,
                base_weight=w, congestion=0, effective_weight=w,
                waypoints=waypoints
            )

    # ------------------------------------------------------------------
    # Dynamic congestion prediction based on current IST time
    # ------------------------------------------------------------------
    def generate_dynamic_predictions(self):
        """
        Generates time-varying congestion predictions for every edge in the
        graph based on the **current** India Standard Time (IST / UTC+5:30).

        Temporal features used:
            • hour          – 0-23
            • hour_sin/cos  – cyclical encoding so 23:00 and 00:00 are close
            • day_of_week   – 0 (Mon) … 6 (Sun)
            • is_peak_hour  – binary flag for morning (8-10) & evening (17-20) rush
            • is_weekend    – binary flag for Saturday / Sunday
            • minute_frac   – fractional minute contribution for sub-hour variation

        Returns a dict mapping (u, v) -> congestion_value that can be fed
        directly into ``update_edge_weights()``.
        """
        now_ist = datetime.now(IST)
        hour = now_ist.hour
        minute = now_ist.minute
        day_of_week = now_ist.weekday()          # 0=Mon … 6=Sun

        # Cyclical hour encoding (matches the preprocessing the RF was trained on)
        hour_sin = math.sin(2 * math.pi * hour / 24.0)
        hour_cos = math.cos(2 * math.pi * hour / 24.0)

        # Binary flags (same logic as preprocessing.py)
        is_peak_hour = 1 if hour in (8, 9, 10, 17, 18, 19, 20) else 0
        is_weekend = 1 if day_of_week >= 5 else 0

        # Sub-hour fraction so predictions shift even within the same hour
        minute_frac = minute / 60.0

        # --- Per-edge congestion formula ---
        # Each edge gets a unique phase offset derived from the hash of its
        # node-pair, making different roads peak at different times.
        predictions = {}
        for u, v, data in self.graph.edges(data=True):
            base_w = data['base_weight']
            # Unique-per-edge phase offset in [0, 2π)
            edge_phase = (hash((u, v)) % 360) * (math.pi / 180.0)

            # Sinusoidal component: produces a smooth wave over 24 hours
            wave = 0.5 * (1.0 + math.sin(
                2 * math.pi * (hour + minute_frac) / 24.0 + edge_phase
            ))

            # Peak-hour surge: congestion spikes during rush hours
            peak_boost = 1.5 * is_peak_hour

            # Weekend discount: less traffic on Sat/Sun
            weekend_factor = 0.6 if is_weekend else 1.0

            # Combine into a congestion value scaled relative to base_weight
            congestion = base_w * (wave + peak_boost) * weekend_factor

            predictions[(u, v)] = round(congestion, 4)

        return predictions

    def update_edge_weights(self, predictions_dict):
        """
        Updates the congestion weights of edges based on ML predictions.
        
        :param predictions_dict: Dictionary mapping edge tuple (u, v) to congestion value.
        """
        for (u, v), congestion_val in predictions_dict.items():
            if self.graph.has_edge(u, v):
                # Update the edge data. Using the max of 0 and predicted congestion.
                self.graph[u][v]['congestion'] = max(0, congestion_val)
                # Compute effective weight: base_weight + congestion effect
                base_w = self.graph[u][v]['base_weight']
                self.graph[u][v]['effective_weight'] = base_w + max(0, congestion_val)
            elif self.graph.has_edge(v, u): # undirected graph handles reverse lookup
                self.graph[v][u]['congestion'] = max(0, congestion_val)
                base_w = self.graph[v][u]['base_weight']
                self.graph[v][u]['effective_weight'] = base_w + max(0, congestion_val)

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
            return shortest_path, path_weight
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
                'base_weight': data.get('base_weight', 0),
                'congestion': data.get('congestion', 0),
                'effective_weight': data.get('effective_weight', 0),
                'waypoints': data.get('waypoints', [])
            })

        return {'nodes': nodes, 'edges': edges}
