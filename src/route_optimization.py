import networkx as nx

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

        # Add edges (u, v, base_weight)
        edges = [
            ('A', 'B', 5), ('A', 'C', 10),
            ('B', 'C', 2), ('B', 'D', 8),
            ('C', 'E', 4),
            ('D', 'E', 6), ('D', 'F', 3),
            ('E', 'F', 2), ('E', 'G', 7),
            ('F', 'G', 4)
        ]
        
        for u, v, w in edges:
            # We initialize effective_weight to base_weight by default
            self.graph.add_edge(u, v, base_weight=w, congestion=0, effective_weight=w)

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
                'effective_weight': data.get('effective_weight', 0)
            })
            
        return {'nodes': nodes, 'edges': edges}
