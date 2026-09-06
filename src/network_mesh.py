import networkx as nx
import time

class TacticalMeshNetwork:
    def __init__(self):
        # Initialize an undirected graph representing the mesh topology
        self.graph = nx.Graph()
        
        # Build a 5-node simulated mesh network layout
        # [Laptop_A] <---> [Node_Relay_1] <---> [Node_Relay_2] <---> [Laptop_B]
        #     │                                       │
        #     └────────────────> [Node_Relay_3] ──────┘
        
        nodes = ["Laptop_A", "Node_Relay_1", "Node_Relay_2", "Node_Relay_3", "Laptop_B"]
        for node in nodes:
            self.graph.add_node(node, is_alive=True)

        # Edges with latency weights (in milliseconds)
        edges = [
            ("Laptop_A", "Node_Relay_1", 5),
            ("Node_Relay_1", "Node_Relay_2", 8),
            ("Node_Relay_2", "Laptop_B", 4),
            ("Laptop_A", "Node_Relay_3", 12),
            ("Node_Relay_3", "Laptop_B", 15)
        ]
        
        for u, v, latency in edges:
            self.graph.add_edge(u, v, weight=latency)

    def route_packet(self, source="Laptop_A", destination="Laptop_B"):
        """
        Calculates the optimal path dynamically based on active node status 
        and link weights (latency).
        """
        # Filter out disabled or "dead" nodes
        active_subgraph = self.graph.subgraph(
            [n for n, attr in self.graph.nodes(data=True) if attr['is_alive']]
        )

        try:
            # Dijkstra's Shortest Path algorithm based on latency weights
            path = nx.shortest_path(active_subgraph, source=source, target=destination, weight='weight')
            path_latency = nx.shortest_path_length(active_subgraph, source=source, target=destination, weight='weight')
            
            return {
                "status": "SUCCESS",
                "path": path,
                "total_hops": len(path) - 1,
                "estimated_latency_ms": path_latency
            }
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return {
                "status": "FAILED",
                "path": [],
                "error": "No available route between source and destination (Network Partitioned)"
            }

    def set_node_status(self, node_id, is_alive=True):
        """Simulates node failure or recovery (e.g., node destroyed or out of range)."""
        if node_id in self.graph.nodes:
            self.graph.nodes[node_id]['is_alive'] = is_alive
            status_str = "ACTIVE" if is_alive else "DEAD/DISCONNECTED"
            print(f"[MESH NETWORK] Node '{node_id}' status updated: {status_str}")

    def get_topology_snapshot(self):
        """Returns the current state of all nodes and active links."""
        return {
            "nodes": {n: attr['is_alive'] for n, attr in self.graph.nodes(data=True)},
            "edges": list(self.graph.edges())
        }