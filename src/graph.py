import networkx as nx
import random

class Graph:
    def __init__(self):
        # Initialize an empty graph with an adjacency list representation
        self.adjacency_list = {}
        self.graph_nx = None  # Placeholder for NetworkX graph
        self.pos = None       # Placeholder for node positions

    def add_edge(self, node1, node2, weight=None):
        # Ensure both nodes exist in the adjacency list before adding the edge
        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = {}
        if node2 not in self.adjacency_list:
            self.adjacency_list[node2] = {}

        # If a weight is provided, treat it as a weighted graph; otherwise, assume it's unweighted (default weight 1)
        if weight is None:
            weight = 1  # Default weight for unweighted graphs

        # Add the edge in both directions since it's an undirected graph
        self.adjacency_list[node1][node2] = weight
        self.adjacency_list[node2][node1] = weight  # Symmetric edge for undirected graph

    def get_neighbors(self, node):
        # Get the neighboring nodes of a given node
        return self.adjacency_list.get(node, [])

    def get_nodes(self):
        # Get all nodes in the graph
        return list(self.adjacency_list.keys())

    def create_nx_graph(self):
        """Create a NetworkX graph for visualization compatibility."""
        # Initialize self.graph_nx only if it has not been initialized
        if self.graph_nx is None:
            self.graph_nx = nx.Graph()

            # Convert adjacency list to NetworkX graph
            for node1, neighbors in self.adjacency_list.items():
                for node2, weight in neighbors.items():
                    self.graph_nx.add_edge(node1, node2, weight=weight)

            # Generate random positions for nodes if not already set
            if self.pos is None:
                self.pos = nx.spring_layout(self.graph_nx)  # Use spring layout for consistency

