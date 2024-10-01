class Graph:
    # Initialize an empty graph with an adjacency list representation
    def __init__(self):
        self.adjacency_list = {}

    # Add an edge between two nodes in the graph
    def add_edge(self, node1, node2):
        # Ensure both nodes exist in the adjacency list before adding the edge
        if node1 not in self.adjacency_list:
            self.adjacency_list[node1] = []
        if node2 not in self.adjacency_list:
            self.adjacency_list[node2] = []
        # Add the edge in both directions since it's an undirected graph
        self.adjacency_list[node1].append(node2)
        self.adjacency_list[node2].append(node1)  # Symmetric edge for undirected graph

    # Get the neighboring nodes of a given node
    def get_neighbors(self, node):
        # Return an empty list if the node doesn't exist in the graph
        return self.adjacency_list.get(node, [])

    # Get all nodes in the graph
    def get_nodes(self):
        # Return a list of all nodes (keys in the adjacency list)
        return list(self.adjacency_list.keys())