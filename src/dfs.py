from collections import deque

def dfs(graph, start_node):
    visited = set()  # Set to track visited nodes
    parent_map = {}  # Store parent-child relationships
    parent_map[start_node] = None  # Start node has no parent

    def dfs_recursive(node):
        """Inner recursive function for DFS."""
        visited.add(node)
        for neighbor in graph.adjacency_list.get(node, []):
            if neighbor not in visited:
                parent_map[neighbor] = node
                dfs_recursive(neighbor)

    dfs_recursive(start_node)
    return parent_map
