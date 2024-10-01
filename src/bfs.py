from collections import deque

def bfs(graph, start_node):
    visited = set()  # Set to keep track of visited nodes
    queue = deque([start_node])  # Initialize a queue with the start node
    parent_map = {}  # To store parent-child relationships

    parent_map[start_node] = None  # Start node has no parent

    while queue:
        node = queue.popleft()  # Get the first node from the queue
        if node not in visited:
            visited.add(node)  # Mark the node as visited

            # Queue up all unvisited neighbors and record their parent
            for neighbor in graph.adjacency_list.get(node, []):
                if neighbor not in visited and neighbor not in queue:
                    queue.append(neighbor)
                    parent_map[neighbor] = node  # Record the parent

    return parent_map  # Return parent relationships instead of traversal order
