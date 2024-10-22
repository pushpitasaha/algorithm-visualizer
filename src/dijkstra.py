import heapq

def dijkstra(graph, start_node):
    # Initialize distances and priority queue (min-heap)
    distances = {node: float('infinity') for node in graph.adjacency_list}
    distances[start_node] = 0

    priority_queue = [(0, start_node)]  # (distance, node)
    heap_snapshots = []  # For visualization of heap progress

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        # Capture the snapshot of priority queue at this step
        heap_snapshots.append(list(priority_queue))

        # If we find a shorter path to a node, we skip this longer path
        if current_distance > distances[current_node]:
            continue

        # Check neighbors
        for neighbor, weight in graph.adjacency_list[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))

    return distances, heap_snapshots
