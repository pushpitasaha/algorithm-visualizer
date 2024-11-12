import tkinter as tk
from tkinter import messagebox, simpledialog

from src.graph import Graph
from src.bfs import bfs
from src.dfs import dfs
from src.dijkstra import dijkstra
from src.tsp import TSPGeneticAlgorithm
from src.predefined_graphs import get_all_predefined_graphs
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation
import heapq  # Import heapq for the priority queue

class AlgorithmVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Visualizer")
        
        self.algorithm_var = tk.StringVar(value="BFS")

        # Add Dropdown for selecting algorithms
        self.algorithm_dropdown = tk.OptionMenu(root, self.algorithm_var, "BFS", "DFS", "Dijkstra", "TSP")
        self.algorithm_dropdown.pack()
        
        # Initialize GUI elements
        self.custom_graph_button = tk.Button(root, text="Input Custom Graph", command=self.custom_graph_input)
        self.custom_graph_button.pack()

        self.predefined_graph_button = tk.Button(root, text="Select Predefined Graph", command=self.select_predefined_graph)
        self.predefined_graph_button.pack()

        self.run_algorithm_button = tk.Button(root, text="Run Algorithm", command=self.run_algorithm)
        self.run_algorithm_button.pack()

        # Store the graph and traversal path
        self.graph = None
        self.traversal_path = []
        self.traversal_index = 0  # Current step in the traversal
        self.parent_map = {}  # Store BFS parent-child relationships
        
        self.distance_map = {}  # Stores the final shortest distance from the start node
        self.dijkstra_states = []  # Store the state of Dijkstra's steps for backtracking
        
        self.previous_node = {}  # Stores the previous node in the shortest path for each node
        self.shortest_path_edges = []  # Initialize the shortest path edges list



    def custom_graph_input(self):
        edge_input = simpledialog.askstring("Input Graph", "Enter edges as node pairs separated by spaces (e.g., 'A B C D' for edges A-B and C-D):")
        if edge_input:
            self.graph = Graph()
            edge_list = edge_input.split()
            for i in range(0, len(edge_list) - 1, 2):
                self.graph.add_edge(edge_list[i], edge_list[i+1])
            messagebox.showinfo("Custom Graph", f"Graph created with {len(self.graph.adjacency_list)} nodes.")

    def select_predefined_graph(self):
        predefined_graphs = get_all_predefined_graphs()
        graph_names = list(predefined_graphs.keys())

        def on_graph_selected(selected_graph):
            self.graph = predefined_graphs[selected_graph]
            messagebox.showinfo("Graph Selected", f"{selected_graph} selected.")

        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Predefined Graph")
        for graph_name in graph_names:
            button = tk.Button(selection_window, text=graph_name, command=lambda name=graph_name: on_graph_selected(name))
            button.pack()

    def run_algorithm(self):
        if not self.graph:
            messagebox.showwarning("No Graph", "Please select or input a graph first.")
            return

        algorithm = self.algorithm_var.get()  # Get selected algorithm
        start_node = simpledialog.askstring("Input Start Node", "Enter the start node:")

        if start_node and start_node in self.graph.adjacency_list:
            if algorithm == "BFS":
                self.parent_map = bfs(self.graph, start_node)
            elif algorithm == "DFS":
                from src.dfs import dfs  # Import DFS when needed
                self.parent_map = dfs(self.graph, start_node)
            elif algorithm == "Dijkstra":
                self.run_dijkstra(start_node)
            elif algorithm == "TSP":
                self.run_tsp()
                return

            self.traversal_index = 0  # Reset traversal index
            messagebox.showinfo(f"{algorithm} Traversal", f"Traversal order: {list(self.parent_map.keys())}")
            self.show_bfs_progress()  # Use the same BFS visualization for both BFS and DFS
        else:
            messagebox.showerror("Invalid Node", "Start node not found in graph.")

    def show_bfs_progress(self):
        bfs_window = tk.Toplevel(self.root)
        bfs_window.title("BFS Traversal")

        # Set up a Matplotlib canvas for visualization
        figure = plt.Figure(figsize=(5, 5), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=bfs_window)
        canvas.get_tk_widget().pack()

        # Visualization variables
        graph_nx = nx.Graph(self.graph.adjacency_list)
        pos = nx.spring_layout(graph_nx)
        
        # Draw initial graph with labels
        ax = figure.add_subplot(111)
        nx.draw(graph_nx, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15, ax=ax)

        # Highlight traversed edges in red
        traversed_edges = []  # Keep track of the edges traversed in BFS order

        def update_graph_display():
            """Updates the graph display showing the edges traversed up to the current step."""
            ax.clear()  # Clear previous drawing
            # Redraw the base graph
            nx.draw(graph_nx, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15, ax=ax)

            # Draw all traversed edges so far
            edge_labels = {edge: str(i + 1) for i, edge in enumerate(traversed_edges)}
            if traversed_edges:  # Highlight edges traversed so far
                nx.draw_networkx_edges(graph_nx, pos, edgelist=traversed_edges, edge_color='r', width=2.5, ax=ax)
                nx.draw_networkx_edge_labels(graph_nx, pos, edge_labels=edge_labels, ax=ax)

            canvas.draw()  # Update the canvas with the new drawing

        def next_step():
            """Proceed to the next step in the BFS traversal."""
            if self.traversal_index < len(self.parent_map) - 1:
                # Get the current node and its parent from parent_map
                current_node = list(self.parent_map.keys())[self.traversal_index + 1]
                parent_node = self.parent_map[current_node]

                if parent_node is not None:  # Avoid None for root node
                    traversed_edges.append((parent_node, current_node))  # Add the edge to the traversed list

                self.traversal_index += 1
                update_graph_display()  # Update visualization with new edge

        def prev_step():
            """Go back to the previous step in the BFS traversal."""
            if self.traversal_index > 0:
                traversed_edges.pop()  # Remove the last traversed edge
                self.traversal_index -= 1
                update_graph_display()  # Update visualization without the last edge

        # Add buttons for next and previous steps in the same window
        control_frame = tk.Frame(bfs_window)
        control_frame.pack()

        next_button = tk.Button(control_frame, text="Next ->", command=next_step)
        next_button.pack(side=tk.RIGHT)

        prev_button = tk.Button(control_frame, text="<- Previous", command=prev_step)
        prev_button.pack(side=tk.LEFT)

        update_graph_display()  # Initial display

    def run_dijkstra(self, start_node):
        """Runs Dijkstra's algorithm with step-by-step visualization."""
        if not self.graph or not start_node:
            return

        dijkstra_window = tk.Toplevel(self.root)
        dijkstra_window.title("Dijkstra Traversal")

        # Set up a Matplotlib canvas for visualization
        figure = plt.Figure(figsize=(6, 5), dpi=100)
        canvas = FigureCanvasTkAgg(figure, master=dijkstra_window)
        canvas.get_tk_widget().pack(side=tk.LEFT)

        # Create NetworkX graph for visualization
        graph_nx = nx.Graph(self.graph.adjacency_list)
        pos = nx.spring_layout(graph_nx)  # Positions for visualization

        # Draw initial graph with edge weights as labels
        ax = figure.add_subplot(111)
        nx.draw(graph_nx, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15, ax=ax)
        
        # Display edge weights
        edge_labels = {(u, v): f"{d}" for u, adj in self.graph.adjacency_list.items() for v, d in adj.items()}
        nx.draw_networkx_edge_labels(graph_nx, pos, edge_labels=edge_labels, ax=ax)
        canvas.draw()

        # Distance array for all nodes, initialized to infinity (INF)
        nodes = list(self.graph.adjacency_list.keys())
        distance = {node: float('inf') for node in nodes}
        distance[start_node] = 0

        # Priority Queue (Min-Heap) for Dijkstra's algorithm
        min_heap = [(0, start_node)]
        heapq.heapify(min_heap)

        visited_edges = []  # To track edges we've already visited
        visited_nodes = set()
        previous_node = {}  # To track the shortest path

        # State history for forward and backward navigation
        dijkstra_states = []

        # Create frames for Min-Heap and Distance Array
        right_frame = tk.Frame(dijkstra_window)
        right_frame.pack(side=tk.RIGHT, padx=10)

        heap_frame = tk.LabelFrame(right_frame, text="Min-Heap", padx=10, pady=10)
        heap_frame.pack(fill="both", expand="yes")

        distance_frame = tk.LabelFrame(right_frame, text="Distance Array", padx=10, pady=10)
        distance_frame.pack(fill="both", expand="yes")

        # Heap and Distance Array Labels
        heap_label = tk.Label(heap_frame, text=str(min_heap))
        heap_label.pack()

        distance_label = tk.Label(distance_frame, text=str(distance))
        distance_label.pack()

        # Redraw the graph with new highlights
        def update_graph_display():
            """Updates the graph display to show current state, with blue and red highlights."""
            if not self.graph:
                return

            ax.clear()  # Clear previous drawing
            # Redraw the base graph with edge weights
            nx.draw(graph_nx, pos, with_labels=True, node_color='skyblue', node_size=700, font_size=15, ax=ax)
            nx.draw_networkx_edge_labels(graph_nx, pos, edge_labels=edge_labels, ax=ax)

            # Highlight visited edges and nodes in blue
            if visited_edges:
                nx.draw_networkx_edges(graph_nx, pos, edgelist=visited_edges, edge_color='blue', width=2.5, ax=ax)

            # Highlight shortest path edges in red after completion
            if self.shortest_path_edges:
                nx.draw_networkx_edges(graph_nx, pos, edgelist=self.shortest_path_edges, edge_color='red', width=2.5, ax=ax)

            canvas.draw()

        # Update the Min-Heap and Distance Array Display
        def update_heap_and_distance():
            heap_label.config(text=str(min_heap))
            distance_label.config(text=str(distance))

        def save_state():
            """Save the current state of the algorithm."""
            state = {
                "min_heap": list(min_heap),
                "distance": distance.copy(),
                "visited_edges": list(visited_edges),
                "visited_nodes": set(visited_nodes)
            }
            dijkstra_states.append(state)

        def restore_state(state):
            """Restore the algorithm to a previous state."""
            min_heap.clear()
            min_heap.extend(state['min_heap'])
            for node in distance:
                distance[node] = state['distance'][node]
            visited_edges.clear()
            visited_edges.extend(state['visited_edges'])
            visited_nodes.clear()
            visited_nodes.update(state['visited_nodes'])
            update_graph_display()
            update_heap_and_distance()

        # Step forward in the algorithm
        def run_step(forward=True):
            if forward:
                save_state()  # Save the current state before processing

            if not min_heap:
                # If no more steps, highlight the shortest path and show message
                highlight_shortest_path()  # Call the method to highlight the path
                messagebox.showinfo("Dijkstra Complete", f"Shortest path distances: {distance}")
                return

            # Pop the node with the smallest distance from the heap
            current_distance, current_node = heapq.heappop(min_heap)
            visited_nodes.add(current_node)

            # Traverse neighbors
            for neighbor, weight in self.graph.adjacency_list[current_node].items():
                if neighbor in visited_nodes:
                    continue

                new_distance = current_distance + weight

                # If the new calculated distance is smaller, update it
                if new_distance < distance[neighbor]:
                    distance[neighbor] = new_distance
                    heapq.heappush(min_heap, (new_distance, neighbor))
                    visited_edges.append((current_node, neighbor))
                    previous_node[neighbor] = current_node  # Track the previous node for backtracking

            update_graph_display()
            update_heap_and_distance()

        # Highlight the final shortest path in red
        def highlight_shortest_path():
            """Backtrack from all nodes to highlight the shortest path in red."""
            self.shortest_path_edges = []
            for node in previous_node:
                current = node
                while current in previous_node:
                    prev = previous_node[current]
                    self.shortest_path_edges.append((prev, current))
                    current = prev
            update_graph_display()  # Redraw the graph with red highlights for shortest path

        # Add buttons for controlling the algorithm step-by-step
        control_frame = tk.Frame(dijkstra_window)
        control_frame.pack()

        prev_button = tk.Button(control_frame, text="<- Previous Step", command=lambda: restore_state(dijkstra_states.pop()))
        prev_button.pack(side=tk.LEFT)

        next_button = tk.Button(control_frame, text="Next Step ->", command=lambda: run_step(True))
        next_button.pack(side=tk.RIGHT)

        # Start the initial display and step
        update_graph_display()
        run_step()

    def run_tsp(self):
        """Run the TSP genetic algorithm with step-by-step visualization."""
        # Get parameters from user input
        population_size = int(simpledialog.askstring("Population Size", "Enter population size:"))
        mutation_rate = float(simpledialog.askstring("Mutation Rate", "Enter mutation rate (0-1):"))
        generations = int(simpledialog.askstring("Generations", "Enter number of generations:"))

        # Initialize TSP genetic algorithm
        tsp_ga = TSPGeneticAlgorithm(self.graph, population_size, mutation_rate, generations)

        # Call create_nx_graph to initialize the graph and layout
        self.graph.create_nx_graph()

        # Setup Tkinter window for visualization
        tsp_window = tk.Toplevel(self.root)
        tsp_window.title("TSP Solution")
        figure, ax = plt.subplots(figsize=(6, 6))
        canvas = FigureCanvasTkAgg(figure, tsp_window)
        canvas.get_tk_widget().pack()

        # Display parameters
        parameter_label = tk.Label(
            tsp_window,
            text=f"Population Size: {population_size}, Mutation Rate: {mutation_rate}, Generations: {generations}",
            font=("Arial", 10)
        )
        parameter_label.pack()
        
        # Label to show the current generation and best fitness
        status_label = tk.Label(
            tsp_window,
            text=f"Generation: 0, Best Fitness: N/A",
            font=("Arial", 10)
        )
        status_label.pack()

        # Set up variables to keep track of the current generation and best path
        current_generation = 1
        best_path = tsp_ga.population[0]  # Initial best path

        def update_visualization():
            """Update the visualization to show the current generation's best path."""
            ax.clear()
            ax.set_title(f"Generation {current_generation}")
            
            # Visualize the current best path
            tsp_ga.visualize_path(best_path, ax)
            canvas.draw()
            
            # Update status label with the current generation and best fitness
            best_fitness = tsp_ga.calculate_fitness(best_path)
            status_label.config(text=f"Generation: {current_generation}, Best Fitness: {best_fitness:.2f}")

        def run_generations():
            """Run through all generations and update visualization and terminal output."""
            nonlocal current_generation, best_path
            
            # Check if we've reached the end
            if current_generation < generations:
                tsp_ga.evolve_population()
                best_path = max(tsp_ga.population, key=tsp_ga.calculate_fitness)
                best_fitness = tsp_ga.calculate_fitness(best_path)
                
                # Print the current fitness in terminal
                print(f"Generation {current_generation}: Best fitness = {best_fitness}")

                # Update the visualization
                update_visualization()

                # Increment the generation count and set the loop to call itself
                current_generation += 1
                tsp_window.after(500, run_generations)  # Call this function again after 500 ms
            else:
                print("Completed all generations.")

        # Button to start the generation updates
        start_button = tk.Button(tsp_window, text="Start Generations", command=run_generations)
        start_button.pack()

        # Initialize the first visualization
        update_visualization()

        # Make sure the Tkinter event loop is running
        tsp_window.protocol("WM_DELETE_WINDOW", tsp_window.quit)
        tsp_window.mainloop()



        
if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmVisualizerApp(root)
    root.mainloop()
