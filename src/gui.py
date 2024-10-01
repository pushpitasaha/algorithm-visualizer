import tkinter as tk
from tkinter import messagebox, simpledialog
from src.graph import Graph
from src.bfs import bfs
from src.predefined_graphs import get_all_predefined_graphs
import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class AlgorithmVisualizerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Algorithm Visualizer")

        # Initialize GUI elements
        self.custom_graph_button = tk.Button(root, text="Input Custom Graph", command=self.custom_graph_input)
        self.custom_graph_button.pack()

        self.predefined_graph_button = tk.Button(root, text="Select Predefined Graph", command=self.select_predefined_graph)
        self.predefined_graph_button.pack()

        self.run_bfs_button = tk.Button(root, text="Run BFS", command=self.run_bfs)
        self.run_bfs_button.pack()

        # Store the graph and traversal path
        self.graph = None
        self.traversal_path = []
        self.traversal_index = 0  # Current step in the traversal
        self.parent_map = {}  # Store BFS parent-child relationships

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

    def run_bfs(self):
        if not self.graph:
            messagebox.showwarning("No Graph", "Please select or input a graph first.")
            return

        start_node = simpledialog.askstring("Input Start Node", "Enter the start node for BFS:")
        if start_node and start_node in self.graph.adjacency_list:
            self.parent_map = bfs(self.graph, start_node)
            self.traversal_index = 0  # Reset traversal index
            messagebox.showinfo("BFS Traversal Order", f"Traversal order: {list(self.parent_map.keys())}")
            self.show_bfs_progress()  # Open BFS visualization window
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


if __name__ == "__main__":
    root = tk.Tk()
    app = AlgorithmVisualizerApp(root)
    root.mainloop()
