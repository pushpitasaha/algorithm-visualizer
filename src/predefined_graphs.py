import os
import json
from tkinter import messagebox, simpledialog
from src.graph import Graph

# Correct path to the JSON file
JSON_PATH = os.path.join(os.path.dirname(__file__), '../assets/sample_graph.json')

def load_graph_from_edges(edges):
    graph = Graph()
    for edge in edges:
        graph.add_edge(*edge)
    return graph

def get_all_predefined_graphs():
    # Ensure sample_graphs.json is present in the correct path
    with open(JSON_PATH, 'r') as file:
        graph_data = json.load(file)
    
    predefined_graphs = {}
    for graph_name, edges in graph_data.items():
        predefined_graphs[graph_name] = load_graph_from_edges(edges)
    
    return predefined_graphs

def select_predefined_graph(self):
    # Ensure this function is correctly bound to the button in the UI
    try:
        predefined_graphs = get_all_predefined_graphs()
        graph_names = list(predefined_graphs.keys())

        # Show a selection window or prompt for predefined graphs
        selected_graph_name = simpledialog.askstring("Select Predefined Graph", "Available graphs: " + ", ".join(graph_names))
        if selected_graph_name and selected_graph_name in predefined_graphs:
            self.graph = predefined_graphs[selected_graph_name]
            self.visualize_graph()
        else:
            messagebox.showerror("Error", "Invalid graph selection")
    except FileNotFoundError as e:
        messagebox.showerror("Error", f"Predefined graphs file not found: {str(e)}")

