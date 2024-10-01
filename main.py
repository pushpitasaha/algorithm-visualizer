from src.gui import AlgorithmVisualizerApp
import tkinter as tk

if __name__ == "__main__":
    
    print("Starting the Algorithm Visualizer...")

    # Create the main application window
    root = tk.Tk()
    # Initialize the Algorithm Visualizer application
    app = AlgorithmVisualizerApp(root)
    # Start the main event loop of the application
    root.mainloop()