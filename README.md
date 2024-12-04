# Algorithm Visualizer

Algorithm Visualizer is a Python desktop application designed to help students and educators understand fundamental graph and genetic algorithms. Through interactive, step-by-step visualizations, it provides insights into algorithms such as Breadth-First Search (BFS), Depth-First Search (DFS), Dijkstra's Algorithm, and the Traveling Salesman Problem (TSP) using genetic algorithms. This tool enhances the learning experience by making complex algorithmic processes tangible and accessible.

![Algorithm Visualizer](assets/bfs.gif)

## Table of Contents
1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [License](#license)

## Features

- **Algorithm Visualizations**:
  - **Graph Algorithms**:
    - Breadth-First Search (BFS): Step-by-step traversal showing each node visited.
    - Depth-First Search (DFS): Depth-oriented traversal with real-time backtracking.
    - Dijkstra's Algorithm: Visualizes shortest path calculations with priority queue updates.
  - **Genetic Algorithm for TSP**:
    - Demonstrates stages such as selection, crossover, mutation, and evolution.
    - Adjustable parameters for population size, mutation rate, and generations.
- **User Interface**:
  - Easy-to-use GUI built with Tkinter for selecting algorithms and configuring settings.
  - Customizable graph input, including the option to input custom graphs or use predefined examples.
  - Playback controls for pausing, restarting, and stepping through algorithms.
- **Real-Time Metrics**:
  - Displays algorithm time and space complexity, along with fitness metrics for genetic algorithms.
- **Export Options**:
  - Export visualizations as images (e.g., PNG) or text summaries (e.g., TXT).

## Technologies Used

- **Programming Language**: Python 3
- **Libraries**:
  - **Tkinter**: GUI development.
  - **Matplotlib**: Graph visualizations and animations.
- **Development Tools**:
  - **IDE**: Visual Studio Code (v1.93).
  - **Version Control**: Git and GitHub.

## Installation

To get the Algorithm Visualizer up and running locally:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/pushpitasaha/algorithm-visualizer.git
   cd algorithm-visualizer
2. **Install Dependencies:** Ensure Python 3 is installed. Then install the necessary libraries -
   ```bash
   pip install -r requirements.txt
3. **Run the Application**: Start the application by running -
   ```bash
   .\venv\Scripts\activate
   python main.py

## Usage

1. Selecting Algorithms:
    - Choose from BFS, DFS, Dijkstra's Algorithm, or the TSP genetic algorithm using the main dashboard.
      
2. Inputting Graphs:
  - Enter a custom graph as node pairs or select a predefined template. For genetic algorithms (TSP), adjust parameters like population size, mutation rate, and number of generations.
      
3. Starting Visualization:
  - Pass a "Start" node to initiate the visualization. Use "Back" and "Next "controls to reset, or step through the algorithm.

4.  Viewing Metrics:
   - Real-time performance metrics, including time and space complexity, will be displayed during the algorithm execution (work-in-progress).

5. Exporting Visualizations:
   - Save visualizations as images or text summaries for further study (work-in-progress).

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Project Timeline and Development Plan
The project is developed with a series of checkpoints to showcase completed functionalities. Key milestones for Q3 2024 include:

  - Checkpoint 1 (Oct 1): Demonstration of the BFS algorithm and basic GUI layout.
  - Checkpoint 2 (Oct 22): Completion of DFS and Dijkstra's Algorithm visualizations.
  - Checkpoint 3 (Nov 12): TSP genetic algorithm visualization and parameter adjustment.
  - Checkpoint 4 (Dec 3): Final user interface, export options, and performance metrics.
    
Future enhancements may include additional algorithms and expanded visualization features.
