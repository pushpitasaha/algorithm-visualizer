import random
import networkx as nx
import matplotlib.pyplot as plt

class TSPGeneticAlgorithm:
    def __init__(self, graph, population_size=10, mutation_rate=0.01, generations=100):
        self.graph = graph  # Assumes graph is an object with adjacency_list, get_nodes(), and networkx Graph
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.generations = generations
        self.population = self.initialize_population()
        self.fitness_history = []

    def initialize_population(self):
        """Generate initial population with random paths."""
        nodes = list(self.graph.get_nodes())
        return [random.sample(nodes, len(nodes)) for _ in range(self.population_size)]

    def calculate_fitness(self, path):
        """Calculate the fitness of a path (inverse of total path distance)."""
        distance = 0
        for i in range(len(path)):
            # Get distance from adjacency_list; assume undirected graph with symmetric weights
            try:
                distance += self.graph.adjacency_list[path[i]][path[(i + 1) % len(path)]]
            except KeyError:
                print(f"Error: Edge ({path[i]}, {path[(i + 1) % len(path)]}) not found.")
                return 0
        return 1 / distance if distance > 0 else 0

    def selection(self):
        """Select paths based on fitness for crossover."""
        selected = sorted(self.population, key=self.calculate_fitness, reverse=True)[:self.population_size // 2]
        return selected

    def crossover(self, parent1, parent2):
        """Perform ordered crossover."""
        start, end = sorted(random.sample(range(len(parent1)), 2))
        child = [None] * len(parent1)
        child[start:end] = parent1[start:end]
        
        pointer = end
        for gene in parent2:
            if gene not in child:
                if pointer >= len(child):
                    pointer = 0
                child[pointer] = gene
                pointer += 1
        return child

    def mutate(self, path):
        """Mutate a path by swapping two cities."""
        if random.random() < self.mutation_rate:
            i, j = random.sample(range(len(path)), 2)
            path[i], path[j] = path[j], path[i]

    def evolve_population(self):
        """Run one generation of selection, crossover, and mutation."""
        selected = self.selection()
        children = []
        while len(children) < self.population_size:
            parent1, parent2 = random.sample(selected, 2)
            child = self.crossover(parent1, parent2)
            self.mutate(child)
            children.append(child)
        self.population = children

    def find_best_path(self):
        """Run the genetic algorithm and return the best path found."""
        best_path = None
        best_fitness = 0
        for generation in range(1, self.generations + 1):
            self.evolve_population()
            current_best_path = max(self.population, key=self.calculate_fitness)
            current_best_fitness = self.calculate_fitness(current_best_path)
            if current_best_fitness > best_fitness:
                best_fitness = current_best_fitness
                best_path = current_best_path
            self.fitness_history.append(best_fitness)
            if generation % 10 == 0:
                print(f"Generation {generation}: Best fitness = {best_fitness}")
        return best_path

    def visualize_path(self, path, ax):
        """Visualize the best path on the graph."""
        # Draw all nodes and edges in a light color
        nx.draw_networkx(
            self.graph.graph_nx, self.graph.pos, ax=ax, with_labels=True, node_color='skyblue', edge_color='lightgrey'
        )

        # Highlight the edges in the current best path
        path_edges = [(path[i], path[i + 1]) for i in range(len(path) - 1)] + [(path[-1], path[0])]
        nx.draw_networkx_edges(
            self.graph.graph_nx, self.graph.pos, edgelist=path_edges, ax=ax, edge_color='red', width=2
        )

