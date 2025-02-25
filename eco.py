import numpy as np
import networkx as nx
import heapq
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import random

# Simulating IoT Sensor Data for Smart Bins
class SmartBin:
    def __init__(self, bin_id, location):
        self.bin_id = bin_id
        self.location = location  # (x, y) coordinates
        self.fill_level = random.randint(0, 100)  # Simulated fill level (0-100%)

    def update_fill_level(self):
        """ Simulates the change in bin fill level over time """
        self.fill_level = min(100, self.fill_level + random.randint(5, 20))

# Generating a set of bins
bins = [SmartBin(i, (random.randint(0, 100), random.randint(0, 100))) for i in range(10)]

# Simulated dataset for waste prediction model
X = np.array([random.randint(0, 100) for _ in range(1000)]).reshape(-1, 1)  # Previous fill levels
y = np.array([min(100, x + random.randint(5, 20)) for x in X.flatten()])  # Next fill levels

# Train a simple Linear Regression Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)

# Predict fill levels and evaluate performance
y_pred = model.predict(X_test)
print("Model Mean Absolute Error:", mean_absolute_error(y_test, y_pred))

# Graph Representation for Route Optimization
G = nx.Graph()

# Adding bins as graph nodes
for bin in bins:
    G.add_node(bin.bin_id, pos=bin.location)

# Creating random connections between bins (simulated roads)
for i in range(len(bins)):
    for j in range(i + 1, len(bins)):
        if random.random() > 0.5:  # Randomly connect some bins
            distance = np.linalg.norm(np.array(bins[i].location) - np.array(bins[j].location))
            G.add_edge(bins[i].bin_id, bins[j].bin_id, weight=distance)

# Dijkstra's Algorithm for Route Optimization
def dijkstra(graph, start_bin):
    """ Finds shortest paths from the start_bin to all other bins using Dijkstra's algorithm """
    pq = [(0, start_bin)]  # Priority queue: (distance, bin_id)
    distances = {node: float('inf') for node in graph.nodes}
    distances[start_bin] = 0
    visited = set()

    while pq:
        current_dist, current_bin = heapq.heappop(pq)

        if current_bin in visited:
            continue
        visited.add(current_bin)

        for neighbor in graph.neighbors(current_bin):
            weight = graph[current_bin][neighbor]['weight']
            new_dist = current_dist + weight

            if new_dist < distances[neighbor]:
                distances[neighbor] = new_dist
                heapq.heappush(pq, (new_dist, neighbor))

    return distances

# Finding the shortest route from the first bin
shortest_routes = dijkstra(G, bins[0].bin_id)
print("Optimized Collection Routes:", shortest_routes)
