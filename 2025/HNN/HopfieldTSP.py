import numpy as np
import matplotlib.pyplot as plt

class HopfieldTSP:
    def __init__(self, num_cities, distance_matrix, A=500, B=500, C=200):
        self.N = num_cities
        self.d = distance_matrix
        self.A = A
        self.B = B
        self.C = C
        self.V = np.random.rand(self.N, self.N)  # Initial neuron activations

    def energy(self):
        """Compute the total energy function."""
        term1 = self.A * np.sum((np.sum(self.V, axis=1) - 1) ** 2)  # Each row has one 1
        term2 = self.B * np.sum((np.sum(self.V, axis=0) - 1) ** 2)  # Each column has one 1
        term3 = 0  # Distance minimization
        for i in range(self.N):
            for j in range(self.N):
                if i != j:
                    term3 += self.d[i, j] * np.sum(self.V[i, :] * self.V[j, :])
        return term1 + term2 + self.C * term3

    def update(self, iterations=1000, lr=0.1):
        """Train the Hopfield network using gradient descent."""
        for _ in range(iterations):
            delta_V = -self.A * (np.sum(self.V, axis=1, keepdims=True) - 1)  # Enforce one per row
            delta_V += -self.B * (np.sum(self.V, axis=0) - 1)  # Enforce one per column
            delta_V += -self.C * (np.dot(self.d, self.V))  # Distance minimization
            self.V += lr * delta_V
            self.V = np.tanh(self.V)  # Activation function
        return self.get_route()

    def get_route(self):
        """Extract route from the neuron states."""
        route = []
        for row in self.V:
            route.append(np.argmax(row))  # Select the most activated neuron in each row
        return route

# Generate Random Cities and Distance Matrix
np.random.seed(42)
num_cities = 10
city_coords = np.random.rand(num_cities, 2) * 100  # Cities in a 100x100 grid
distance_matrix = np.linalg.norm(city_coords[:, np.newaxis] - city_coords[np.newaxis, :], axis=2)

# Run Hopfield TSP
hopfield_tsp = HopfieldTSP(num_cities, distance_matrix)
route = hopfield_tsp.update()

print("Optimized Route:", route)
print("Total Energy:", hopfield_tsp.energy())

# Plot the TSP Solution
def plot_tsp(route, city_coords):
    plt.figure(figsize=(8, 8))
    for i in range(len(route) - 1):
        plt.plot([city_coords[route[i]][0], city_coords[route[i + 1]][0]],
                 [city_coords[route[i]][1], city_coords[route[i + 1]][1]], 'bo-')
    plt.plot([city_coords[route[-1]][0], city_coords[route[0]][0]], 
             [city_coords[route[-1]][1], city_coords[route[0]][1]], 'bo-')  # Connect last to first
    plt.scatter(city_coords[:, 0], city_coords[:, 1], c='red', marker='o', label='Cities')
    for i, (x, y) in enumerate(city_coords):
        plt.text(x, y, str(i), fontsize=12, ha='right')
    plt.title("Optimized TSP Route using Hopfield Network")
    plt.legend()
    plt.grid()
    plt.show()

# Visualize the optimized route
plot_tsp(route, city_coords)
