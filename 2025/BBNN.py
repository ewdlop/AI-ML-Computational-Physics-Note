import networkx as nx
import matplotlib.pyplot as plt

# Create a small-scale Boltzmann Brain Neural Network
G = nx.DiGraph()

# Define nodes (representing neurons)
nodes = ["Input1", "Input2", "Hidden1", "Hidden2", "Output"]

# Add nodes to graph
G.add_nodes_from(nodes)

# Define edges (representing synapses)
edges = [
    ("Input1", "Hidden1"), ("Input1", "Hidden2"),
    ("Input2", "Hidden1"), ("Input2", "Hidden2"),
    ("Hidden1", "Output"), ("Hidden2", "Output"),
    ("Hidden1", "Hidden2"), ("Hidden2", "Hidden1")  # Recurrent connections
]

# Add edges to graph
G.add_edges_from(edges)

# Draw the network
plt.figure(figsize=(6, 6))
pos = nx.spring_layout(G, seed=42)  # Layout for better visualization
nx.draw(G, pos, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10, font_weight="bold")
plt.title("Boltzmann Brain Neural Network (Simplified)")
plt.show()
