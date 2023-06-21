import networkx as nx
import matplotlib.pyplot as plt
import random

# Set the number of nodes and edges
N = 30
num_edges_per_node = 5

# Create an undirected Barabasi-Albert graph with 5 edges per node
BA = nx.barabasi_albert_graph(N, num_edges_per_node)

# Convert the graph to a directed graph
G_before = nx.DiGraph(BA)

# Create a directed graph for the modified version
G_after = nx.DiGraph()

# Randomly add malicious agents to the graph
num_malicious = 15
malicious_nodes = random.sample(range(N), num_malicious)

# Iterate through the edges of the original graph and modify them in the new graph
for u, v in BA.edges():
    if random.random() < 0.5:
        G_after.add_edges_from([(u, v)])
    else:
        G_after.add_edges_from([(v, u)])

# Set the positions of the nodes for visualization
pos = nx.spring_layout(G_before)

# Create a figure with two subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

# Draw the graph before adding malicious agents
nx.draw_networkx(G_after, pos, node_color='lightblue',
                 edge_color='gray', arrows=True, ax=ax1, with_labels=False)
# ax1.set_title("Graph Before Adding Malicious Agents")

# Draw the graph after adding malicious agents
node_colors = [
    'red' if node in malicious_nodes else 'lightblue' for node in G_after.nodes]
nx.draw_networkx(G_after, pos, node_color=node_colors,
                 edge_color='gray', arrows=True, ax=ax2, with_labels=False)
# ax2.set_title("Graph After Adding Malicious Agents")

# Create a legend
legend_labels = {'Benign Agents': 'lightblue', 'Malicious Agents': 'red'}
legend_handles = [plt.Line2D([], [], marker='o', color='lightblue', linestyle='None', markersize=10),
                  plt.Line2D([], [], marker='o', color='red', linestyle='None', markersize=10)]
fig.legend(legend_handles, legend_labels, loc='upper center')

# Adjust spacing between subplots
plt.tight_layout()

# Display the figure
plt.show()
