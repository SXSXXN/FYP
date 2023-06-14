import networkx as nx
import matplotlib.pyplot as plt
import random

# Create a Barabási-Albert (BA) graph
BA = nx.barabasi_albert_graph(n=20, m=2)

# Probability of being a one-way edge
p = 0.5

# create a directed graph with edges pointing in one direction
ba_graph = nx.DiGraph()

for u, v in BA.edges():
    if random.random() < p:
        # Randomly choose the direction of the edge
        if random.random() < 0.5:
            ba_graph.add_edges_from([(u, v)])
        else:
            ba_graph.add_edges_from([(v, u)])
    else:
        # Keep the edge as a bidirectional edge
        ba_graph.add_edge(u, v)
        ba_graph.add_edge(v, u)


# Create an Erdős-Rényi (ER) graph
er_graph = nx.erdos_renyi_graph(n=20, p=0.1, directed=True)

# Draw the Barabási-Albert graph
plt.figure()
nx.draw(ba_graph, with_labels=False, node_color='lightblue',
        node_size=100, edge_color='gray', arrows=True)
plt.title('Barabási-Albert (BA) Graph')
plt.show()

# Draw the Erdős-Rényi graph
plt.figure()
nx.draw(er_graph, with_labels=False, node_color='lightblue',
        node_size=100, edge_color='gray', arrows=True)
plt.title('Erdős-Rényi (ER) Graph')
plt.show()
