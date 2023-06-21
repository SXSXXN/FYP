import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import math

import adaption


random.seed(3)

# Set the number of nodes
N = 100

# Probability of being a one-way edge
p = 0.5

# p = 2 * math.log(N) / N

# preferential attachment

# Create an undirected Barabasi-Albert graph with 5 edges per node
BA = nx.barabasi_albert_graph(N, 5)

# Convert the graph to a directed graph
# BA = nx.DiGraph(BA)

# create a directed graph with edges pointing in one direction
G = nx.DiGraph()

for u, v in BA.edges():
    if random.random() < p:
        # Randomly choose the direction of the edge
        if random.random() < 0.5:
            G.add_edges_from([(u, v)])
        else:
            G.add_edges_from([(v, u)])
    else:
        # Keep the edge as a bidirectional edge
        G.add_edge(u, v)
        G.add_edge(v, u)

# Generate an Erdős-Rényi random graph
# G = nx.erdos_renyi_graph(N, p, directed=True)

# Set the number of malicious agents to be 1.5x number of nodes and select their nodes
num_malicious = 60
malicious_nodes = random.sample(range(N), num_malicious)

# Draw the graph with the malicious nodes highlighted
# nx.draw(G, node_color=[
#     'r' if node in malicious_nodes else 'b' for node in G.nodes], with_labels=True)
# plt.show()


# Define the true weights and bias for the linear equation y = wx + b
w_true = 2.0
b_true = 2.0

# generate data for simple linear regression problem
x = np.linspace(0, 10, 100).reshape(-1, 1)
y = w_true*x + b_true + np.random.randn(100, 1)

# initialize model parameters
w = np.random.randn(N, 1)
b = np.random.randn(N, 1)

# initialize local model parameters
w_local = np.zeros((N, 1))
b_local = np.zeros((N, 1))

# initialize learning rate
learning_rate = 0.01

# initialize number of local updates
n_iterations = 50

T = 100  # number of time steps


beta = np.zeros((N, N))

p = np.ones((N, N))

# Initialize stochastic observations of trust
alpha = {}

# Initialize the error list
w_error_list = []
b_error_list = []

average_w_error_list = [0]*T
average_b_error_list = [0]*T

final_w_error_list = []
final_b_error_list = []

trials = 10  # number of trials

# loop over trials
for trial in range(trials):
    # Loop over time steps
    for t in range(T):
        # Model the trust observations alpha
        for node in G.nodes:
            if node in malicious_nodes:
                # malicious node：E[α] = 0.45
                alpha[node] = random.uniform(0.25, 0.65)
            else:
                # benign node: E[α] = 0.55
                alpha[node] = random.uniform(0.35, 0.75)

        # Determine in-neighbors trust vector
        for i in range(N):
            for j in G.predecessors(i):
                # benign agent (follows the strategy)
                if i not in malicious_nodes:
                    # Calculate aggregate trust value βij(t) for the link (j, i) at time t
                    beta[i, j] += alpha[j] - 0.5

                    if beta[i, j] > 0:  # if q ∈ N_in and βij(t) ≥ 0
                        p[i, j] = 1
                    else:  # if q ∈ N_in and βiq(t) < 0.
                        p[i, j] = 0
                else:
                    # malicious agent send the opposite of the true trustworthiness
                    if j in malicious_nodes:
                        p[i, j] = 1
                    else:
                        p[i, j] = 0

        # Determine out-neighbors trust vector
        for i in range(N):
            # for agents that are out-neighbors of i but not in-neighbors of i
            for j in G.successors(i):
                if j not in G.predecessors(i):
                    if i not in malicious_nodes:
                        count = 0
                        for q in G.predecessors(i):
                            if j in G.predecessors(q):
                                p[i, j] = 0
                                p[i, j] += p[q, j]
                                count += 1
                        if count != 0:
                            p[i, j] /= count
                        else:
                            p[i, j] = 1
                    else:
                        # malicious agent send the opposite of the true trustworthiness
                        if j in malicious_nodes:
                            p[i, j] = 1
                        else:
                            p[i, j] = 0

        G_copy = G.copy()

        # Remove edges
        for i in range(N):
            if i not in malicious_nodes:
                for j in G.predecessors(i):
                    if p[i, j] < 0.5:
                        if G_copy.has_edge(j, i):
                            G_copy.remove_edge(j, i)
                for j in G.successors(i):
                    if p[i, j] < 0.5:
                        if G_copy.has_edge(i, j):
                            G_copy.remove_edge(i, j)
                # print edges of node i

        # print("Edge Count G:", G.number_of_edges())
        # print("Edge Count G_copy:", G_copy.number_of_edges())

        # Local SGD update and aggregation
        for i in range(N):
            if i not in malicious_nodes:
                w_local[i], b_local[i] = adaption.sgd(
                    x, y, w[i], b[i], learning_rate, n_iterations)
            else:
                w_local[i], b_local[i] = np.random.normal(
                    10, 5), np.random.normal(10, 5)

        # Aggregate local updates
        # ? for one agent
        for i in range(N):
            if i not in malicious_nodes:
                for j in G_copy.predecessors(i):
                    w_local[i] += w_local[j]
                    b_local[i] += b_local[j]
                if len(list(G_copy.predecessors(i))) > 0:
                    w_local[i] /= (len(list(G_copy.predecessors(i)))+1)
                    b_local[i] /= (len(list(G_copy.predecessors(i)))+1)
                else:
                    w_local[i] = w_local[i]
                    b_local[i] = b_local[i]

        # calculate the average error of w and b for only benign nodes
        benign_nodes = [
            node for node in G.nodes if node not in malicious_nodes]
        w_error = np.abs(w_local[benign_nodes] - w_true).mean()
        b_error = np.abs(b_local[benign_nodes] - b_true).mean()

        # save the average error of w and b
        w_error_list.append(w_error)
        b_error_list.append(b_error)

        # print the average error of w and b
        # print("w_error:", w_error)
        # print("b_error:", b_error)

        # update w and b
        w = w_local
        b = b_local

        """
        # Get the in-neighbors and out-neighbors of node 0
        node = 8
        in_neighbours_before = list(G.predecessors(node))
        in_neighbours_after = list(G_copy.predecessors(node))

        for k in G_copy.predecessors(node):
            w_local[node] += w_local[k]
            b_local[node] += b_local[k]

        w = w_local[node] / len(list(G_copy.predecessors(node)))
        b = b_local[node] / len(list(G_copy.predecessors(node)))
        """
    for k in range(0, T-1):
        average_w_error_list[k] = (average_w_error_list[k] + w_error_list[k])
        average_b_error_list[k] = (average_b_error_list[k] + b_error_list[k])

for i in range(0, T-1):
    final_w_error_list.append((average_w_error_list[i])/trials)
    final_b_error_list.append((average_b_error_list[i])/trials)


"""
# print node in-neighbors before and after the attack
print("In-neighbors before:", in_neighbours_before)
print("In-neighbors after:", in_neighbours_after)

# print node out-neighbors before and after the attack
print("Out-neighbors before:", list(G.successors(node)))
print("Out-neighbors after:", list(G_copy.successors(node)))

"""


# plot error versus the number of iterations
plt.plot(final_w_error_list, label="w_error")
plt.plot(final_b_error_list, label="b_error")
plt.title("Malicious nodes = 60")
plt.xlabel("Number of iterations")
plt.ylabel("Error")
plt.yscale("log")  # set y-axis to log scale
plt.legend()
plt.grid(True)  # turn on grid
plt.show()


# print alpha
# print("alpha:", alpha)

# print in-neighbors nodes
# print("In-neighbors:", list(G.predecessors(2)))

# print in-neighbors nodes alfter cut
# print("In-neighbors after cut:", list(G_copy.predecessors(2)))

# print out-neighbors nodes
# print("Out-neighbors:", list(G.successors(2)))

# print out-neighbors nodes alfter cut
# print("Out-neighbors after cut:", list(G_copy.successors(2)))

# print beta vector for node
# print("beta:", beta[2, 17])

# print trust vector p for node
# print("p:", p[2, 17])

# print("w:", w)
