import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 784) / 255.0  # Reshape and normalize
X_test = X_test.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)  # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test)

# plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
# plt.show()

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(128, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Set the number of nodes
N = 100

initial_model_weights = model.get_weights()

zero_model_weights = np.zeros_like(initial_model_weights)

local_model_weights = np.tile(initial_model_weights, (N, 1, 1))
model_weights = np.tile(initial_model_weights, (N, 1, 1))

batch_size = 600
epochs = 10
"""
# Determine the number of batches
num_batches = int(len(X_train) / batch_size)

# Shuffle the training data and labels randomly
indices = tf.range(start=0, limit=tf.shape(X_train)[0], dtype=tf.int32)
shuffled_indices = tf.random.shuffle(indices)
x_train = tf.gather(X_train, shuffled_indices)
y_train = tf.gather(y_train, shuffled_indices)

# Divide the training data and labels into batches for each agent
x_train_batches = tf.split(x_train, num_batches)
y_train_batches = tf.split(y_train, num_batches)

# Assign each batch to a different agent
agent_batches = []
for i in range(100):
    agent_batches.append((x_train_batches[i], y_train_batches[i]))

"""

# split the data into 100 batches
agent_batches = np.split(X_train, N)
agent_batches_labels = np.split(y_train, N)


# Probability of being a one-way edge
p = 0.5

# Create an undirected Barabasi-Albert graph with 5 edges per node
BA = nx.barabasi_albert_graph(N, 5)

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


# Set the number of malicious agents to be 1.5x number of nodes and select their nodes
num_malicious = 50
malicious_nodes = random.sample(range(N), num_malicious)

T = 50  # number of time steps


beta = np.zeros((N, N))

p = np.ones((N, N))

# Initialize stochastic observations of trust
alpha = {}

# Initialize accuracy list
accuracy = []

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
            if i not in malicious_nodes:  # benign agent (follows the strategy)
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

    print("Edge Count G:", G.number_of_edges())
    print("Edge Count G_copy:", G_copy.number_of_edges())

    # Local training
    for i in range(N):
        if i not in malicious_nodes:
            model.set_weights(model_weights[i, 0])
            model.fit(agent_batches[i], agent_batches_labels[i],
                      epochs=epochs, verbose=0)
            local_model_weights[i] = model.get_weights()
        else:
            local_model_weights[i] = [np.random.normal(
                size=w.shape) for w in model.get_weights()]

    # Aggregation
    for i in range(N):
        if i not in malicious_nodes:
            for j in G_copy.predecessors(i):
                local_model_weights[i] = local_model_weights[i] + \
                    local_model_weights[j]
            if len(list(G_copy.predecessors(i))) > 0:
                local_model_weights[i] = np.divide(
                    local_model_weights[i], (len(list(G_copy.predecessors(i)))+1))
        else:
            local_model_weights[i] = local_model_weights[i]

    # calculate the average accuracy for only benign nodes
    total_acc = 0

    for i in range(N):
        if i not in malicious_nodes:
            model.set_weights(local_model_weights[i, 0])
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            # average accuracy
            total_acc += acc
    total_acc /= (N-num_malicious)

    accuracy.append(total_acc)

    print("Accuracy:", total_acc)

    # Update the global model weights
    model_weights = local_model_weights


# Plot the accuracy
plt.plot(accuracy)
plt.xlabel("Number of iterations")
plt.ylabel("Accuracy")
plt.title("MNIST Accuracy")
plt.grid(True)  # turn on grid
plt.show()
