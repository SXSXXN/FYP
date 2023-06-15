import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import tensorflow as tf


# Load the MNIST dataset
mnist = tf.keras.datasets.mnist  # 28x28 images of hand-written digits 0-9
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# Preprocess the data
X_train = X_train.reshape(-1, 784) / 255.0  # Reshape and normalize
X_test = X_test.reshape(-1, 784) / 255.0
y_train = tf.keras.utils.to_categorical(y_train)  # One-hot encode the labels
y_test = tf.keras.utils.to_categorical(y_test)

# plt.imshow(X_train[0].reshape(28, 28), cmap='gray')
# plt.show()


# Set the number of nodes
N = 100

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
# Randomly select split points
split_points = np.sort(np.random.choice(
    np.arange(len(X_train)), N-1, replace=False))

# Add the start and end points to the split points
split_points = np.concatenate(([0], split_points, [len(X_train)]))

# Split the array based on the split points
agent_batches = [X_train[split_points[i]:split_points[i+1]] for i in range(N)]
agent_batches_labels = [
    y_train[split_points[i]:split_points[i+1]] for i in range(N)]

test_batches = [X_test[split_points[i]:split_points[i+1]] for i in range(N)]
test_batches_labels = [
    y_test[split_points[i]:split_points[i+1]] for i in range(N)]


# Define the range of resolutions for agents
min_resolution = 8
max_resolution = 28


agent_batches_resized = [[[] for _ in range(N)] for _ in range(N)]

# Randomly assign resolutions to each agent
resolutions = []
for _ in range(100):  # Assuming 100 agents
    resolution = random.randint(min_resolution, max_resolution)
    resolutions.append((resolution, resolution))

model = []
local_model_weights = []
# Resize images for each agent
resized_agent_data = []
for i in range(len(agent_batches)):
    agent_images = []
    for vector in agent_batches[i]:
        image = np.reshape(vector, (28, 28))
        # Add channel dimension if missing
        image = np.expand_dims(image, axis=-1)
        resolution = resolutions[i]
        resized_image = tf.image.resize(image, resolution)
        resized_vector = np.reshape(resized_image, (-1,))
        agent_images.append(resized_vector)
    resized_agent_data.append(
        (np.array(agent_images)))

    # Define the neural network architecture
    model.append(tf.keras.models.Sequential([
        tf.keras.layers.Dense(128, activation='relu',
                              input_shape=(resolution[0]*resolution[1],)),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ]))

    model[i].compile(optimizer='adam', loss='categorical_crossentropy',
                     metrics=['accuracy'])

    local_model_weights.append(model[i].get_weights())
    model_weights = local_model_weights

batch_size = 600
epochs = 10

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

alpha = np.zeros((N, N))
beta = np.zeros((N, N))

p = np.ones((N, N))


# Initialize accuracy list
accuracy = []

# Initialize an empty list to store the test images and labels
test_images_subset = [[] for _ in range(N)]
test_labels_subset = [[] for _ in range(N)]

# Define the desired number of images per label
images_per_label = 1


for node in G.nodes:
    # Count the number of images per label for each agent
    label_counts = {label: 0 for label in range(10)}
    # Iterate through the test data
    for image, label in zip(resized_agent_data[node], agent_batches_labels[node]):
        # Check if we have already collected the desired number of images per label
        if all(count >= images_per_label for count in label_counts.values()):
            break

        # Check if the current label has been collected enough times
        label_index = np.argmax(label)
        if label_counts[label_index] < images_per_label:
            # Add the image and label to the subset
            test_images_subset[node].append(image)
            test_labels_subset[node].append(label)
            label_counts[label_index] += 1

    # Convert the test image subset to a numpy array
    test_images_subset[node] = np.array(test_images_subset[node])
    test_labels_subset[node] = np.array(test_labels_subset[node])

# Loop over time steps
for t in range(T):

    # Determine in-neighbors trust vector
    for i in range(N):

        for j in G.predecessors(i):

            if i not in malicious_nodes:  # benign agent (follows the strategy)
                weights_A = local_model_weights[i]
                weights_B = local_model_weights[j]

                # Check if the shapes of the weights are compatible
                if weights_A[0].shape != weights_B[0].shape:
                    # Resize the weights of Agent B to match the shape of Agent A's weights
                    resized_weights_B = []
                    for weight_A, weight_B in zip(weights_A, weights_B):
                        resized_weight_B = np.resize(weight_B, weight_A.shape)
                        resized_weights_B.append(resized_weight_B)
                    weights_B = resized_weights_B

                model[i].set_weights(weights_B)
                _, alpha[i, j] = model[i].evaluate(
                    test_images_subset[i], test_labels_subset[i], verbose=0)

                # Calculate aggregate trust value βij(t) for the link (j, i) at time t
                beta[i, j] += alpha[i, j] - 0.5

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
            model[i].set_weights(model_weights[i])
            model[i].fit(resized_agent_data[i], agent_batches_labels[i],
                         epochs=epochs, verbose=0)
            local_model_weights[i] = model[i].get_weights()
            test_labels_subset[i] = tf.keras.utils.to_categorical(np.argmax(
                model[i].predict(test_images_subset[i], verbose=0), axis=-1), num_classes=10)
        else:
            local_model_weights[i] = [np.random.normal(
                size=w.shape) for w in model[i].get_weights()]

    # Aggregation
    for i in range(N):
        if i not in malicious_nodes:
            for j in G_copy.predecessors(i):
                # benign agent (follows the strategy)
                if i not in malicious_nodes:
                    weights_A = local_model_weights[i]
                    weights_B = local_model_weights[j]

                    # Check if the shapes of the weights are compatible
                    if weights_A[0].shape != weights_B[0].shape:
                        # Resize the weights of Agent B to match the shape of Agent A's weights
                        resized_weights_B = []
                        for weight_A, weight_B in zip(weights_A, weights_B):
                            resized_weight_B = np.resize(
                                weight_B, weight_A.shape)
                            resized_weights_B.append(resized_weight_B)
                        weights_B = resized_weights_B

                weights_A = weights_A + \
                    weights_B
            if len(list(G_copy.predecessors(i))) > 0:
                local_model_weights[i] = np.divide(
                    local_model_weights[i], (len(list(G_copy.predecessors(i)))+1))
        else:
            local_model_weights[i] = local_model_weights[i]

    # calculate the average accuracy for only benign nodes
    total_acc = 0

    for i in range(N):
        if i not in malicious_nodes:
            model[i].set_weights(local_model_weights[i])
            # resize testing images X_test
            # Create an empty list to store the resized/reshaped images
            resized_images = []

            # Iterate over each image in 'X_test' and resize/reshape it
            for image in X_test:
                resized_image = np.resize(
                    image, (local_model_weights[i][0].shape[0]))
                resized_images.append(resized_image)

            # Convert the list of resized/reshaped images into a numpy array
            resized_images = np.array(resized_images)

            _, acc = model[i].evaluate(resized_images, y_test, verbose=0)
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