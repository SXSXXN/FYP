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


def is_array_in_list(target_array, array_list):
    for array in array_list:
        if np.array_equal(target_array, array):
            return True
    return False


# Initialize an empty list to store the test images and labels
test_images_subset = []
test_labels_subset = []

# Iterate through the test data
for image, label in zip(X_test, y_test):
    if len(test_images_subset) == 10:
        break

    # Check if the label is not already in the subset
    if not is_array_in_list(label, test_labels_subset):
        # Add the image and label to the subset
        test_images_subset.append(image)
        test_labels_subset.append(label)

# Convert the test image subset to a numpy array
test_images_subset = np.array(test_images_subset)
test_labels_subset = np.array(test_labels_subset)

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
agent_batches = np.split(X_train, 10*N)
agent_batches_labels = np.split(y_train, 10*N)


G = nx.Graph()

agents = range(N)  # Assuming agent IDs are integers from 0 to 99
G.add_nodes_from(agents)

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
