import numpy as np


def mean(w, neighbor_weights):
    # Update the model by averaging the weights of the neighbor models
    for neighbor_w in neighbor_weights:
        w += neighbor_w

    w = np.divide(w, len(neighbor_weights) + 1)
    return w
