import combination
import numpy as np

w = np.array([2, 5])
neighbor_weights = np.array([1, 2]), np.array([3, 4]), np.array([5, 6])


w_new = combination.mean(w, neighbor_weights)
print(w_new)
