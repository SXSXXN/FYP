import random
import numpy as np


def load():
    x = np.random.rand(1000, 10)  # generate random input data
    # generate random output data
    y = np.dot(x, np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
    return x, y


# Compare this snippet from test.py:
train_x, train_y = load()
print(train_x)
