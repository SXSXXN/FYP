import numpy as np
import matplotlib.pyplot as plt
from adaption import local_sgd

# Generate some random data with a trend
X = np.linspace(0, 10, 100).reshape(-1, 1)
y = 2*X + np.random.randn(100, 1)

# Initialize the model parameters
w = np.zeros((1, 1))
b = 0
lr = 0.01
batch_size = 10
local_updates = 10

# Perform local SGD
w, b = local_sgd(X, y, w, b, lr, batch_size, local_updates)

# Compute the predictions for the entire dataset
y_pred = np.dot(X, w) + b

# Plot the data and the regression line
plt.scatter(X, y, color='blue')
plt.plot(X, y_pred, color='red')
plt.show()
