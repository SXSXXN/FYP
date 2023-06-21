import numpy as np


def objective_function(x, y, w, b):
    y_hat = np.dot(x, w) + b
    error = y - y_hat
    mse = np.mean(error**2)
    return mse


def gradient_w(x, y, w, b):
    y_hat = np.dot(x, w) + b
    error = y - y_hat
    grad_w = -2 * np.mean(error * x, axis=0)
    return grad_w


def gradient_b(x, y, w, b):
    y_hat = np.dot(x, w) + b
    error = y - y_hat
    grad_b = -2 * np.mean(error)
    return grad_b


def sgd(x, y, w, b, learning_rate, n_iterations):
    for i in range(n_iterations):
        # Randomly sample a data point
        random_index = np.random.randint(0, len(x))
        x_i = x[random_index]
        y_i = y[random_index]

        w = w - learning_rate * gradient_w(x_i, y_i, w, b)
        b = b - learning_rate * gradient_b(x_i, y_i, w, b)

    return w, b
