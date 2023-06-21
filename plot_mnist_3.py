import random
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

# Load the MNIST dataset
(train_images, train_labels), (_, _) = mnist.load_data()

# Define the number of example digits to display
num_examples = 20  # Change this value to display a different number of random digits

# Select a random sequence of example digits from the dataset
random.seed(24)  # Set a seed for reproducibility
random_digits = random.choices(range(10), k=num_examples)

# Create a grid of subplots for the example digits
num_rows = (num_examples - 1) // 10 + 1
fig, axs = plt.subplots(num_rows, 10, figsize=(10, 2))

# Plot the example digits
for i, digit in enumerate(random_digits):
    # Find all occurrences of the digit in the dataset
    indices = [j for j, label in enumerate(train_labels) if label == digit]

    # Select a random image of the digit
    index = random.choice(indices)
    image = train_images[index]

    # Reduce the resolution of the image
    low_res_image = image[::3, ::3]

    # Plot the low-resolution digit image in the appropriate subplot
    axs[i // 10, i % 10].imshow(low_res_image, cmap='gray')
    # axs[i // 4, i % 4].set_title(f"Digit {digit}")
    axs[i // 10, i % 10].axis('off')

# Adjust the spacing between subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

# Show the graph
plt.show()
