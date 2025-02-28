import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

# Load the Fashion-MNIST dataset
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

# Define the Fashion-MNIST classes
class_names = [
    "T-shirt/top",
    "Trouser",
    "Pullover",
    "Dress",
    "Coat",
    "Sandal",
    "Shirt",
    "Sneaker",
    "Bag",
    "Ankle boot"
]

# Prepare a figure with 2 rows and 5 columns (for 10 total classes)
plt.figure(figsize=(10, 5))

# Find and plot one sample per class
for label in range(10):
    # Get the first index in X_train that corresponds to the current label
    idx = np.where(y_train == label)[0][0]

    plt.subplot(2, 5, label + 1)
    plt.imshow(X_train[idx], cmap='gray')
    plt.title(class_names[label])
    plt.axis('off')  # Hide axis ticks

plt.tight_layout()
plt.show()
