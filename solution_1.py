import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist
import wandb

wandb.init(project="DA6401_Deep_Learning_Assignment1_DA24S020")

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

images = []
for i in range(len(class_names)):
    idx = np.where(y_train == i)[0][0]
    images.append(X_train[idx])

wandb.log({"Question 1": [wandb.Image(img, caption=caption) for img, caption in zip(images, class_names)]})