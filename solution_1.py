import wandb
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import fashion_mnist

wandb.login()
# 1. Initialize wandb
wandb.init(
    project="DA24S020_DA6401_Deep_Learning_Assignment1",
    name="solution_1: Fashion-MNIST samples",
)

# 2. Load Fashion-MNIST data
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

# 3. Create a mapping from class indices to class names
class_names = {
    0: 'T-shirt/Top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle Boot'
}

# 4. Find one sample per class
sample_indices = []
for class_idx in range(10):
    idx = np.where(y_train == class_idx)[0][0]
    sample_indices.append(idx)

# 5. Plot a 2×5 grid of these samples (locally)
fig, axes = plt.subplots(2, 5, figsize=(10, 5))
axes = axes.flatten()
for i, ax in enumerate(axes):
    idx = sample_indices[i]
    ax.imshow(x_train[idx], cmap='gray')
    ax.set_title(class_names[y_train[idx]])
    ax.axis('off')
plt.tight_layout()
plt.show()

# 6. Log the entire figure to W&B (optional)
wandb.log({"fashion_mnist_grid": wandb.Image(fig, caption="Fashion-MNIST Grid")})

# 7. Log each image individually so you get a scrollable gallery in W&B
images = []
for idx in sample_indices:
    images.append(
        wandb.Image(
            x_train[idx],
            caption=f"Class: {class_names[y_train[idx]]}"
        )
    )

wandb.log({"fashion_mnist_samples": images})
