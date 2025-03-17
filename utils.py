import numpy as np
from keras.datasets import fashion_mnist


def load_fashion_mnist():
    """
    Loads the Fashion-MNIST dataset using keras.datasets.fashion_mnist,
    returns training and test sets.
    """
    (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

    # X_train: (60000, 28, 28)
    # y_train: (60000,)
    # X_test:  (10000, 28, 28)
    # y_test:  (10000,)

    # Flatten each 28x28 image into a 784-dim vector
    X_train = X_train.reshape(-1, 784).astype(np.float32)  # -> (60000, 784)
    X_test = X_test.reshape(-1, 784).astype(np.float32)  # -> (10000, 784)

    # Normalize to [0, 1]
    X_train /= 255.0
    X_test /= 255.0

    return X_train, y_train, X_test, y_test


def one_hot_encode(labels, num_classes=10):
    """
    Converts integer labels into one-hot vectors.
    labels:      1D array of shape (N,)
    num_classes: number of categories (default 10 for Fashion-MNIST)
    Returns:     2D array of shape (N, num_classes)
    """
    one_hot = np.zeros((labels.shape[0], num_classes))  # (N, num_classes) -> (60000, 10)
    one_hot[np.arange(labels.shape[0]), labels] = 1.0
    return one_hot


def get_minibatches(X, y, batch_size, shuffle=True):
    """
    Generates mini-batches of data.
      - X: input features (N, d)
      - y: labels (N, num_classes) or (N,)
      - batch_size: size of each batch
      - shuffle: whether to shuffle data before batching
    Yields tuples (X_batch, y_batch) each with shape:
      X_batch: (batch_size, d)
      y_batch: (batch_size, num_classes) or (batch_size,)
    """
    N = X.shape[0]
    indices = np.arange(N)
    if shuffle:
        np.random.shuffle(indices)

    for start_idx in range(0, N, batch_size):
        end_idx = start_idx + batch_size
        batch_indices = indices[start_idx:end_idx]
        yield X[batch_indices], y[batch_indices]


def evaluate_model(model, X_data, y_data, batch_size):
    """
    Returns (loss, accuracy) for model on (X_data, y_data).
    We'll do it in mini-batches to avoid memory issues on large sets.
    """
    losses = []
    correct = 0
    total = X_data.shape[0]

    minibatches = get_minibatches(X_data, y_data, batch_size, shuffle=False)
    for Xb, Yb in minibatches:
        caches = model.forward(Xb)
        H_last = caches[f'H{model.num_layers}']
        loss = model.compute_loss(H_last, Yb)
        losses.append(loss)

        preds = np.argmax(H_last, axis=1)
        actual = np.argmax(Yb, axis=1)
        correct += np.sum(preds == actual)

    avg_loss = np.mean(losses)
    acc = correct / total
    return avg_loss, acc