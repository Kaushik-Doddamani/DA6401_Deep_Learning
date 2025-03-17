import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX', etc. Use "Agg" for non-interactive plot
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from model import NeuralNetwork
from utils import get_minibatches, evaluate_model, load_fashion_mnist, one_hot_encode
from optimizers import create_optimizer_from_config


# Best hyperparams obtained from the sweeps of the previous tasks
best_params = {
    "activation": "relu",
    "batch_size": 64,
    "epochs": 10,
    "hidden_size": 128,
    "learning_rate": 0.001,
    "num_hidden_layers": 4,
    "optimizer": "nadam",
    "weight_decay": 0.0,
    "weight_init": "xavier"
}


def build_best_model(best_params):
    """
    Creates and returns a NeuralNetwork with hyperparams from `best_params`.
    best_params = dict with keys:
        - activation, batch_size, epochs, hidden_size, learning_rate,
          num_hidden_layers, optimizer, weight_decay, weight_init
        e.g.:
            {
                "activation": "relu", -
                "batch_size": 64,
                "epochs": 10,
                "hidden_size": 128, -
                "learning_rate": 0.001, -
                "num_hidden_layers": 4, -
                "optimizer": "nadam", -
                "weight_decay": 0.0, -
                "weight_init": "xavier" -
            }
    Returns: an untrained NeuralNetwork instance.
    """
    # 1) Construct layer sizes: e.g. 784 -> multiple hidden layers -> 10
    hidden_layers = [best_params["hidden_size"]] * best_params["num_hidden_layers"]
    layer_sizes = [784] + hidden_layers + [10]

    # 2) Create the network
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=best_params["activation"],
        optimizer=None,  # We'll set it below
        seed=42
    )

    # 3) Weight init
    if best_params["weight_init"] == "xavier":
        for i in range(nn.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            nn.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            nn.params[f'b{i}'] = np.zeros(out_dim)
    # else: default random init is used (already implemented in Neural Network constructor)

    # 4) Create the optimizer
    opt_name = best_params["optimizer"]
    lr = best_params["learning_rate"]
    wd = best_params["weight_decay"]

    optimizer = create_optimizer_from_config(opt_name, lr, weight_decay=wd)
    nn.optimizer = optimizer
    return nn


def train_model(model, X_train, y_train, batch_size, epochs):
    """
    Trains the given model on (X_train, y_train) for 'epochs' epochs in mini-batches.
    We assume y_train is one-hot.
    """
    for epoch in range(epochs):
        minibatches = get_minibatches(X_train, y_train, batch_size, shuffle=True)
        for X_batch, Y_batch in minibatches:
            loss = model.train_batch(X_batch, Y_batch)
        print(f"Epoch {epoch+1}, loss: {loss:.4f}")
    print("Training complete!")


def plot_confusion_matrix(model, X_data, y_data, class_labels=None, batch_size=64):
    """
    Plots a confusion matrix for `model` predictions on (X_data, y_data).
    If class_labels is provided (e.g. ["T-shirt/top", "Trouser", ...]),
    we label the axes accordingly.
    """
    # 1) Predict
    caches = model.forward(X_data)
    H_last = caches[f'H{model.num_layers}']
    preds = np.argmax(H_last, axis=1)
    true = np.argmax(y_data, axis=1)

    # 2) Confusion matrix
    cm = confusion_matrix(true, preds)

    # 3) Plot
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix")
    if class_labels is not None:
        plt.xticks(ticks=np.arange(len(class_labels))+0.5, labels=class_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(class_labels))+0.5, labels=class_labels, rotation=45)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.show()


def plot_confusion_matrix_creative(model, X_data, Y_data, class_labels, batch_size=64):
    """
    - Plots TWO confusion matrices side by side:
      1) Raw counts
      2) Row-normalized (percentage)
    - Adds color coding, axis labels, etc.
    - 'class_labels' is a list of strings for the 10 classes.
    """
    # Step A: gather predictions for the entire dataset
    all_preds = []
    all_actual = []
    minibatches = get_minibatches(X_data, Y_data, batch_size, shuffle=False)
    for Xb, Yb in minibatches:
        caches = model.forward(Xb)
        H_last = caches[f'H{model.num_layers}']
        preds = np.argmax(H_last, axis=1)
        actual = np.argmax(Yb, axis=1)

        all_preds.extend(preds)
        all_actual.extend(actual)

    all_preds = np.array(all_preds)
    all_actual = np.array(all_actual)

    # Step B: compute confusion matrix
    cm = confusion_matrix(all_actual, all_preds)

    # Step C: create a normalized version (row-wise)
    cm_norm = cm.astype(np.float32)
    for i in range(cm_norm.shape[0]):
        row_sum = np.sum(cm_norm[i])
        if row_sum > 0:
            cm_norm[i] = cm_norm[i] / row_sum

    # Step D: set up a figure with 2 subplots
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (1) Raw Confusion Matrix
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    # rotate x-ticks if needed
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(45)
    for tick in axes[0].get_yticklabels():
        tick.set_rotation(45)

    # (2) Normalized Confusion Matrix
    # We'll show percentages with 2 decimal places
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[1].set_title("Confusion Matrix (Row %)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)
    for tick in axes[1].get_yticklabels():
        tick.set_rotation(45)

    # add some tight layout
    plt.tight_layout()
    plt.show()


def main():
    # 1) Load data
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    y_train_onehot = one_hot_encode(y_train, 10)
    y_test_onehot  = one_hot_encode(y_test, 10)

    # 2) Build the best model
    model = build_best_model(best_params)

    # 3) Train
    train_model(model, X_train, y_train_onehot,
                batch_size=best_params["batch_size"],
                epochs=best_params["epochs"])

    # 4) Evaluate test set
    test_loss, test_acc = evaluate_model(model, X_test, y_test_onehot, best_params["batch_size"])
    print(f"Best Model Test accuracy: {test_acc*100:.2f}%")

    # 5) Plot confusion matrix
    class_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]
    plot_confusion_matrix(model, X_test, y_test_onehot, class_labels=class_labels,
                          batch_size=best_params["batch_size"])
    plot_confusion_matrix_creative(model, X_test, y_test_onehot, class_labels, batch_size=best_params["batch_size"])


if __name__ == "__main__":
    main()
