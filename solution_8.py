import numpy as np
import matplotlib
import time

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import confusion_matrix
from model import NeuralNetwork
from utils import one_hot_encode, load_fashion_mnist, evaluate_model, get_minibatches
from optimizers import create_optimizer_from_config

# Best params obtained from previous tasks:
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

all_optimizers = ["sgd", "momentum", "nesterov", "rmsprop", "adam", "nadam"]

loss_types = ["cross_entropy", "mse"]

class_labels = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
]


# Define the best model
def build_model(loss_type, opt_name, best_params):
    """
    Creates a NeuralNetwork using best_params for #layers, hidden_size, etc.
    But sets 'loss_type' = cross_entropy/mse, 
    and sets the chosen optimizer by name.
    Also does 'xavier' init if requested in best_params.
    """
    # 1) Layer sizes
    hidden_layers = [best_params["hidden_size"]] * best_params["num_hidden_layers"]
    layer_sizes = [784] + hidden_layers + [10]

    # 2) Create the network
    nn = NeuralNetwork(layer_sizes=layer_sizes,
                       activation=best_params["activation"],
                       optimizer=None,
                       loss_type=loss_type,
                       seed=42)

    # 3) Weight_init = 'xavier'
    if best_params["weight_init"] == "xavier":
        for i in range(nn.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            nn.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            nn.params[f'b{i}'] = np.zeros(out_dim)
    # else: default random init is used (already implemented in Neural Network constructor)

    # 4) Create optimizer
    optimizer = create_optimizer_from_config(
        opt_name,
        best_params["learning_rate"],
        best_params["weight_decay"]
    )
    nn.optimizer = optimizer
    return nn


def train_model(model, X_train, Y_train, X_val, Y_val, epochs, batch_size):
    """
    Train the model, track train/val curves. Return dict with logs.
    """
    N = X_train.shape[0]
    train_loss_hist, train_acc_hist = [], []
    val_loss_hist, val_acc_hist = [], []

    for epoch in range(epochs):
        # Stratified is done once, we re-shuffle each epoch if we want
        idx = np.arange(N)
        np.random.shuffle(idx)
        X_train = X_train[idx]
        Y_train = Y_train[idx]

        # Mini-batch pass
        total_loss = 0.0
        total_count = 0
        correct = 0
        for Xb, Yb in get_minibatches(X_train, Y_train, batch_size, shuffle=False):
            loss = model.train_batch(Xb, Yb)
            total_loss += loss * Xb.shape[0]
            total_count += Xb.shape[0]

            # Prediction
            caches = model.forward(Xb)
            Y_hat = caches[f'H{model.num_layers}']
            preds = np.argmax(Y_hat, axis=1)
            actual = np.argmax(Yb, axis=1)
            correct += np.sum(preds == actual)

        train_loss = total_loss / total_count
        train_acc = correct / total_count
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)

        # Validation
        val_loss, val_acc = evaluate_model(model, X_val, Y_val, batch_size)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        print(f"Epoch {epoch + 1}, Train Loss={train_loss:.4f}, Train Acc={train_acc:.4f}, "
              f"Val Loss={val_loss:.4f}, Val Acc={val_acc:.4f}")

    return {
        "train_loss": train_loss_hist,
        "val_loss": val_loss_hist,
        "train_acc": train_acc_hist,
        "val_acc": val_acc_hist
    }


def plot_confusion_matrix_creative(net, X_data, Y_data, class_names, batch_size=64):
    """
    A creative confusion matrix with raw counts + row-normalized side by side.
    """
    from utils import get_minibatches
    preds_all, actual_all = [], []
    for Xb, Yb in get_minibatches(X_data, Y_data, batch_size, shuffle=False):
        caches = net.forward(Xb)
        Y_hat = caches[f'H{net.num_layers}']
        preds = np.argmax(Y_hat, axis=1)
        actual = np.argmax(Yb, axis=1)
        preds_all.extend(preds)
        actual_all.extend(actual)

    preds_all = np.array(preds_all)
    actual_all = np.array(actual_all)
    cm = confusion_matrix(actual_all, preds_all)

    # row-normalized
    cm_norm = cm.astype(np.float32)
    for i in range(cm_norm.shape[0]):
        row_sum = np.sum(cm_norm[i])
        if row_sum > 0:
            cm_norm[i] = cm_norm[i] / row_sum

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # (1) raw cm
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # (2) normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                xticklabels=class_names, yticklabels=class_names)
    axes[1].set_title("Confusion Matrix (Row %)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    plt.show()


def plot_results(results, epochs):
    """
    Plot train/val loss and accuracy for all results.
    """
    colors = {
        "sgd": "b",
        "momentum": "g",
        "nesterov": "r",
        "rmsprop": "c",
        "adam": "m",
        "nadam": "y"
    }
    line_styles = {
        "cross_entropy": "-",
        "mse": "--"
    }

    epoch_range = np.arange(1, epochs + 1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    ax_tl, ax_vl = axes[0, 0], axes[0, 1]  # train_loss, val_loss
    ax_ta, ax_va = axes[1, 0], axes[1, 1]  # train_acc, val_acc

    for (opt_name, lt), logs in results.items():
        c = colors.get(opt_name, "k")  # fallback black
        ls = line_styles.get(lt, "-")
        label_str = f"{opt_name}-{lt}"

        # train_loss
        ax_tl.plot(epoch_range, logs["train_loss"], color=c, linestyle=ls, marker='o', label=label_str)
        # val_loss
        ax_vl.plot(epoch_range, logs["val_loss"], color=c, linestyle=ls, label=label_str)
        # train_acc
        ax_ta.plot(epoch_range, logs["train_acc"], color=c, linestyle=ls, label=label_str)
        # val_acc
        ax_va.plot(epoch_range, logs["val_acc"], color=c, linestyle=ls, label=label_str)

    ax_tl.set_title("Train Loss")
    ax_vl.set_title("Val Loss")
    ax_ta.set_title("Train Accuracy")
    ax_va.set_title("Val Accuracy")

    for ax in [ax_tl, ax_vl, ax_ta, ax_va]:
        ax.set_xlabel("Epoch")
        ax.legend()

    plt.tight_layout()
    plt.show()


def main():
    # Load data
    X_train_full, y_train_full, X_test, y_test = load_fashion_mnist()

    # Stratified 90/10 train/validation split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, val_idx in sss.split(X_train_full, y_train_full):
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

    # one-hot
    y_train_onehot = one_hot_encode(y_train, 10)
    y_val_onehot = one_hot_encode(y_val, 10)
    y_test_onehot = one_hot_encode(y_test, 10)

    # Store results => { (opt_name, loss_type) : logs }
    results = {}

    # We fix the epochs and batch_size from best_params
    epochs = best_params["epochs"]
    batch_size = best_params["batch_size"]

    # We'll do a big loop
    for opt_name in all_optimizers:
        for lt in loss_types:
            print(f"\n===== Training with opt={opt_name}, loss_type={lt} =====")
            model = build_model(lt, opt_name, best_params)
            logs = train_model(model, X_train, y_train_onehot, X_val, y_val_onehot,
                               epochs=epochs, batch_size=batch_size)

            # final test eval
            test_loss, test_acc = evaluate_model(model, X_test, y_test_onehot, batch_size)
            logs["test_loss"] = test_loss
            logs["test_acc"] = test_acc
            results[(opt_name, lt)] = logs

    # Now we'll do some rich PLOTTING to compare them
    plot_results(results, epochs)

    # Pick "best" final combo from results: highest test_acc
    best_combo = None
    best_acc = 0
    for (opt_name, lt), logs in results.items():
        if logs["test_acc"] > best_acc:
            best_acc = logs["test_acc"]
            best_combo = (opt_name, lt)

    print(f"\nBest combo on test set: {best_combo} with test_acc={best_acc * 100:.2f}%")
    print(f"Creating a creative confusion matrix for the best combo ...")

    # CONFUSION MATRIX for the best combo
    best_opt, best_lt = best_combo
    best_model = build_model(best_lt, best_opt, best_params)

    # Re-train it (since we haven't stored the model above)
    print(f"\nRe-training best combo {best_combo} from scratch for final confusion matrix ...")
    logs = train_model(best_model, X_train, y_train_onehot, X_val, y_val_onehot, epochs=epochs, batch_size=batch_size)

    plot_confusion_matrix_creative(best_model, X_test, y_test_onehot, class_labels, batch_size=batch_size)


if __name__ == "__main__":
    main()
