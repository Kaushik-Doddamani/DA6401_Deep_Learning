import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX', etc.
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import wandb

from model import NeuralNetwork
from utils import get_minibatches, evaluate_model, load_fashion_mnist, one_hot_encode
from optimizers import create_optimizer_from_config

# Best hyperparams from previous tasks
best_params = {
    "activation": "relu",
    "batch_size": 64,
    "epochs": 10,
    "hidden_size": 128,
    "learning_rate": 0.001,
    "num_hidden_layers": 3,
    "optimizer": "adam",
    "weight_decay": 0.0,
    "weight_init": "xavier"
}

def build_best_model(best_params):
    # 1) Construct layer sizes
    hidden_layers = [best_params["hidden_size"]] * best_params["num_hidden_layers"]
    layer_sizes = [784] + hidden_layers + [10]

    # 2) Create the network
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=best_params["activation"],
        optimizer=None,
        seed=42
    )

    # 3) Weight init if xavier
    if best_params["weight_init"] == "xavier":
        for i in range(nn.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i+1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            nn.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            nn.params[f'b{i}'] = np.zeros(out_dim)

    # 4) Create the optimizer
    opt_name = best_params["optimizer"]
    lr       = best_params["learning_rate"]
    wd       = best_params["weight_decay"]
    optimizer = create_optimizer_from_config(opt_name, lr, weight_decay=wd)
    nn.optimizer = optimizer
    return nn

def train_model(model, X_train, y_train, batch_size, epochs):
    for epoch in range(epochs):
        minibatches = get_minibatches(X_train, y_train, batch_size, shuffle=True)
        for X_batch, Y_batch in minibatches:
            loss = model.train_batch(X_batch, Y_batch)
        print(f"Epoch {epoch+1}, loss: {loss:.4f}")
    print("Training complete!")

def plot_confusion_matrix(model, X_data, y_data, class_labels=None, batch_size=64):
    # 1) Predict
    caches = model.forward(X_data)
    H_last = caches[f'H{model.num_layers}']
    preds = np.argmax(H_last, axis=1)
    true  = np.argmax(y_data, axis=1)

    # 2) Conf matrix
    cm = confusion_matrix(true, preds)

    # 3) Plot
    fig = plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion Matrix (Simple)")
    if class_labels is not None:
        plt.xticks(ticks=np.arange(len(class_labels))+0.5,
                   labels=class_labels, rotation=45)
        plt.yticks(ticks=np.arange(len(class_labels))+0.5,
                   labels=class_labels, rotation=45)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    return fig

def plot_confusion_matrix_creative(model, X_data, Y_data, class_labels, batch_size=64):
    # gather predictions
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
    all_actual= np.array(all_actual)

    cm = confusion_matrix(all_actual, all_preds)
    cm_norm = cm.astype(np.float32)
    for i in range(cm_norm.shape[0]):
        row_sum = np.sum(cm_norm[i])
        if row_sum>0:
            cm_norm[i] /= row_sum

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # (1) raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[0],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("True")

    # (2) normalized
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Greens", ax=axes[1],
                xticklabels=class_labels, yticklabels=class_labels)
    axes[1].set_title("Confusion Matrix (Row %)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("True")

    plt.tight_layout()
    return fig

def main():
    # 1) Start wandb
    wandb.init(project="DA24S020_DA6401_Deep_Learning_Assignment1",
               name="solution_7: Confusion Matrix with wandb.plot")

    # 2) Load data
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    y_train_onehot = one_hot_encode(y_train, 10)
    y_test_onehot  = one_hot_encode(y_test,  10)

    # 3) Build best model
    model = build_best_model(best_params)

    # 4) Train
    train_model(model, X_train, y_train_onehot,
                batch_size=best_params["batch_size"],
                epochs=best_params["epochs"])

    # 5) Evaluate test
    test_loss, test_acc = evaluate_model(model, X_test, y_test_onehot, best_params["batch_size"])
    print(f"Best Model Test accuracy: {test_acc*100:.2f}%")
    wandb.log({"Test Accuracy": test_acc, "Test Loss": test_loss})

    # 6) Confusion matrix class labels
    class_labels = [
        "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
        "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
    ]

    # 7) We do an entire forward pass on X_test to get predictions
    caches_test = model.forward(X_test)
    Y_hat_test  = caches_test[f'H{model.num_layers}']
    preds_test  = np.argmax(Y_hat_test, axis=1)
    true_test   = np.argmax(y_test_onehot, axis=1)

    # 7a) Native wandb confusion matrix
    wandb.log({
        "conf_mat_wandbplot": wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_test,
            preds=preds_test,
            class_names=class_labels
        )
    })

    # 8) Simple confusion matrix image
    fig_simple = plot_confusion_matrix(
        model, X_test, y_test_onehot,
        class_labels=class_labels,
        batch_size=best_params["batch_size"]
    )
    plt.savefig("confusion_matrix_simple.png")
    wandb.log({"Confusion_Matrix_Simple": wandb.Image("confusion_matrix_simple.png")})
    plt.show()

    # 9) Creative confusion matrix image
    fig_creative = plot_confusion_matrix_creative(
        model, X_test, y_test_onehot,
        class_labels,
        batch_size=best_params["batch_size"]
    )
    plt.savefig("confusion_matrix_creative.png")
    wandb.log({"Confusion_Matrix_Creative": wandb.Image("confusion_matrix_creative.png")})
    plt.show()

    wandb.finish()

if __name__ == "__main__":
    main()
