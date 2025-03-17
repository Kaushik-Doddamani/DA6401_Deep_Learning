import numpy as np
from keras.datasets import mnist
from sklearn.model_selection import StratifiedShuffleSplit
from model import NeuralNetwork
import wandb
from utils import load_mnist, one_hot_encode, get_minibatches, evaluate_model
from optimizers import create_optimizer_from_config

configs_q10 = [
    {
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 10,
        "hidden_size": 128,
        "learning_rate": 0.001,
        "num_hidden_layers": 3,
        "optimizer": "nadam",
        "weight_decay": 0.0,
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    },
    {
        "activation": "relu",
        "batch_size": 64,
        "epochs": 10,
        "hidden_size": 128,
        "learning_rate": 0.001,
        "num_hidden_layers": 3,
        "optimizer": "adam",
        "weight_decay": 0.0,
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    },
    {
        "activation": "tanh",
        "batch_size": 64,
        "epochs": 10,
        "hidden_size": 128,
        "learning_rate": 0.001,
        "num_hidden_layers": 3,
        "optimizer": "rmsprop",
        "weight_decay": 0.0,
        "weight_init": "xavier",
        "loss_type": "cross_entropy"
    }
]


def main():
    wandb_project = "DA24S020_DA6401_Deep_Learning_Assignment1"

    # 1) Load MNIST
    X_full, y_full, X_test, y_test = load_mnist()

    # Stratified 90/10 for train/val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for tr_idx, val_idx in sss.split(X_full, y_full):
        X_train = X_full[tr_idx]
        y_train = y_full[tr_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

    y_train_oh = one_hot_encode(y_train, 10)
    y_val_oh = one_hot_encode(y_val, 10)
    y_test_oh = one_hot_encode(y_test, 10)

    # 2) For each of the 3 recommended configs
    for i, config in enumerate(configs_q10, start=1):
        run_name = f"Run{i}_{config['optimizer']}_{config['activation']}"

        # Start wandb run
        wandb.init(project=wandb_project, name=run_name)
        # Build the network
        layer_sizes = [784] + [config["hidden_size"]] * config["num_hidden_layers"] + [10]
        opt = create_optimizer_from_config(config["optimizer"], lr=config["learning_rate"])
        net = NeuralNetwork(
            layer_sizes=layer_sizes,
            activation=config["activation"],
            loss_type=config["loss_type"],
            optimizer=opt,
            seed=42
        )

        # If xavier
        if config["weight_init"] == 'xavier':
            for j in range(net.num_layers):
                in_dim = layer_sizes[j]
                out_dim = layer_sizes[j + 1]
                limit = np.sqrt(6.0 / (in_dim + out_dim))
                net.params[f'W{j}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
                net.params[f'b{j}'] = np.zeros(out_dim, dtype=np.float32)

        epochs = config["epochs"]
        batch_size = config["batch_size"]
        N = X_train.shape[0]

        best_val_acc = 0.0
        for epoch in range(epochs):
            # Shuffle train each epoch
            idx = np.arange(N)
            np.random.shuffle(idx)
            X_train = X_train[idx]
            y_train_oh = y_train_oh[idx]

            # Train in mini-batches
            total_loss = 0.0
            correct = 0
            for Xb, Yb in get_minibatches(X_train, y_train_oh, batch_size, shuffle=False):
                loss = net.train_batch(Xb, Yb)
                total_loss += loss * Xb.shape[0]
                # quick train acc check
                c = net.forward(Xb)
                Y_hat = c[f'H{net.num_layers}']
                preds = np.argmax(Y_hat, axis=1)
                actual = np.argmax(Yb, axis=1)
                correct += np.sum(preds == actual)

            train_loss = total_loss / N
            train_acc = correct / N

            # Evaluate val
            val_loss, val_acc = evaluate_model(net, X_val, y_val_oh, batch_size)

            # Log
            wandb.log({
                "Q10_epoch": epoch,
                "Q10_train_loss": train_loss,
                "Q10_train_accuracy": train_acc,
                "Q10_val_loss": val_loss,
                "Q10_val_accuracy": val_acc
            })

            print(f"Run{i}, epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        # final test
        test_loss, test_acc = evaluate_model(net, X_test, y_test_oh, batch_size)
        wandb.log({"Q10_test_loss": test_loss, "Q10_test_acc": test_acc})
        print(f"(Run{i}) FINAL TEST => loss={test_loss:.4f}, acc={test_acc * 100:.2f}%")

        wandb.finish()


if __name__ == "__main__":
    main()
