#!/usr/bin/env python

import argparse
import numpy as np
import wandb
from sklearn.model_selection import StratifiedShuffleSplit

# Import your own modules:
from utils import load_fashion_mnist, load_mnist, one_hot_encode, get_minibatches, evaluate_model
from model import NeuralNetwork
from optimizers import create_optimizer_from_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a feedforward network on MNIST or Fashion-MNIST with custom hyperparams."
    )
    # WandB arguments
    parser.add_argument("-we", "--wandb_entity", default="myname", type=str,
                        help="W&B Entity (username or team). Default 'myname'")
    parser.add_argument("-wp", "--wandb_project", default="myprojectname", type=str,
                        help="W&B Project name. Default 'myprojectname'")

    # Dataset
    parser.add_argument("-d", "--dataset", default="fashion_mnist", type=str,
                        choices=["mnist", "fashion_mnist"],
                        help="Which dataset to train on (mnist or fashion_mnist).")

    # Basic hyperparameters
    parser.add_argument("-e", "--epochs", default=10, type=int,
                        help="Number of epochs to train neural network.")
    parser.add_argument("-b", "--batch_size", default=64, type=int,
                        help="Batch size used to train neural network.")
    parser.add_argument("-l", "--loss", default="cross_entropy", type=str,
                        choices=["mean_squared_error", "cross_entropy"],
                        help="Loss function to optimize. Default cross_entropy.")
    parser.add_argument("-o", "--optimizer", default="nadam", type=str,
                        choices=["sgd", "momentum", "nag", "rmsprop", "adam", "nadam"],
                        help="Choice of optimizer. Default 'adam'.")
    parser.add_argument("-lr", "--learning_rate", default=0.001, type=float,
                        help="Learning rate used for optimizing. Default 0.001.")
    parser.add_argument("-m", "--momentum", default=0.9, type=float,
                        help="Momentum for momentum / nag optimizers. Default 0.9.")
    parser.add_argument("-beta", "--beta", default=0.9, type=float,
                        help="Beta for RMSProp. Default 0.9.")
    parser.add_argument("-beta1", "--beta1", default=0.9, type=float,
                        help="Beta1 for Adam / Nadam. Default 0.9.")
    parser.add_argument("-beta2", "--beta2", default=0.999, type=float,
                        help="Beta2 for Adam / Nadam. Default 0.999.")
    parser.add_argument("-eps", "--epsilon", default=1e-8, type=float,
                        help="Epsilon for numerical stability. Default 1e-8.")
    parser.add_argument("-w_d", "--weight_decay", default=0.0, type=float,
                        help="Weight decay (L2 regularization). Default 0.0.")

    # Network architecture
    parser.add_argument("-w_i", "--weight_init", default="xavier", type=str,
                        choices=["random", "Xavier"],
                        help="Weight initialization. Default 'xavier'.")
    parser.add_argument("-nhl", "--num_layers", default=3, type=int,
                        help="Number of hidden layers. Default 3.")
    parser.add_argument("-sz", "--hidden_size", default=128, type=int,
                        help="Number of neurons in each hidden layer. Default 128.")
    parser.add_argument("-a", "--activation", default="tanh", type=str,
                        choices=["identity", "sigmoid", "tanh", "ReLU"],
                        help="Activation function. Default 'relu'.")

    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    # 1) Initialize W&B
    wandb.init(
        entity=args.wandb_entity,
        project=args.wandb_project,
        config=vars(args)  # log all hyperparams to wandb
    )
    wandb.run.name = (f"train_run Dataset: {args.dataset}, Epochs: {args.epochs}, BS: {args.batch_size}, "
                      f"Loss: {args.loss}, Optimizer: {args.optimizer}, LR: {args.learning_rate}, "
                      f"Weight_Decay: {args.weight_decay}, Weight_Init: {args.weight_init}, "
                      f"Num_Layers: {args.num_layers}, Hidden_Size: {args.hidden_size}, Activation: {args.activation}")

    # 2) Load dataset
    if args.dataset == "mnist":
        X_full, y_full, X_test, y_test = load_mnist()
    else:
        X_full, y_full, X_test, y_test = load_fashion_mnist()

    # 3) Stratified 90/10 train/val split
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, val_idx in sss.split(X_full, y_full):
        X_train = X_full[train_idx]
        y_train = y_full[train_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

    # 4) One-hot encode
    y_train_oh = one_hot_encode(y_train, 10)
    y_val_oh = one_hot_encode(y_val, 10)
    y_test_oh = one_hot_encode(y_test, 10)

    # 5) Build the feedforward network
    #    (map "mean_squared_error" -> "mse", "cross_entropy" -> "cross_entropy")
    if args.loss == "mean_squared_error":
        loss_type = "mse"
    else:
        loss_type = "cross_entropy"

    # Similarly, map "nag" -> "nesterov" in our code
    optimizer_name = args.optimizer
    if optimizer_name == "nag":
        optimizer_name = "nesterov"

    # If user typed "ReLU", let's unify to lowercase "relu" for our code
    activation_name = args.activation.lower()
    if activation_name == "identity":
        activation_name = "identity"  # if you actually want to implement identity
    # else: 'sigmoid', 'tanh', or 'relu' are normal

    # Hidden layers
    layer_sizes = [784] + [args.hidden_size] * args.num_layers + [10]

    # create optimizer
    opt_instance = create_optimizer_from_config(
        optimizer_name,
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    # if it's momentum or nesterov, set momentum= args.momentum
    # if it's RMSProp, set beta= args.beta
    # if it's adam/nadam, set beta1=..., beta2=..., eps=...
    # For brevity, let's do a quick override if needed:
    # (In a real code, you'd adapt create_optimizer_from_config to handle these).
    if hasattr(opt_instance, "momentum") and (optimizer_name in ["momentum", "nesterov"]):
        opt_instance.beta = args.momentum
    if optimizer_name == "rmsprop" and hasattr(opt_instance, "beta"):
        opt_instance.beta = args.beta
    if optimizer_name in ["adam", "nadam"]:
        if hasattr(opt_instance, "beta1"):
            opt_instance.beta1 = args.beta1
        if hasattr(opt_instance, "beta2"):
            opt_instance.beta2 = args.beta2
        if hasattr(opt_instance, "eps"):
            opt_instance.eps = args.epsilon

    # Build the model
    model = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=activation_name,
        loss_type=loss_type,
        optimizer=opt_instance,
        seed=42
    )

    # If xavier
    if args.weight_init == "Xavier":
        for i in range(model.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            model.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            model.params[f'b{i}'] = np.zeros(out_dim)

    # 6) Train
    epochs = args.epochs
    batch_size = args.batch_size
    N = X_train.shape[0]

    best_val_acc = 0.0
    for epoch in range(epochs):
        # Shuffle data each epoch
        idx = np.arange(N)
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train_oh = y_train_oh[idx]

        # mini-batch
        losses = []
        correct = 0
        for Xb, Yb in get_minibatches(X_train, y_train_oh, batch_size, shuffle=False):
            loss = model.train_batch(Xb, Yb)
            losses.append(loss)

            # quick train acc check
            caches = model.forward(Xb)
            Y_hat = caches[f'H{model.num_layers}']
            preds = np.argmax(Y_hat, axis=1)
            actual = np.argmax(Yb, axis=1)
            correct += np.sum(preds == actual)

        train_loss = np.mean(losses)
        train_acc = correct / N

        # val
        val_loss, val_acc = evaluate_model(model, X_val, y_val_oh, batch_size)

        wandb.log({
            "final_epoch": epoch,
            "final_train_loss": train_loss,
            "final_train_acc": train_acc,
            "final_val_loss": val_loss,
            "final_val_acc": val_acc
        })
        print(f"Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc

    # 7) Final test
    test_loss, test_acc = evaluate_model(model, X_test, one_hot_encode(y_test, 10), batch_size)
    wandb.log({"final_test_loss": test_loss, "final_test_acc": test_acc})
    print(f"FINAL TEST => loss={test_loss:.4f}, acc={test_acc * 100:.2f}%")

    wandb.finish()


if __name__ == "__main__":
    main()
