import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg', 'MacOSX', etc. Use "Agg" for non-interactive plot
import matplotlib.pyplot as plt
from utils import load_fashion_mnist, one_hot_encode, get_minibatches
from model import NeuralNetwork
from optimizers import (
    SGDOptimizer,
    MomentumOptimizer,
    NesterovOptimizer,
    RMSPropOptimizer,
    AdamOptimizer,
    NadamOptimizer
)


def predict(nn, X):
    """
    Generate class predictions (0–9) by taking the argmax of the final activation H.
    X: (N, 784)
    Returns: (N,) array of predicted class indices
    """
    caches = nn.forward(X)
    H_last = caches[f'H{nn.num_layers}']  # final output probabilities with shape: (N, 10)
    return np.argmax(H_last, axis=1)


def main():
    # 1) Load and prepare data
    X_train, y_train, X_test, y_test = load_fashion_mnist()
    # X_train: (60000, 784), y_train: (60000,)
    # X_test:  (10000, 784), y_test: (10000,)

    # One-hot encode
    y_train_onehot = one_hot_encode(y_train, num_classes=10)  # (60000, 10)
    y_test_onehot = one_hot_encode(y_test, num_classes=10)  # (10000, 10)

    # 2) Define hyperparameters
    layer_sizes = [784, 128, 64, 10]  # Example 2-hidden-layer architecture
    num_epochs = 3  # Train for 3 epochs
    batch_size = 64

    # 3) List of all optimizers we want to test
    optimizers_to_run = {
        "SGD": SGDOptimizer(lr=0.01),
        "Momentum": MomentumOptimizer(lr=0.01, beta=0.9),
        "Nesterov": NesterovOptimizer(lr=0.01, beta=0.9),
        "RMSProp": RMSPropOptimizer(lr=0.001, beta=0.9),
        "Adam": AdamOptimizer(lr=0.001, beta1=0.9, beta2=0.999),
        "Nadam": NadamOptimizer(lr=0.001, beta1=0.9, beta2=0.999)
    }

    # 4) Dictionary to store the loss curves for each optimizer
    all_losses = {}
    # Dictionary to store final test accuracies for each optimizer
    final_accuracies = {}

    for opt_name, opt_instance in optimizers_to_run.items():
        # For pure SGD, use batch_size=1; otherwise, 64
        if opt_name == "SGD":
            batch_size = 1
        else:
            batch_size = 64

        print(f"\n=== Training with {opt_name} ===")
        print("Mini-Batch size: ", batch_size)

        # Create a new model for each optimizer
        nn = NeuralNetwork(layer_sizes=layer_sizes,
                           activation='relu',
                           optimizer=opt_instance,
                           seed=42)

        # List to keep track of the loss after every update (step)
        loss_values = []

        # 5) Training loop (over epochs)
        for epoch in range(num_epochs):
            loss_values_per_epoch = []
            # Use get_minibatches to iterate over all training data in mini-batches
            for X_batch, Y_batch in get_minibatches(X_train, y_train_onehot, batch_size, shuffle=True):
                # Perform one update step
                loss = nn.train_batch(X_batch, Y_batch)
                # Record the loss
                loss_values.append(loss)
                loss_values_per_epoch.append(loss)
            # Print last-batch loss for the epoch
            print(f"Epoch {epoch + 1}/{num_epochs} done. Last batch loss = {loss:.4f}")
            # Print average loss over the epoch
            print(f"Average loss for epoch {epoch + 1}/{num_epochs} over all "
                  f"mini-batches: {np.mean(loss_values_per_epoch):.4f}")


        # Store the loss history for plotting
        all_losses[opt_name] = loss_values

        # After training, evaluate on the test set
        test_preds = predict(nn, X_test)
        test_acc = np.mean(test_preds == y_test)
        final_accuracies[opt_name] = test_acc
        print(f"{opt_name} final test accuracy: {test_acc * 100:.2f}%")

    # # 6) Plot the loss curves for each optimizer
    # plt.figure(figsize=(8, 6))
    # for opt_name, loss_list in all_losses.items():
    #     plt.plot(loss_list, label=opt_name)
    # plt.title("Training Loss over Updates for Different Optimizers")
    # plt.xlabel("Update Step")
    # plt.ylabel("Loss")
    # plt.legend()
    # plt.show()

    # 7) Print final accuracies summary
    print("\n=== Final Test Accuracies ===")
    for opt_name, acc in final_accuracies.items():
        print(f"{opt_name}: {acc*100:.2f}%")


if __name__ == "__main__":
    main()
