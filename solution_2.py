from data_utils import load_fashion_mnist, one_hot_encode
from model import NeuralNetwork


def main():
    # 1) Load the Fashion-MNIST data
    #    X_train: (60000, 784), y_train: (60000,)
    #    X_test:  (10000, 784), y_test:  (10000,)
    X_train, y_train, X_test, y_test = load_fashion_mnist()

    # 2) Define a feedforward network architecture
    #    Here we have 2 hidden layers: 128 and 64 neurons,
    #    and an output layer of size 10 (one for each class).
    layer_sizes = [784, 128, 64, 10]
    nn = NeuralNetwork(layer_sizes=layer_sizes, activation='relu', seed=42)

    # 3) Run a forward pass on a small batch of images
    #    For demonstration, let's pick the first 5 training examples
    #    and see the network's output distribution (unnormalized at first).
    X_batch = X_train[:5]  # shape: (5, 784)

    # 4) Forward pass -> output is a probability distribution over 10 classes
    caches = nn.forward(X_batch)
    # The final output is in caches['A{num_layers}']
    probs = caches[f'H{nn.num_layers}']  # shape: (5, 10)

    # 5) Print the resulting probabilities
    print("Shape of output probabilities:", probs.shape)  # Should be (5, 10)
    print("Output probabilities for the first 5 images:")
    print(probs)  # Each row sums to 1 across 10 classes


if __name__ == "__main__":
    main()
