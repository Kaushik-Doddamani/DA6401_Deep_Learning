import numpy as np
from optimizers import BaseOptimizer, SGDOptimizer


class NeuralNetwork:
    """
    A flexible feedforward neural network built using only NumPy for matrix ops.
    Backprop is implemented manually, and the optimizer is pluggable.

    Example usage:
        nn = NeuralNetwork(layer_sizes=[784, 128, 64, 10], activation='relu')
        # Then call nn.forward(...) and nn.backward(...) or use train_batch(...)
    Here:
        - "A" refers to the PRE-activation (linear combination).
        - "H" refers to the activation (post-activation).
    """

    def __init__(self, layer_sizes, activation='relu', optimizer=None, seed=42):
        """
        layer_sizes: list of layer dimensions, e.g. [784, 128, 64, 10], where layer_sizes[i] = number of units in layer i
        activation:  'relu' or 'sigmoid' for hidden layers
        optimizer:   any object with an update(params, grads) method (e.g., from optimizers.py)
        seed:        random seed for reproducibility
        """
        np.random.seed(seed)

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1
        self.activation_name = activation

        # By default, if no optimizer is given, use plain SGD
        if optimizer is None:
            self.optimizer = SGDOptimizer(lr=0.01)
        else:
            self.optimizer = optimizer

        # Initialize parameters W{i}, b{i} for i in [0, num_layers-1]
        self.params = {}
        for i in range(self.num_layers):
            in_dim = layer_sizes[i]  # number of units in layer i
            out_dim = layer_sizes[i + 1]  # number of units in layer i+1
            # Xavier-like initialization
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            self.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            # W{i}: (in_dim, out_dim)
            self.params[f'b{i}'] = np.zeros(out_dim)
            # b{i}: (out_dim,)

    def activation(self, a, derivative=False):
        """
        Applies the chosen activation function (ReLU or Sigmoid) on 'a'.
        a: (batch_size, layer_sizes[i+1])
        derivative=True => returns derivative w.r.t. 'a'
        """
        if self.activation_name == 'relu':
            if derivative:
                return (a > 0).astype(float)
            return np.maximum(0, a)
        elif self.activation_name == 'sigmoid':
            if derivative:
                sig = 1.0 / (1.0 + np.exp(-a))
                return sig * (1.0 - sig)
            return 1.0 / (1.0 + np.exp(-a))
        else:
            raise ValueError("Unsupported activation function. Use 'relu' or 'sigmoid'.")

    def softmax(self, logits):
        """
        Softmax function for the output layer.
        logits: (batch_size, num_classes)
        Returns: (batch_size, num_classes) with each row summing to 1
        """
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shifted)  # use the shifted values to avoid overflow.
        return exps / np.sum(exps, axis=1, keepdims=True)

    def forward(self, X, param_offset=None):
        """
        Forward pass with optional param_offset (for Nesterov).
          - If param_offset is not None, we add it to the stored params before computing.
        X: (batch_size, layer_sizes[0])  i.e. (batch_size, 784) for MNIST

        Returns:
         - caches: dict of intermediate A (pre-activations) and H (activations) for each layer
           e.g. caches['H0'] = X (input)
                caches['A1'], caches['H1']
                ...
                caches['A{num_layers}'], caches['H{num_layers}']
        """
        if param_offset is None:
            # default offset of zero
            param_offset = {k: np.zeros_like(v) for k, v in self.params.items()}

        caches = {}
        H = X  # input activation for layer 0 i.e. (batch_size, layer_sizes[0])
        caches['H0'] = H

        for i in range(self.num_layers):
            W_key = f'W{i}'
            b_key = f'b{i}'

            # "Effective" W and b, possibly shifted by offset
            W_eff = self.params[W_key] + param_offset[W_key]  # (layer_sizes[i], layer_sizes[i+1])
            b_eff = self.params[b_key] + param_offset[b_key]  # (layer_sizes[i+1],)

            # A{i+1}: (batch_size, layer_sizes[i+1])
            A = H @ W_eff + b_eff
            # (batch_size, layer_sizes[i]) * (layer_sizes[i], layer_sizes[i+1]) + (layer_sizes[i+1],)

            if i < self.num_layers - 1:
                H = self.activation(A, derivative=False)  # (batch_size, layer_sizes[i+1])
            else:
                # last layer => softmax
                H = self.softmax(A)  # (batch_size, 10)

            caches[f'A{i + 1}'] = A
            caches[f'H{i + 1}'] = H

        return caches

    def compute_loss(self, Y_hat, Y):
        """
        Cross-entropy loss for softmax outputs.
        Y_hat: (batch_size, layer_sizes[-1]) i.e. (batch_size, 10) predicted probabilities
        Y:      (batch_size, layer_sizes[-1]) i.e. (batch_size, 10) one-hot labels
        """
        eps = 1e-9
        log_probs = np.log(Y_hat + eps)  # (batch_size, 10)
        loss = -np.mean(np.sum(Y * log_probs, axis=1))
        return loss

    def backward(self, caches, X, Y):
        """
        Backpropagation: returns dict of gradients for all parameters.

        caches: forward pass outputs (Z and A for each layer)
        X:      (batch_size, layer_sizes[0])
        Y:      (batch_size, layer_sizes[-1]) one-hot
        """
        grads = {}
        batch_size = X.shape[0]

        # final layer's activation
        H_last = caches[f'H{self.num_layers}']  # (batch_size, 10)

        # derivative of cross-entropy w.r.t. final layer logits => (A_last - Y)
        dA = H_last - Y  # (batch_size, 10)

        # loop backward over layers
        for i in reversed(range(self.num_layers)):
            H_prev = caches[f'H{i}']  # (batch_size, layer_sizes[i])
            W = self.params[f'W{i}']  # (layer_sizes[i], layer_sizes[i+1])

            # dW: (layer_sizes[i], layer_sizes[i+1])
            dW = (H_prev.T @ dA) / batch_size
            # db: (layer_sizes[i+1],)
            db = np.sum(dA, axis=0) / batch_size

            grads[f'dW{i}'] = dW
            grads[f'db{i}'] = db

            if i > 0:
                # propagate error to previous layer
                A_prev = caches[f'A{i}']  # (batch_size, layer_sizes[i])
                dH_prev = dA @ W.T  # (batch_size, layer_sizes[i])
                dA = dH_prev * self.activation(A_prev, derivative=True)

        return grads

    def train_batch(self, X_batch, Y_batch):
        """
        A single training step on one batch:
          1. param_offset = optimizer.get_offset(...) if available
          2. forward pass
          3. compute loss
          4. backward pass
          5. optimizer update

        X_batch: (batch_size, layer_sizes[0])
        Y_batch: (batch_size, layer_sizes[-1])
        Returns: loss for this batch
        """
        # 1) If the optimizer has a 'get_offset' method (NAG), use it
        if hasattr(self.optimizer, 'get_offset'):
            offset = self.optimizer.get_offset(self.params)
        else:
            offset = None

        # 2) Forward pass
        caches = self.forward(X_batch, param_offset=offset)

        # 2) Compute loss
        H_last = caches[f'H{self.num_layers}']  # (batch_size, 10)
        loss = self.compute_loss(H_last, Y_batch)

        # 3) Backprop
        grads = self.backward(caches, X_batch, Y_batch)

        # 4) Update params using the chosen optimizer
        self.optimizer.update(self.params, grads)
        return loss
