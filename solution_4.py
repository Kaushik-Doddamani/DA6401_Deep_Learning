import wandb
import yaml
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from utils import one_hot_encode, load_fashion_mnist, evaluate_model, get_minibatches
from model import NeuralNetwork
from optimizers import create_optimizer_from_config


def build_model(config):
    """
        Constructs and returns a NeuralNetwork instance
        with the specified hyperparams (num_hidden_layers, hidden_size, activation, etc.).
        Applies xavier init if requested.
    """
    hidden_layers = [config.hidden_size] * config.num_hidden_layers
    layer_sizes = [784] + hidden_layers + [10]

    # Instantiate the neural network
    nn = NeuralNetwork(
        layer_sizes=layer_sizes,
        activation=config.activation,
        optimizer=None,  # will be set later
        seed=42
    )

    # Weight initialization: use Xavier if specified
    if config.weight_init == 'xavier':
        for i in range(nn.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            nn.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            nn.params[f'b{i}'] = np.zeros(out_dim)
    # else: default initialization from the constructor is used
    return nn


def train_model():
    """
        The function wandb.agent will call to run a single hyperparameter trial.
    """
    # 1. Initialize a new wandb run
    wandb.init()
    config = wandb.config

    # 2. Load and prepare data
    X_train_full, y_train_full, X_test, y_test = load_fashion_mnist()

    # 3. Create a StratifiedShuffleSplit for 1 split => 90% train, 10% val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for train_idx, val_idx in sss.split(X_train_full, y_train_full):
        X_train = X_train_full[train_idx]
        y_train = y_train_full[train_idx]
        X_val = X_train_full[val_idx]
        y_val = y_train_full[val_idx]

    # 4. One-hot encode
    y_train_onehot = one_hot_encode(y_train, 10)  # (N-val_count, 10)
    y_val_onehot = one_hot_encode(y_val, 10)  # (val_count, 10)
    y_test_onehot = one_hot_encode(y_test, 10)  # (10000, 10)

    # 4. Build the model
    nn = build_model(config)

    # 5. L2 weight decay
    nn.optimizer = create_optimizer_from_config(config.optimizer, config.learning_rate, config.weight_decay)

    # 6. Training
    batch_size = config.batch_size
    epochs = config.epochs

    for epoch in range(epochs):
        losses = []
        minibatches = get_minibatches(X_train, y_train, batch_size, shuffle=True)
        for X_batch, Y_batch in minibatches:
            loss = nn.train_batch(X_batch, Y_batch)
            losses.append(loss)
        print(f"Epoch {epoch+1}, loss: {loss:.4f}")

        # Evaluate on train & val
        train_loss, train_acc = evaluate_model(nn, X_train, y_train_onehot, batch_size)
        val_loss, val_acc = evaluate_model(nn, X_val, y_val_onehot, batch_size)

        # Log to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'val_loss': val_loss,
            'val_accuracy': val_acc
        })

    # 7. Final test metrics
    test_loss, test_acc = evaluate_model(nn, X_test, y_test_onehot, batch_size)
    wandb.log({
        'test_loss': test_loss,
        'test_accuracy': test_acc
    })

    # 8. Rename the run so it’s easy to read
    run_name = (f"Epochs:{config.epochs}, Num_HL:{config.num_hidden_layers}, HL_Size:{config.hidden_size}, "
                f"Wt_Decay:{config.weight_decay}, LR:{config.learning_rate}, Optimizer:{config.optimizer}, "
                f"BS:{config.batch_size}, Wt_Init:{config.weight_init}, Activation:{config.activation}")
    wandb.run.name = run_name


def main():
    """
        Main entry point: load config.yaml, create the sweep, run it for some number of trials.
    """
    # 1. Read the sweep config from config.yaml
    with open("config.yaml", "r") as f:
        sweep_config = yaml.safe_load(f)

    # 2. Create sweep in wandb
    sweep_id = wandb.sweep(sweep_config, project="random_sweep")

    # 3. Launch the sweep.
    wandb.agent(sweep_id, function=train_model, count=250)

    wandb.finish()


if __name__ == "__main__":
    main()
