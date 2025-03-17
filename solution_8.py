import numpy as np
import wandb
from utils import load_fashion_mnist
from utils import one_hot_encode, get_minibatches, evaluate_model
from model import NeuralNetwork
from optimizers import create_optimizer_from_config

sweep_config = {
    'method': 'random',
    'metric': {
        'name': 'val_accuracy',
        'goal': 'maximize'
    },
    'parameters': {
        'epochs': {
            'values': [5, 10]
        },
        'num_hidden_layers': {
            'values': [3, 4, 5]
        },
        'hidden_size': {
            'values': [32, 64, 128]
        },
        'weight_decay': {
            'values': [0.0, 0.0005, 0.5]
        },
        'learning_rate': {
            'values': [1e-3, 1e-4]
        },
        'optimizer': {
            'values': ['sgd', 'momentum', 'nesterov', 'rmsprop', 'adam', 'nadam']
        },
        'batch_size': {
            'values': [16, 32, 64]
        },
        'weight_init': {
            'values': ['random', 'xavier']
        },
        'activation': {
            'values': ['sigmoid', 'tanh', 'relu']
        },
        'loss_type': {
            'values': ['cross_entropy', 'mse']
        }
    }
}


def sweep_train():
    # Initialize run
    wandb.init()
    config = wandb.config

    # Load data
    X_full, y_full, X_test, y_test = load_fashion_mnist()

    # Stratified 90/10 train/val
    from sklearn.model_selection import StratifiedShuffleSplit
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
    for tr_idx, val_idx in sss.split(X_full, y_full):
        X_train = X_full[tr_idx]
        y_train = y_full[tr_idx]
        X_val = X_full[val_idx]
        y_val = y_full[val_idx]

    # one-hot
    y_train_oh = one_hot_encode(y_train, 10)
    y_val_oh = one_hot_encode(y_val, 10)
    y_test_oh = one_hot_encode(y_test, 10)

    # 2) Build the model from the config
    # layer sizes
    hidden_layers = [config.hidden_size] * config.num_hidden_layers
    layer_sizes = [784] + hidden_layers + [10]

    # create net
    net = NeuralNetwork(layer_sizes=layer_sizes,
                        activation=config.activation,
                        optimizer=None,
                        loss_type=config.loss_type,
                        seed=42)

    # weight init
    if config.weight_init == 'xavier':
        for i in range(net.num_layers):
            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]
            limit = np.sqrt(6.0 / (in_dim + out_dim))
            net.params[f'W{i}'] = np.random.uniform(-limit, limit, (in_dim, out_dim))
            net.params[f'b{i}'] = np.zeros(out_dim, dtype=np.float32)

    # create optimizer
    opt = create_optimizer_from_config(config.optimizer,
                                       config.learning_rate,
                                       config.weight_decay)
    net.optimizer = opt

    # 3) Training loop
    epochs = config.epochs
    batch_size = config.batch_size

    best_val_acc = 0.0
    for epoch in range(epochs):
        # shuffle train
        idx = np.arange(X_train.shape[0])
        np.random.shuffle(idx)
        X_train = X_train[idx]
        y_train_oh = y_train_oh[idx]

        # mini-batch pass
        losses = []
        correct = 0
        count = X_train.shape[0]
        for Xb, Yb in get_minibatches(X_train, y_train_oh, batch_size, shuffle=False):
            loss = net.train_batch(Xb, Yb)
            losses.append(loss)

            # quick check for training accuracy on this batch
            caches = net.forward(Xb)
            Y_hat = caches[f'H{net.num_layers}']
            preds = np.argmax(Y_hat, axis=1)
            actual = np.argmax(Yb, axis=1)
            correct += np.sum(preds == actual)

        train_loss = np.mean(losses)
        train_acc = correct / count

        # val
        val_loss, val_acc = evaluate_model(net, X_val, y_val_oh, batch_size)

        # wandb log
        wandb.log({
            'Q8_epoch': epoch,
            'Q8_train_loss': train_loss,
            'Q8_train_accuracy': train_acc,
            'Q8_val_loss': val_loss,
            'Q8_val_accuracy': val_acc
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        print(f"Epoch {epoch + 1}/{epochs}, train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")

    # final test
    test_loss, test_acc = evaluate_model(net, X_test, y_test_oh, batch_size)
    wandb.log({'Q8_test_loss': test_loss, 'Q8_test_accuracy': test_acc})
    print(f"FINAL TEST:  test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")

    # 8. Rename the run so it’s easy to read
    run_name = (f"Epochs:{config.epochs}, Num_HL:{config.num_hidden_layers}, HL_Size:{config.hidden_size}, "
                f"Wt_Decay:{config.weight_decay}, LR:{config.learning_rate}, Optimizer:{config.optimizer}, "
                f"BS:{config.batch_size}, Wt_Init:{config.weight_init}, Activation:{config.activation}, "
                f"Loss:{config.loss_type}")
    wandb.run.name = run_name


############################################
# 5) Main: Create & Run the Sweep
############################################
def main():
    # 1) Create the sweep
    sweep_id = wandb.sweep(sweep_config, project="DA24S020_DA6401_Deep_Learning_Assignment1")

    # 2) Launch the sweep with N random runs, e.g. 15
    wandb.agent(sweep_id, function=sweep_train, count=50)


if __name__ == "__main__":
    main()
