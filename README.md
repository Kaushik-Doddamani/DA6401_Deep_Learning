# DA6401_Deep_Learning
Source code for assignments of course DA6401

Name: Kaushik Ningappa Doddamani

## Optimizers and Loss Functions

Optimizers implemented:
- SGD - Stochastic Gradient Descent
- Momentum - Momentum SGD
- NAG - Nesterov Accelerated Gradient (optimized version)
- RMSProp - Root Mean Square Propagation
- Adam - Adaptive Moment Estimation
- Nadam - Nesterov Adaptive Moment Estimation

Loss functions implemented:
- Cross Entropy
- Mean Squared Error


## Dataset

The dataset is from Fashion MNIST available in keras in the keras.datasets module.

# Solutions:

## Question 1
The solution for Question 1 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_1.py).
The program displays one sample from each class of Fashion MNIST dataset and logs the same on `wandb.ai`

## Questions 2
The solution for Question 2 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_2.py).
The program Implements a feedforward neural network which takes images from the fashion-mnist data as input and outputs a probability distribution over the 10 classes

## Questions 3
The solution for Question 3 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_3.py).
The program implements the backpropagation algorithm with support for the following optimisation functions 
- sgd
- momentum based gradient descent
- nesterov accelerated gradient descent
- rmsprop
- adam
- nadam

## Questions 4
The solution for Question 4 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_4.py).
The program uses the sweep functionality provided by `wandb` inorder to perform hyperparameter tunning and selection the optimal hyperparameter configuration

The list of hyperparameter configs are as follows:
- number of epochs: 5, 10
- number of hidden layers:  3, 4, 5
- size of every hidden layer:  32, 64, 128
- weight decay (L2 regularisation): 0, 0.0005,  0.5
- learning rate: 1e-3, 1 e-4 
- optimizer:  sgd, momentum, nesterov, rmsprop, adam, nadam
- batch size: 16, 32, 64
- weight initialisation: random, Xavier
- activation functions: sigmoid, tanh, ReLU

## Questions 5-6
The solution for Question 5 is available [here]().
The required plots have been generated using `wandb`

## Questions 7
The solution for Question 7 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_7.py). 
The program uses the test data prediction to generate a confusion matrix.

```python
wandb.log({
        "conf_mat_wandbplot": wandb.plot.confusion_matrix(
            probs=None,
            y_true=true_test,
            preds=preds_test,
            class_names=class_labels
        )
    })
```

## Questions 8
The solution for Question 8 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_8.py).
The program trains the models using `mse` and `cross_entropy` loss functions.


## Questions 10
The solution for Question 10 is available [here](https://github.com/Kaushik-Doddamani/DA6401_Deep_Learning/blob/main/solution_10.py).
The program implements model training on `mnist` data using 2 different configs which are as follows:

- Configuration 1: 
    - `activation`: "tanh",
    - `batch_size`: 64,
    - `epochs`: 10,
    - `hidden_size`: 128,
    - `learning_rate`: 0.001,
    - `num_hidden_layers`: 3,
    - `optimizer`: "nadam",
    - `weight_decay`: 0.0,
    - `weight_init`: "xavier",
    - `loss_type`: "cross_entropy"

- Configuration 2: 
    - `activation`: "relu",
    - `batch_size`: 64,
    - `epochs`: 10,
    - `hidden_size`: 128,
    - `learning_rate`: 0.001,
    - `num_hidden_layers`: 3,
    - `optimizer`: "adam",
    - `weight_decay`: 0.0,
    - `weight_init`: "xavier",
    - `loss_type`: "cross_entropy"

- Configuration 3: 
    - `activation`: "tanh",
    - `batch_size`: 64,
    - `epochs`: 10,
    - `hidden_size`: 128,
    - `learning_rate`: 0.001,
    - `num_hidden_layers`: 3,
    - `optimizer`: "rmsprop",
    - `weight_decay`: 0.0,
    - `weight_init`: "xavier",
    - `loss_type`: "cross_entropy"

