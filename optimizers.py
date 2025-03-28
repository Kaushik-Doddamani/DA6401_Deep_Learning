import numpy as np


class BaseOptimizer:
    """
    Base class to define the interface for custom optimizers.
    Each optimizer will maintain internal state (e.g. momentum vectors)
    and implement an update(params, grads) method.
    """

    def update(self, params, grads):
        raise NotImplementedError


# ------------------------------------------------------------------------------
# 1. Plain Stochastic Gradient Descent
# ------------------------------------------------------------------------------
class SGDOptimizer(BaseOptimizer):
    def __init__(self, lr=0.01, weight_decay=0.0):
        """
        lr: learning rate
        weight_decay: L2 regularization coefficient (default=0 )
        """
        self.lr = lr
        self.weight_decay = weight_decay

    def update(self, params, grads):
        """
        Performs in-place update of params with plain SGD:
          param = param - lr * grad
        Then apply weight decay: W = W - weight_decay * W
        """
        for key in params:
            params[key] -= self.lr * grads[f'd{key}']
            # apply weight decay only to weights, not biases
            if key.startswith('W') and self.weight_decay > 0.0:
                params[key] -= self.weight_decay * params[key]


# ------------------------------------------------------------------------------
# 2. Momentum-based Gradient Descent
# ------------------------------------------------------------------------------
class MomentumOptimizer(BaseOptimizer):
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        """
        lr: learning rate
        beta: momentum coefficient (beta)
        weight_decay: L2 regularization coefficient (default=0)
        """
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.u = {}  # velocity dictionary

    def update(self, params, grads):
        """
        u_t = beta * u_{t-1} + grad
        w_{t+1} = w_t - lr * u_t
        weight decay: w = w - decay*w
        """
        for key in params:
            if key not in self.u:
                self.u[key] = np.zeros_like(params[key])
            self.u[key] = self.beta * self.u[key] + self.lr * grads[f'd{key}']
            params[key] -= self.u[key]
            # weight decay
            if key.startswith('W') and self.weight_decay > 0:
                params[key] -= self.weight_decay * params[key]


# ------------------------------------------------------------------------------
# 3. Nesterov Accelerated Gradient
# ------------------------------------------------------------------------------
class NesterovOptimizer(BaseOptimizer):
    def __init__(self, lr=0.01, beta=0.9, weight_decay=0.0):
        """
        lr: learning rate
        beta: momentum coefficient
        """
        self.lr = lr
        self.beta = beta
        self.weight_decay = weight_decay
        self.u = {}  # velocity dictionary

    def get_offset(self, params):
        """
        Returns the offset = -beta * u_{t-1},
        so we can compute gradients at (w_t + offset).
        """
        offset = {}
        for key in params:
            if key not in self.u:
                self.u[key] = np.zeros_like(params[key])
            offset[key] = -self.beta * self.u[key]
        return offset

    def update(self, params, grads):
        """
        NAG update step:
          u_t = beta*u_{t-1} + grads
          w_{t+1} = w_t - lr*u_t
          Then weight decay
        """
        for key in params:
            if key not in self.u:
                self.u[key] = np.zeros_like(params[key])
            # velocity update
            self.u[key] = self.beta * self.u[key] + self.lr * grads[f'd{key}']
            # parameter update
            params[key] -= self.u[key]

            if key.startswith('W') and self.weight_decay > 0:
                params[key] -= self.weight_decay * params[key]


# ------------------------------------------------------------------------------
# 4. RMSProp
# ------------------------------------------------------------------------------
class RMSPropOptimizer(BaseOptimizer):
    def __init__(self, lr=0.001, beta=0.9, eps=1e-8, weight_decay=0.0):
        """
        lr: learning rate
        beta: decay term for moving average of squared gradients
        eps: numerical stability
        """
        self.lr = lr
        self.beta = beta
        self.eps = eps
        self.weight_decay = weight_decay
        self.eg2 = {}  # moving average of grad^2

    def update(self, params, grads):
        """
        eg2[key] = beta * eg2[key] + (1-beta) * (grad^2)
        param = param - lr * grad / (sqrt(eg2[key]) + eps)
        Then weight decay
        """
        for key in params:
            if key not in self.eg2:
                self.eg2[key] = np.zeros_like(params[key])
            # accumulate squared gradients
            self.eg2[key] = self.beta * self.eg2[key] + (1 - self.beta) * (grads[f'd{key}'] ** 2)
            # update
            params[key] -= self.lr * grads[f'd{key}'] / (np.sqrt(self.eg2[key]) + self.eps)

            if key.startswith('W') and self.weight_decay > 0:
                params[key] -= self.weight_decay * params[key]


# ------------------------------------------------------------------------------
# 5. Adam
# ------------------------------------------------------------------------------
class AdamOptimizer(BaseOptimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """
        lr: learning rate
        beta1, beta2: decay rates for moment estimates
        eps: numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}  # first moment
        self.v = {}  # second moment
        self.t = 0  # time step

    def update(self, params, grads):
        """
        t = t + 1
        m = beta1*m + (1 - beta1)*grad
        v = beta2*v + (1 - beta2)*(grad^2)
        m_hat = m / (1 - beta1^t)
        v_hat = v / (1 - beta2^t)
        param = param - lr * m_hat / (sqrt(v_hat) + eps)
        Then weight decay
        """
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * grads[f'd{key}']
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (grads[f'd{key}'] ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            params[key] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            if key.startswith('W') and self.weight_decay > 0:
                params[key] -= self.weight_decay * params[key]


# ------------------------------------------------------------------------------
# 6. Nadam
# ------------------------------------------------------------------------------
class NadamOptimizer(BaseOptimizer):
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        """
        lr: learning rate
        beta1, beta2: decay rates for moment estimates
        eps: numerical stability
        """
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}
        self.t = 0

    def update(self, params, grads):
        """
        Nadam is basically Adam + Nesterov for the first moment.
        We can implement it as:
          t = t + 1
          g_t = grad
          m = beta1*m + (1 - beta1)*g_t
          v = beta2*v + (1 - beta2)*(g_t^2)
          m_hat = m / (1 - beta1^t)
          v_hat = v / (1 - beta2^t)
          m_nadam = beta1*m_hat + (1 - beta1)*g_t / (1 - beta1^t)
          param = param - lr * m_nadam / (sqrt(v_hat) + eps)
          Then weight decay
        """
        self.t += 1
        for key in params:
            if key not in self.m:
                self.m[key] = np.zeros_like(params[key])
                self.v[key] = np.zeros_like(params[key])

            g_t = grads[f'd{key}']

            self.m[key] = self.beta1 * self.m[key] + (1 - self.beta1) * g_t
            self.v[key] = self.beta2 * self.v[key] + (1 - self.beta2) * (g_t ** 2)

            m_hat = self.m[key] / (1 - self.beta1 ** self.t)
            v_hat = self.v[key] / (1 - self.beta2 ** self.t)

            # Nadam effective gradient
            m_nadam = self.beta1 * m_hat + (1 - self.beta1) * g_t / (1 - self.beta1 ** self.t)

            params[key] -= self.lr * m_nadam / (np.sqrt(v_hat) + self.eps)

            if key.startswith('W') and self.weight_decay > 0:
                params[key] -= self.weight_decay * params[key]


def create_optimizer_from_config(opt_name, lr, weight_decay=0.0):
    """
    Utility function that returns an instance of the chosen optimizer with the given LR.
    """
    if opt_name == 'sgd':
        return SGDOptimizer(lr=lr, weight_decay=weight_decay)
    elif opt_name == 'momentum':
        return MomentumOptimizer(lr=lr, beta=0.9, weight_decay=weight_decay)
    elif opt_name == 'nesterov':
        return NesterovOptimizer(lr=lr, beta=0.9, weight_decay=weight_decay)
    elif opt_name == 'rmsprop':
        return RMSPropOptimizer(lr=lr, beta=0.9, weight_decay=weight_decay)
    elif opt_name == 'adam':
        return AdamOptimizer(lr=lr, beta1=0.9, beta2=0.999, weight_decay=weight_decay)
    elif opt_name == 'nadam':
        return NadamOptimizer(lr=lr, beta1=0.9, beta2=0.999, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {opt_name}")
