import torch
import math
import jax.numpy as jnp
from . import Proximal
import jax

def abs_torch_complex(x):
    a, b = x[..., 0], x[..., 1]
    return jnp.sqrt(a ** 2 + b ** 2)


def l_forward(x):
    """

    :rtype: y -> shape = (m, n)
    :type x -> shape = (2, m, n) where p = x[0] and q = x[1].
    """
    # y = x.clone()
    y = x.copy()

    # y[0, 1:, :] = x[0, 1:, :] - x[0, :-1, :]
    y = jax.ops.index_update(y, jax.ops.index[0, 1:, :], x[0, 1:, :] - x[0, :-1, :])

    # y[1, :, 1:] = x[1, :, 1:] - x[1, :, :-1]
    y = jax.ops.index_update(y, jax.ops.index[1, :, 1:], x[1, :, 1:] - x[1, :, :-1])

    y = y[0, :, :] + y[1, :, :]

    return y


def l_transpose(x):
    """

    :rtype: y -> shape = (2, m, n) where p = y[0] and q = y[1].
    :type x -> shape = (m, n)
    """
    if x.shape[-1] != 2:
        m, n = x.shape
        y = jnp.zeros((2, m, n))

        # y[0, :-1, :] = x[:-1, :] - x[1:, :]
        y = jax.ops.index_update(y, jax.ops.index[0, :-1, :], x[:-1, :] - x[1:, :])

        # y[1, :, :-1] = x[:, :-1] - x[:, 1:]
        y = jax.ops.index_update(y, jax.ops.index[1, :, :-1], x[:, :-1] - x[:, 1:])

        return y

    else:
        m, n, _ = x.shape
        y = torch.zeros((2, m, n, 2))

        # y[0, :-1, :] = x[:-1, :] - x[1:, :]
        y = jax.ops.index_update(y, jax.ops.index[0, :-1, :], x[:-1, :] - x[1:, :])

        # y[1, :, :-1] = x[:, :-1] - x[:, 1:]
        y = jax.ops.index_update(y, jax.ops.index[1, :, :-1], x[:, :-1] - x[:, 1:])

        return y


def pc(x, constant):
    if x.shape[-1] != 2:
        # x[x < constant[0]] = constant[0]
        x = jax.ops.index_update(x, jax.ops.index[x < constant[0]], constant[0])

        # x[x > constant[1]] = constant[1]
        x = jax.ops.index_update(x, jax.ops.index[x > constant[1]], constant[1])

    else:
        if constant != (-float('inf'), float('inf')):
            raise NotImplementedError()

    return x


def pp(x):
    if x.shape[-1] != 2:
        bottom = jnp.sqrt(x[0, :, :] ** 2 + x[1, :, :] ** 2)
        # bottom[bottom < 1] = 1
        bottom = jax.ops.index_update(bottom, jax.ops.index[bottom < 1], 1)

        return x / bottom

    else:
        bottom = jnp.sqrt(abs_torch_complex(x[0, :, :]) ** 2 + abs_torch_complex(x[1, :, :]) ** 2)
        # bottom[bottom < 1] = 1
        bottom = jax.ops.index_update(bottom, jax.ops.index[bottom < 1], 1)

        # return x / bottom.unsqueeze(-1)
        return x / jnp.expand_dims(bottom, axis=-1)


class TVOperator(Proximal):
    def __init__(self, lambda_, num_iter):
        """

        :param lambda_: regularization parameters
        :param num_iter: Number of iterations
        """
        super().__init__()

        self.lambda_ = lambda_
        self.N = num_iter

    def eval(self, x, **kwargs):
        y = l_transpose(x)
        y = jnp.sum(jnp.sqrt(y[0, :, :] ** 2 + y[1, :, :] ** 2))
        y = self.lambda_ * y

        return y

    def prox(self, x, **kwargs):
        return self.fit(x, lambda_=self.lambda_, num_iter=self.N)

    @staticmethod
    def fit(b, lambda_, num_iter, c=(-float('inf'), float('inf')), verbose=False):
        if b.shape[-1] != 2:
            m, n = b.shape

            pq = jnp.zeros((2, m, n))  # p and q are concat together.
            rs = jnp.zeros((2, m, n))  # r and s are concat together.

        else:
            m, n, _ = b.shape

            pq = jnp.zeros((2, m, n, 2))  # p and q are concat together.
            rs = jnp.zeros((2, m, n, 2))  # r and s are concat together.

        t = 1

        # iter_ = tqdm(range(num_iter), desc='TVDenoiser') if verbose else range(num_iter)

        for i in range(num_iter):
            t_last = t
            pq_last = pq.copy()

            pq = pp(rs + (1 / (8 * lambda_)) * l_transpose(pc(b - lambda_ * l_forward(rs), c)))
            t = (1 + jnp.sqrt(1 + 4 * (t ** 2))) / 2  # !!! possible error
            rs = pq + (t_last - 1) / t * (pq - pq_last)

        x_start = pc(b - lambda_ * l_forward(pq), c)

        del pq
        del rs

        return x_start
