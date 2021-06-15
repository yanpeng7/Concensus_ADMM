from tqdm import tqdm
import torch
import math

from . import Proximal
from ...utility.torch_complex_op import abs_torch_complex


def l_forward(x: torch.Tensor):
    """

    :rtype: y -> shape = (m, n)
    :type x -> shape = (2, m, n) where p = x[0] and q = x[1].
    """
    y = x.clone()

    y[0, 1:, :] = x[0, 1:, :] - x[0, :-1, :]
    y[1, :, 1:] = x[1, :, 1:] - x[1, :, :-1]

    y = y[0, :, :] + y[1, :, :]

    return y


def l_transpose(x: torch.Tensor):
    """

    :rtype: y -> shape = (2, m, n) where p = y[0] and q = y[1].
    :type x -> shape = (m, n)
    """
    if x.shape[-1] != 2:
        m, n = x.shape
        y = torch.zeros(2, m, n)

        if x.is_cuda:
            y = y.cuda()

        y[0, :-1, :] = x[:-1, :] - x[1:, :]
        y[1, :, :-1] = x[:, :-1] - x[:, 1:]

        return y

    else:
        m, n, _ = x.shape
        y = torch.zeros(2, m, n, 2)

        if x.is_cuda:
            y = y.cuda()

        y[0, :-1, :] = x[:-1, :] - x[1:, :]
        y[1, :, :-1] = x[:, :-1] - x[:, 1:]

        return y


def pc(x: torch.Tensor, constant):
    if x.shape[-1] != 2:
        x[x < constant[0]] = constant[0]
        x[x > constant[1]] = constant[1]

    else:
        if constant != (-float('inf'), float('inf')):
            raise NotImplementedError()

    return x


def pp(x: torch.Tensor):
    if x.shape[-1] != 2:
        bottom = torch.sqrt(x[0, :, :] ** 2 + x[1, :, :] ** 2)
        bottom[bottom < 1] = 1

        return x / bottom

    else:
        bottom = torch.sqrt(abs_torch_complex(x[0, :, :]) ** 2 + abs_torch_complex(x[1, :, :]) ** 2)
        bottom[bottom < 1] = 1

        return x / bottom.unsqueeze(-1)


class TVOperator(Proximal):
    def __init__(self, lambda_, num_iter):
        """

        :param lambda_: regularization parameters
        :param num_iter: Number of iterations
        """
        super().__init__()

        self.lambda_ = lambda_
        self.N = num_iter

    def eval(self, x,  **kwargs):
        y = l_transpose(x)
        y = torch.sum(torch.sqrt(y[0, :, :] ** 2 + y[1, :, :] ** 2))
        y = self.lambda_ * y

        return y

    def prox(self, x,  **kwargs):
        return self.fit(x, lambda_=self.lambda_, num_iter=self.N)

    @staticmethod
    def fit(b, lambda_, num_iter, c=(-float('inf'), float('inf')), verbose=False):
        if b.shape[-1] != 2:
            m, n = b.shape

            pq = torch.zeros(2, m, n)  # p and q are concat together.
            rs = torch.zeros(2, m, n)  # r and s are concat together.

        else:
            m, n, _ = b.shape

            pq = torch.zeros(2, m, n, 2)  # p and q are concat together.
            rs = torch.zeros(2, m, n, 2)  # r and s are concat together.

        if b.is_cuda:
            pq = pq.cuda()
            rs = rs.cuda()

        t = 1

        iter_ = tqdm(range(num_iter), desc='TVDenoiser') if verbose else range(num_iter)

        for _ in iter_:
            t_last = t
            pq_last = pq.clone()

            pq = pp(rs + (1 / (8 * lambda_)) * l_transpose(pc(b - lambda_ * l_forward(rs), c)))
            t = (1 + math.sqrt(1 + 4 * (t ** 2))) / 2
            rs = pq + (t_last - 1) / t * (pq - pq_last)

        x_start = pc(b - lambda_ * l_forward(pq), c)

        del pq
        del rs

        return x_start
