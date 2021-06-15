import torch
import numpy as np


def np2torch_complex(x: np.ndarray):
    assert isinstance(x, np.ndarray)

    return torch.stack([
        torch.from_numpy(x.real), torch.from_numpy(x.imag)
    ], -1)


def torch_complex2np(x: torch.Tensor):
    x_real, x_imag = x[..., 0].detach().cpu().numpy(), x[..., 1].detach().cpu().numpy()

    return x_real + x_imag * 1j


def new_torch_complex(real, imag):
    if isinstance(real, np.ndarray):
        real = torch.from_numpy(real)

    if isinstance(imag, np.ndarray):
        imag = torch.from_numpy(imag)

    return torch.stack([real, imag], -1)


def add_torch_complex(x: torch.Tensor, y: torch.Tensor):
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)

    if x.dim() == y.dim():
        return x + y

    elif x.dim() > y.dim():
        return x + torch.stack([y, y], -1)

    else:
        return torch.stack([x, x], -1) + y


def sub_torch_complex(x: torch.Tensor, y: torch.Tensor):
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor)

    if x.dim() == y.dim():
        return x - y

    elif x.dim() > y.dim():
        return x - torch.stack([y, y], -1)

    else:
        return torch.stack([x, x], -1) - y


def mm_torch_complex(x: torch.Tensor, y: torch.Tensor):
    a, b = x[..., 0], x[..., 1]
    c, d = y[..., 0], y[..., 1]

    return torch.stack([
        a.mm(c) - b.mm(d), a.mm(d) + b.mm(c)
    ], -1)


def mul_torch_complex(x, y):
    if (isinstance(x, float) and isinstance(y, torch.Tensor)) or (isinstance(y, float) and isinstance(x, torch.Tensor)):
        return x * y

    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

        if x.dim() == y.dim():
            a, b = x[..., 0], x[..., 1]
            c, d = y[..., 0], y[..., 1]

            return torch.stack([
                a * c - b * d, a * d + b * c
            ], -1)

        elif x.dim() > y.dim():
            return x * y.unsqueeze(-1)

        else:
            return x.unsqueeze(-1) * y


def tran_torch_complex(x: torch.Tensor, dim0, dim1):
    return x.transpose(dim0, dim1)


def conj_torch_complex(x: torch.Tensor):
    return torch.stack([x[..., 0], -1 * x[..., 1]], -1)


def div_torch_complex(x, y):
    if (isinstance(x, float) and isinstance(y, torch.Tensor)) or (isinstance(y, float) and isinstance(x, torch.Tensor)):
        return x / y

    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):

        if x.dim() == y.dim():
            a, b = x[..., 0], x[..., 1]
            c, d = y[..., 0], y[..., 1]

            denominator = c ** 2 + d ** 2

            return torch.stack([
                (a * c + b * d)/denominator, (b * c - a * d)/denominator
            ], -1)

        elif x.dim() > y.dim():
            return x / y.unsqueeze(-1)

        else:
            return x.unsqueeze(-1) / y


def reshape_torch_complex(x, shape):
    return x.reshape(shape + [2])


def abs_torch_complex(x):
    a, b = x[..., 0], x[..., 1]

    return torch.sqrt(a ** 2 + b ** 2)
