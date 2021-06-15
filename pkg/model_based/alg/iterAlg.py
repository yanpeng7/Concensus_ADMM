import torch
from tqdm import tqdm

from ..linop import LinearOperator
from ..prox import Proximal
from ...utility.measure import compare_snr
import numpy as np
import os


def iter_alg(
        linop: LinearOperator,
        prox: Proximal,
        y, method='ISTA',
        verbose=False,
        num_iter=100,
        step=1,
        x_true=None,
        desc=None
):
    if method != 'ISTA':
        raise NotImplementedError()

    iter_ = tqdm(range(num_iter), desc=desc)

    prox.lambda_ = prox.lambda_ * step

    snr_list = []

    x = linop.ftran(y)  # Set Initial Point
    for i in iter_:
        if verbose:
            d_full = torch.sum((linop.fmult(x) - y) ** 2).item()
            r_full = prox.eval(x)

            verbose_info = "[%.3d/%.3d] dFull: [%.5f] rFull: [%.5f] Objective: [%.5f]" % (
                i + 1, num_iter, d_full, prox.lambda_ * r_full, d_full + prox.lambda_ * r_full
            )

            snr = compare_snr(x, x_true).item()

            if x_true is not None:
                verbose_info += " SNR: [%.3f]" % snr

            snr_list.append(snr)

            iter_.write(verbose_info)
            iter_.update()

        x = x - step * linop.ftran(linop.fmult(x) - y)
        x = prox.prox(x)

    # np.save(os.path.join('saved_results', 'baseline'), snr_list)

    return x
