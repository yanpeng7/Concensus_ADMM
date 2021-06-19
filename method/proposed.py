import jax
import numpy as np
import scipy.io as sio
import cv2
from scico.linop.radon import ParallelBeamProj
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pkg.model_based.prox.TV import TVOperator
from scico import metric
import os
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack
import time
import torch
from concensus_SR_demo import ConsensusSR
from pkg.torch.callback import Tensorboard, check_and_mkdir
import jsonargparse.core as jc
import pprint


def run(config):
    """
    CT parameters
    """

    N = 256  # dimensions and number of detectors per dimension (so 1 detector per pixel)
    n_projection = 180  # number of angles

    matContent = sio.loadmat('dataset/CT_images_preprocessed.mat', squeeze_me=True)
    ct = matContent['img_cropped'][:, :, 0]
    down_ct = cv2.resize(ct, (N, N))

    """
    Configure CT projection operator and generate synthetic measurements
    """

    num_iter = int(config.method.proposed.num_iter)
    gamma = float(config.method.proposed.gamma)
    rho = float(config.method.proposed.rho)
    tau = float(config.method.proposed.tau)
    lambda_TV = float(config.method.proposed.lambda_TV)
    num_iter_TV = int(config.method.proposed.num_iter_TV)

    tb_writer = Tensorboard(file_path=config.setting.root_path + config.setting.proj_name + '/')

    # num_iter = 100
    # gamma = 1e-6
    # rho = 10
    # tau = 1
    # lambda_TV = 1
    # num_iter_TV = 100

    linop = ConsensusSR(4, kernel=None)

    xin = jax.device_put(down_ct)  # Convert to jax type, push to GPU
    """
    Configure CT projection operator and generate synthetic measurements
    """

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

    A_original = ParallelBeamProj(xin.shape, 1, N, angles)  # Radon transform operator
    A = ParallelBeamProj((int(N / 2), int(N / 2)), 0.5, N, angles)  # Radon transform operator

    d = A_original @ xin  # Sinogram

    # initialize x_hat to be the back-projection of d
    x_hat = A_original.fbp(d)
    y_hat = linop.fmult(x_hat)
    lambda_hat = jnp.zeros_like(y_hat)

    def A_i(x):
        return A @ (x * 2)

    # def A_i_batched(x_batched):
    #     return jax.pmap(A_i)(x_batched)

    def A_i_batched(x_batched):
        x_list = []
        for j in range(4):
            x_list.append(A_i(x_batched[j]))
        return jnp.stack(x_list, axis=0)

    def A_i_adj(x):
        return (A.adj(x))/2

    # def A_i_adj_batched(x_batched):
    #     return jax.pmap(A_i_adj)(x_batched)

    def A_i_adj_batched(x_batched):
        x_list = []
        for j in range(4):
            x_list.append(A_i_adj(x_batched[j]))
        return jnp.stack(x_list, axis=0)

    def jax2torch(x):
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x))

    def torch2jax(x):
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x))


    # prior config
    prior = TVOperator(lambda_TV, num_iter_TV)
    prior.lambda_ = prior.lambda_ / rho * tau

    x_hat_start = x_hat

    tb_writer.manual_add(global_step=0,
                         image={'per_iter/x0': jax2torch(x_hat).detach().cpu(),
                                'per_iter/xin': jax2torch(xin).detach().cpu()},
                         text={'config': pprint.pformat(jc.namespace_to_dict(config))})

    @jax.jit
    def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
        result = l_lambda_hat + l_y_hat - linop.fmult(l_x_hat)
        return result

    # print("algorithm started")
    start_time = time.time()

    # def oracle_d(x):
    #     x = linop.fmult(x)
    #     x_list = []
    #     for j in range(4):
    #         x_list.append(A @ x[j])
    #     return jnp.stack(x_list, axis=0)
    #
    # d = oracle_d(xin)


    # main algorithm
    for i in range(num_iter):
        # y step
        y_hat = y_hat - gamma * A_i_adj_batched(A_i_batched(y_hat) - d) - gamma * rho * (
                y_hat - linop.fmult(x_hat) + lambda_hat)

        # if i % 4 == 0:
        #     plt.imshow(x_hat, cmap="gray")
        #     plt.colorbar()
        #     plt.title("before prox")
        #     plt.show()

        # x step tau * rho
        prox_in = x_hat - tau * linop.ftran(linop.fmult(x_hat) - y_hat - lambda_hat)
        prox_in = jax2torch(prox_in)
        prox_out = prior.prox(prox_in)
        x_hat = torch2jax(prox_out)

        # if i % 4 == 0:
        #     plt.imshow(x_hat, cmap="gray")
        #     plt.colorbar()
        #     plt.title("after prox")
        #     plt.show()

        # lambda step
        lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)

        tb_writer.manual_add(global_step=i + 1, image={'per_iter/x_hat': jax2torch(x_hat)},
                             log={'per_iter/snr': jax2torch(metric.snr(xin, x_hat)).item()})

        print(f'iteration {i} SNR: {metric.snr(xin, x_hat)}')
        # if i == num_iter - 1:
        #     print(f'iteration {i} SNR: {metric.snr(xin, x_hat)}')

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)

    # fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    # im1 = axes[0].imshow(x_hat_start, cmap="gray")
    # im2 = axes[1].imshow(x_hat, cmap="gray")
    # im3 = axes[2].imshow(xin, cmap="gray")
    # axes[0].title.set_text('input')
    # axes[1].title.set_text('output')
    # axes[2].title.set_text('ground truth')
    # fig.colorbar(im1, ax=axes[0])
    # fig.colorbar(im2, ax=axes[1])
    # fig.colorbar(im3, ax=axes[2])
    # plt.show()

    tb_writer.manual_add(global_step=num_iter, hparams={
        'hparam_dict': {
            "num_iter": config.method.proposed.num_iter,
            "gamma": config.method.proposed.gamma,
            "rho": config.method.proposed.rho,
            "tau": config.method.proposed.tau,
            "lambda_TV": config.method.proposed.lambda_TV,
            "num_iter_TV": config.method.proposed.num_iter_TV,
        },

        'metric_dict': {
            "metrics/snr": jax2torch(metric.snr(xin, x_hat)).item(),
            "metrics/running_time": running_time
        }
    })
