import jax
import jax.numpy as jnp
import numpy as np
import scipy.io as sio
import cv2
from scico.linop.radon import ParallelBeamProj
import matplotlib.pyplot as plt
from scico import metric
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack
import time
from pkg.torch.callback import Tensorboard
import jsonargparse.core as jc
import pprint
import JAX_TV
from jax.ops import index, index_update
import os
from B_impl import B_jax, B_T_jax, generate_weight_matrices
import astra
from A_impl import A_batched, A_adj_batched


def run(config):
    num_iter = int(config.method.proposed.num_iter)
    gamma = float(config.method.proposed.gamma)
    rho = float(config.method.proposed.rho)
    tau = float(config.method.proposed.tau)
    lambda_TV = float(config.method.proposed.lambda_TV)
    num_iter_TV = int(config.method.proposed.num_iter_TV)
    N = int(config.dataset.ct_sample.img_dim)
    n_projection = int(config.dataset.ct_sample.num_projection)
    num_detector = int(config.dataset.ct_sample.num_detector)
    A_scaling = float(config.method.proposed.A_scaling)
    kernel_type = str(config.method.proposed.kernel_type)
    A_impl = str(config.method.proposed.A_impl)
    """
    Load image
    """

    matContent = sio.loadmat('dataset/CT_images_preprocessed.mat', squeeze_me=True)
    ct = matContent['img_cropped'][:, :, 0]
    down_ct = cv2.resize(ct, (N, N))
    xin = jax.device_put(down_ct)  # Convert to jax type, push to GPU

    tb_writer = Tensorboard(file_path=config.setting.root_path + config.setting.proj_name + '/')

    """
    Configure CT 
    """

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

    A_original = ParallelBeamProj(xin.shape, 1.0, num_detector, angles)  # Radon transform operator
    # A = ParallelBeamProj((int(N / 2), int(N / 2)), 0.5, N, angles)  # Radon transform operator

    d = A_original @ xin  # Sinogram

    mat_list, init_point_list = generate_weight_matrices()
    print(mat_list)
    print(init_point_list)

    # initialize x_hat to be the back-projection of d
    x_hat_in = A_original.fbp(d)
    y_hat_in = B_jax(x_hat_in, mat_list)
    lambda_hat_in = jnp.zeros_like(y_hat_in)

    A = None
    r = 0.5

    """
    A operators
    """

    def astra_A_i(projector, x):
        x = np.array(x)
        proj_geom = astra.create_proj_geom('parallel', 1.0, num_detector, np.linspace(0, np.pi, 180))
        vol_geom_y = astra.create_vol_geom(N // 2, N // 2, -N * r, N * r, -N * r, N * r)
        projector = astra.create_projector("line", proj_geom, vol_geom_y)
        astra_id, result = astra.create_sino(x, projector)
        astra.data2d.delete(astra_id)
        result = jnp.array(result)
        return result

    def astra_A_i_adj(projector, x):
        x = np.array(x)
        proj_geom = astra.create_proj_geom('parallel', 1.0, num_detector, np.linspace(0, np.pi, 180))
        vol_geom_y = astra.create_vol_geom(N // 2, N // 2, -N * r, N * r, -N * r, N * r)
        projector = astra.create_projector("line", proj_geom, vol_geom_y)
        astra_id, result = astra.create_backprojection(x, projector)
        astra.data2d.delete(astra_id)
        result = jnp.array(result)
        return result

    def A_i_batched_seq(A_func, x_batched):
        x_list = []
        for j in range(4):
            x_list.append(astra_A_i(A_func, x_batched[j]))
        return jnp.stack(x_list, axis=0)

    def A_i_adj_batched_seq(A_func, x_batched):
        x_list = []
        for j in range(4):
            x_list.append(astra_A_i_adj(A_func, x_batched[j]))
        return jnp.stack(x_list, axis=0)

    if A_impl == "seq":
        A_i_batched = A_i_batched_seq
        A_i_adj_batched = A_i_adj_batched_seq
    else:
        raise NotImplementedError('specify either seq or parallel')

    """
    Utility functions
    """

    def jax2torch(x):
        return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x))

    def torch2jax(x):
        return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x))

    def oracle_d(x):
        x = B_jax(x, mat_list)
        x_list = []
        for j in range(4):
            x_list.append(A @ x[j])
        return jnp.stack(x_list, axis=0)

    # d = oracle_d(xin)

    """
    Main Algorithm 
    """
    x_hat_start = x_hat_in
    tb_writer.manual_add(global_step=0,
                         image={'per_iter/x0': jax2torch(x_hat_in).detach().cpu(),
                                'per_iter/xin': jax2torch(xin).detach().cpu()},
                         text={'config': pprint.pformat(jc.namespace_to_dict(config))})

    def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
        result = l_lambda_hat + l_y_hat - B_jax(l_x_hat, mat_list)
        return result

    # @jax.jit
    # def x_step(x_x_hat, x_y_hat, x_lambda_hat):
    #     return x_x_hat - tau * B_T(B(x_x_hat) - x_y_hat - x_lambda_hat)
    #
    # def main_body_func(iteration, init_vals):
    #     x_hat, y_hat, lambda_hat = init_vals
    #
    #     # y step can not jit due to A being implemented in Astra
    #     y_hat = y_hat - gamma * A_i_adj_batched(A, A_i_batched(A, y_hat) - d) - gamma * rho * (
    #             y_hat - B(x_hat) + lambda_hat)
    #
    #     # x step tau * rho
    #     prox_in = x_step(x_hat, y_hat, lambda_hat)
    #
    #     x_hat = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)
    #
    #     # lambda step
    #     lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)
    #
    #     return x_hat, y_hat, lambda_hat

    snr_list = []

    print("algorithm started!")
    start_time = time.time()

    # x_hat_out, _, _ = jax.lax.fori_loop(1, num_iter, body_fun=main_body_func,
    #                                     init_val=(x_hat_in, y_hat_in, lambda_hat_in))
    #
    # x_hat_out.block_until_ready()

    for i in range(num_iter):
        # y step
        y_hat_in = y_hat_in - gamma * A_adj_batched(A_batched(y_hat_in, init_point_list) - d,
                                                    init_point_list) - gamma * rho * (
                           y_hat_in - B_jax(x_hat_in, mat_list) + lambda_hat_in)

        # y_hat_in = y_hat_in - gamma * A_i_adj_batched(A, A_i_batched(A, y_hat_in) - d) - gamma * rho * (
        #                  y_hat_in - B_jax(x_hat_in, mat_list) + lambda_hat_in)

        # if i % 1 == 0:
        #
        #     # plt.imshow(d)
        #     # plt.title('d')
        #     # plt.colorbar()
        #     # plt.show()
        #
        #     A_y = A_i_batched(A, y_hat_in)[0]
        #     # plt.imshow(A_y)
        #     # plt.title('A y')
        #     # plt.colorbar()
        #     # plt.show()
        #
        #     diff = d - A_y
        #     print(f'diff: {np.sum(diff)}')
        #     # plt.imshow(diff)
        #     # plt.title('diff')
        #     # plt.colorbar()
        #     # plt.show()

        # x step tau * rho
        prox_in = x_hat_in - tau * B_T_jax(B_jax(x_hat_in, mat_list) - y_hat_in - lambda_hat_in, mat_list)
        x_hat_in = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)
        # lambda step
        lambda_hat_in = lambda_step(x_hat_in, y_hat_in, lambda_hat_in)
        print(f'iteration {i} SNR: {metric.snr(xin, x_hat_in)}')
        # snr_list.append(metric.snr(xin, x_hat_in))

    # for i in range(num_iter):
    #     # y step
    #     y_hat_in = y_hat_in - gamma * A_i_adj_batched(A, A_i_batched(A, y_hat_in) - d) - gamma * rho * (
    #             y_hat_in - B_jax(x_hat_in, mat_list) + lambda_hat_in)
    #
    #     # x step tau * rho
    #     prox_in = x_hat_in - tau * B_T_jax(B_jax(x_hat_in, mat_list) - y_hat_in - lambda_hat_in, mat_list)
    #     x_hat_in = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)
    #     # lambda step
    #     lambda_hat_in = lambda_step(x_hat_in, y_hat_in, lambda_hat_in)
    #     print(f'iteration {i} SNR: {metric.snr(xin, x_hat_in)}')
    #     # snr_list.append(metric.snr(xin, x_hat_in))

    x_hat_out = x_hat_in

    x_hat_out.block_until_ready()

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)
    print(f'final SNR: {metric.snr(xin, x_hat_out)}')

    # np.save(os.path.join('saved', '0.25-75'), snr_list)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    im1 = axes[0].imshow(x_hat_start, cmap="gray")
    im2 = axes[1].imshow(x_hat_out, cmap="gray")
    im3 = axes[2].imshow(xin, cmap="gray")
    axes[0].title.set_text(f'input SNR: {"{:.2f}".format(metric.snr(xin, x_hat_start))}')
    axes[1].title.set_text(f'output SNR: {"{:.2f}".format(metric.snr(xin, x_hat_out))}')
    axes[2].title.set_text('ground truth')
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.show()

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
            "metrics/snr": jax2torch(metric.snr(xin, x_hat_out)).item(),
            "metrics/running_time": running_time
        }
    })
