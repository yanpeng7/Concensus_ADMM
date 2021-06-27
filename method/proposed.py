import jax
import numpy as np
import scipy.io as sio
import cv2
from scico.linop.radon import ParallelBeamProj
import jax.numpy as jnp
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


def run(config):
    num_iter = int(config.method.proposed.num_iter)
    gamma = float(config.method.proposed.gamma)
    rho = float(config.method.proposed.rho)
    tau = float(config.method.proposed.tau)
    lambda_TV = float(config.method.proposed.lambda_TV)
    num_iter_TV = int(config.method.proposed.num_iter_TV)
    N = int(config.dataset.ct_sample.img_dim)
    n_projection = int(config.dataset.ct_sample.num_projection)
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
    B operators
    """

    # convolution returns uninitialized array if move to another file for some reason
    # SNR stuck at 12 if passed through convolution func and is on GPU, but isn't stuck when convolution is done on CPU?
    # GPU and CPU output the same if no convolution function is used?

    if kernel_type == "none":
        kernel = None
    elif kernel_type == "gaussian":
        kernel = jnp.asarray([[[[1 / 16, 1 / 8, 1 / 16], [1 / 8, 1 / 4, 1 / 8], [1 / 16, 1 / 8, 1 / 16]]]])
    elif kernel_type == "identity":
        kernel = jnp.asarray([[[[0, 0, 0], [0, 1.0, 0], [0, 0, 0]]]])
    else:
        raise NotImplementedError('specify either none, gaussian, or identity kernel')

    def B(x, conv_kernel=kernel, K=4):
        if conv_kernel is not None:
            x = jnp.expand_dims(jnp.expand_dims(x, axis=0), axis=0)
            x = jax.lax.conv_general_dilated(x,  # lhs = image tensor
                                             conv_kernel,  # rhs = conv kernel tensor
                                             (1, 1),  # window strides
                                             "SAME",  # padding mode
                                             None,  # lhs/image dilation
                                             None)  # rhs/kernel dilation
            x = jnp.squeeze(x, (0, 1))
        ret = []
        for a in range(K // 2):
            for j in range(K // 2):
                ret.append(x[a::K // 2, j::K // 2])
        ret = jnp.stack(ret, axis=0)
        return ret

    def B_T(x, conv_kernel=kernel, K=4):
        _, width, height = x.shape
        ret = jnp.zeros((width * K // 2, height * K // 2))
        index_x = 0
        for a in range(K // 2):
            for j in range(K // 2):
                ret = index_update(ret, index[a::K // 2, j::K // 2], x[index_x])
                index_x = index_x + 1

        if conv_kernel is not None:
            ret = jnp.expand_dims(jnp.expand_dims(ret, axis=0), axis=0)
            kernel_rot = jnp.rot90(jnp.rot90(conv_kernel, axes=(2, 3)), axes=(2, 3))
            ret = jax.lax.conv_general_dilated(ret,  # lhs = image tensor
                                               kernel_rot,  # rhs = conv kernel tensor
                                               (1, 1),  # window strides
                                               "SAME",  # padding mode
                                               None,  # lhs/image dilation
                                               None)  # rhs/kernel dilation
            ret = jnp.squeeze(ret, (0, 1))
        return ret

    """
    Configure CT 
    """

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

    A_original = ParallelBeamProj(xin.shape, 1, N, angles)  # Radon transform operator
    A = ParallelBeamProj((int(N / 2), int(N / 2)), 0.5, N, angles)  # Radon transform operator

    d = A_original @ xin  # Sinogram

    # initialize x_hat to be the back-projection of d
    x_hat_in = A_original.fbp(d)
    y_hat_in = B(x_hat_in)
    lambda_hat_in = jnp.zeros_like(y_hat_in)

    """
    A operators
    """
    def A_i(A_func, x):
        return A_func @ (x * A_scaling)

    def A_i_adj(A_func, x):
        return (A_func.adj(x)) / A_scaling

    def A_i_batched_parallel(A_func, x_batched):
        return jax.pmap(A_i, static_broadcasted_argnums=0)(A_func, x_batched)

    def A_i_batched_seq(A_func, x_batched):
        x_list = []
        for j in range(4):
            x_list.append(A_i(A_func, x_batched[j]))
        return jnp.stack(x_list, axis=0)

    def A_i_adj_batched_seq(A_func, x_batched):
        x_list = []
        for j in range(4):
            x_list.append(A_i_adj(A_func, x_batched[j]))
        return jnp.stack(x_list, axis=0)

    def A_i_adj_batched_parallel(A_func, x_batched):
        return jax.pmap(A_i_adj, static_broadcasted_argnums=0)(A_func, x_batched)

    if A_impl == "seq":
        A_i_batched = A_i_batched_seq
        A_i_adj_batched = A_i_adj_batched_seq
    elif A_impl == "parallel":
        A_i_batched = A_i_batched_parallel
        A_i_adj_batched = A_i_adj_batched_parallel
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
        x = B(x)
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

    @jax.jit
    def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
        result = l_lambda_hat + l_y_hat - B(l_x_hat)
        return result

    @jax.jit
    def x_step(x_x_hat, x_y_hat, x_lambda_hat):
        return x_x_hat - tau * B_T(B(x_x_hat) - x_y_hat - x_lambda_hat)

    def main_body_func(iteration, init_vals):
        x_hat, y_hat, lambda_hat = init_vals

        # y step can not jit due to A being implemented in Astra
        y_hat = y_hat - gamma * A_i_adj_batched(A, A_i_batched(A, y_hat) - d) - gamma * rho * (
                y_hat - B(x_hat) + lambda_hat)

        # x step tau * rho
        prox_in = x_step(x_hat, y_hat, lambda_hat)

        x_hat = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)

        # lambda step
        lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)

        return x_hat, y_hat, lambda_hat,

    # snr_list = []

    print("algorithm started!")
    start_time = time.time()

    x_hat_out, _, _ = jax.lax.fori_loop(1, num_iter, body_fun=main_body_func,
                                        init_val=(x_hat_in, y_hat_in, lambda_hat_in))

    x_hat_out.block_until_ready()

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)
    print(f'final SNR: {metric.snr(xin, x_hat_out)}')

    # np.save(os.path.join('saved', 'proposed_gau_slow'), snr_list)

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
