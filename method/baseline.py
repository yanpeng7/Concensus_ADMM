import jax
import numpy as np
import scipy.io as sio
import cv2
from scico.linop.radon import ParallelBeamProj
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scico import metric
import time
import JAX_TV
import os

def run(config):
    num_iter = int(config.method.baseline.num_iter)
    gamma = float(config.method.baseline.gamma)
    rho = float(config.method.baseline.rho)
    tau = float(config.method.baseline.tau)
    lambda_TV = float(config.method.baseline.lambda_TV)
    num_iter_TV = int(config.method.baseline.num_iter_TV)
    N = int(config.dataset.ct_sample.img_dim)
    n_projection = int(config.dataset.ct_sample.num_projection)
    num_detector = int(config.dataset.ct_sample.num_detector)

    """
    Load CT image
    """

    matContent = sio.loadmat('dataset/CT_images_preprocessed.mat', squeeze_me=True)
    ct = matContent['img_cropped'][:, :, 0]
    down_ct = cv2.resize(ct, (N, N))

    xin = jax.device_put(down_ct)  # Convert to jax type, push to GPU

    """
    Configure CT projection operator and generate synthetic measurements
    """

    angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

    A = ParallelBeamProj(xin.shape, 1.0, num_detector, angles)  # Radon transform operator
    d = A @ xin  # Sinogram

    # initialize x_hat to be the back-projection of d
    x_hat_in = A.fbp(d)
    y_hat_in = x_hat_in
    lambda_hat_in = jnp.zeros_like(y_hat_in)

    def A_i(A_func, x):
        return A_func @ x

    def A_i_adj(A_func, x):
        return A_func.adj(x)

    @jax.jit
    def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
        result = l_lambda_hat + l_y_hat - l_x_hat
        return result

    @jax.jit
    def x_step(x_y_hat, x_lambda_hat):
        return x_y_hat + x_lambda_hat

    def main_body_func(iteration, init_vals):
        x_hat, y_hat, lambda_hat = init_vals

        # y step can not jit due to A being implemented in Astra
        y_hat = y_hat - gamma * A_i_adj(A_i(y_hat) - d) - gamma * rho * (
                y_hat - x_hat + lambda_hat)

        # x step tau * rho
        prox_in = x_step(y_hat, lambda_hat)
        x_hat = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)

        # lambda step
        lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)

        return x_hat, y_hat, lambda_hat

    snr_list = []

    print("algorithm started")
    start_time = time.time()

    for i in range(num_iter):
        # y step
        y_hat_in = y_hat_in - gamma * A_i_adj(A, A_i(A, y_hat_in) - d) - gamma * rho * (
                y_hat_in - x_hat_in + lambda_hat_in)

        if i % 5 == 0:
            # plt.imshow(d)
            # plt.title('d')
            # plt.colorbar()
            # plt.show()

            A_y = A_i(A, y_hat_in)
            # plt.imshow(A_y)
            # plt.title('A y')
            # plt.colorbar()
            # plt.show()

            diff = d - A_y
            plt.imshow(diff)
            plt.title('diff')
            plt.colorbar()
            plt.show()

        # x step tau * rho
        prox_in = x_step(y_hat_in, lambda_hat_in)
        x_hat_in = JAX_TV.TotalVariation_Proximal(prox_in, lambda_TV / rho * tau, num_iter_TV)
        # lambda step
        lambda_hat_in = lambda_step(x_hat_in, y_hat_in, lambda_hat_in)
        print(f'iteration {i} SNR: {metric.snr(xin, x_hat_in)}')
        snr_list.append(metric.snr(xin, x_hat_in))

    x_hat_out = x_hat_in

    # x_hat_out, _, _ = jax.lax.fori_loop(1, num_iter, body_fun=main_body_func,
    #                                     init_val=(x_hat_in, y_hat_in, lambda_hat_in))

    x_hat_out.block_until_ready()

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)
    print(f'final SNR: {metric.snr(xin, x_hat_out)}')

    np.save(os.path.join('saved', 'baseline'), snr_list)

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    im1 = axes[0].imshow(x_hat_in, cmap="gray")
    im2 = axes[1].imshow(x_hat_out, cmap="gray")
    im3 = axes[2].imshow(xin, cmap="gray")
    axes[0].title.set_text(f'input SNR: {"{:.2f}".format(metric.snr(xin, x_hat_in))}')
    axes[1].title.set_text(f'output SNR: {"{:.2f}".format(metric.snr(xin, x_hat_out))}')
    axes[2].title.set_text('ground truth')
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.show()
