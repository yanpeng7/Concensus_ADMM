import numpy as np
# from B_impl import B_numpy, B_T_numpy
from numpy_impl.numpy_B_impl import B_numpy, B_T_numpy
import jax
import jax.numpy as jnp
import scipy.io as sio
import cv2
import JAX_TV
from scico import metric
import matplotlib.pyplot as plt
from jax.ops import index, index_update
import os

if __name__ == '__main__':
    N = 256
    num_iter = 300
    gamma = 1e-0
    matContent = sio.loadmat('../dataset/CT_images_preprocessed.mat', squeeze_me=True)
    shift_list = np.asarray([[0, 0.25], [0, 0.75], [0.25, 0], [0.75, 0]])
    ct = matContent['img_cropped'][:, :, 0]
    true = cv2.resize(ct, (N, N))
    interpolated = cv2.resize(true, (N//2, N//2))
    interpolated = cv2.resize(interpolated, (N, N))
    y = B_numpy(true, shift_list)
    x_init = jnp.zeros((N,N))
    x = x_init
    x_prev = x
    theta_prev = 1
    for i in range(num_iter):
        theta = 0.5 * (1 + jnp.sqrt(1 + 4 * theta_prev * theta_prev))

        s = np.array(x + ((x_prev - 1) / theta) * (x - x_prev))

        x_next = JAX_TV.TotalVariation_Proximal(jnp.array(s - gamma*B_T_numpy(B_numpy(s, shift_list) - y, shift_list)), 9e-5)

        x_prev = x
        x = x_next
        theta_prev = theta

        print(f'SNR: {"{:.2f}".format(metric.snr(true, x))}')

    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(12, 5))
    im1 = axes[0].imshow(x, cmap="gray")
    im2 = axes[1].imshow(interpolated, cmap="gray")
    im3 = axes[2].imshow(true, cmap="gray")
    axes[0].title.set_text(f'output SNR: {"{:.2f}".format(metric.snr(true, x))}')
    axes[1].title.set_text(f'interpolated SNR: {"{:.2f}".format(metric.snr(interpolated, x))}')
    axes[2].title.set_text('ground truth')
    # fig.colorbar(im1, ax=axes[0])
    # fig.colorbar(im2, ax=axes[1])
    # fig.colorbar(im3, ax=axes[2])
    plt.show()



