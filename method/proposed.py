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

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

"""
CT parameters
"""

N = 256  # dimensions and number of detectors per dimension (so 1 detector per pixel)
n_projection = 180  # number of angles

matContent = sio.loadmat('../dataset/CT_images_preprocessed.mat', squeeze_me=True)
ct = matContent['img_cropped'][:, :, 0]
down_ct = cv2.resize(ct, (N, N))

"""
Configure CT projection operator and generate synthetic measurements
"""

num_iter = 100
gamma = 1e-6
rho = 10
tau = 1e-6
lambda_TV = 1
num_iter_TV = 100

xin = jax.device_put(down_ct)  # Convert to jax type, push to GPU

"""
Configure CT projection operator and generate synthetic measurements
"""

angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

A = ParallelBeamProj(xin.shape, 1, N, angles)  # Radon transform operator
d = A @ xin  # Sinogram


def B(x):
    return x


def B_adj(x):
    return x


# initialize x_hat to be the back-projection of d
x_hat = A.fbp(d)
y_hat = B(x_hat)
lambda_hat = jnp.zeros_like(y_hat)

# prior config
prior = TVOperator(lambda_TV, num_iter_TV)
prior.lambda_ = prior.lambda_ / rho

plt.imshow(x_hat, cmap="gray")
plt.title("Input Image")
plt.show()


@jax.jit
def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
    result = l_lambda_hat + l_y_hat - l_x_hat
    return result

start_time = time.time()

# main algorithm
for i in range(num_iter):
    # y step
    y_hat = y_hat - gamma * A.adj(A @ y_hat - d) - gamma * rho * (y_hat - B(x_hat) + lambda_hat)

    # x step

    #  naive conversion
    # torch_prox_input = torch.from_numpy(np.array(y_hat + lambda_hat))
    # x_hat = (prior.prox(torch_prox_input)).numpy()
    # x_hat = jax.device_put(x_hat)

    #  around 35% faster when using dlpack
    prox_in = B_adj(y_hat + lambda_hat) # is this right? Also tau, pho not multiplied cus it screws up the result still need to tune
    prox_in = torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(prox_in))
    prox_out = prior.prox(prox_in)
    x_hat = jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(prox_out))

    # lambda step
    lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)
    print(f'iteration {i} SNR: {metric.psnr(xin, x_hat)}')

print("--- %s seconds ---" % (time.time() - start_time))

plt.imshow(x_hat, cmap="gray")
plt.title("output image")
plt.colorbar()
plt.show()
