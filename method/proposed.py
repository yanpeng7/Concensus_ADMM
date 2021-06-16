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
import json

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
with open('../config.json') as File:
    config = json.load(File)

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

num_iter = config['method']['proposed']['num_iter']
gamma = config['method']['proposed']['gamma']
rho = config['method']['proposed']['rho']
tau = config['method']['proposed']['tau']
lambda_TV = config['method']['proposed']['lambda_TV']
num_iter_TV = config['method']['proposed']['num_iter_TV']

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
d = A_original @ xin  # Sinogram

A = ParallelBeamProj((int(N / 2), int(N / 2)), 1, N, angles)  # Radon transform operator


def A_i(x):
    return A @ x


def A_i_batched(x_batched):
    return jax.pmap(A_i)(x_batched)


def A_i_adj(x):
    return A.adj(x)


def A_i_adj_batched(x_batched):
    return jax.pmap(A_i_adj)(x_batched)


def jax2torch(x):
    return torch_dlpack.from_dlpack(jax_dlpack.to_dlpack(x))


def torch2jax(x):
    return jax_dlpack.from_dlpack(torch_dlpack.to_dlpack(x))


# initialize x_hat to be the back-projection of d
x_hat = A_original.fbp(d)
y_hat = linop.fmult(x_hat)
lambda_hat = jnp.zeros_like(y_hat)

# prior config
prior = TVOperator(lambda_TV, num_iter_TV)
prior.lambda_ = prior.lambda_ / rho

plt.imshow(x_hat, cmap="gray")
plt.title("Input Image")
plt.colorbar()
plt.show()


@jax.jit
def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
    result = l_lambda_hat + l_y_hat - linop.fmult(l_x_hat)
    return result


print("algorithm started now fucking wait")
start_time = time.time()

# main algorithm
for i in range(num_iter):
    # y step
    y_hat = y_hat - gamma * A_i_adj_batched(A_i_batched(y_hat) - d) - gamma * rho * (
            y_hat - linop.fmult(x_hat) + lambda_hat)

    # x step
    prox_in = 1 * 1 * (x_hat - linop.ftran(linop.fmult(x_hat) - y_hat - lambda_hat))
    prox_in = jax2torch(prox_in)
    prox_out = prior.prox(prox_in)
    x_hat = torch2jax(prox_out)

    # lambda step
    lambda_hat = lambda_step(x_hat, y_hat, lambda_hat)
    print(f'iteration {i} SNR: {metric.psnr(xin, x_hat)}')

    # if i % 10 == 0:
    #     plt.imshow()

print("--- %s seconds ---" % (time.time() - start_time))

plt.imshow(x_hat, cmap="gray")
plt.title("output image")
plt.colorbar()
plt.show()
