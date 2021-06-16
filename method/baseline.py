import jax
import numpy as np
import scipy.io as sio
import cv2
from scico.linop.radon import ParallelBeamProj
import jax.numpy as jnp
import matplotlib.pyplot as plt
from pkg.model_based.prox.TV import TVOperator
import torch
from scico import admm, functional, linop, loss, metric, plot
import os
from jax import dlpack as jax_dlpack
from torch.utils import dlpack as torch_dlpack
import time
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

xin = jax.device_put(down_ct)  # Convert to jax type, push to GPU

"""
Configure CT projection operator and generate synthetic measurements
"""

angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

A = ParallelBeamProj(xin.shape, 1, N, angles)  # Radon transform operator
d = A @ xin  # Sinogram

# initialize x_hat to be the back-projection of d
x_hat = A.fbp(d)
y_hat = x_hat
lambda_hat = jnp.zeros_like(y_hat)

# prior config
prior = TVOperator(lambda_TV, num_iter_TV)
prior.lambda_ = prior.lambda_ / rho


@jax.jit
def lambda_step(l_x_hat, l_y_hat, l_lambda_hat):
    result = l_lambda_hat + l_y_hat - l_x_hat
    return result


plt.imshow(x_hat, cmap="gray")
plt.title("Input Image")
plt.show()


# TV_prox solver using built in solver- doesn't work well
def prox_TV(x):
    λ = 5e-0  # L1 norm regularization parameter
    ρ = 5e-0  # ADMM penalty parameter
    maxiter = 20  # Number of ADMM iterations

    g_list = [λ * functional.L1Norm()]  # Regularization functionals Fi
    C_list = [linop.FiniteDifference(input_shape=x.shape)]  # Analysis operators Ci
    rho_list = [ρ]  # ADMM parameters

    f = loss.SquaredL2Loss(y=x, A=None)

    admm_ = admm.ADMMQuadraticLoss(
        f=f,
        g_list=g_list,
        C_list=C_list,
        rho_list=rho_list,
        x_solver_kwargs={"maxiter": 18},
        maxiter=maxiter,
        x0=x,
    )
    admm_.solve()
    return admm_.x

start_time = time.time()

# main algorithm
for i in range(num_iter):
    # y step
    y_hat = y_hat - gamma * A.adj(A @ y_hat - d) - gamma * rho * (y_hat - x_hat + lambda_hat)

    # x step

    #  naive conversion
    # torch_prox_input = torch.from_numpy(np.array(y_hat + lambda_hat))
    # x_hat = (prior.prox(torch_prox_input)).numpy()
    # x_hat = jax.device_put(x_hat)

    #  around 35% faster when using dlpack
    prox_in = y_hat + lambda_hat
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
