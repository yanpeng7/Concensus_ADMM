import numpy as np

import jax

from xdesign import Foam, discrete_phantom

import scico.numpy as snp
from scico import admm, functional, linop, loss, metric, plot
from scico.linop.radon import ParallelBeamProj
import scipy.io as sio
import cv2
import os

"""
Using SCICO ADMM SOLVER
"""

"""
Configure problem size, number of projections, and solver parameters
"""
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

N = 256  # Phantom size
n_projection = 180  # number of projections

matContent = sio.loadmat('dataset/CT_images_preprocessed.mat', squeeze_me=True)
ct = matContent['img_cropped'][:,:,0]
down_ct = cv2.resize(ct,(N,N))

λ = 2e-0  # L1 norm regularization parameter
ρ = 5e-0  # ADMM penalty parameter
maxiter = 50  # Number of ADMM iterations


"""
Create input image, convert to jax array
"""

xin = down_ct

xin = jax.device_put(xin)  # Convert to jax type, push to GPU


"""
Configure CT projection operator and generate synthetic measurements
"""

angles = np.linspace(0, np.pi, n_projection)  # evenly spaced projection angles

A = ParallelBeamProj(xin.shape, 1, N, angles)  # Radon transform operator
y = A @ xin  # Sinogram


"""
Set up ADMM solver object
"""
g_list = [λ * functional.L1Norm()]  # Regularization functionals Fi
C_list = [linop.FiniteDifference(input_shape=xin.shape)]  # Analysis operators Ci
rho_list = [ρ]  # ADMM parameters

f = loss.SquaredL2Loss(y=y, A=A)

x0 = A.fbp(y)

admm_ = admm.ADMMQuadraticLoss(
    f=f,
    g_list=g_list,
    C_list=C_list,
    rho_list=rho_list,
    x_solver_kwargs={"maxiter": 20},
    maxiter=maxiter,
    x0=x0
)

"""
Run the solver
"""
admm_.solve()
hist = admm_.itstat_object.history(transpose=True)
admm_.x = snp.clip(admm_.x, 0, 1)

"""
Show recovered image
"""

fig, ax = plot.subplots(nrows=1, ncols=3, figsize=(15, 6))
plot.imview(xin, title="Ground truth", fig=fig, ax=ax[0])
plot.imview(x0, title="FBP Reconstruction: %.2f (dB)" % metric.psnr(xin, x0), fig=fig, ax=ax[1])
plot.imview(
    admm_.x, title="TV Reconstruction: %.2f (dB)" % metric.psnr(xin, admm_.x), fig=fig, ax=ax[2]
)
fig.show()

"""
Plot convergence statistics
"""

fig, ax = plot.subplots(nrows=1, ncols=2, figsize=(12, 5))
plot.plot(
    hist.Objective,
    title="Objective function",
    xlbl="Iteration",
    ylbl="Functional value",
    fig=fig,
    ax=ax[0],
)
plot.plot(
    snp.vstack((hist.Primal_Rsdl, hist.Dual_Rsdl)).T,
    ptyp="semilogy",
    title="Residuals",
    xlbl="Iteration",
    lgnd=("Primal", "Dual"),
    fig=fig,
    ax=ax[1],
)
fig.show()

quit()

