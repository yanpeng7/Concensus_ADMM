import numpy as np
import cv2
import astra
from scico import metric
import JAX_TV
import matplotlib.pyplot as plt
from numpy_impl.old_B_impl import B_numpy, B_T_numpy
from numpy_impl.old_A_impl import A_batched, A_adj_batched
import scipy.io as sio
import time
import jax.numpy as jnp

if __name__ == '__main__':
    num_iter = 100
    gamma = 1e-5
    rho = 1e1
    tau = 0.1
    lambda_TV = 0.1
    num_iter_TV = 50
    N = 256
    num_projection = 180  # need to set this in other places too, will fix later
    num_detector = 256  # need to set this in other places too, will fix later
    # shift_list = np.asarray([[0, 0.25], [0, 0.75], [0.25, 0], [0.75, 0]])
    # shift_list = np.asarray([[0.10, 0], [0.05, 0], [0, 0.10], [0, 0.05]])
    shift_list = np.asarray([[0, 0]])

    # load image
    matContent = sio.loadmat('../dataset/CT_images_preprocessed.mat', squeeze_me=True)
    ct = matContent['img_cropped'][:, :, 0]
    plt.imshow(ct)
    plt.title('0')
    plt.show()
    quit()

    # ct = np.pad(ct, 50)
    # plt.imshow(ct)
    # plt.show()

    x_star = cv2.resize(ct, (N, N))

    # y = B_numpy(x_star, shift_list)[0]
    # print(y.shape)
    # plt.imshow(y)
    # plt.title('max shift y')
    # plt.show()
    # quit()


    # generate d
    proj_geom = astra.create_proj_geom('parallel', 1.0, num_detector, np.linspace(0, np.pi, num_projection))
    vol_geom = astra.create_vol_geom(N, N)
    projector = astra.create_projector("line", proj_geom, vol_geom)
    _, d = astra.create_sino(x_star, projector)

    # initialize x as fbp of d
    rec_id = astra.data2d.create("-vol", vol_geom)
    sino_id = astra.data2d.create("-sino", proj_geom, d)
    cfg = astra.astra_dict("FBP")
    cfg["ReconstructionDataId"] = rec_id
    cfg["ProjectorId"] = projector
    cfg["ProjectionDataId"] = sino_id
    cfg["option"] = {"FilterType": 'Ram-Lak'}
    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)
    x_init = astra.data2d.get(rec_id)
    astra.algorithm.delete(alg_id)
    astra.data2d.delete(rec_id)

    x_hat = x_init
    y_hat = B_numpy(x_hat, shift_list)
    lambda_hat = np.zeros_like(y_hat)

    # main algorithm
    print("algorithm started!")
    start_time = time.time()
    for i in range(num_iter):
        y_hat = y_hat - gamma * A_adj_batched(A_batched(y_hat, shift_list) - d, shift_list)\
                   - gamma * rho * (y_hat - B_numpy(x_hat, shift_list) + lambda_hat)

        prox_in = x_hat - tau * B_T_numpy(B_numpy(x_hat, shift_list) - y_hat - lambda_hat, shift_list)
        x_hat = np.array(JAX_TV.TotalVariation_Proximal(jnp.array(prox_in), lambda_TV / rho * tau, num_iter_TV))
        lambda_hat = lambda_hat + y_hat - B_numpy(x_hat, shift_list)
        print(f'iteration {i} SNR: {metric.snr(x_star, x_hat)}')

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)
    print(f'final SNR: {metric.snr(x_star, x_hat)}')

    # visualize results
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    im1 = axes[0].imshow(x_init, cmap="gray")
    im2 = axes[1].imshow(x_hat, cmap="gray")
    im3 = axes[2].imshow(x_star, cmap="gray")
    axes[0].title.set_text(f'input SNR: {"{:.2f}".format(metric.snr(x_star, x_init))}')
    axes[1].title.set_text(f'output SNR: {"{:.2f}".format(metric.snr(x_star, x_hat))}')
    axes[2].title.set_text('ground truth')
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.show()
