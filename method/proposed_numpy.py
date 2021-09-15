import numpy as np
import cv2
import astra
from scico import metric
import JAX_TV
import matplotlib.pyplot as plt
from numpy_impl.numpy_B_impl import OperatorB
from numpy_impl.numpy_A_impl import OperatorA
import scipy.io as sio
import time
import jax.numpy as jnp
from pkg.torch.callback import Tensorboard
import jsonargparse.core as jc
import pprint
from skimage.data import shepp_logan_phantom
import os

def run(config):
    num_iter = int(config.method.proposed_numpy.num_iter)
    gamma = float(config.method.proposed_numpy.gamma)
    rho = float(config.method.proposed_numpy.rho)
    tau = float(config.method.proposed_numpy.tau)
    lambda_TV = float(config.method.proposed_numpy.lambda_TV)
    num_iter_TV = int(config.method.proposed_numpy.num_iter_TV)
    N = int(config.dataset.ct_sample.img_dim)
    num_projection = int(config.dataset.ct_sample.num_projection)
    num_detector = int(config.dataset.ct_sample.num_detector)
    tb_writer = Tensorboard(file_path=config.setting.root_path + config.setting.proj_name + '/')

    shift_list = np.asarray([[0, 0.25], [0, 0.75], [0.25, 0], [0.75, 0]])
    # shift_list = np.asarray([[0, 0.25], [0, 0.75]])
    # shift_list = np.asarray([[1, 1]])

    A = OperatorA(input_size=N, num_detector=num_detector, num_angles=num_projection, init_point_list=shift_list)
    B = OperatorB(init_point_list=shift_list)

    # load image
    image_number = 7
    matContent = sio.loadmat('dataset/CT_images_preprocessed.mat', squeeze_me=True)
    ct = matContent['img_cropped'][:, :, image_number]

    x_star = cv2.resize(ct, (N, N))

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
    y_hat = B.fmult(x_hat)
    lambda_hat = np.zeros_like(y_hat)

    best_SNR = 0
    best_iter = 0

    snr_list = []

    # main algorithm
    print("algorithm started!")
    tb_writer.manual_add(global_step=0, text={'config': pprint.pformat(jc.namespace_to_dict(config))})
    start_time = time.time()

    for i in range(num_iter):
        # rho is optional
        y_hat = y_hat - gamma * A.adj_v2(A.fmult_v2(y_hat) - d) - gamma * rho * (y_hat - B.fmult(x_hat) + lambda_hat)
        prox_in = x_hat - tau * B.adj(B.fmult(x_hat) - y_hat - lambda_hat)
        x_hat = np.array(JAX_TV.TotalVariation_Proximal(jnp.array(prox_in), lambda_TV / rho * tau, num_iter_TV))
        lambda_hat = lambda_hat + y_hat - B.fmult(x_hat)
        x_hat_masked = x_hat.copy()
        # x_hat_masked[background_mask == 0] = 0
        snr = metric.snr(x_star, x_hat_masked)
        if snr > best_SNR:
            best_SNR = snr
            best_iter = i
        print(f'iteration {i} SNR: {snr}')
        snr_list.append(snr)

    running_time = time.time() - start_time
    print("--- %s seconds ---" % running_time)
    x_hat_masked = x_hat.copy()
    # x_hat_masked[background_mask == 0] = 0
    final_SNR = metric.snr(x_star, x_hat_masked)
    print(f'final SNR: {final_SNR}')
    print(f'best SNR: {best_SNR}')
    print(f'best SNR iteration: {best_iter + 1}')

    np.save(os.path.join('saved', f'proposed_1_1_{image_number}_output'), x_hat)
    np.save(os.path.join('saved', f'proposed_1_1_{image_number}'), snr_list)

    # visualize results
    fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
    im1 = axes[0].imshow(x_init, cmap="gray")
    im2 = axes[1].imshow(x_hat, cmap="gray")
    im3 = axes[2].imshow(x_star, cmap="gray")
    x_init_masked = x_init
    # x_init_masked[background_mask == 0] = 0
    axes[0].title.set_text(f'input SNR: {"{:.2f}".format(metric.snr(x_star, x_init_masked))}')
    axes[1].title.set_text(f'output SNR: {"{:.2f}".format(final_SNR)}')
    axes[2].title.set_text('ground truth')
    fig.colorbar(im1, ax=axes[0])
    fig.colorbar(im2, ax=axes[1])
    fig.colorbar(im3, ax=axes[2])
    plt.show()

    # difference map
    diff_map = x_star - x_hat
    plt.imshow(np.absolute(diff_map), cmap='gray')
    plt.title('difference map')
    plt.colorbar()
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
            "metrics/final_SNR": np.asarray(final_SNR),
            "metrics/best_SNR": np.asarray(best_SNR)
        }
    })
