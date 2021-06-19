import jax
import jax.numpy as jnp
from scico.metric import snr


@jax.jit
def __L_fwd(pq):

    p_diff = jnp.pad(pq[0, 1:, :] - pq[0, :-1, :], ((1, 0), (0, 0)))
    q_diff = jnp.pad(pq[1, :, 1:] - pq[1, :, :-1], ((0, 0), (1, 0)))

    return p_diff + q_diff


@jax.jit
def __L_tran(x):

    return jnp.stack([
        jnp.pad(x[:-1, :] - x[1:, :], ((0, 1), (0, 0))), jnp.pad(x[:, :-1] - x[:, 1:], ((0, 0), (0, 1)))
        ], axis=0)


@jax.jit
def __PC(x, C):

    x = jnp.maximum(x, C[0])
    x = jnp.minimum(x, C[1])

    return x


@jax.jit
def __PP(x):

    denominator = jnp.sqrt(x[0, :, :] ** 2 + x[1, :, :] ** 2)
    denominator = jnp.maximum(denominator, 1)

    return x / denominator


@jax.jit
def __iter_body_fn(i, val):
    pq, rs, t, x, lambda_, C = val

    pq_i = __PP(rs + (1 / (8 * lambda_)) * __L_tran(__PC(x - lambda_ * __L_fwd(rs), C)))
    t_i = (1 + jnp.sqrt(1 + 4 * (t ** 2))) / 2

    rs = pq_i + (t - 1) / t_i * (pq_i - pq)

    t = t_i
    pq = pq_i

    return pq, rs, t, x, lambda_, C


def TotalVariation_Proximal(
        x: jax.abstract_arrays,
        lambda_: float,
        max_iter: int = 100,
        C: tuple = (-float('inf'), float('inf'))
) -> jax.abstract_arrays:

    """
    solve the proximal of the isotropic Total Variation with the form of

    f(x) = \arg\min_z \frac{1}{2}||x - z||^2_2 + \lambda * TV_I(x) ,   (1)

    where:

        x\in\mathbb{R}^{m\times n}, and
        TV_I(x) = \sum_i^m \sum_j^n \sqrt{(x_{i,j}-x_{i+1,j})^2+(x_{i,j}-x_{i,j+1})^2}

    reference: https://ieeexplore.ieee.org/document/5173518

    Args:
        x: input of the algorithm, corresponding to x in (1)
        lambda_: trade-off parameter of the TV regularization, corresponding to \lambda in (1)
        max_iter (optional): maximum number of iteration
        C (optional): pre-defined feasible set

    Returns:
        x_hat: restored image
    """
    m, n = x.shape

    pq_init, rs_init, t_init = jnp.zeros(shape=[2, m, n]), jnp.zeros(shape=[2, m, n]), jnp.array(1, float)

    pq_final, rs_final, t_final, x, lambda_, C = \
        jax.lax.fori_loop(1, max_iter, body_fun=__iter_body_fn, init_val=(pq_init, rs_init, t_init, x, lambda_, C))

    x_hat = __PC(x - lambda_ * __L_fwd(pq_final), C)

    return x_hat


if __name__ == '__main__':

    from skimage.data import camera
    import numpy as np
    import matplotlib.pyplot as plt

    x_true = camera()
    x_true = x_true / 255

    y = x_true + np.random.randn(512, 512) * 20 / 255

    plt.subplot(1, 3, 1)
    plt.imshow(x_true, cmap='gray')
    plt.axis('off')
    plt.title('x_true')

    plt.subplot(1, 3, 2)
    plt.imshow(y, cmap='gray')
    plt.axis('off')
    plt.title('y')

    x_est = TotalVariation_Proximal(jnp.array(y), 0.075)
    plt.subplot(1, 3, 3)
    plt.imshow(x_est, cmap='gray')
    plt.axis('off')
    plt.title('x_est SNR: %.5f' % snr(jnp.array(x_true), x_est))

    plt.show()
