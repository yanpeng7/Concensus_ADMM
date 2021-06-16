import torch
from pkg.model_based.linop import LinearOperator
from pkg.model_based.alg.iterAlg import iter_alg
from pkg.model_based.prox.TV import TVOperator
from skimage.data import camera
import torch
import torch.nn.functional as F
import jax.numpy as jnp
from jax.ops import index, index_add, index_update
import jax
from functools import partial


class ConsensusSR(LinearOperator):
    def __init__(self, K, kernel=None):

        self.K = K
        self.kernel = kernel

    @partial(jax.jit, static_argnums=(0,))
    def fmult(self, x):

        # if self.kernel is not None:
        #     x = F.conv2d(x.unsqueeze(0).unsqueeze(0), self.kernel, padding=1).squeeze(0).squeeze(0)

        ret = []
        for i in range(self.K // 2):
            for j in range(self.K // 2):
                ret.append(x[i::self.K // 2, j::self.K // 2])

        # ret = torch.stack(ret, 0)
        ret = jnp.stack(ret, axis=0)

        return ret

    @partial(jax.jit, static_argnums=(0,))
    def ftran(self, x):

        _, width, height = x.shape
        # ret = torch.zeros(width * self.K // 2, height * self.K // 2)
        ret = jnp.zeros((width * self.K // 2, height * self.K // 2))

        index_x = 0
        for i in range(self.K // 2):
            for j in range(self.K // 2):
                # ret[i::self.K // 2, j::self.K // 2] = x[index_x]
                ret = index_update(ret, index[i::self.K // 2, j::self.K // 2], x[index_x])
                index_x = index_x + 1

        # if self.kernel is not None:
        #     ret = F.conv_transpose2d(ret.unsqueeze(0).unsqueeze(0), self.kernel, padding=1).squeeze(0).squeeze(0)

        return ret

# x_true = torch.from_numpy(camera()).to(torch.float32) / 255
# kernel_true = torch.ones(3, 3).unsqueeze(0).unsqueeze(0) / 9
#
# linop = ConsensusSR(4, kernel_true)
# prox = TVOperator(lambda_=1e-7, num_iter=100)
#
# y = linop.fmult(x_true)
#
# x_pre = alg = iter_alg(linop, prox, y, verbose=True, x_true=x_true)
#
# import matplotlib.pyplot as plt
#
# plt.subplot(1, 2, 1)
# plt.imshow(x_pre, cmap='gray')
# plt.axis('off')
#
# plt.subplot(1, 2, 2)
# plt.imshow(x_true, cmap='gray')
# plt.axis('off')
#
# plt.show()
