import torch
import numpy as np

from ...utility.torch_complex_op import mul_torch_complex, div_torch_complex, abs_torch_complex, conj_torch_complex
from ...ext.torchkbnufft import KbNufft, AdjKbNufft


class MCNUFFT:
    def __init__(self, IM_SIZE=640, GRID_SIZE=960):
        self.IM_SIZE = IM_SIZE
        self.GRID_SIZE = GRID_SIZE

        self.adjkbnufft = AdjKbNufft(im_size=(
            self.IM_SIZE, self.IM_SIZE), grid_size=(self.GRID_SIZE, self.GRID_SIZE)).to(torch.float32)
        self.kbnufft = KbNufft(im_size=(
            self.IM_SIZE, self.IM_SIZE), grid_size=(self.GRID_SIZE, self.GRID_SIZE)).to(torch.float32)

    def ftran(self, y_, b1_, w_, ktraj_):
        n_spoke = (w_.shape[-1] * w_.shape[-2]) / self.IM_SIZE

        y_ = self.torch_flatten_last_two_dimensions(y_)
        w_ = self.torch_flatten_last_two_dimensions(w_)
        ktraj_ = self.torch_flatten_last_two_dimensions(ktraj_)

        res_ = y_ * w_.unsqueeze(1).unsqueeze(1)
        res_ = self.adjkbnufft(res_, ktraj_)

        res_ = res_.permute([0, 3, 4, 1, 2])
        res_ = res_ * np.pi / 2 / n_spoke

        b1_rep = conj_torch_complex(b1_.unsqueeze(0))
        b1_abs = torch.sum(abs_torch_complex(b1_) ** 2, -1).unsqueeze(0)

        res_ = mul_torch_complex(res_, b1_rep)
        res_ = torch.sum(res_, -2)
        res_ = div_torch_complex(res_, b1_abs)

        return res_

    def fmult(self, img_, b1_, ktraj_):
        _, _, length_spoke, num_spoke = ktraj_.shape

        img_ = img_.unsqueeze(-2)
        b1_ = b1_.unsqueeze(0)
        ktraj_ = self.torch_flatten_last_two_dimensions(ktraj_)

        res_ = mul_torch_complex(img_, b1_)
        res_ = res_.permute([0, 3, 4, 1, 2])

        res_ = self.kbnufft(res_, ktraj_) / self.IM_SIZE

        res_ = res_.reshape(list(res_.shape[:-1] + (length_spoke, num_spoke)))
        return res_

    def cuda(self):
        self.adjkbnufft = self.adjkbnufft.cuda()
        self.kbnufft = self.kbnufft.cuda()

    @staticmethod
    def torch_flatten_last_two_dimensions(x: torch.Tensor):
        assert x.dim() >= 3

        new_shape = list(x.shape[:-2] + (-1, ))
        return x.reshape(new_shape)
