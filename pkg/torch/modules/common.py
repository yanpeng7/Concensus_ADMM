import torch
import torch.nn as nn
from .involution_cuda import involution

activation_fn = {
    'relu': lambda: nn.ReLU(inplace=True),
    'lrelu': lambda: nn.LeakyReLU(inplace=True),
    'prelu': lambda: nn.PReLU()
}


class ConvBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, times=1, is_bn=False, activation='relu', kernel_size=3):
        super().__init__()

        if dimension == 3:
            conv_fn = lambda in_c: torch.nn.Conv3d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda in_c: torch.nn.Conv2d(in_channels=in_c,
                                                   out_channels=out_channels,
                                                   kernel_size=kernel_size,
                                                   padding=kernel_size // 2
                                                   )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        layers = []
        for i in range(times):
            if i == 0:
                layers.append(conv_fn(in_channels))
            else:
                layers.append(conv_fn(out_channels))

            if is_bn:
                layers.append(bn_fn())

            if activation is not None:
                layers.append(activation_fn[activation]())

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class ConvtranBnActivation(nn.Module):
    def __init__(self, in_channels, out_channels, dimension=2, is_bn=False, activation='relu', kernel_size=3):
        self.is_bn = is_bn
        super().__init__()
        if dimension == 3:
            conv_fn = lambda: torch.nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=(1, 2, 2),
                padding=kernel_size // 2,
                output_padding=(0, 1, 1)
            )
            bn_fn = lambda: torch.nn.BatchNorm3d(out_channels)

        elif dimension == 2:
            conv_fn = lambda: torch.nn.ConvTranspose2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=2,
                padding=kernel_size // 2,
                output_padding=1
            )
            bn_fn = lambda: torch.nn.BatchNorm2d(out_channels)
        else:
            raise ValueError()

        self.net1 = conv_fn()
        if self.is_bn:
            self.net2 = bn_fn()
        self.net3 = activation_fn[activation]()

    def forward(self, x):
        ret = self.net1(x)
        if self.is_bn:
            ret = self.net2(ret)

        ret = self.net3(ret)

        return ret


class UNet(nn.Module):
    def __init__(self, dimension, i_nc=1, o_nc=1, f_root=32, conv_times=3, is_bn=False, activation='relu',
                 is_residual=False, up_down_times=3):

        self.is_residual = is_residual
        self.up_down_time = up_down_times

        super().__init__()

        if dimension == 2:
            self.down_sample = nn.MaxPool2d(2)
        elif dimension == 3:
            self.down_sample = nn.MaxPool3d((1, 2, 2))
        else:
            raise ValueError()

        self.conv_in = ConvBnActivation(
            in_channels=i_nc,
            out_channels=f_root,
            is_bn=is_bn,
            activation=activation,
            dimension=dimension)

        self.conv_out = ConvBnActivation(
            in_channels=f_root,
            out_channels=o_nc,
            kernel_size=1,
            dimension=dimension,
            times=1,
            is_bn=False,
            activation=None
        )

        self.bottom = ConvBnActivation(
            in_channels=f_root * (2 ** (up_down_times - 1)),
            out_channels=f_root * (2 ** up_down_times),
            times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension)

        self.down_list = nn.ModuleList([
                                           ConvBnActivation(
                                               in_channels=f_root * 1,
                                               out_channels=f_root * 1,
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension)
                                       ] + [
                                           ConvBnActivation(
                                               in_channels=f_root * (2 ** i),
                                               out_channels=f_root * (2 ** (i + 1)),
                                               times=conv_times, is_bn=is_bn, activation=activation,
                                               dimension=dimension)
                                           for i in range(up_down_times - 1)
                                       ])

        self.up_conv_list = nn.ModuleList([
            ConvBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                times=conv_times, is_bn=is_bn, activation=activation, dimension=dimension)
            for i in range(up_down_times)
        ])

        self.up_conv_tran_list = nn.ModuleList([
            ConvtranBnActivation(
                in_channels=f_root * (2 ** (up_down_times - i)),
                out_channels=f_root * (2 ** (up_down_times - i - 1)),
                is_bn=is_bn, activation=activation, dimension=dimension)
            for i in range(up_down_times)
        ])

    def forward(self, x):

        input_ = x

        x = self.conv_in(x)

        skip_layers = []
        for i in range(self.up_down_time):
            x = self.down_list[i](x)

            skip_layers.append(x)
            x = self.down_sample(x)

        x = self.bottom(x)

        for i in range(self.up_down_time):
            x = self.up_conv_tran_list[i](x)
            x = torch.cat([x, skip_layers[self.up_down_time - i - 1]], 1)
            x = self.up_conv_list[i](x)

        x = self.conv_out(x)

        ret = input_ - x if self.is_residual else x

        return ret


class DnCNN(nn.Module):
    def __init__(self, dimension, depth=13, n_channels=64, i_nc=1, o_nc=1, kernel_size=3, is_bn=False,
                 is_residual=False):
        self.is_bn = is_bn
        self.is_residual = is_residual

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        else:
            raise ValueError()

        super().__init__()
        padding = kernel_size // 2

        layers = [conv_fn(
            in_channels=i_nc, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)]

        for _ in range(depth - 2):
            if self.is_bn:
                layers.append(conv_fn(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                    bias=False))
                layers.append(bn_fn(n_channels, eps=0.0001, momentum=0.95))
            else:
                layers.append(conv_fn(
                    in_channels=n_channels, out_channels=n_channels, kernel_size=kernel_size, padding=padding,
                    bias=True))

            layers.append(nn.ReLU(inplace=True))

        layers.append(
            conv_fn(in_channels=n_channels, out_channels=o_nc, kernel_size=kernel_size, padding=padding, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.net(x)

        ret = y - out if self.is_residual else out
        return ret


class InDnCNN(nn.Module):
    def __init__(self, depth=13, n_channels=64, i_nc=1, o_nc=1, kernel_size=3, is_residual=False):
        self.is_residual = is_residual

        conv_fn = nn.Conv2d

        super().__init__()
        padding = kernel_size // 2

        layers = [conv_fn(
            in_channels=i_nc, out_channels=n_channels, kernel_size=kernel_size, padding=padding, bias=True),
            nn.ReLU(inplace=True)]

        for _ in range(depth - 2):
            layers.append(involution(
                channels=n_channels,
                kernel_size=3,
                stride=1
            ))
            layers.append(nn.ReLU(inplace=True))

        layers.append(
            conv_fn(in_channels=n_channels, out_channels=o_nc, kernel_size=kernel_size, padding=padding, bias=False))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        y = x
        out = self.net(x)

        ret = y - out if self.is_residual else out
        return ret


class ResBlock(nn.Module):
    def __init__(self, dimension, n_feats, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_scale=1):

        if dimension == 2:
            conv_fn = nn.Conv2d
            bn_fn = nn.BatchNorm2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
            bn_fn = nn.BatchNorm3d
        else:
            raise ValueError()

        super(ResBlock, self).__init__()
        m = []
        for i in range(2):
            m.append(conv_fn(n_feats, n_feats, kernel_size, padding=kernel_size // 2, bias=bias))
            if bn:
                m.append(bn_fn(n_feats))
            if i == 0:
                m.append(act)

        self.body = nn.Sequential(*m)
        self.res_scale = res_scale

    def forward(self, x):
        res = self.body(x).mul(self.res_scale)
        res += x

        return res


class EDSR(nn.Module):
    def __init__(self, dimension, n_resblocks, n_feats, res_scale, in_channels=1, out_channels=1, act='relu'):
        super().__init__()

        if dimension == 2:
            conv_fn = nn.Conv2d
        elif dimension == 3:
            conv_fn = nn.Conv3d
        else:
            raise ValueError()

        m_head = [conv_fn(in_channels, n_feats, 3, padding=3 // 2)]

        m_body = [
            ResBlock(
                dimension, n_feats, 3, res_scale=res_scale, act=activation_fn[act](),
            ) for _ in range(n_resblocks)
        ]

        m_body.append(conv_fn(n_feats, n_feats, 3, padding=3 // 2))

        m_tail = [
            conv_fn(n_feats, out_channels, 3, padding=3 // 2)
        ]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        x = self.head(x)

        res = self.body(x)
        res += x

        x = self.tail(res)

        return x
