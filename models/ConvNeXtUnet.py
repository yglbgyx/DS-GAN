# -*- coding: utf-8 -*-
# @Time : 2023/4/8 15:16
# @Author : Yao Guoliang

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
from torch.autograd import Variable
from models.init_weights import init_weights

from init_weights import init_weights
from models.FTNet import get_segmentation_model

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 利用1x1卷积代替全连接
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.PReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class cbam_block(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(cbam_block, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# ConvNeXt_Unet_GAN

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., layer_scale_init_value=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        # self.norm = LayerNorm(dim, eps=1e-6)
        self.norm = nn.InstanceNorm2d(dim)
        # self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1, stride=1, bias=False)
        # self.act = nn.GELU()
        self.act = nn.LeakyReLU()
        # self.pwconv2 = nn.Linear(4 * dim, dim)
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1, stride=1, bias=False)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = self.norm(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        # x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class upSample(nn.Module):
    def __init__(self, in_channel, out_channel, isAttention = False, norm_layer = nn.InstanceNorm2d):
        super(upSample, self).__init__()
        activatoin = nn.LeakyReLU()
        # (H - 1) * S + K - 2P
        model = [nn.ConvTranspose2d(in_channel, out_channel, kernel_size=3, stride=2, padding=1, output_padding=1),
                 norm_layer(out_channel), activatoin]
        self.model = nn.Sequential(*model)

        # 注意力
        self.isAttention = isAttention
        if self.isAttention:
            self.attention = cbam_block(out_channel)

    def forward(self, x, feature_map):
        out = self.model(x)
        if self.isAttention:
            out = self.attention(out)
        # 加上前一层map
        return torch.cat((out, feature_map), dim=1)

class downSample(nn.Module):
    def __init__(self):
        super(downSample, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        return self.max_pool(x)

class Bottleneck(nn.Module):

    def __init__(self, in_channel, out_channel, norm_layer=nn.InstanceNorm2d):
        super(Bottleneck, self).__init__()
        #
        self.encoderBlock = nn.Sequential(
            Block(in_channel),
            Block(in_channel)
        )
        self.changeChannel = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False),
            LayerNorm(out_channel, eps=1e-6, data_format="channels_first"),
            nn.LeakyReLU()
        )

    def forward(self, x):
        res = self.encoderBlock(x)
        res = self.changeChannel(res)
        return res


class ConvNeXtUnet(nn.Module):
    def __init__(self):
        super(ConvNeXtUnet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 1)

        self.c1 = Bottleneck(64, 64)
        self.d1 = downSample()
        self.c2 = Bottleneck(64, 128)
        self.d2 = downSample()
        self.c3 = Bottleneck(128, 256)
        self.d3 = downSample()
        self.c4 = Bottleneck(256, 512)
        self.d4 = downSample()
        self.c5 = Bottleneck(512, 1024)

        self.u1 = upSample(1024, 512)
        self.uc1 = Bottleneck(1024, 512)
        self.u2 = upSample(512, 256)
        self.uc2 = Bottleneck(512, 256)
        self.u3 = upSample(256, 128)
        self.uc3 = Bottleneck(256, 128)
        self.u4 = upSample(128, 64)
        self.uc4 = Bottleneck(128, 64)

        self.convto3 = nn.Conv2d(64, 3, kernel_size=3, padding=1)



    def forward(self, x):
        # 64 * 256 * 256
        out = self.conv(x)

        # 64 * 256 * 256
        R1 = self.c1(out)
        # 128 * 128 * 128
        R2 = self.c2(self.d1(R1))
        # 256 * 64 * 64
        R3 = self.c3(self.d2(R2))
        # 512 * 32 * 32
        R4 = self.c4(self.d3(R3))
        # 1024 * 16 * 16
        R5 = self.c5(self.d4(R4))

        # 1024 * 16 * 16 -> 512 * 32 * 32 -> 1024 * 32 * 32 - > 512 * 32 * 32
        O1 = self.uc1(self.u1(R5, R4))
        # 256 * 64 * 64
        O2 = self.uc2(self.u2(O1, R3))
        # 128 * 128 * 128
        O3 = self.uc3(self.u3(O2, R2))
        # 64 * 256 * 256
        O4 = self.uc4(self.u4(O3, R1))

        # 3 * 256 # 256
        result = self.convto3(O4)

        return result

import time
if __name__ == '__main__':
    pass
    # model = models.resnet50().cuda()
    model = ConvNeXtUnet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    start = time.time()
    for i in range(1):
        input = torch.randn([2, 3, 256, 256])
        print(model(input).shape)
    end = time.time()
    print(end - start)