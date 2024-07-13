# -*- coding: utf-8 -*-
# @Time : 2023/4/9 20:04
# @Author : Yao Guoliang
import torch
import torch.nn as nn
import math
from models.ACmixModel.SEWeightModule import SEWeightModule
from models.ACmix import ACmix

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

def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1, groups=1):
    """standard convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, groups=groups, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class PSAModule(nn.Module):

    def __init__(self, inplans, planes, conv_kernels=[3, 5, 7, 9], stride=1, conv_groups=[1, 4, 8, 16]):
        super(PSAModule, self).__init__()
        self.conv_1 = conv(inplans, planes//4, kernel_size=conv_kernels[0], padding=conv_kernels[0]//2,
                            stride=stride, groups=conv_groups[0])
        self.conv_2 = conv(inplans, planes//4, kernel_size=conv_kernels[1], padding=conv_kernels[1]//2,
                            stride=stride, groups=conv_groups[1])
        self.conv_3 = conv(inplans, planes//4, kernel_size=conv_kernels[2], padding=conv_kernels[2]//2,
                            stride=stride, groups=conv_groups[2])
        self.conv_4 = conv(inplans, planes//4, kernel_size=conv_kernels[3], padding=conv_kernels[3]//2,
                            stride=stride, groups=conv_groups[3])
        self.se = SEWeightModule(planes // 4)
        self.split_channel = planes // 4
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)
        x3 = self.conv_3(x)
        x4 = self.conv_4(x)

        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])

        x1_se = self.se(x1)
        x2_se = self.se(x2)
        x3_se = self.se(x3)
        x4_se = self.se(x4)

        x_se = torch.cat((x1_se, x2_se, x3_se, x4_se), dim=1)
        attention_vectors = x_se.view(batch_size, 4, self.split_channel, 1, 1)
        attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats * attention_vectors
        for i in range(4):
            x_se_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_se_weight_fp
            else:
                out = torch.cat((x_se_weight_fp, out), 1)

        return out


class Block(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=True, norm_layer=None, conv_kernels=[3, 5, 7, 9],
                 conv_groups=[1, 4, 8, 16]):
        super(Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.InstanceNorm2d
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.layer1 = nn.Sequential(
            conv1x1(inplanes, planes),
            norm_layer(planes),
            # PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups),
            nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1),
            norm_layer(planes),
            # conv1x1(planes, planes * self.expansion),
            # norm_layer(planes * self.expansion),
            nn.LeakyReLU()
        )

        self.layer2 = nn.Sequential(
            # conv1x1(planes * self.expansion, planes),
            # norm_layer(planes),
            PSAModule(planes, planes, stride=stride, conv_kernels=conv_kernels, conv_groups=conv_groups),
            norm_layer(planes),
            # conv1x1(planes, planes),
            # norm_layer(planes)
        )

        self.relu = nn.LeakyReLU()

        self.downsample = downsample
        if self.downsample:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride = 1, bias=False),
                norm_layer(planes)
            )
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.layer1(x)
        out = self.layer2(out)

        if self.downsample:
            identity = self.shortcut(x)

        out += identity
        out = self.relu(out)
        return out

class downSample(nn.Module):
    def __init__(self):
        super(downSample, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2)

    def forward(self,x):
        return self.max_pool(x)

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

class ACmix_module(nn.Module):
    def __init__(self, inplanes, planes, k_att=7, head=4, k_conv=3, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        super(ACmix_module, self).__init__()
        self.acmixBlock = nn.Sequential(
            conv1x1(inplanes, 128),

            ACmix(128, 128, k_att, head, k_conv, stride, dilation),
            norm_layer(128),
            nn.LeakyReLU(),
            ACmix(128, 128, k_att, head, k_conv, stride, dilation),
            norm_layer(128),
            nn.LeakyReLU(),
            ACmix(128, 128, k_att, head, k_conv, stride, dilation),
            norm_layer(128),
            nn.LeakyReLU(),

            conv1x1(128, planes),
            norm_layer(planes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return self.acmixBlock(x)


class EPSA_ACmixUnet(nn.Module):
    def __init__(self):
        super(EPSA_ACmixUnet, self).__init__()
        self.conv = nn.Conv2d(3, 64, 1)

        self.c1 = Block(64, 64)
        self.d1 = downSample()
        self.c2 = Block(64, 128)
        self.d2 = downSample()
        self.c3 = Block(128, 256)
        self.d3 = downSample()
        self.c4 = Block(256, 512)
        # self.d4 = downSample()
        # self.c5 = Block(512, 1024)
        #
        # self.u1 = upSample(1024, 512)
        # self.uc1 = Block(1024, 512)
        self.u2 = upSample(512, 256)
        self.uc2 = Block(512, 256)
        self.u3 = upSample(256, 128)
        self.uc3 = Block(256, 128)
        self.u4 = upSample(128, 64)
        self.uc4 = Block(128, 64)

        self.res = nn.Sequential(
            nn.Conv2d(64, 3, kernel_size=3, padding=1),
            nn.InstanceNorm2d(3),
            nn.LeakyReLU()
        )

        # self.acmix = ACmix_module(3, 64)

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
        # R5 = self.c5(self.d4(R4))

        # 1024 * 16 * 16 -> 512 * 32 * 32 -> 1024 * 32 * 32 - > 512 * 32 * 32
        # O1 = self.uc1(self.u1(R5, R4))

        # 256 * 64 * 64
        O2 = self.uc2(self.u2(R4, R3))
        # 128 * 128 * 128
        O3 = self.uc3(self.u3(O2, R2))
        # 64 * 256 * 256
        O4 = self.uc4(self.u4(O3, R1))

        # ans = self.acmix(x)
        # ans = torch.cat((O4, ans), dim=1)
        # 3 * 256 # 256
        result = self.res(O4)


        return result


import time
if __name__ == '__main__':
    pass
    # model = models.resnet50().cuda()
    model = EPSA_ACmixUnet()
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

    # start = time.time()
    # for i in range(100):
    #     print(i)
    #     input = torch.randn([1, 64, 224, 224]).cuda()
    #     res = model(input)
    #     print(res.shape)
    # end = time.time()
    # print(end - start)
    #
    # print(model(input).shape)