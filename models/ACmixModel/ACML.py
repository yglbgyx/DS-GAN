# -*- coding: utf-8 -*-
# @Time : 2023/4/21 17:20
# @Author : Yao Guoliang
import torch
import torch.nn as nn
from models.ACmix import ACmix

class Bottleneck(nn.Module):
    def __init__(self,  inplanes, planes, k_att=7, head=4, k_conv=3, stride=1, dilation=1, norm_layer=nn.InstanceNorm2d):
        super(Bottleneck, self).__init__()

        self.shortcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, 1, bias=False),
            norm_layer(planes),
            nn.LeakyReLU()
        )

        self.model = nn.Sequential(
            ACmix(inplanes, inplanes, k_att, head, k_conv, stride, dilation),
            norm_layer(inplanes),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inplanes, planes, 3),
            norm_layer(planes),
            nn.LeakyReLU()
        )

    def forward(self, x):
        out = self.model(x)
        return out + self.shortcut(x)

class CA(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(CA, self).__init__()
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

class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = self.conv1(max_out)
        return self.sigmoid(out)

class Attent(nn.Module):
    def __init__(self, channel, ratio=8, kernel_size=7):
        super(Attent, self).__init__()
        self.channelattention = CA(channel, ratio=ratio)
        self.spatialattention = SA(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)
        x = x * self.spatialattention(x)
        return x

# class DecoderBlock(nn.Module):
#     def __init__(self, inplanes, planes, norm_layer=nn.InstanceNorm2d):
#         super(DecoderBlock, self).__init__()
#
#         self.shorcut = nn.Sequential(
#             nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
#             norm_layer(planes),
#             nn.LeakyReLU()
#         )
#
#         self.model = nn.Sequential(
#             nn.ReflectionPad2d(3),
#             nn.Conv2d(inplanes, inplanes, kernel_size=7),
#             norm_layer(planes),
#             nn.Conv2d(inplanes, planes, kernel_size=1),
#             nn.LeakyReLU()
#         )
#
#         self.attn = Attent(planes)
#
#     def forward(self, x):
#         out = self.model(x)
#         out = self.attn(out)
#         return out + self.shorcut(x)

class DecoderBlock(nn.Module):
    def __init__(self,  inplanes, planes, norm_layer=nn.InstanceNorm2d):
        super(DecoderBlock, self).__init__()
        self.shorcut = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, bias=False),
            norm_layer(planes),
            nn.LeakyReLU()
        )

        self.model = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(inplanes, inplanes, 3),
            norm_layer(inplanes),
            nn.LeakyReLU(),
            nn.ReflectionPad2d(1),
            nn.Conv2d(inplanes, planes, 3),
            norm_layer(planes),
            nn.LeakyReLU()
        )
        self.attn = Attent(planes)

    def forward(self, x):
        out = self.model(x)
        out = self.attn(out)
        return out + self.shorcut(x)

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
            self.attention = Attent(out_channel)

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

class MLKA(nn.Module):
    def __init__(self, dim):
        super(MLKA, self).__init__()

        self.norm = nn.InstanceNorm2d(4 * dim)
        self.dim = dim
        self.activate = nn.LeakyReLU()
        # Multiscale Large Kernel Attention
        self.LKA9 = self._make_layer(dim, 9, 5)
        self.LKA7 = self._make_layer(dim, 7, 4)
        self.LKA5 = self._make_layer(dim, 5, 3)
        self.LKA3 = self._make_layer(dim, 3, 2)

        self.X3 = nn.Conv2d(dim, dim, 3, 1, 1)
        self.X5 = nn.Conv2d(dim, dim, 5, 1, 5 // 2)
        self.X7 = nn.Conv2d(dim, dim, 7, 1, 7 // 2)
        self.X9 = nn.Conv2d(dim, dim, 9, 1, 9 // 2)

        self.shortcut = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1, bias=False),
            nn.InstanceNorm2d(4 * dim),
        )

        self.conv = nn.Conv2d(dim * 4, dim, kernel_size=3, stride=1, padding=1)

    def _make_layer(self, dim, ks, scaling):
        layer = [
            nn.Conv2d(dim, dim, kernel_size=ks, stride=1, padding=(ks // 2), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(ks + 2), stride=1, padding=((ks + 2) // 2) * scaling, groups=dim, dilation=scaling),
            nn.Conv2d(dim, dim, 1)
        ]
        return nn.Sequential(*layer)

    def forward(self, x):
        out = torch.cat((self.LKA3(x) * self.X3(x), self.LKA5(x) * self.X5(x), self.LKA7(x) * self.X7(x), self.LKA9(x) * self.X9(x)),dim=1)
        out = self.norm(out)
        out = out + self.shortcut(x)
        out = self.activate(out)

        out = self.conv(out)
        return out

class downSkip(nn.Module):
    def __init__(self):
        super(downSkip, self).__init__()
        # 64 - > 128
        self.to2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU()
        )
        # 64 - > 256
        self.to4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU()
        )
        # 64 - > 512
        self.to8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU()
        )
        # 64 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16),
            nn.Conv2d(64, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.LeakyReLU()
        )

    def forward(self, x):
        return [self.to2(x), self.to4(x), self.to8(x), self.to16(x)]


class ACML(nn.Module):
    def __init__(self, isAttention = False, norm_layer=nn.InstanceNorm2d):
        super(ACML, self).__init__()
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
        self.uc1 = DecoderBlock(1024, 512)
        self.u2 = upSample(512, 256)
        self.uc2 = DecoderBlock(512, 256)
        self.u3 = upSample(256, 128)
        self.uc3 = DecoderBlock(256, 128)
        self.u4 = upSample(128, 64)
        self.uc4 = DecoderBlock(128, 64)

        self.down = downSkip()

        self.local = MLKA(64)
        self.res = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 64 * 256 * 256
        out = self.conv(x)

        # 128 * 128 * 128 , 256 * 64 * 64, 512 * 32 * 32, 1024 * 16 * 16
        downto =self.down(out)

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
        O1 = self.uc1(self.u1(R5 + downto[3], R4))
        # 256 * 64 * 64
        O2 = self.uc2(self.u2(O1 + downto[2], R3))
        # 128 * 128 * 128
        O3 = self.uc3(self.u3(O2 + downto[1], R2))
        # 64 * 256 * 256
        O4 = self.uc4(self.u4(O3 + downto[0], R1))

        loc = self.local(out)
        O4 = O4 + loc
        # 3 * 256 # 256
        result = self.res(O4)

        return result

import time
if __name__ == '__main__':
    # block = Bottleneck(64, 128)
    # print(block(img).shape)
    model = ACML()
    # model = EPSA_ACmixUnet()
    total_params = sum(p.numel() for p in model.parameters())
    print(f'{total_params:,} total parameters.')
    total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'{total_trainable_params:,} training parameters.')
    start = time.time()
    for i in range(10):
        input = torch.randn([2, 3, 256, 256])
        print(model(input).shape)
    end = time.time()
    print(end - start)

    # torchsummary.summary(model, (1, 256, 256))
