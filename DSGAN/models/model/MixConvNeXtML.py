
import torch
import torch.nn as nn

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

class upSample(nn.Module):
    def __init__(self, in_channel, out_channel, isAttention = False, norm_layer = nn.InstanceNorm2d):
        super(upSample, self).__init__()
        activatoin = nn.GELU()
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

class MidMLKA(nn.Module):
    def __init__(self, dim):
        super(MidMLKA, self).__init__()

        self.norm = nn.InstanceNorm2d(dim)
        self.dim = dim
        self.activate = nn.GELU()

        # self.conv1 = nn.Conv2d(4 * dim, 4 * dim, kernel_size=1, groups=4 * dim ,bias=False)
        self.conv = nn.Conv2d(dim, dim, kernel_size=1)

        self.attn = CA(dim)

        # self.LKA9 = self._make_layer(dim, 9, 5)
        # self.LKA7 = self._make_layer(dim, 7, 4)
        # self.LKA5 = self._make_layer(dim, 5, 3)
        # self.LKA3 = self._make_layer(dim, 3, 2)

        self.X3 = nn.Conv2d(dim // 4, dim // 4, 3, 1, 1, groups=dim // 4)
        self.X5 = nn.Conv2d(dim // 4, dim // 4, 5, 1, 5 // 2, groups=dim // 4)
        self.X7 = nn.Conv2d(dim // 4, dim // 4, 7, 1, 7 // 2, groups=dim // 4)
        self.X9 = nn.Conv2d(dim // 4, dim // 4, 9, 1, 9 // 2, groups=dim // 4)


    def _make_layer(self, dim, ks, scaling):
        layer = [
            nn.Conv2d(dim, dim, kernel_size=ks, stride=1, padding=(ks // 2), groups=dim),
            nn.Conv2d(dim, dim, kernel_size=(ks + 2), stride=1, padding=((ks + 2) // 2) * scaling, groups=dim, dilation=scaling),
            nn.Conv2d(dim, dim, 1)
        ]
        return nn.Sequential(*layer)


    def forward(self, x):
        x1, x2, x3, x4 = torch.chunk(x, 4, dim=1)
        out = torch.cat((self.X3(x1), self.X5(x2), self.X7(x3), self.X9(x4)), dim=1)
        out = self.conv(out)
        out = out * self.attn(out)
        out = self.norm(out)
        out += x
        out = self.activate(out)
        return out

class OriginMLKA(nn.Module):
    def __init__(self):
        super(OriginMLKA, self).__init__()
        self.to32 = nn.Conv2d(3, 32, kernel_size=1, bias=False)
        self.max_pool1 = downSample()
        self.mid32 = MidMLKA(32)

        self.to64 = nn.Conv2d(32, 64, kernel_size=1, bias=False)
        self.max_pool2 = downSample()
        self.mid64 = MidMLKA(64)

        self.to128 = nn.Conv2d(64, 128, kernel_size=1, bias=False)
        self.max_pool3 = downSample()
        self.mid128 = MidMLKA(128)

        self.to256 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
        self.max_pool4 = downSample()
        self.mid256 = MidMLKA(256)

        self.up1 = upSample(256, 128)
        self.upc1 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=1, bias=False),
            MidMLKA(128)
        )
        self.up2 = upSample(128, 64)
        self.upc2 = MidMLKA(128)

        self.up3 = upSample(128, 64)
        self.upc3 = MidMLKA(128)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64)
        )

        self.activate = nn.GELU()

        self.shortcut = nn.Sequential(
            nn.Conv2d(3, 64, 1, bias=False),
            nn.InstanceNorm2d(64)
        )

    def forward(self, x):
        # 32 * 256 * 256
        d1 = self.to32(x)
        # 32 * 128 * 128
        d2 = self.mid32(self.max_pool1(d1))
        # 64 * 128 * 128
        d3 = self.to64(d2)
        # 64 * 64 * 64
        d4 = self.mid64(self.max_pool2(d3))
        # 128 * 64 * 64
        d5 = self.to128(d4)
        # 128 * 32 * 32
        d6 = self.mid128(self.max_pool3(d5))
        # 256 * 32 * 32
        d7 = self.to256(d6)
        # 256 * 16 * 16
        d8 = self.mid256(self.max_pool4(d7))

        # 128 * 32 * 32 ->  256 * 32 * 32 -> 128 * 32 * 32
        u1 = self.upc1(self.up1(d8, d6))
        # 64 * 64 * 64 -> 128 * 64 * 64 -> 128 * 64 * 64
        u2 = self.upc2(self.up2(u1, d4))
        # 64 * 128 * 128 -> 128 * 128 * 128 -> 128 * 128 * 128
        u3 = self.upc3(self.up3(u2, d3))
        # 64 * 256 * 256
        u4 = self.up4(u3)
        out = u4 + self.shortcut(x)
        out = self.activate(out)
        return out


def down_make_layer(inplans, plans, scale):
    to = [
        nn.MaxPool2d(kernel_size=scale),
        nn.Conv2d(inplans, plans, kernel_size=1, stride=1, bias=False),
        nn.InstanceNorm2d(plans),
        nn.GELU()
    ]
    return nn.Sequential(*to)



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

    def __init__(self, dim, plans, drop_path=0., ldimayer_scale_init_value=1e-6):
        super().__init__()

        self.shortcut = nn.Conv2d(dim, plans, kernel_size=1, bias=False)

        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.InstanceNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, plans)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                          requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()


    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        x = self.shortcut(input) + x
        return x


class midBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """

    def __init__(self, dim, drop_path=0., ldimayer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.InstanceNorm2d(dim)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        # self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)),
        #                          requires_grad=True) if layer_scale_init_value > 0 else None
        # self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        # x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        # if self.gamma is not None:
        #     x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # x = self.atten(x)
        x = input + x
        return x


class MLKA(nn.Module):
    def __init__(self, dim):
        super(MLKA, self).__init__()

        self.norm = nn.InstanceNorm2d(4 * dim)
        self.dim = dim
        self.activate = nn.GELU()
        # Multiscale Large Kernel Attention
        self.LKA9 = self._make_layer(dim, 9, 5)
        self.LKA7 = self._make_layer(dim, 7, 4)
        self.LKA5 = self._make_layer(dim, 5, 3)
        self.LKA3 = self._make_layer(dim, 3, 2)

        self.X3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)
        self.X5 = nn.Conv2d(dim, dim, 5, 1, 5 // 2, groups=dim)
        self.X7 = nn.Conv2d(dim, dim, 7, 1, 7 // 2, groups=dim)
        self.X9 = nn.Conv2d(dim, dim, 9, 1, 9 // 2, groups=dim)

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
            nn.GELU()
        )
        # 64 - > 256
        self.to4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(64, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
        # 64 - > 512
        self.to8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(64, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.GELU()
        )
        # 64 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=16),
            nn.Conv2d(64, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.GELU()
        )

    def forward(self, x):
        return [self.to2(x), self.to4(x), self.to8(x), self.to16(x)]

class downSkip128(nn.Module):
    def __init__(self):
        super(downSkip128, self).__init__()
        # 128 - > 256
        self.to4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(128, 256, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(256),
            nn.GELU()
        )
        # 128 - > 512
        self.to8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(128, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.GELU()
        )
        # 128 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=8),
            nn.Conv2d(128, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.GELU()
        )


    def forward(self, x):
        return [self.to4(x), self.to8(x), self.to16(x)]

class downSkip256(nn.Module):
    def __init__(self):
        super(downSkip256, self).__init__()
        # 256 - > 512
        self.to8 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(256, 512, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(512),
            nn.GELU()
        )
        # 256 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=4),
            nn.Conv2d(256, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.GELU()
        )

    def forward(self, x):
        return [self.to8(x), self.to16(x)]

class downSkip512(nn.Module):
    def __init__(self):
        super(downSkip512, self).__init__()
        # 512 - > 1024
        self.to16 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(512, 1024, kernel_size=1, stride=1, bias=False),
            nn.InstanceNorm2d(1024),
            nn.GELU()
        )


    def forward(self, x):
        return [self.to16(x)]

class MixConvNeXtML(nn.Module):
    def __init__(self):
        super(MixConvNeXtML, self).__init__()
        # self.conv = nn.Conv2d(3, 64, kernel_size=7, padding=3)

        self.c1 = Block(3, 64)
        self.d1 = downSample()
        self.c2 = Block(64, 128)
        self.d2 = downSample()
        self.c3 = Block(128, 256)
        self.d3 = downSample()
        self.c4 = Block(256, 512)
        self.d4 = downSample()
        self.c5 = Block(512, 1024)

        self.u1 = upSample(1024, 512)
        self.uc1 = Block(1024, 512)
        self.u2 = upSample(512, 256)
        self.uc2 = Block(512, 256)
        self.u3 = upSample(256, 128)
        self.uc3 = Block(256, 128)
        self.u4 = upSample(128, 64)
        self.uc4 = Block(128, 64)

        self.down64 = downSkip()
        self.down128 = downSkip128()
        self.down256 = downSkip256()
        self.down512 = downSkip512()

        self.local = OriginMLKA()

        self.res = nn.Conv2d(64, 3, kernel_size=3, padding=1)

    def forward(self, x):
        # 64 * 256 * 256
        # out = self.conv(x)

        # 64 * 256 * 256
        R1 = self.c1(x)
        # 128 * 128 * 128
        R2 = self.c2(self.d1(R1))
        # 256 * 64 * 64
        R3 = self.c3(self.d2(R2))
        # 512 * 32 * 32
        R4 = self.c4(self.d3(R3))
        # 1024 * 16 * 16
        R5 = self.c5(self.d4(R4))

        down64 = self.down64(R1)
        down128 = self.down128(R2)
        down256 = self.down256(R3)
        down512 = self.down512(R4)

        # 1024 * 16 * 16 -> 512 * 32 * 32 -> 1024 * 32 * 32 - > 512 * 32 * 32
        O1 = self.uc1(self.u1(R5 + down64[3] + down128[2] + down256[1] + down512[0], R4))
        # 256 * 64 * 64
        O2 = self.uc2(self.u2(O1 + down64[2] + down128[1] + down256[0], R3))
        # 128 * 128 * 128
        O3 = self.uc3(self.u3(O2 + down64[1] + down128[0], R2))
        # 64 * 256 * 256
        O4 = self.uc4(self.u4(O3 + down64[0], R1))

        Loc = self.local(x)
        # 3 * 256 * 256
        result = self.res(O4 + Loc)

        return result

