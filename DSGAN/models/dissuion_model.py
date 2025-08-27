
import os
import cv2
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .patch_embed import PatchEmbed
import pytorch_msssim as torchssim


if not os.path.exists('samples'):
    os.mkdir('samples')

# 基本配置
imgs = os.listdir('/root/CelebA-HQ/train/')
imgs += os.listdir('/root/CelebA-HQ/valid/', 'png')
np.random.shuffle(imgs)
img_size = 128  # 如果只想快速实验，可以改为64
batch_size = 32  # 如果显存不够，可以降低为16，但不建议低于16
embedding_size = 128
channels = [1, 1, 2, 2, 4, 4]
blocks = 2  # 如果显存不够，可以降低为1



# 超参数选择
T = 1000
alpha = np.sqrt(1 - 0.02 * np.arange(1, T + 1) / T)
beta = np.sqrt(1 - alpha**2)
bar_alpha = np.cumprod(alpha)
bar_beta = np.sqrt(1 - bar_alpha**2)
sigma = beta.copy()
# sigma *= np.pad(bar_beta[:-1], [1, 0]) / bar_beta


def imread(f, crop_size=None):
    """读取图片
    """
    x = cv2.imread(f)
    height, width = x.shape[:2]
    if crop_size is None:
        crop_size = min([height, width])
    else:
        crop_size = min([crop_size, height, width])
    height_x = (height - crop_size + 1) // 2
    width_x = (width - crop_size + 1) // 2
    x = x[height_x:height_x + crop_size, width_x:width_x + crop_size]
    if x.shape[:2] != (img_size, img_size):
        x = cv2.resize(x, (img_size, img_size))
    x = x.astype('float32')
    x = x / 255 * 2 - 1
    return x

def imwrite(path, figure):
    """归一化到了[-1, 1]的图片矩阵保存为图片
    """
    figure = (figure + 1) / 2 * 255
    figure = np.round(figure, 0).astype('uint8')
    cv2.imwrite(path, figure)





def data_generator():
    """图片读取
    """
    batch_imgs = []
    while True:
        for i in np.random.permutation(len(imgs)):
            batch_imgs.append(imread(imgs[i]))
            if len(batch_imgs) == batch_size:
                batch_imgs = np.array(batch_imgs)
                batch_steps = np.random.choice(T, batch_size)
                batch_bar_alpha = bar_alpha[batch_steps][:, None, None, None]
                batch_bar_beta = bar_beta[batch_steps][:, None, None, None]
                batch_noise = np.random.randn(*batch_imgs.shape)
                batch_noisy_imgs = batch_imgs * batch_bar_alpha + batch_noise * batch_bar_beta
                yield [batch_noisy_imgs, batch_steps[:, None]], batch_noise
                batch_imgs = []




class GroupNorm(nn.Module):
    """定义GroupNorm，默认groups=32
    """
    def __init__(self):
        super(GroupNorm, self).__init__()
    def forward(self, inputs):
        inputs = inputs.view((-1, 32), -1)
        mean=torch.mean(inputs,dim=[1,2,3],keepdim=True)
        variance = torch.var(inputs,dim=[1,2,3],keepdim=True)
        inputs = (inputs - mean) * torch.rsqrt(variance + 1e-6)
        inputs = torch.nn.flatten(inputs, -2)
        return inputs



def dense(x, out_dim):
    return torch.nn.Linear(x.shape[1],out_dim,bias=False)

def conv2d(x, out_dim):
    """Conv2D包装
    """
    input=x.shape[1]
    return torch.nn.Conv2d(input,out_dim,3,1,1,bias=False)

def swish(x,beta=1):
    return x*torch.nn.Sigmoid()(x*beta)

def residual_block(x, ch, t):
    """残差block
    """
    in_dim = x.shape[-1]
    out_dim = ch * embedding_size
    if in_dim == out_dim:
        xi = x
    else:
        xi = dense(x, out_dim)(x)
    x = GroupNorm(x)
    x = swish(x)
    x = conv2d(x, out_dim)
    x = x+dense(t, x.shape[-1])(x)
    x = GroupNorm(x)
    x = swish(x)
    x = conv2d(x, out_dim)(x)
    x = x+xi
    return x

def l2_loss(y_true, y_pred):
    """用l2距离为损失，不能用mse代替
    """
    return torch.sum((y_true - y_pred)**2, dim=[1, 2, 3])

# 搭建去噪模型
x_in = x = Input(shape=(img_size, img_size, 3))
t_in = Input(shape=(1,))
t = torch.nn.Linear(
    input_dim=T,
    output_dim=embedding_size,
)(t_in)
t = dense(t, embedding_size * 4, 'swish')
t = dense(t, embedding_size * 4, 'swish')
t = Lambda(lambda t: t[:, None])(t)

x = conv2d(x, embedding_size)
inputs = [x]



