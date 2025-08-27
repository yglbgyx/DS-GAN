# DS-GAN
DS-GAN, A GAN for thermal infrared image colorazation

# NetWork
![image](https://github.com/yglbgyx/DS-GAN/blob/main/img/overall.png)

# Data Preparation
[KAIST-MS](https://github.com/SoonminHwang/rgbt-ped-detection/blob/master/data/README.md) and [IRVI](https://pan.baidu.com/s/1og7bcuVDModuBJhEQXWPxg?pwd=IRVI) dataset. 
The resolution of all images in the input network is 256 × 256.

# Colorization results
## KAIST dataset
![image](https://github.com/yglbgyx/DS-GAN/blob/main/img/KAIST.png)
(a) TIR image. (b) CycleGAN. (c) PearlGAN. (d) Pix2pix. (e)TIC-CGAN. (f) LKAT-GAN. (g) MUGAN. (h) Ours. (i) Ground truth.

## IRVI dataset
![image](https://github.com/yglbgyx/DS-GAN/blob/main/img/traffic.png)
(a) TIR image. (b) PearlGAN. (c) Pix2pix. (d) I2VGAN. (e)TIC-CGAN. (f) LKAT-GAN. (g) MUGAN. (h) Ours. (i) Ground truth.
![image](https://github.com/yglbgyx/DS-GAN/blob/main/img/monti.png)
(a) TIR image. (b) PearlGAN. (c) Pix2pix. (d) I2VGAN. (e)TIC-CGAN. (f) LKAT-GAN. (g) MUGAN. (h) Ours. (i) Ground truth.

# Acknowledgments
This code borrows heavily from TICCGAN.
