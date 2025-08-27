# -*- coding: utf-8 -*-
# import xlsxwriter as xw
import os
import time
from options.train_options import TrainOptions
from data import CreateDataLoader
from models import create_model
from util.visualizer import Visualizer
from tqdm import tqdm
from options.test_options import TestOptions
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

import csv
import numpy as np
import cv2
import math
import torch
import random

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def check_img_data_range(img):
    if img.dtype == np.uint8:
        return 255
    else:
        return 1.0

def cal_psnr(img1, img2):
    if type(img1) == torch.Tensor:
        img1 = img1.cpu().data.numpy()
    if type(img2) == torch.Tensor:
        img2 = img2.cpu().data.numpy()
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)
    if mse < 1.0e-10:
        return 100
    return peak_signal_noise_ratio(img1, img2, data_range=check_img_data_range(img1))

def cal_ssim(img1, img2):
    return structural_similarity(img1, img2, multichannel = (len(img1.shape) == 3), data_range = check_img_data_range(img1))


if __name__ == '__main__':
    setup_seed(20)
    dataset_path = r'you are dataset path'
    cur_path = os.path.abspath('..')
    test_filepath = 'resext50_vision1' #
    model_test_result_path =os.path.join(cur_path, test_filepath)
    if os.path.exists(model_test_result_path) is not True:
        os.makedirs(model_test_result_path)
    Train_save_img_path = os.path.join(model_test_result_path, 'train_img')
    if os.path.exists(Train_save_img_path) is not True:
            os.makedirs(Train_save_img_path)
    Test_save_img_path = os.path.join(model_test_result_path, 'test_img')
    if os.path.exists(Test_save_img_path) is not True:
        os.makedirs(Test_save_img_path)
    output_freq = 100

    opt = TrainOptions().parse(dataset_path, model_test_result_path)
    opt2 = TestOptions().parse(dataset_path, model_test_result_path)
    data_loader = CreateDataLoader(opt, 'train')
    dataset = data_loader.load_data()
    data_loader_test = CreateDataLoader(opt2, 'test')
    dataset_test = data_loader_test.load_data()

    dataset_size = len(data_loader)
    dataset_size_test = len(data_loader_test)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    #加载保存参数
    model.setup(opt)
    visualizer = Visualizer(opt)

    best_psnr = 0
    # test(model,14,dataset_test)
    # exit()

    for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):

        print(opt.epoch_count, opt.niter + opt.niter_decay + 1)
        ssim_sum = 0
        psnr_sum = 0
        img_num = 0
        total_steps = 0
        with tqdm(total=math.ceil(len(dataset)/opt.batchSize), ascii=True) as tt:
            tt.set_description('epoch: {}/{}'.format(epoch, 21))
            epoch_start_time = time.time()
            iter_data_time = time.time()
            epoch_iter = 0

            for i, data in enumerate(dataset):
                # count = count + 1
                img_num = img_num + 1

                iter_start_time = time.time()
                # if total_steps % opt.print_freq == 0:
                if total_steps % output_freq == 0:
                    t_data = iter_start_time - iter_data_time
                # visualizer.reset()
                total_steps += opt.batchSize
                epoch_iter += opt.batchSize
                model.set_input(data)
                model.optimize_parameters()  # 是pass！！！

                tir = model.get_img_tir(data)
                tirs = np.clip(tir.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                fake = model.get_img_gen(data)
                result = np.clip(fake.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)
                label = model.get_img_label(data)
                labels = np.clip(label.cpu().data.numpy()[0], 0.0, 255.0).transpose([1, 2, 0]).astype(np.uint8)

                ssim = cal_ssim(labels, result)
                psnr = cal_psnr(labels, result)
                ssim_sum = ssim_sum + ssim
                psnr_sum = psnr_sum + psnr


                tt.update(1)
                ssim_avg = ssim_sum / img_num
                psnr_avg = psnr_sum / img_num

                if ((i+1) % output_freq==0):#保存训练集部分图像
                    color_img_sum = np.hstack([tirs, result, labels])
                    color_img_sum = cv2.cvtColor(color_img_sum, cv2.COLOR_BGR2RGB) #将BGR转换成RGB
                    cv2.imwrite(Train_save_img_path + '/train_' + 'Re' + str(epoch) + '_' + str(i+1) + ".png",
                                color_img_sum)

                # if (i+1) % opt.display_freq == 0:
                if ((i+1) % output_freq == 0):
                    save_result = (i+1) % opt.update_html_freq == 0

                    # visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

                # if (i+1) % opt.print_freq == 0:
                if ((i+1) % output_freq == 0):
                    losses = model.get_current_losses()
                    t = (time.time() - iter_start_time) / opt.batchSize
                    visualizer.print_current_losses(epoch, epoch_iter, losses, t, t_data,ssim_avg,psnr_avg)
                    # print('SSIM:%lf'%ssim)
                    # print(img_num)

                    f = open(os.path.join(model_test_result_path, 'result.csv'), 'a',newline='')  #把w改成a就可以不覆盖原有数据

                    # 2. 基于文件对象构建 csv写入对象
                    csv_writer = csv.writer(f)
                    message = ""
                    for k, v in losses.items():
                            message += '%s: %.3f ' % (k, v)
                    message += '  '

                    # 4. 写入csv文件内容
                    csv_writer.writerow([epoch,message,ssim_avg,psnr_avg])
                    f.close()
                iter_data_time = time.time()


            print(img_num)
            ssim_avg = ssim_sum / img_num
            psnr_avg = psnr_sum / img_num
            f2 = open(os.path.join(model_test_result_path, 'each_epoch.csv'), 'a', newline='')  # 把w改成a就可以不覆盖原有数据
            # 2. 基于文件对象构建 csv写入对象
            csv_writer = csv.writer(f2)
            csv_writer.writerow([epoch, 'train', ssim_avg, psnr_avg]) #这里是写入一个epoch最终的数据

            # 5. 关闭文件
            f2.close()

            if epoch <= opt.niter + opt.niter_decay:  # 保存当前最佳模型
                best_psnr = psnr_avg
                print('saving the model at the end of epoch %d, iters %d' %
                      (epoch, (i+1)))
                # model.save_networks('latest')
                model.save_networks(epoch)

            print('End of epoch %d / %d \t Time Taken: %d sec' %
                  (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))


            model.update_learning_rate()
