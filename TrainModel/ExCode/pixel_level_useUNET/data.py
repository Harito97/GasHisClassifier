# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 19:59:47 2019

@author: zjh
"""
import cv2 as cv
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import os
import skimage.io as io
import glob
import scipy.io as sio


#归一化图像，制作好gt，分前景后景
def adjustData(original,mask):
    original = original/255
    mask = mask/255
    mask[mask > 0.5] = 1
    mask[mask < 0.5] =0
    return(original,mask)


#生成训练图像
def trainGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,#图片转换为灰度图像
            target_size = target_size,#data_size = (256, 256)相当于resize的功能
            batch_size = batch_size, #BATCH_SIZE = 8
            save_to_dir = aug_image_save_dir,#把变动之后的图保存在Aug_originall文件夹里面
            save_prefix = original_aug_prefix,
            seed = seed)
    # myGene = trainGenerator(BATCH_SIZE, train_path, type, "GTM", aug_dict, target_size=data_size,
                                #aug_image_save_dir=Aug_originall, aug_mask_save_dir=Aug_GTM1)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)

    train_generator = zip(original_generator,mask_generator)

    #zip()是Python的一个内建函数，它接受一系列可迭代的对象作为参数，将对象中对应的元素打包成一个个tuple（元组），然后返回由这
    # 些tuples组成的list（列表）。也就是说，该函数返回一个以元组为元素的列表，其中第 i 个元组包含每个参数序列的第 i 个元素。
    #原图GT一一对应打好包
    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)
    #按批次返回数据
def validationGenerator(batch_size,train_path,original_dir,mask_dir,aug_dict,target_size,image_color_mode = "grayscale",aug_image_save_dir=None,aug_mask_save_dir=None,original_aug_prefix="image",mask_aug_prefix="mask",seed=1):
    original_datagen = ImageDataGenerator(**aug_dict)
    mask_datagen = ImageDataGenerator(**aug_dict)
    original_generator = original_datagen.flow_from_directory(
            train_path,
            classes = [original_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_image_save_dir,
            save_prefix = original_aug_prefix,
            seed = seed)
    mask_generator = mask_datagen.flow_from_directory(
            train_path,
            classes = [mask_dir],
            class_mode = None,
            color_mode = image_color_mode,
            target_size = target_size,
            batch_size = batch_size,
            save_to_dir = aug_mask_save_dir,
            save_prefix = mask_aug_prefix,
            seed = seed)



    train_generator = zip(original_generator,mask_generator)



    for (original,mask) in train_generator:
        original,mask = adjustData(original,mask)
        yield (original,mask)


def testGenerator(test_path, num_image, target_size):
    for pngfile in glob.glob(test_path + "/*.png"):#查找有关目录下的所有文件，pngfile为路径
        img = cv.imread(pngfile, cv.IMREAD_GRAYSCALE)#在OpenCV第一个参数是指定需要读取的图片的路径和图片名，另一个参数，常用的就是"IMREAD_UNCHANGED"、"IMREAD_GRAYSCALE"、"IMREAD_COLOR"三个属性
        img = img / 255
        img = cv.resize(img, target_size)#缩放图片
        img = np.reshape(img, img.shape + (1,))
        img = np.reshape(img, (1,) + img.shape)#前后增加一维
        yield img


#


def saveResult(save_path, result, flag_multi_class=False, num_class=2):
    for i, item in enumerate(result):
        img = item[:, :, 0]
        # io.imsave(os.path.join(save_path,"%d_predict.png"%i),img)     #使用该函数保存时不用做归一化
        # cv.normalize(img, img, 0, 255, cv.NORM_MINMAX)  # 使用opencvimwrite保存图片时，需做归一化，将原先归一化至0-1的图片复原出来，否则结果全黑
        # img = cv.resize(img, (2048,2048))
        img = np.array(img)

        # cv.imwrite(save_path + "/" + str(i) + "_predict.mat", img)
        # sio.savemat(save_path + "/" + ("%04d" % i) + ".mat", {'img': img}) #val
        sio.savemat(save_path + "/" + ("%05d" % i) + ".mat", {'img': img})  # val

