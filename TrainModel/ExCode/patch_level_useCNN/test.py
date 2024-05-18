import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Flatten, Dropout
from keras.models import Model
from keras.optimizers import Adam, SGD
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
from keras.models import load_model, Sequential
import numpy as np

img = image.load_img('1_0.png', target_size=(64, 64))
x = image.img_to_array(img)    # x.shape: (224, 224, 3)
x = np.expand_dims(x, axis=0)  # x.shape: (1, 224, 224, 3)
x = tf.convert_to_tensor(x, tf.float32)
# x = tf.read_file("1_0.png",target_size=(224, 224))
# # 将图像文件解码为Tensor
# x = tf.image.decode_jpeg(x)
# Inp = Input((224, 224, 3))
vgg16_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))
features = vgg16_model(x)
features = GlobalAveragePooling2D()(features)
model = load_model('D:\\PycharmProject\\GastricCarcinoma\\my_weight.h5')
preds_acc = model.predict(features)