import time
import math, json, os, sys
import matplotlib.pyplot as plt
import keras
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Dense, GlobalAveragePooling2D, Input, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing import image
import pandas as pd
import tensorflow as tf
# num_classes=2
# def mycrossentropy(y_true, y_pred, e=0.14):
#        return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

"""
#1,数据路径
#2，模型保存路径名称,注意更改加载模型
#3，修改存储 history 路径
#4，存储fig 的标题以及路径
"""
DATA_DIR = 'E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\'

TRAIN_DIR = os.path.join(DATA_DIR, 'train')

VALID_DIR = os.path.join(DATA_DIR, 'val')

SIZE = (64, 64)
BATCH_SIZE = 128


def save_history(History):
    acc = pd.Series(History.history['acc'], name='acc')
    loss = pd.Series(History.history['loss'], name='loss')
    val_acc = pd.Series(History.history['val_acc'], name='val_acc')
    val_loss = pd.Series(History.history['val_loss'], name='val_loss')
    com = pd.concat([acc, loss, val_acc, val_loss], axis=1)
    # 注意存储位置！！
    com.to_csv('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\history.csv')


# 画出acc loss曲线
def plot_history(history):
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    # 在此处更改 图标的标题！！
    plt.title("vgg16 model")
    plt.ylabel("acc-loss")
    plt.xlabel("epoch")
    plt.legend([" acc", "val acc", " loss", "val loss"], loc="upper right")
    # plt.show()
    # 注意修改存储名称！！！
    plt.savefig("E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\vgg16_model.png")


if __name__ == "__main__":
    start = time.clock()
    num_train_samples = sum([len(files) for r, d, files in os.walk(TRAIN_DIR)])
    num_valid_samples = sum([len(files) for r, d, files in os.walk(VALID_DIR)])

    num_train_steps = math.floor(num_train_samples / BATCH_SIZE)#取整，把训练图像分拨放入网络
    num_valid_steps = math.floor(num_valid_samples / BATCH_SIZE)

    gen = keras.preprocessing.image.ImageDataGenerator()
    #数据集拓充，详见https: // blog.csdn.net / jacke121 / article / details / 79245732
    #val_gen = keras.preprocessing.image.ImageDataGenerator(horizontal_flip=True, vertical_flip=True)
    val_gen = keras.preprocessing.image.ImageDataGenerator()

    batches = gen.flow_from_directory(TRAIN_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                      batch_size=BATCH_SIZE)
    #使用了flow_from_directory方法从硬盘读取图像数据
    val_batches = val_gen.flow_from_directory(VALID_DIR, target_size=SIZE, class_mode='categorical', shuffle=True,
                                              batch_size=BATCH_SIZE)
    #可以用来批量处理数据
    classes = list(iter(batches.class_indices))

    # 构建不带分类器 （include_top=False）的预训练模型，只提取特征
    base_model = keras.applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3))


    for layer in base_model.layers:
        layer.trainable = False  # 锁住所有 InceptionV3 的卷积层
    Inp = Input((64, 64, 3))
    x = base_model(Inp)
    # x = base_model(Inp).layers[-1].output
    # 添加全局平均池化层
    x = GlobalAveragePooling2D()(x)
    # 添加一个全连接层
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)
    # 添加一个分类器，假设我们有classes长度个类
    predictions = Dense(len(classes), activation="softmax")(x)
    # 构建我们需要训练的完整模型
    finetuned_model = Model(inputs=Inp, outputs=predictions)
    # 首先，我们只训练顶部的几层（随机初始化的层）
    finetuned_model.compile(optimizer=Adam(lr=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
    for c in batches.class_indices:
        classes[batches.class_indices[c]] = c
    finetuned_model.classes = classes

    early_stopping = EarlyStopping(patience=15)

    checkpointer = ModelCheckpoint('E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\model_12\\vgg16_best.h5',
                                   verbose=1, save_best_only=True)
    # 在新的数据集上训练几代，base_model参数保持不变，只有增加的最后一层参数更新
    History = finetuned_model.fit_generator(batches, steps_per_epoch=num_train_steps, epochs=15,
                                            callbacks=[early_stopping, checkpointer], validation_data=val_batches,
                                            validation_steps=num_valid_steps)
    # end1 = time.clock()
    # print("save model before", end1)
    # finetuned_model.save('D:\\PycharmProject\\GastricCarcinoma\\vgg16_best.h5')
    save_history(History)
    plot_history(History)

    end2 = time.clock()
    print("final is in ", end2 - start)