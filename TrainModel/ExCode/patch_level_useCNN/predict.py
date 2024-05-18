import os
import time
import numpy as np
import pandas as pd
from keras import backend as K
from keras.applications.imagenet_utils import preprocess_input
from keras.models import load_model
from keras.preprocessing import image

#num_classes=2
# def mycrossentropy(y_true, y_pred, e=0.86):
#     return (1-e)*K.categorical_crossentropy(y_pred,y_true) + e*K.categorical_crossentropy(y_pred, K.ones_like(y_pred)/num_classes)

# =============================================================================
# #预设路径。
# =============================================================================
#模型路径
model_path = r'E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\vgg16-best.h5'
#图像数据路径
pic_path = r'E:\20190429-Gastric-Carcinoma-Cancer-Subset-Division\test\patch_level\\'
#预测结果以及混淆矩阵存储路径
save_path = r'E:\\20190429-Gastric-Carcinoma-Cancer-Subset-Division\\vgg16\\'

#不需要修改以下内容
#high_path = pic_path + 'test-set\\high\\'
#mid_path = pic_path + 'test-set\\mid\\'
#low_path = pic_path + 'test-set\\low\\'
#不需要修改以下内容
cancer_path = pic_path + 'cancer\\'
no_cancer_path = pic_path + 'no_cancer\\'




def predict(img_path, model):
    """
    预测单条数据的结果，返回值为分类代号
    [0]:cancer
    [1]:no_cancer

    """
    img = image.load_img(img_path, target_size=(64, 64))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    preds_acc = model.predict(x)
    #predict_test = model.predict_classes(x)
    #print(predict_test)
    return np.argmax(preds_acc,axis=1), preds_acc

def get_filename(file_dir):
    """
    获取指定路径内的所有文件绝对路径
    返回值类型为 list
    """
    L = []
    for root, dirs, files in os.walk(file_dir):
        for file in files:
            if os.path.splitext(file)[1] == '.png':#指定文件类型
                L.append(os.path.join(root, file))
    return L

def mytest(model,cancer,no_cancer):
    """
    预测high,low,mid三个文件夹内的数据结果。
    将结果写入为csv格式。
    """
    cancer_name=[]
    cancer_answer=[]
    no_cancer_name=[]
    no_cancer_answer=[]
    cancer_pred_cancer_acc_col_1=[]
    cancer_pred__no_cancer_acc_col_2=[]
    no_cancer_pred_cancer_acc_col_1=[]
    no_cancer_pred_no_cancer_acc_col_2=[]

    for i in cancer:
        cancer_name.append(i.split('\\')[-1])
        pred_label1, pred_acc1 = predict(i,model)
        #pred_label1 = predict(i, model)
        cancer_answer.append(pred_label1)
        cancer_pred_cancer_acc_col_1.append(pred_acc1[0,0])
        cancer_pred__no_cancer_acc_col_2 .append(pred_acc1[0,1])
        #cancer_answer.append(pred_acc1[])
    for j in no_cancer:
        no_cancer_name.append(j.split('\\')[-1])
        pred_label2,pred_acc2,=predict(j,model)
        #pred_label2= predict(j, model)
        no_cancer_answer.append(pred_label2)
        no_cancer_pred_cancer_acc_col_1 .append(pred_acc2[0,0])
        no_cancer_pred_no_cancer_acc_col_2 .append(pred_acc2[0,1])
        #no_cancer_answer.append(pred_acc2)

    cancer_name_col = pd.Series(cancer_name, name='cancer_name')
    cancer_pred_col = pd.Series(cancer_answer, name='cancer_pred')
    cancer_pred_acc_col_1 = pd.Series(cancer_pred_cancer_acc_col_1, name='cancer_pred_pro')
    cancer_pred_acc_col_2 = pd.Series(cancer_pred__no_cancer_acc_col_2, name='no_cancer_pred_pro')

    no_cancer_name_col = pd.Series(no_cancer_name, name='no_cancer_name')
    no_cancer_pred_col = pd.Series(no_cancer_answer, name='no_cancer_pred')
    no_cancer_pred_acc_col_1 = pd.Series(no_cancer_pred_cancer_acc_col_1, name='cancer_pred_pro')
    no_cancer_pred_acc_col_2 = pd.Series(no_cancer_pred_no_cancer_acc_col_2 , name='no_cancer_pred_pro')

    predictions = pd.concat([cancer_name_col, cancer_pred_acc_col_1,cancer_pred_acc_col_2,cancer_pred_col, no_cancer_name_col,no_cancer_pred_acc_col_1, no_cancer_pred_acc_col_2, no_cancer_pred_col], axis=1)
    predictions.to_csv(save_path+'pred.csv')

if __name__ == "__main__":
    start = time.clock()
    """
    程序入口。
    """
    model = load_model('vgg16-best-2.h5')
    cancer=get_filename(cancer_path)
    no_cancer=get_filename(no_cancer_path)
    mytest(model,cancer,no_cancer)
    print('预测完毕')

    end2 = time.clock()
    print("final is in ", end2 - start)