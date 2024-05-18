import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import h5py
from PIL import Image

def makedir(path):

    folder = os.path.exists(path)

    if folder is True:
        print("文件夹已存在")
    else:
        os.makedirs(path)
        print("文件夹创建中")
        print("完成")


def Dice(SEG,GT):
    intersection = SEG & GT
    num_intersection = intersection.sum()
    num_sum = SEG.sum() + GT.sum()
    DICE = 2 * num_intersection / num_sum
    return DICE
def VOE(SEG,GT):
    mistake = SEG ^ GT
    num_mistake = mistake.sum()
    num_sum = SEG.sum() + GT.sum()
    voe = num_mistake/num_sum
    return voe
def RVD(SEG,GT):
    num_SEG = SEG.sum()
    num_GT = GT.sum()
    rvd = abs(num_SEG/num_GT - 1)
    return rvd
def IoU(SEG,GT):
    intersection = SEG & GT
    union = SEG|GT
    num_intersection = intersection.sum()
    num_union = union.sum()
    IOU = num_intersection/num_union
    return IOU
def Precision(SEG,GT):
    intersection = SEG & GT
    num_positive = SEG.sum()
    num_intersection = intersection.sum()
    precision = num_intersection/num_positive
    return precision
def Recall(SEG,GT):
    intersection = SEG & GT
    num_gt = GT.sum()
    num_intersection = intersection.sum()
    recall = num_intersection/num_gt
    return recall
def Specificity(SEG,GT,length,width):
    sum_totally = length*width
    union = SEG|GT
    num_union = union.sum()
    num_gt = GT.sum()
    num_gt_negative = sum_totally - num_gt
    num_seg_negative = sum_totally - num_union
    specificity = num_seg_negative/num_gt_negative
    return specificity
def Accuracy(SEG,GT,length,width):
    sum_totally = length * width
    right = (SEG ^ GT)
    # print(right)
    num_union = right.sum()
    accuracy = 1-num_union/sum_totally
    return accuracy

def view(mat,num,img_path,GT_path,prd_path):
    name_list=["D","I","P","R","S","V","RV","Acc"]
    for i in range(num):
        num_list = mat[i]
        # predict_path = img_path
        # gt_path =  GT_path


        ################################每个预测图片与GT对比图，及各项指标####################################
        save_fig_path = prd_path
        makedir(save_fig_path)
        fig = plt.figure(figsize=(10,3))
        data = h5py.File(img_path+"/"+("%03d" % (i+1))+".mat", 'r')
        SEG1 = data['binary_pic'][:]
        SEG1 = np.transpose(SEG1)


        SEG1[SEG1 > 0.5] = 1
        SEG1[SEG1 < 0.5] = 0
        # GT1[GT1 > 0.5] = 1
        # GT1[GT1 < 0.5] = 0
        SEG2 = SEG1.astype(np.uint8)
        SEG2 = Image.fromarray( SEG2)
        gt = cv.imread(GT_path + "/" + ("%03d" %(i+1)) + ".png", cv.IMREAD_GRAYSCALE)
        gt = cv.resize(gt,(256,256))
        ax1 = fig.add_subplot(131)
        ax2 = fig.add_subplot(132)

        ax1.imshow(gt)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title("GroundTruth")

        ax2.imshow(SEG2)
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_title("Predict")

        ax3 = fig.add_subplot(133)
        ax3.bar(range(len(num_list)),num_list,color = ['#4F94CD','#CAE1FF'])
        plt.xticks(range(len(num_list)), name_list)
        ax3.set_title("Evaluation")
   #     plt.show()
        plt.savefig(save_fig_path+"/"+str(i)+'_evaluation.jpg',dpi=300)


        with open(save_fig_path + "/" + 'evaluation.txt', 'a', encoding='utf-8') as f:
            f.write("predict_image_"+ str(i) + "_index:" + str(num_list) + "\n" )
    with open(save_fig_path + "/" + 'evaluation.txt', 'a', encoding='utf-8') as f:
        f.write("predict_image_average_index:" + str(sum(mat)/num) + "\n" )

    ##############################################图片指标总图################################################
    # num_list = mat[0]  # 为了获取指标个数
    color = ['#4F94CD', '#CAE1FF']
    # l = 15 * x_num  # 水平方向总图大小
    # h = 1 * y_num  # 竖直方向总图大小
    # fig1 = plt.figure(figsize=(l, h))  # 总图大小
    #
    # for f in range(num):
    #     ax1 = fig1.add_subplot(x_num, y_num, f + 1)
    #     ax1.bar(range(len(num_list)), mat[f], color=color)
    #     plt.xticks(range(len(num_list)), name_list)
    #     ax1.set_title(str(f) + "_index")
    # plt.savefig(save_fig_path + "/total_evaluation.jpg", dpi=400)

    average_list = sum(mat) / num
    fig2 = plt.figure()
    plt.bar(range(len(average_list)), average_list, color=color)
    plt.xticks(range(len(average_list)), name_list)
    plt.title("Average_Index")
    plt.savefig(save_fig_path + "/Average_Index.jpg", dpi=300)
    ##############################################各类别图片指标总图################################################
    """"
    for i in range(1):
        num_list = mat[0]  # 为了获取指标个数
        color = ['#4F94CD', '#CAE1FF']
        help_list = np.linspace(i*10,(i+1)*10-1,10)
        help_list = list(map(int, help_list))
        help_mat = mat[help_list]
        help_average_list = sum(help_mat) / 10
        fig2 = plt.figure()
        plt.bar(range(len(help_average_list)), help_average_list, color=color)
        plt.xticks(range(len(help_average_list)), name_list)
        plt.title("Average_Index")
        plt.savefig(save_fig_path + "/Kind_"+str(i)+"_Average_Index.jpg", dpi=300)
        with open(save_fig_path + '.txt', 'a', encoding='utf-8') as f:
            f.write("Kind_"+str(i)+"_Average_Index:" + str(help_average_list) + "\n")
    """

def evaluation(img_path, GT_path,prd_path):
    img_list = os.listdir(img_path)
    GT_list = os.listdir(GT_path)
    num_img = len(img_list)
    mat = np.zeros(shape=(num_img, 8))
    for i in range(num_img):
        data = h5py.File(img_path+"/"+("%03d" % (i+1))+".mat", 'r')
        SEG1 = data['binary_pic'][:]
        SEG1 = np.transpose(SEG1)
        GT = cv.imread(GT_path + "/" + ("%03d" % (i+1)) + ".png", cv.IMREAD_GRAYSCALE)
        # img = cv.imread(img_path+"/"+str(i)+".png", cv.IMREAD_GRAYSCALE)
        # GT = cv.imread(GT_path + "/" + str(i) + ".png", cv.IMREAD_GRAYSCALE)
        # img_shape = img.shape
        # GT = cv.resize(GT,img_shape)

        GT1 = GT/255

        SEG1[SEG1 > 0.5] = 1
        SEG1[SEG1 < 0.5] = 0
        GT1[GT1 > 0.5] = 1
        GT1[GT1 < 0.5] = 0
        SEG2 = SEG1.astype(np.int16)
        GT2 = GT1.astype(np.int16)

        D = Dice(SEG2, GT2)
        I = IoU(SEG2, GT2)
        P = Precision(SEG2, GT2)
        R = Recall(SEG2, GT2)
        S = Specificity(SEG2, GT2, 2048, 2048)
        V = VOE(SEG2, GT2)
        RV = RVD(SEG2, GT2)
        Acc = Accuracy(SEG2, GT2, 2048, 2048)
        mat[i] = [D, I, P, R, S, V, RV, Acc]
    view(mat,num_img,img_path,GT_path,prd_path)


evaluation("E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/test/unet/patch/1/test/binary_result/binary48_seg_result_mat","E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/test/unet/GTM","E:/20190429-Gastric-Carcinoma-Cancer-Subset-Division/test/unet/patch/1/test/binary_result/binary48_evaluation")