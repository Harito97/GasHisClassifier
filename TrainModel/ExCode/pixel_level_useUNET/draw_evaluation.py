import cv2 as cv
import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.io
import h5py
from PIL import Image

def autolabel(rects,prd_path_2):

    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.-0.4, 1.02*height, '%.4f' % height)
    plt.yticks(np.arange(0, 2.2, 0.2))
    plt.savefig(prd_path_2 + "/pixel_test_binary"
                             ".jpg", dpi=300)
def view(prd_path):
    name_list=["D","I","P","R","S","RV","A"]
    save_fig_path = prd_path
    color = ['#4F94CD']
    #num_list = [33, 44, 53, 16, 11, 17, 17, 10]
    average_list=[0.45688971, 0.32025873, 0.40174932, 0.6913869,  0.77957416,
 1.6718464,  0.76886455]
    autolabel(plt.bar(range(len(average_list)), average_list, color=color,tick_label=name_list), save_fig_path)
    #plt.bar(range(len(average_list)), average_list, color=color)
    #plt.xticks(range(len(average_list)), name_list)




def evaluation(prd_path):

    view(prd_path)


evaluation("C:\\Users\\sch23\\Desktop\\paper_image")