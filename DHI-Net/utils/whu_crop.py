import cv2
import torch
import math
import os
from pathlib import Path


file_dir = "./../data/whu/val/"
val=os.listdir(file_dir)
# print(val)
save_path = './../../data/whu/val/'
# Path(save_path).mkdir(parents=True, exist_ok=True)
#
# block size
height_crop = 256
width_crop = 256
#
# overlap （如果不想重叠，可以置为0）
over_x = 0
over_y = 0
# 下上左右填充
pad_b,pad_u,pad_l,pad_r = 0,0,0,0
# h_val = height - over_x
# w_val = width - over_y
#
# # Set whether to discard an image that does not meet the size
# mandatory = False
for k,name in enumerate(val):
    filename = file_dir + name + '/'

    file_list = os.listdir(filename)

    # 保存每个图片除去后缀的名字
    imagename = []
    for i in file_list:
        imagename.append(i.split('.')[0])

    # 保存最终切割图片
    images = []
    for file in file_list:
        img2 = cv2.imread(filename + file)
        # img2 = cv2.copyMakeBorder(img, pad_u, pad_b, pad_l, pad_r, borderType=cv2.BORDER_CONSTANT, value=0)

        height,width = img2.shape[0],img2.shape[1]
        # 切割的行列个数
        max_row = height // height_crop
        max_line = width // width_crop
        # break
        image = []
        # img2 = torch.Tensor(img2)
        # img2 = img2.transpose(0,2).transpose(1,2)
        # img2 = list(img2)
        for line in range(max_line):
            for row in range(max_row):
                # print(img2.shape)
                # print(row * width_crop)
                # print((row + 1)*width_crop)
                img_ = img2[line * height_crop:(line+1) * height_crop,row * width_crop : (row + 1)*width_crop,:]
                # print(img_)
                # print(img_.shape)
                image.append(img_)
        images.append(image)

    save_ = save_path + name + '/'
    for i , img_big in enumerate(images):
        for j , img_small in enumerate(img_big):
            cv2.imwrite(save_+imagename[i] + '_' + str(j) + '.png' , img_small)
