import os
import PIL.Image as Image
import numpy as np
import cv2
# import glob


def selet_pic(labpath):
    pic_path = os.listdir(labpath)
    list = []
    for alldir in pic_path:
        child = os.path.join(labpath, alldir)
        img = cv2.imread(child)
        lab = Image.fromarray(cv2.cvtColor(lab, cv2.COLOR_BGR2GRAY)).convert('1')
        pic_arrays = np.array(img)  # 将图片转化成数组
        print(pic_arrays)
        # 临界值设为4
        if np.mean(pic_arrays) <= 3:
            # print("图片为黑色",alldir)
            list.append(alldir)

    return list


def del_pic(list, img_del_path, lab_del_path):
    for i in list:
        img_path = os.path.join(img_del_path, i)
        lab_path = os.path.join(lab_del_path, i)
        # print(lab_path)
        os.remove(img_path)  # 直接删除
        os.remove(lab_path)
        # shutil.move(img_path, img_del_path)
        # shutil.move(lab_path,lab_del_path)         #移到指定位置


if __name__ == "__main__":
    imgpath = '../data/whu/train/img'
    labpath = '../data/whu/train/lab'
    list = selet_pic(labpath)
    del_pic(list,imgpath,labpath)
    #del_pic(list)