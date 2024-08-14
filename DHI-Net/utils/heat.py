import os
import torch
import torchvision as tv
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import argparse
import skimage.data
import skimage.io
import skimage.transform
import numpy as np
import matplotlib.pyplot as plt
import torchvision.models as models
from PIL import Image
import cv2
import level_inter as li
def make_dirs(path):
    if os.path.exists(path) is False:
        os.makedirs(path)

def get_feature():
    transform = li.transforms.Compose([
        li.transforms.ToTensor(),

    ])

    test_path = './data/whu'
    # model_path = 'mdl/mul/whu_w/1/2/mul21.mdl'
    model_path = './result/wh/ml18.mdl'
    test_dataset = li.test_dataset(test_path, transform=transform)
    test_dataloader = li.DataLoader(test_dataset, batch_size=1)

    my_model = li.CHINet()
    my_model.load_state_dict(li.torch.load(model_path))

    device = 'cpu' if li.torch.cuda.is_available() else 'cpu'
    my_model.to(device)

    for i, (image, label) in enumerate(test_dataloader):
        print(image.shape)
        image, label = image.to(device), label.to(device)
        my_model.eval()
        output = my_model(image)

    # 这里主要是一些参数，比如要提取的网络，网络的权重，要提取的层，指定的图像放大的大小，存储路径等等。
    print(output[0].shape)
    dst = './img/heat-1'
    therd_size = 256

    # 这段主要是存储图片，为每个层创建一个文件夹将特征图以JET的colormap进行按顺序存储到该文件夹，
    # 并且如果特征图过小也会对特征图放大同时存储原始图和放大后的图。
    for k in range(len(output)):
        if k == 7:
            continue
        features = output[k][0]
        iter_range = features.shape[0]
        for i in range(iter_range):
            # plt.imshow(x[0].data.numpy()[0,i,:,:],cmap='jet')

            feature = features.data.cpu().numpy()
            feature_img = feature[i, :, :]
            feature_img = np.asarray(feature_img * 255, dtype=np.uint8)

            dst_path = os.path.join(dst, str(k))

            make_dirs(dst_path)
            feature_img = cv2.applyColorMap(feature_img, cv2.COLORMAP_JET)
            if feature_img.shape[0] < therd_size:
                tmp_file = os.path.join(dst_path, str(i) + '_' + str(therd_size) + '.jpg')
                tmp_img = feature_img.copy()
                tmp_img = cv2.resize(tmp_img, (therd_size, therd_size), interpolation=cv2.INTER_NEAREST)
                cv2.imwrite(tmp_file, tmp_img)

            dst_file = os.path.join(dst_path, str(i) + '.jpg')
            cv2.imwrite(dst_file, feature_img)
if __name__ == '__main__':
    get_feature()