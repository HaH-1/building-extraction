import torch
import torch.nn as nn

# 单个残差块
class BottleNeck(nn.Module):
    def __init__(self, num_fea):
        super(BottleNeck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.BatchNorm2d(num_features=num_fea, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(num_fea, 64, kernel_size=1, stride=1, padding='valid')
        )
        self.conv2 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same')
        )
        self.conv3 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1,padding='same')
        )
        self.conv4 = nn.Sequential(
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, num_fea, kernel_size=1, stride=1, padding='valid')
        )


    def forward(self, x):
        residual = x
        # 降维
        out = self.conv1(x)
        # 提取特征
        out = self.conv2(out)
        out = self.conv3(out)
        # 恢复维度
        out = self.conv4(out)
        x = torch.add(residual, out)
        return x


# 卷积块
class Conv_Block(nn.Module):
    def __init__(self, num_fea):
        super(Conv_Block, self).__init__()
        self.bottle1 = BottleNeck(num_fea)
        self.bottle2 = BottleNeck(num_fea)
        self.bottle3 = BottleNeck(num_fea)
        self.bottle4 = BottleNeck(num_fea)

    def forward(self, x):
        out = self.bottle1(x)
        out = self.bottle2(out)
        out = self.bottle3(out)
        out = self.bottle4(out)
        return out


# 金字塔池化模块
class PyramidPoolingBlock(nn.Module):
    def __init__(self, bin_sizes):
        super(PyramidPoolingBlock, self).__init__()
        self.h = 256
        self.c = 64 * 7
        self.pool0 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.h // bin_sizes[0], stride=self.h // bin_sizes[0]),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=self.h // bin_sizes[0])
        )
        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.h // bin_sizes[1], stride=self.h // bin_sizes[1]),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=self.h // bin_sizes[1])
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.h // bin_sizes[2], stride=self.h // bin_sizes[2]),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=self.h // bin_sizes[2])
        )
        self.pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.h // bin_sizes[3], stride=self.h // bin_sizes[3]),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.UpsamplingBilinear2d(scale_factor=self.h // bin_sizes[3])
        )

    def forward(self, x):
        pool0 = self.pool0(x)
        pool1 = self.pool1(x)
        pool2 = self.pool2(x)
        pool3 = self.pool3(x)
        pool = torch.cat([pool0, pool1, pool2, pool3], dim=1)
        return pool

# 通道压缩模块
class ChannelSqueeze(nn.Module):
    def __init__(self):
        super(ChannelSqueeze, self).__init__()
        self.h = 256
        self.c = 64 * 7
        self.squeeze = nn.Sequential(
            nn.AvgPool2d(kernel_size=self.h),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(self.c, self.c, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        fc = self.squeeze(x)
        result = fc.view(-1, self.c, 1, 1)
        # print(x.shape)
        # print(fc.shape)
        # print(result.shape)
        return x * result

class MapNet(nn.Module):
    def __init__(self):
        super(MapNet, self).__init__()
        # 对于输入的处理
        self.in_ = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(num_features=64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.gen1 = Conv_Block(num_fea=64)
        self.path1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            Conv_Block(num_fea=64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding='same'),
            Conv_Block(num_fea=64)
        )
        self.gen2 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            Conv_Block(num_fea=128)
        )
        self.path2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding='same'),
            Conv_Block(num_fea=128),
            nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.path3 = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=1, stride=1),
            Conv_Block(num_fea=256),
            nn.UpsamplingBilinear2d(scale_factor=4)
        )

        self.enhance = nn.Sequential(
            nn.BatchNorm2d(num_features=448 * 5, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(448 * 5, 128, kernel_size=1, padding='same')
        )

        # 最后上采样阶段
        self.out = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.BatchNorm2d(128, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(64, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(32, momentum=0.99, eps=1e-3),
            nn.ReLU(),
            nn.Conv2d(32, 1, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.channle_squeeze = ChannelSqueeze()
        self.pyramid_pooling_block = PyramidPoolingBlock([1, 2, 4, 8])

    def forward(self, x):
        in_ = self.in_(x)
        gen1 = self.gen1(in_)
        path1 = self.path1(gen1)
        gen2 = self.gen2(gen1)
        path2 = self.path2(gen2)
        path3 = self.path3(gen2)
        fuse = torch.cat([path1, path2, path3], dim=1)
        squeeze = self.channle_squeeze(fuse)
        print(squeeze.shape)
        spatial = self.pyramid_pooling_block(squeeze)
        new_feature = torch.cat([spatial, squeeze], dim=1)
        result = self.enhance(new_feature)
        final = self.out(result)
        # print(final)
        return final

# import numpy as np
# x = np.ones((2,3,256,256))
# x = torch.Tensor(x)
# print(MapNet()(x))
from torchsummary import summary
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MapNet().to('cuda')
    import numpy
    summary(model,(3,256,256))