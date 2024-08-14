import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv(in_channels, out_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=2,stride=2)

    def forward(self, x):
        x = self.conv(x)
        return x,self.maxpool(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if True:
            self.up = nn.UpsamplingBilinear2d(scale_factor=2)
            self.conv = Conv(in_channels, out_channels, in_channels // 2)
        # else:
        #     self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        #     self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        # diffY = x2.size()[2] - x1.size()[2]
        # diffX = x2.size()[3] - x1.size()[3]
        #
        # x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
        #                 diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv((x))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        return self.sigmoid(self.conv(x))


class ASPP(nn.Module):
    def __init__(self, in_channel, depth):
        super(ASPP, self).__init__()
        # global average pooling : init nn.AdaptiveAvgPool2d ;also forward torch.mean(,,keep_dim=True)
        self.mean = nn.AdaptiveAvgPool2d((1, 1))
        self.conv = nn.Conv2d(in_channel, depth, 1, 1)
        # k=1 s=1 no pad
        self.atrous_block1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.atrous_block12 = nn.Conv2d(in_channel, depth, 3, 1, padding=8, dilation=8)
        self.atrous_block18 = nn.Conv2d(in_channel, depth, 3, 1, padding=12, dilation=12)

        self.conv_1x1_output = nn.Conv2d(depth * 5, depth, 1, 1)

    def forward(self, x):
        size = x.shape[2:]
        # mean.shape = torch.Size([8, 3, 1, 1])
        image_features = self.mean(x)
        # conv.shape = torch.Size([8, 3, 1, 1])
        image_features = self.conv(image_features)
        # upsample.shape = torch.Size([8, 3, 32, 32])
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        # block1.shape = torch.Size([8, 3, 32, 32])
        atrous_block1 = self.atrous_block1(x)

        # block6.shape = torch.Size([8, 3, 32, 32])
        atrous_block6 = self.atrous_block6(x)

        # block12.shape = torch.Size([8, 3, 32, 32])
        atrous_block12 = self.atrous_block12(x)

        # block18.shape = torch.Size([8, 3, 32, 32])
        atrous_block18 = self.atrous_block18(x)

        # torch.cat.shape = torch.Size([8, 15, 32, 32])
        # conv_1x1.shape = torch.Size([8, 3, 32, 32])
        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net

class IDSC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(IDSC, self).__init__()
        self.conv0 = nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1)
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same', groups=in_channels, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same', bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        # print(x.shape)
        x = self.bn1(self.relu(self.conv0(x)))
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x

class RefineUnet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(RefineUnet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.idsc1 = IDSC(64,64)
        self.idsc2 = IDSC(128,128)
        self.idsc3 = IDSC(256,256)
        self.idsc4 = IDSC(512, 512)
        self.aspp = ASPP(512,512)
        self.down1 = (Down(3, 64))
        self.down2 = (Down(64, 128))
        self.down3 = (Down(128, 256))
        self.down4 = (Down(256, 512))
        self.up1 = (Up(1024, 256))
        self.up2 = (Up(512, 128))
        self.up3 = (Up(256, 64))
        self.up4 = (Up(128, 64))
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        x1,x1_ = self.down1(x)
        x2,x2_ = self.down2(x1_)
        x3,x3_ = self.down3(x2_)
        x4,x4_ = self.down4(x3_)
        x5 = self.aspp(x4_)
        x4 = self.idsc4(x4)
        x3 = self.idsc3(x3)
        x2 = self.idsc2(x2)
        x1 = self.idsc1(x1)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits
from torchsummary import summary
if __name__ == '__main__':
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RefineUnet(3,1).to('cuda')
    import numpy
    summary(model,(3,256,256))