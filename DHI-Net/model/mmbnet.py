import torch
import os
import logging
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import level_inter.utils.layers as be

# from thop import profile

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, 1, 0),
                nn.BatchNorm2d(planes * self.expansion),
                nn.ReLU()
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or inplanes != planes * self.expansion:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * self.expansion, 1, 1, 0),
                nn.BatchNorm2d(planes * self.expansion),
                nn.ReLU()
            )
        else:
            self.downsample = None
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class Block(nn.Module):
    def __init__(self, inchannel, channel, block):
        super(Block, self).__init__()
        self.conv1 = block(inplanes=inchannel, planes=channel)


    def forward(self, x):
        x = self.conv1(x)

        return x




class BasicRFB_a(nn.Module):
    def __init__(self, in_planes, out_planes, stride=1, scale = 0.1):
        super(BasicRFB_a, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes //4
        self.branch0 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=1,relu=False)
                )
        self.branch1 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(3,1), stride=1, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch2 = nn.Sequential(
                BasicConv(in_planes, inter_planes, kernel_size=1, stride=1),
                BasicConv(inter_planes, inter_planes, kernel_size=(1,3), stride=stride, padding=(0,1)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=3, dilation=3, relu=False)
                )
        self.branch3 = nn.Sequential(
                BasicConv(in_planes, inter_planes//2, kernel_size=1, stride=1),
                BasicConv(inter_planes//2, (inter_planes//4)*3, kernel_size=(1,3), stride=1, padding=(0,1)),
                BasicConv((inter_planes//4)*3, inter_planes, kernel_size=(3,1), stride=stride, padding=(1,0)),
                BasicConv(inter_planes, inter_planes, kernel_size=3, stride=1, padding=5, dilation=5, relu=False)
                )
        self.ConvLinear = BasicConv(4*inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self,x):

        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)

        out = torch.cat((x0,x1,x2,x3),1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out*self.scale + short
        out = self.relu(out)
        return out

class SCM(nn.Module ):

    def __init__(self,inchannel,channel):
        super(SCM, self).__init__()

        self.conv=Block (3,128,BasicBlock )

        self.conv1=nn.Sequential (
            nn.Conv2d (inchannel,channel,1,1,0,bias=False ),
            nn.BatchNorm2d (channel)
        )
        self.sig=nn.Sigmoid ()

    def forward(self,image,x):
        _,_,h,w=image.shape
        x1=F.interpolate(x,size=(h,w),mode='bilinear',align_corners= True)

        feature=self.conv(image)
        residual=feature

        high_feature=self.conv1(x1)
        high_feature =self.sig(high_feature )
        correct_feature=1-high_feature

        feature=correct_feature *feature
        out=residual+feature

        return out

class MMbNet(nn.Module):
    def __init__(self, inchannel, num_class):
        super(MMbNet, self).__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(inchannel, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv1 = Block(64, 128, BasicBlock)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2 = Block(128, 256, BasicBlock)
        self.conv12 = Block(128, 128, BasicBlock)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3    = Block(256, 512, BasicBlock)
        self.conv23   = Block(256, 256, BasicBlock)
        self.conv13   = Block(128, 128, BasicBlock)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4  = Block(512, 1024, BasicBlock)
        self.conv34 = Block(512, 512, BasicBlock)
        self.conv24 = Block(256, 256, BasicBlock)
        self.conv14 = Block(128, 128, BasicBlock)

        self.in_rf=BasicRFB_a(1024,1024)
        self.scm=SCM (inchannel= 1024,channel= 128)

        self.up1 = be.atten_Up(1024,512,is_deconv= True )
        self.up2 = be.atten_Up(512, 256,is_deconv= True )
        self.up3 = be.atten_Up(256, 128,is_deconv= True )

        self.out = nn.Conv2d(128, num_class, 1, 1, 0)

    def forward(self, x):
        b, c, h, w = x.shape
        downsample=F.interpolate(x,size=(h//4,w//4),mode='bilinear',align_corners= True )

        x = self.stem(x)

        x1 = self.conv1(x)
        x2 = self.maxpool1(x1)

        x2 = self.conv2(x2)
        x1 = self.conv12(x1)
        x3 = self.maxpool2(x2)

        x3 = self.conv3(x3)
        x2 = self.conv23(x2)
        x1 = self.conv13(x1)
        x4 = self.maxpool3(x3)

        x4 = self.conv4(x4)
        x3 = self.conv34(x3)
        x2 = self.conv24(x2)
        x1 = self.conv14(x1)

        in_rf=self.in_rf(x4)


        x = self.up1(in_rf   , x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        scm=self.scm(downsample ,x4  )
        if scm.shape[2]!=x.shape[2]:
            scm = F.pad(scm, [1, 0, 1, 0])
        #
        x=x+scm
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        out = self.out(x)
        out = nn.Sigmoid()(out)
        return out

if __name__=='__main__':
    net=MMbNet (3,2)
    print(net)

    # input = torch.randn(1, 3, 512, 512)
    # macs, params = profile(net, inputs=(input,))
    # print('Total macc:{}, Total params: {}'.format(macs / 1e9, params / 1e6))




