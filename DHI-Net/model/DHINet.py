import torch
nn = torch.nn

class SKConv(nn.Module):
    def __init__(self,in_channels,out_channels,M=2,r=16,L=32):
        super(SKConv,self).__init__()
        self.M = M
        d=max(in_channels//r,L)
        self.out_channels=out_channels
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding='same',dilation=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding='same',dilation=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)
        )
        self.global_pool=nn.AdaptiveAvgPool2d(1) # 自适应pool到指定维度    这里指定为1，实现 GAP
        self.fc1=nn.Sequential(nn.Conv2d(out_channels,d,1,bias=False),
                               nn.BatchNorm2d(d),
                               nn.ReLU(inplace=True))   # 降维
        self.fc2=nn.Conv2d(d,out_channels*self.M,1,1,bias=False)  # 升维
        self.softmax=nn.Softmax(dim=1) # 指定dim=1  使得两个全连接层对应位置进行softmax,保证 对应位置a+b+..=1
    def forward(self, input1,input2):
        batch_size = input1.size(0)
        U=input1+input2
        s=self.global_pool(U)
        z=self.fc1(s)
        a_b=self.fc2(z)
        a_b=a_b.reshape(batch_size,self.M,self.out_channels,1)
        a_b=self.softmax(a_b)

        a = a_b[:,0,:,:]
        a = a.view(a.shape[0],a.shape[1],-1,a.shape[2])
        b = a_b[:,1,:,:]
        b = b.view(b.shape[0],b.shape[1],-1,b.shape[2])
        return input1 * a + input2 * b

class Space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels,out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class SKFuse(nn.Module):
    def __init__(self,in_,out_):
        super(SKFuse, self).__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64,32,kernel_size=3,stride=1,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32,1,kernel_size=3,stride=1,padding='same'),
            nn.ReLU(),
            nn.BatchNorm2d(1),
            nn.Conv2d(1, out_, kernel_size=3, stride=1, padding='same'),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        return self.fuse(x)

class PDDM_(nn.Module):
    '''
    Preserve Detail Down Module
    '''
    def __init__(self,in_channels):
        super(PDDM_, self).__init__()
        self.spd = Space_to_depth()
        self.spd_conv = nn.Conv2d(in_channels*4,in_channels,kernel_size=1,stride=1)
        self.max = nn.MaxPool2d(kernel_size=2,stride=2)
        self.s_w = nn.Conv2d(in_channels,1,kernel_size=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self,x):
        spd = self.spd_conv(self.spd(x))
        maxF = self.max(x)
        s_w = self.s_w(maxF)
        s = self.sigmoid(s_w)
        spd_s = spd * s
        return spd_s + maxF


class  CIM(nn.Module):
    '''
        Context Interaction module
    '''
    def __init__(self,in_channels,out_channels):
        super(CIM, self).__init__()

        self.up2 = nn.Sequential(
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1),
        )
        self.sk1 = SKConv(in_channels,in_channels)
        self.sk2 = SKConv(out_channels,out_channels)
        self.down2 = nn.Sequential(
            nn.Conv2d(out_channels, in_channels, kernel_size=1, stride=1),
            nn.MaxPool2d(kernel_size=2,stride=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding='same'),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.fuse = SKFuse(out_channels,2)


    def forward(self,x,y):
        y_en = self.down2(y)
        fuse1 = self.sk1(x,y_en)
        fuse1 = self.up2(fuse1)
        fuse2 = self.sk2(fuse1,y)
        return fuse2

class OutStem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutStem, self).__init__()
        self.up = nn.Sequential(nn.UpsamplingBilinear2d(scale_factor=2),
        nn.Conv2d(in_channels, 64, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=True),)
        self.conv = nn.Conv2d(64, out_channels, kernel_size=1)
        self.conv1x1 = nn.Conv2d(64*2,64,kernel_size=1,stride=1,bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,y):
        x = self.up(x)
        x = self.conv1x1(torch.cat([x,y],dim=1))
        return self.sigmoid(self.conv(x))


class Up_OP(nn.Module):
    def __init__(self,in_Channels,out_channels):
        super(Up_OP, self).__init__()
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv3x3 = nn.Sequential(
            nn.Conv2d(in_Channels+out_channels,out_channels,kernel_size=3,padding='same',stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding='same', stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self,bot,up):
        bot = self.up(bot)
        fuse = torch.cat([up,bot],dim=1)
        return self.conv3x3(fuse)


class DHI(nn.Module):
    def __init__(self):
        super(DHI, self).__init__()
        self.input_ = DoubleConv(3,64)
        self.down1 = PDDM_(64)
        self.fea1 = DoubleConv(64,128)
        self.down2 = PDDM_(128)
        self.fea2 = DoubleConv(128,256)
        self.down3 = PDDM_(256)
        self.fea3 = DoubleConv(256,512)
        self.down4 = PDDM_(512)
        self.fea4 = DoubleConv(512, 512)
        self.cim_up = CIM(256,128)
        self.cim_mid = CIM(512,256)
        self.cim_bottom = CIM(512,512)
        self.out_ = OutStem(128,1)
        self.op1 = Up_OP(512,256)
        self.op2 = Up_OP(256,128)


    def forward(self,x):
        input_ = self.input_(x)
        fea1 = self.fea1(self.down1(input_))
        fea2 = self.fea2(self.down2(fea1))
        fea3 = self.fea3(self.down3(fea2))
        fea4 = self.fea4(self.down4(fea3))

        ci_11 = self.cim_up(fea2,fea1)
        ci_12 = self.cim_mid(fea3,fea2)
        ci_13 = self.cim_bottom(fea4,fea3)
        ci_12 = self.op1(ci_13,ci_12)
        ci_11 = self.op2(ci_12,ci_11)

        ci_21 = self.cim_up(ci_12,ci_11)
        ci_22 = self.cim_mid(ci_13,ci_12)
        ci_21 = self.op2(ci_22,ci_21)

        ci_3 = self.cim_up(ci_22,ci_21)
        out_ = self.out_(ci_3,input_)

        return out_