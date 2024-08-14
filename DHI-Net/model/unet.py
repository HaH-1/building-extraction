import level_inter as bd
torch = bd.torch
nn = torch.nn

class UNet(nn.Module):
    def __init__(self,in_channels,num_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.in_conv = bd.DoubleConv(in_channels, 64)
        self.down1 = bd.Down(64,128)
        self.down2 = bd.Down(128,256)
        self.down3 = bd.Down(256,512)
        self.down4 = bd.Down(512,512)
        self.up1 = bd.Up(1024,256)
        self.up2 = bd.Up(512,128)
        self.up3 = bd.Up(256,64)
        self.up4 = bd.Up(128,32)
        self.out_conv = bd.OutConv(32, num_classes)

    def forward(self, x):
        x1 = self.in_conv(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.out_conv(x)

        return logits