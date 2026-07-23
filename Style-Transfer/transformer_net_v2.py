import torch
from torch import nn
from transformer_net import ConvLayer, ResidualBlock, UpsampleConvLayer

class TransformerNet(nn.Module):
    def __init__(self):
        super(TransformerNet, self).__init__()

        # Downsampling
        self.conv1 = ConvLayer(3, 64, kernel_size=9, stride=1)
        self.in1 = nn.InstanceNorm2d(64, affine=True)

        self.conv2 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in2 = nn.InstanceNorm2d(128, affine=True)

        self.conv3 = ConvLayer(128, 256, kernel_size=3, stride=2)
        self.in3 = nn.InstanceNorm2d(256, affine=True)

        # Residual blocks (7 total)
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        self.res4 = ResidualBlock(256)
        self.res5 = ResidualBlock(256)
        self.res6 = ResidualBlock(256)
        self.res7 = ResidualBlock(256)

        # Upsampling
        self.deconv1 = UpsampleConvLayer(256, 128, kernel_size=3, stride=1, upsample=2)
        self.in4 = nn.InstanceNorm2d(128, affine=True)

        self.deconv2 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in5 = nn.InstanceNorm2d(64, affine=True)

        self.deconv3 = ConvLayer(64, 3, kernel_size=9, stride=1)

        self.relu = nn.ReLU()

    def forward(self, X):
        y = self.relu(self.in1(self.conv1(X)))
        y = self.relu(self.in2(self.conv2(y)))
        y = self.relu(self.in3(self.conv3(y)))

        y = self.res1(y)
        y = self.res2(y)
        y = self.res3(y)
        y = self.res4(y)
        y = self.res5(y)
        y = self.res6(y)
        y = self.res7(y)

        y = self.relu(self.in4(self.deconv1(y)))
        y = self.relu(self.in5(self.deconv2(y)))
        y = self.deconv3(y)

        return y 

