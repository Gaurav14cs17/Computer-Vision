import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path  # if you haven't already done so

file = Path(__file__).resolve()
print(file)
parent, root = file.parent, file.parents[1]
print(parent, root)
sys.path.append(str(root))
from layers.conv_module import *


class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x1, x2), dim=1)
        return out


class Get_block(nn.Module):
    def __init__(self, in_channels):
        super(Get_block, self).__init__()
        self.cfg = [(1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1), (1, 1, 0)]
        self.__in_channels = in_channels
        self.__out_channels = 2 * in_channels

        self.conv_1 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[0][0],
                                    stride=self.cfg[0][1], padding=self.cfg[0][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels =  self.__out_channels//2

        self.conv_2 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[1][0],
                                    stride=self.cfg[1][1], padding=self.cfg[1][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels *2

        self.conv_3 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[2][0],
                                    stride=self.cfg[2][1], padding=self.cfg[2][2], batch_norm='bn', activate='leaky')
        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels//2

        self.conv_4 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[3][0],
                                    stride=self.cfg[3][1], padding=self.cfg[3][2], batch_norm='bn', activate='leaky')
        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels* 2

        self.conv_5 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[4][0],
                                    stride=self.cfg[4][1], padding=self.cfg[4][2], batch_norm='bn', activate='leaky')

    def forward(self, x):
        x = self.conv_1(x)
        print(x.shape)
        x = self.conv_2(x)
        print(x.shape)
        x = self.conv_3(x)
        print(x.shape)
        x = self.conv_4(x)
        print(x.shape)
        x = self.conv_5(x)
        print(x.shape)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = Get_block(512)

    image = torch.randn((1, 512, 50, 50))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.cuda()
    summary(model, (512, 50, 50) )
    output = model(image)
    print(output.shape)
