import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path
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
    def __init__(self, feature_input_channels, out_channels, feature_output_channels):
        super(Get_block, self).__init__()
        self.cfg = [(1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1)]
        self.__in_channels = feature_input_channels
        self.__out_channels = out_channels

        self.conv_1 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[0][0],
                                    stride=self.cfg[0][1], padding=self.cfg[0][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels * 2

        self.conv_2 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[1][0],
                                    stride=self.cfg[1][1], padding=self.cfg[1][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels // 2

        self.conv_3 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[2][0],
                                    stride=self.cfg[2][1], padding=self.cfg[2][2], batch_norm='bn', activate='leaky')
        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels * 2

        self.conv_4 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[3][0],
                                    stride=self.cfg[3][1], padding=self.cfg[3][2], batch_norm='bn', activate='leaky')
        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels // 2

        self.conv_5 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[4][0],
                                    stride=self.cfg[4][1], padding=self.cfg[4][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels = self.__out_channels * 2
        self.conv_6 = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[5][0],
                                    stride=self.cfg[5][1], padding=self.cfg[5][2], batch_norm='bn', activate='leaky')

        self.__in_channels = self.__out_channels
        self.__out_channels = feature_output_channels

        self.last = Convolutional(self.__in_channels, feature_output_channels, kernel_size=1,
                                  stride=1, padding=0, batch_norm='bn', activate='leaky')

    def forward(self, x):
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        x = self.last(x)
        return x


class Feature_block(nn.Module):
    def __init__(self, feature_input_channels, out_channels, feature_output_channels):
        super(Feature_block, self).__init__()
        self.__in_channels = feature_input_channels
        self.__out_channels = out_channels
        self.cfg = [(1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1)]
        self.model_list = nn.ModuleList()
        flag = 0
        for i in range(len(self.cfg)):
            x = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[i][0],
                              stride=self.cfg[i][1], padding=self.cfg[i][2], batch_norm='bn',
                              activate='leaky')
            self.model_list.append(x)
            self.__in_channels = self.__out_channels
            if not flag:
                self.__out_channels = self.__out_channels * 2
            else:
                self.__out_channels = self.__out_channels // 2
            flag = flag ^ 1
        self.my_block = nn.Sequential(*self.model_list)
        self.last_layer = Convolutional(self.__in_channels, feature_output_channels, kernel_size=1,
                                        stride=1, padding=0, batch_norm='bn', activate='leaky')

    def forward(self, x):
        x = self.my_block(x)
        x = self.last_layer(x)
        return x


if __name__ == '__main__':
    from torchsummary import summary

    model = Get_block(54, 128, 53)
    model_1 = Feature_block(54, 128, 53)
    image = torch.randn((1, 54, 50, 50))
    image_1 = image.clone()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image = image.to(device)
    model.cuda()
    summary(model, (54, 50, 50))
    output = model(image)
    print(torch.equal(image_1, image))
    output_1 = model_1(image_1)
    print("\n\n")
    print(output.shape , output_1.shape)
