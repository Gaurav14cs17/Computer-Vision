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


class Feature_block(nn.Module):
    def __init__(self, feature_input_channels, out_channels, feature_output_channels):
        super(Feature_block, self).__init__()
        self.__in_channels = feature_input_channels
        self.__out_channels = out_channels
        self.cfg = [(1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1), (1, 1, 0), (3, 1, 1)]
        self.model_list = nn.ModuleList()
        self.out_layer = None
        flag = 0
        temp_layer = None
        for i in range(len(self.cfg)):
            x = Convolutional(self.__in_channels, self.__out_channels, kernel_size=self.cfg[i][0],
                              stride=self.cfg[i][1], padding=self.cfg[i][2], batch_norm='bn',
                              activate='leaky')
            if i == 5:
                temp_layer = x
            else:
                self.model_list.append(x)
            self.__in_channels = self.__out_channels
            if not flag:
                self.__out_channels = self.__out_channels * 2
            else:
                self.__out_channels = self.__out_channels // 2
            flag = flag ^ 1
        self.my_block = nn.Sequential(*self.model_list)
        self.out_layer = temp_layer
        self.last_layer = Convolutional(self.__in_channels, feature_output_channels, kernel_size=1,
                                        stride=1, padding=0, batch_norm='bn', activate='leaky')

    def forward(self, x):
        x_1 = self.my_block(x)
        x = self.out_layer(x_1)
        x = self.last_layer(x)
        return x_1, x


class FPN_YOLO_V3(nn.Module):
    """
         Feature pyramid network (FPN)
         FPN for yolov3, and is different from original FPN or retinanet' FPN.
    """

    def __init__(self, fileters_in, fileters_out):
        super(FPN_YOLO_V3, self).__init__()
        '''
        feature_input = [1024, 512, 256]
                         samll , med , large
        '''

        fi_0, fi_1, fi_2 = fileters_in
        fo_0, fo_1, fo_2 = fileters_out

        self.neck_1 = Feature_block(fi_0, 512, fo_0)

        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, padding=0,batch_norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()
        self.neck_2 = Feature_block((fi_1 + 256), 256, fo_1)

        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, padding=0,batch_norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()
        self.neck_3 = Feature_block((fi_2 + 128), 128, fo_2)

    def forward(self, x1, x2, x3):
        ro, large_output = self.neck_1(x1)

        c_0 = self.__conv0(ro)
        up_0 = self.__upsample0(c_0)
        merage_0 = self.__route0(x2, up_0)
        r1, medium_output = self.neck_2(merage_0)

        c_1 = self.__conv1(r1)
        up_0 = self.__upsample1(c_1)
        merage_1 = self.__route1(x3, up_0)
        r2, small_output = self.neck_3(merage_1)
        return small_output, medium_output, large_output


if __name__ == '__main__':
    from torchsummary import summary

    fileters_in = [1024, 512, 256]
    fileters_out = [10, 10, 10]
    FPN_YOLO_V3_obj = FPN_YOLO_V3(fileters_in, fileters_out)
    image_1 = torch.randn((1, 1024, 100, 100))
    image_2 = torch.randn((1, 512, 200, 200))
    image_3 = torch.randn((1, 256, 400, 400))
    output = FPN_YOLO_V3_obj(image_1, image_2, image_3)
    print(output[0].shape, output[1].shape, output[2].shape)
