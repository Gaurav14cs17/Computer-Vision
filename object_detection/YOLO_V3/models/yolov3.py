import sys

sys.path.append('..')

import torch.nn as nn
from models.backbones.darknet53 import Darknet53
from models.necks.neck import FPN_YOLO_V3
from models.head.yolo_head import Yolo_head

from models.layers.conv_module import Convolutional
import config.yolov3_config_voc as cfg
import numpy as np
from utils.tools import *


class YoloV3(nn.Module):
    def __init__(self, init_weights=True):
        super(YoloV3, self).__init__()

        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5)

        self.__backbone = Darknet53()
        self.__fpn = FPN_YOLO_V3(fileters_in=[1024, 512, 256],
                                 fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0])
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1])
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        output = []
        x_s, x_m, x_l = self.__backbone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)
        output.append(self.__head_s(x_s))
        output.append(self.__head_m(x_m))
        output.append(self.__head_l(x_l))

        if self.training:
            p, p_d = list(zip(*output))

            return p, p_d
        else:
            p, p_d = list(zip(*output))
            return p, torch.cat(p_d, 0)

    def __init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
                print("initing {}".format(m))


if __name__ == '__main__':
    net = YoloV3()
    print(net)

    in_img = torch.randn(12, 3, 416, 416)
    p, p_d = net(in_img)

    for i in range(3):
        print(p[i].shape)
        print(p_d[i].shape)
        print("\n\n\n")

