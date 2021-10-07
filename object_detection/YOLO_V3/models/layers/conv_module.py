import torch
import torch.nn as nn
import torch.nn.functional as F
from .activate import *

norm_name = {
    "bn": nn.BatchNorm2d
}

activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "mish": Mish,
    "swish": Swish
}


class Convolutional(nn.Module):
    def __init__(self, filter_in, filter_out, kernel_size, stride, padding, batch_norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.batch_norm = batch_norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filter_in, out_channels=filter_out, kernel_size=kernel_size, stride=stride,
                                padding=padding, bias=not batch_norm)
        if batch_norm:
            assert batch_norm in norm_name.keys(), "Norm not define in given list"
            if batch_norm == "bn":
                self.__norm = norm_name[batch_norm](num_features=filter_out)
        if activate:
            assert activate in activate_name.keys(), "activate name is not define"
            if activate == 'leaky':
                self.__activate = nn.LeakyReLU(negative_slope=0.1, inplace=True)
            if activate == 'relu':
                self.__activate = nn.ReLU(inplace=True)
            if activate == "mish":
                self.__activate = Mish()
            if activate == "swish":
                self.__activate = Swish()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x
