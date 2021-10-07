
import torch.nn as nn
from ..layers.conv_module import Convolutional

class Residual_block(nn.Module):
    def __init__(self , in_filters , out_filters , medium_filters):
        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filter_in=in_filters , filter_out=medium_filters , kernel_size=1 , stride=1 , padding=0, batch_norm="bn" , activate="leaky")
        self.__conv2 = Convolutional(filter_in=medium_filters , filter_out=out_filters , kernel_size=3 , stride=1, padding=1, batch_norm="bn" , activate="leaky")

    def forward(self , x):
        '''
        x - > ( -1 , c , n , m )

        '''
        r = self.__conv1(x)
        r = self.__conv2(r)
        return x + r
