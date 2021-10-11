
import torch.nn as nn
from ..layers.conv_module import Convolutional

class Residual_block(nn.Module):
    def __init__(self , filters_in = None  , filters_out = None  , filters_medium =  None ):
        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in , filters_out=filters_medium , kernel_size=1 , stride=1 , padding=0, batch_norm="bn" , activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_medium , filters_out=filters_out , kernel_size=3 , stride=1, padding=1, batch_norm="bn" , activate="leaky")

    def forward(self , x):
        '''
        x - > ( -1 , c , n , m )

        '''
        r = self.__conv1(x)
        r = self.__conv2(r)
        return x + r
