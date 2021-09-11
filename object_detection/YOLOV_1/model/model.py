import torch
import torch.nn as nn

architecture_config = [
    (7, 64, 2, 3),
    "M",
    (3, 192, 1, 1),
    "M",
    (1, 128, 1, 0),
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],
    (3, 1024, 1, 1),
    (3, 1024, 2, 1),
    (3, 1024, 1, 1),
    (3, 1024, 1, 1),
]


class Conv_Layer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, use_batch_norm=False, **kwargs):
        super(Conv_Layer, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, **kwargs)
        if use_batch_norm :
            self.batch_norm = nn.BatchNorm2d(num_features=out_channels)
        self.activation_layer = nn.ReLU(True)
        self.use_batch_norm = use_batch_norm

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.batch_norm(x)
        x = self.activation_layer(x)
        return x


class Darknet(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Darknet, self).__init__()
        self.arch_config = architecture_config
        self.in_channels = in_channels
        self.creat_layer = self._create_conv_layers()

    def forward(self, x):
        x = self.creat_layer(x)
        return x

    def _create_conv_layers(self):
        layers = nn.ModuleList()
        in_channels = self.in_channels
        for x in self.arch_config:
            if isinstance(x, tuple):
                output_channels = x[1]
                kernel_size = x[0]
                stride = x[2]
                padding = x[3]
                layers.append(Conv_Layer(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_size,stride=stride, padding=padding))
                in_channels = output_channels
            elif isinstance(x, str):
                layers.append(nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2)))
            elif isinstance(x, list):
                repeats_block = x[2]
                conv_1 = x[0]
                conv_2 = x[1]
                for _ in range(repeats_block):
                    output_channels = conv_1[1]
                    kernel_size = conv_1[0]
                    stride = conv_1[2]
                    padding = conv_1[3]
                    layers.append(
                        Conv_Layer(in_channels=in_channels, out_channels=output_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding))
                    output_channels = conv_2[1]
                    kernel_size = conv_2[0]
                    stride = conv_2[2]
                    padding = conv_2[3]
                    layers.append(
                        Conv_Layer(in_channels=conv_1[1], out_channels=output_channels, kernel_size=kernel_size,
                                   stride=stride, padding=padding))
                    in_channels = conv_2[1]
        return nn.Sequential(*layers)


class FC_layer(nn.Module):
    def __init__(self, number_of_classes, **kwargs):
        super(FC_layer, self).__init__()
        self.number_of_classes = number_of_classes
        self.S = kwargs['S']
        self.number_of_bbox = kwargs['number_of_bbox']
        self.in_channels = kwargs['in_channels']
        self.out_channels = kwargs['out_channels']
        self.last_channel_output = (self.S * self.S * (self.number_of_classes + 5 * self.number_of_bbox))
        self.flatten_layer = nn.Flatten()
        self.fc_1 = nn.Linear(self.in_channels * self.S * self.S, self.out_channels)
        self.dropout_layer = nn.Dropout(0.1)
        self.activation_layer = nn.LeakyReLU(0.1)
        self.last_layer = nn.Linear(self.out_channels, self.last_channel_output)
        self.activation_layer_1 = nn.Sigmoid()

    def forward(self, x):
        x = self.flatten_layer(x)
        x = self.fc_1(x)
        x = self.dropout_layer(x)
        x = self.activation_layer(x)
        x = self.last_layer(x)
        x = self.activation_layer_1(x)
        return x


class YOLOv1(nn.Module):
    def __init__(self ,S  = 7 ,  B = 2 , number_of_classes = 9  ):
        super(YOLOv1 , self ).__init__()
        self.number_of_bbox = B
        self.number_of_classes = number_of_classes
        self.darknet_layer = Darknet(in_channels = 3 )

        out_channels = 1024 # -1x1024xx7x7
        self.S = S
        second_out_channels = 496
        self.yolo_head = FC_layer(number_of_classes, S=self.S, number_of_bbox=self.number_of_bbox, in_channels=out_channels,out_channels=second_out_channels)

    def forward(self , x ):
        x = self.darknet_layer(x) # (-1 , 1024 , 7, 7)
        x = self.yolo_head(x) # ( -1 , 4 , 1470 )
        #  out = out.reshape(-1, self.S, self.S, self.B * 5 + self.num_classes)
        x = x.reshape( -1 , self.S , self.S , self.number_of_bbox*5 + self.number_of_classes ) #( -1 , 7,7,30)
        return x




if __name__ == '__main__':
    obj_Darknet = Darknet()
    image = torch.randn((4, 3, 448, 448))
    output = obj_Darknet(image)
    print(output.shape)

    number_of_bbox = 2
    number_of_classes = 20
    in_channels = 1024
    out_channels = 496
    S = 7
    last_layer_out = FC_layer(number_of_classes, S=S, number_of_bbox=number_of_bbox, in_channels=in_channels,out_channels=out_channels)
    x = last_layer_out(output)
    print(x.shape)

    yolo_object = YOLOv1()
    print(yolo_object)
    output = yolo_object(image)
    print(output.shape)
    obj = YOLOv1(S , number_of_bbox , number_of_classes)
    output = obj(image)
    print(output.shape)