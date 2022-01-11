import torch.nn as nn
import torch


class Yolo_head(nn.Module):
    def __init__(self, nC = None , anchors = None , stride =None ):
        super(Yolo_head, self).__init__()
        self.__anchors = anchors
        self.__nA = len(self.__anchors)
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        '''
        p = [-1 , 5 + self.__nC ,  x, y]
        '''
        batch_size, nG = p.shape[0], p.shape[-1]
        p = p.view(batch_size, self.__nA, 5 + self.__nC, nG, nG)
        # [ batch_size , 3 , 5 + classes , block_size , block_size]
        p = p.permute(0, 3, 4, 1, 2)
        # [ batch , block_size , block_size , block_size , 3 , 5+classes]
        p_decode = self.__decode(p.clone())
        return (p, p_decode)

    def __decode(self, p):
        '''
        p = [ batch , block_size , block_size , block_size , 3 , 5+classes]
        '''
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)

        conv_raw_dxdy = p[:, :, :, :, 0:2]
        # [ batch_size , blocksize , blocksize , 3 , 2]

        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_conf = p[:, :, :, :, 4:5]
        conv_raw_classes_prob = p[:, :, :, :, 5:]

        x = torch.arange(0, output_size)  # ( outsize)
        x = x.unsqueeze(1)  # ( outsize , 1)
        x = x.repeat(1, output_size)  # ( outsize , outsize)

        y = torch.arange(0, output_size)  # (outsize)
        y = y.unsqueeze(0)  # ( 1, outsize)
        y = y.repeat(output_size, 1)  # ( outsize , outsize)

        grid_xy = torch.stack([x, y], dim=-1)  # ( outsize , outsize , 2)
        '''
        convert grid_xy_shape ---->  conv_raw_dxdy_shape
        grid.shape (outsize , outsize , 2) --> [batch_size , blocksize , blocksize , 3 ,2]
        '''
        grid_xy = grid_xy.unsqueeze(0)  # ( -1 , outsize , outsize , 2 )
        grid_xy = grid_xy.unsqueeze(3)  # (-1 , outsize , outsize , -1 , 2)
        grid_xy = grid_xy.repeat(batch_size, 1, 1, 3, 1)  # ( batch_size , outsize , outsize , 3,2)
        grid_xy = grid_xy.float().to(device)

        pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride  # (batch_size , stride_size , stride_size , 3, 2)
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors) * stride  # (batch_size , stride_size , stride_size , 3, 2)
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)  # (batch_size , stride_size , stride_size , 3, 4)

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_classes_prob)
        pred_bbox = torch.cat([pred_xywh, pred_conf, pred_prob],
                              dim=-1)  # (batch_size , stride_size , stride_size , 3,  5 + classes )
        return pred_bbox.view(-1, 5 + self.__nC) if not self.training else pred_bbox
