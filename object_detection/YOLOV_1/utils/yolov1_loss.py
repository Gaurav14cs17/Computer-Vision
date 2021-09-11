import torch.nn as nn
import torch
from utils.utils import calculate_iou


class Yolo_loss(nn.Module):
    def __init__(self, S=7, B=2):
        super(Yolo_loss, self).__init__()
        self.S = S
        self.B = B

    def forward(self, preds, labels):
        '''
        :param preds: ( -1 , 7 , 7, 30)
        :param labels:
        :return:
        '''
        batch_size, S, S, _ = preds.shape
        reg_loss_xy = 0.
        reg_loss_wh = 0.
        have_object = 0.
        no_object = 0.
        loss_class = 0.
        for i in range(batch_size):
            for x in range(S):
                for y in range(S):
                    if labels[i, x, y, 4] == 1:
                        pred_bbox1 = torch.tensor([preds[i, x, y, 0], preds[i, x, y, 1], preds[i, x, y, 2], preds[i, x, y, 3]])
                        pred_bbox2 = torch.tensor([preds[i, x, y, 5], preds[i, x, y, 6], preds[i, x, y, 7], preds[i, x, y, 8]])
                        label_bbox = torch.Tensor([labels[i, y, x, 0], labels[i, x, y, 1], labels[i, x, y, 2], labels[i, x, y, 3]])
                        iou_1 = calculate_iou(pred_bbox1, label_bbox)
                        iou_2 = calculate_iou(pred_bbox2, label_bbox)
                        if iou_1 > iou_2:
                            reg_loss_xy += 5 * torch.sum((preds[i, x, y, 0:2] - labels[i, x, y, 0:2]) ** 2)
                            reg_loss_wh += torch.sum((preds[i, x, y, 2:4].sqrt() - labels[i, x, y, 2:4].sqrt()) ** 2)
                            have_object += (iou_1 - preds[i, x, y, 4]) ** 2
                            no_object += 0.5 * ((0 - preds[i, x, y, 9]) ** 2)
                        else:
                            reg_loss_xy += 5 * torch.sum((preds[i, x, y, 5:7] - labels[i, x, y, 5:7]) ** 2)
                            reg_loss_wh += torch.sum((preds[i, x, y, 7:9].sqrt() - labels[i, x, y, 7:9].sqrt()) ** 2)
                            have_object += (iou_2 - preds[i, x, y, 9]) ** 2
                            no_object += 0.5 * ((0 - preds[i, x, y, 4]) ** 2)
                        loss_class += torch.sum((labels[i, x, y, 10:] - preds[i, x, y, 10:]) ** 2)

                    else:
                        no_object += 0.5 * torch.sum((0 - preds[i, x, y, [4, 9]]) ** 2)

        loss = reg_loss_xy + reg_loss_wh + have_object + no_object + loss_class
        return loss / batch_size





if __name__ == '__main__':
    pred = torch.randn((4, 7, 7, 30))
    label = torch.randn((4, 7, 7, 30))
    loss = Yolo_loss()
    output = loss(pred, label)

    print("---------------------")
    print(output)

