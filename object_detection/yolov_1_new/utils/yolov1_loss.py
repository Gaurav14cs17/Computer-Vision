import torch
import torch.nn as nn
from .utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, c=2):
        super(YoloLoss, self).__init__()
        self.S = S
        self.B = B
        self.classes = c
        self.lambda_noobj = 0.5
        self.lambda_coord = 5
        self.mse = nn.MSELoss(reduction="sum")

        self.layer_reshape = (-1, self.S, self.S, (self.classes + 5 * self.B))

    def forward(self, predictions, targets):
        predictions = predictions.reshape(self.layer_reshape)

        pred_bbox_1 = predictions[..., 3:7]  # ( -1  , s , s , 4 )
        pred_bbox_2 = predictions[..., 8:12]
        target_bbox = targets[..., 3:7]

        iou_b1 = intersection_over_union(pred_bbox_1, target_bbox)  # (-1, s , s , 1)
        iou_b2 = intersection_over_union(pred_bbox_2, target_bbox)  # (-1, s ,s , 1)
        iou_b1 = iou_b1.unsqueeze(0)  # (1 , -1  , s , s , 1)
        iou_b2 = iou_b2.unsqueeze(0)

        ious = torch.cat([iou_b1, iou_b2], dim=0)  # (2 , -1 , s ,s , 1)
        ious_mx, best_bbox = torch.max(ious, dim=0)  # _ , ( -1 , s , s , 1)

        exists_bbox = targets[..., 2]  # (-1 , s , s  )
        exists_bbox = exists_bbox.unsqueeze(3)  # ( -1 , S , S , 1 )

        bbox_predictions = exists_bbox * (best_bbox * pred_bbox_1 + (1 - best_bbox) * pred_bbox_2)
        bbox_predictions[..., 2:4] = torch.sign(bbox_predictions[..., 2:4]) * torch.sqrt(torch.abs(bbox_predictions[..., 2:4] + 1e-6))
        target_bbox = exists_bbox * target_bbox

        target_bbox[..., 2:4] = torch.sqrt(target_bbox[..., 2:4])  # ( -1 , s ,s ,4  )
        bbox_loss = self.mse(torch.flatten(bbox_predictions, end_dim=-2),
                             torch.flatten(target_bbox, end_dim=-2))

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #

        # print(predictions[..., 2:3].size() , predictions[..., 2].size())
        # output --> torch.Size([4, 7, 7, 1]) torch.Size([4, 7, 7])

        pred_bbox_object = exists_bbox * (best_bbox * predictions[..., 2:3] + (1 - best_bbox) * predictions[..., 7:8])
        target_bbox_object = exists_bbox * (targets[..., 2:3])

        object_loss = self.mse(torch.flatten(pred_bbox_object),
                               torch.flatten(target_bbox_object))

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #

        pred_bbox1_no_object = (1 - exists_bbox) * predictions[..., 2:3]
        pred_bbox2_no_object = (1 - exists_bbox) * predictions[..., 7:8]
        target_bbox_noobject = (1 - exists_bbox) * targets[..., 2:3]

        pred_bbox1_no_object = torch.flatten(pred_bbox1_no_object, start_dim=1)
        pred_bbox2_no_object = torch.flatten(pred_bbox2_no_object, start_dim=1)
        target_bbox_noobject = torch.flatten(target_bbox_noobject, start_dim=1)

        no_object_loss = self.mse(pred_bbox1_no_object, target_bbox_noobject)
        no_object_loss += self.mse(pred_bbox2_no_object, target_bbox_noobject)

        # ================== #
        #   FOR CLASS LOSS   #
        # ================== #
        pred_class = exists_bbox * predictions[..., :2]  # (-1 , s , s , 2)
        target_class = exists_bbox * targets[..., :2]

        pred_class = torch.flatten(pred_class, end_dim=-2)
        target_class = torch.flatten(target_class, end_dim=-2)

        class_loss = self.mse(pred_class, target_class)

        #print(bbox_loss, object_loss, no_object_loss, class_loss)
        loss = (
                self.lambda_coord * bbox_loss  # first two rows in paper
                + object_loss  # third row in paper
                + self.lambda_noobj * no_object_loss  # forth row
                + class_loss  # fifth row
        )

        return loss


if __name__ == '__main__':
    object_loss_yolo = YoloLoss()
    pred = torch.rand((4, 7, 7, 12))
    target = torch.rand((4, 7, 7, 7))
    print(pred.size())
    print(target.size())
    loss = object_loss_yolo(pred, target)
    print(loss)
