import numpy as np
import torch
import yaml


def convert_to_yolo_labels(bbox, S, B, number_of_classes=20):
    label = torch.zeros((S, S, (5 * B + number_of_classes)))
    for box in bbox:
        x, y, w, h, c = box
        x_idx = int(x * S)
        y_idx = int(y * S)

        x_cell = (x * S) - int(x * S)
        y_cell = (y * S) - int(y * S)
        w, h = w * S, h * S
        label[x_idx, y_idx, 0: 5] = torch.tensor([x_cell, y_cell, w, h, 1])
        label[x_idx, y_idx, 5: 10] = torch.tensor([x_cell, y_cell, w, h, 1])
        label[x_idx, y_idx, 10 + c] = 1
    return label


import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from collections import Counter


def intersection_over_union(boxes_preds, boxes_labels, box_format="midpoint"):
    """
    Calculates intersection over union

    Parameters:
        boxes_preds (tensor): Predictions of Bounding Boxes (BATCH_SIZE, 4)
        boxes_labels (tensor): Correct labels of Bounding Boxes (BATCH_SIZE, 4)
        box_format (str): midpoint/corners, if boxes (x,y,w,h) or (x1,y1,x2,y2)

    Returns:
        tensor: Intersection over union for all examples
    """

    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    if box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  # (N, 1)
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # .clamp(0) is for the case when they do not intersect
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))

    return intersection / (box1_area + box2_area - intersection + 1e-6)


def non_max_suppression(bboxes, iou_threshold, threshold, box_format="corners"):
    """
    Does Non Max Suppression given bboxes

    Parameters:
        bboxes (list): list of lists containing all bboxes with each bboxes
        specified as [class_pred, prob_score, x1, y1, x2, y2]
        iou_threshold (float): threshold where predicted bboxes is correct
        threshold (float): threshold to remove predicted bboxes (independent of IoU)
        box_format (str): "midpoint" or "corners" used to specify bboxes

    Returns:
        list: bboxes after performing NMS given a specific IoU threshold
    """

    assert type(bboxes) == list

    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
               or intersection_over_union(
                torch.tensor(chosen_box[2:]),
                torch.tensor(box[2:]),
                box_format=box_format,
            )
               < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold
    """

    # list storing all AP for respective classes
    average_precisions = []

    # used for numerical stability later on
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truths = []

        # Go through all predictions and targets,
        # and only add the ones that belong to the
        # current class c
        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        # find the amount of bboxes for each training example
        # Counter here finds how many ground truth bboxes we get
        # for each training example, so let's say img 0 has 3,
        # img 1 has 5 then we will obtain a dictionary with:
        # amount_bboxes = {0:3, 1:5}
        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        # We then go through each key, val in this dictionary
        # and convert to the following (w.r.t same example):
        # ammount_bboxes = {0:torch.tensor[0,0,0], 1:torch.tensor[0,0,0,0,0]}
        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        # sort by box probabilities which is index 2
        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)

        # If none exists for this class then we can safely skip
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # Only take out the ground_truths that have the same
            # training idx as detection
            ground_truth_img = [
                bbox for bbox in ground_truths if bbox[0] == detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                # only detect ground truth detection once
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # true positive and add this bounding box to seen
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            # if IOU is lower then the detection is a false positive
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + epsilon))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        # torch.trapz for numerical integration
        average_precisions.append(torch.trapz(precisions, recalls))

    return sum(average_precisions) / len(average_precisions)


def plot_image(image, boxes):
    """Plots predicted bounding boxes on the image"""
    im = np.array(image)
    height, width, _ = im.shape

    # Create figure and axes
    fig, ax = plt.subplots(1)
    # Display the image
    ax.imshow(im)

    # box[0] is x midpoint, box[2] is width
    # box[1] is y midpoint, box[3] is height

    # Create a Rectangle potch
    for box in boxes:
        box = box[2:]
        assert len(box) == 4, "Got more values than in x, y, w, h, in a box!"
        upper_left_x = box[0] - box[2] / 2
        upper_left_y = box[1] - box[3] / 2
        rect = patches.Rectangle(
            (upper_left_x * width, upper_left_y * height),
            box[2] * width,
            box[3] * height,
            linewidth=1,
            edgecolor="r",
            facecolor="none",
        )
        # Add the patch to the Axes
        ax.add_patch(rect)

    plt.show()


def get_bboxes(
        loader,
        model,
        iou_threshold,
        threshold,
        pred_format="cells",
        box_format="midpoint",
        device="cuda",
):
    all_pred_boxes = []
    all_true_boxes = []

    # make sure model is in eval before get bboxes
    model.eval()
    train_idx = 0

    for batch_idx, (x, labels) in enumerate(loader):
        x = x.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            predictions = model(x)

        batch_size = x.shape[0]
        true_bboxes = cellboxes_to_boxes(labels)
        bboxes = cellboxes_to_boxes(predictions)

        for idx in range(batch_size):
            nms_boxes = non_max_suppression(
                bboxes[idx],
                iou_threshold=iou_threshold,
                threshold=threshold,
                box_format=box_format,
            )

            # if batch_idx == 0 and idx == 0:
            #    plot_image(x[idx].permute(1,2,0).to("cpu"), nms_boxes)
            #    print(nms_boxes)

            for nms_box in nms_boxes:
                all_pred_boxes.append([train_idx] + nms_box)

            for box in true_bboxes[idx]:
                # many will get converted to 0 pred
                if box[1] > threshold:
                    all_true_boxes.append([train_idx] + box)

            train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes


def convert_cellboxes(predictions, S=7):
    """
    Converts bounding boxes output from Yolo with
    an image split size of S into entire image ratios
    rather than relative to cell ratios. Tried to do this
    vectorized, but this resulted in quite difficult to read
    code... Use as a black box? Or implement a more intuitive,
    using 2 for loops iterating range(S) and convert them one
    by one, resulting in a slower but more readable implementation.
    """

    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    predictions = predictions.reshape(batch_size, 7, 7, 30)
    bboxes1 = predictions[..., 21:25]
    bboxes2 = predictions[..., 26:30]
    scores = torch.cat(
        (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
    )
    best_box = scores.argmax(0).unsqueeze(-1)
    best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
    cell_indices = torch.arange(7).repeat(batch_size, 7, 1).unsqueeze(-1)
    x = 1 / S * (best_boxes[..., :1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = 1 / S * best_boxes[..., 2:4]
    converted_bboxes = torch.cat((x, y, w_y), dim=-1)
    predicted_class = predictions[..., :20].argmax(-1).unsqueeze(-1)
    best_confidence = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(
        -1
    )
    converted_preds = torch.cat(
        (predicted_class, best_confidence, converted_bboxes), dim=-1
    )

    return converted_preds


def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []

        for bbox_idx in range(S * S):
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])




def pred2xywhcc(pred, S, B, num_classes, conf_thresh, iou_thresh):
    # pred is a 7*7*(5*B+C) tensor, default S=7 B=2

    # bboxs = torch.zeros((S * S * B, 5 + num_classes))  # 98*15
    # for x in range(S):
    #     for y in range(S):
    #         # bbox1
    #         # bboxs[B * (x * S + y), 0:4] = torch.Tensor(
    #         #     [(pred[x, y, 0] + x) / S, (pred[x, y, 1] + y) / S, pred[x, y, 2], pred[x, y, 3]])
    #         # bboxs[B * (x * S + y), 4] = pred[x, y, 4]
    #         # bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
    #         #
    #         # # bbox2
    #         # bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor(
    #         #     [(pred[x, y, 5] + x) / S, (pred[x, y, 6] + y) / S, pred[x, y, S], pred[x, y, 8]])
    #         # bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
    #         # bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]
    #
    #         # bbox1
    #         bboxs[B * (x * S + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
    #         bboxs[B * (x * S + y), 4] = pred[x, y, 4]
    #         bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
    #
    #         # bbox2
    #         bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
    #         bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
    #         bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]

    bboxs = torch.zeros((S * S, 5 + num_classes))  # 49*25
    for x in range(S):
        for y in range(S):
            # bbox1
            # bboxs[B * (x * S + y), 0:4] = torch.Tensor(
            #     [(pred[x, y, 0] + x) / S, (pred[x, y, 1] + y) / S, pred[x, y, 2], pred[x, y, 3]])
            # bboxs[B * (x * S + y), 4] = pred[x, y, 4]
            # bboxs[B * (x * S + y), 5:] = pred[x, y, 10:]
            #
            # # bbox2
            # bboxs[B * (x * S + y) + 1, 0:4] = torch.Tensor(
            #     [(pred[x, y, 5] + x) / S, (pred[x, y, 6] + y) / S, pred[x, y, S], pred[x, y, 8]])
            # bboxs[B * (x * S + y) + 1, 4] = pred[x, y, 9]
            # bboxs[B * (x * S + y) + 1, 5:] = pred[x, y, 10:]

            conf1, conf2 = pred[x, y, 4], pred[x, y, 9]
            if conf1 > conf2:
                # bbox1
                bboxs[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
                bboxs[(x * S + y), 4] = pred[x, y, 4]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]
            else:
                # bbox2
                bboxs[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
                bboxs[(x * S + y), 4] = pred[x, y, 9]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]

    # apply NMS to all bboxs
    xywhcc = nms(bboxs, num_classes, conf_thresh, iou_thresh)
    return xywhcc


def nms(bboxs, num_classes, conf_thresh=0.1, iou_thresh=0.3):
    # Non-Maximum Suppression, bboxs is a 98*15 tensor
    bbox_prob = bboxs[:, 5:].clone().detach()  # 98*10
    bbox_conf = bboxs[:, 4].clone().detach().unsqueeze(1).expand_as(bbox_prob)  # 98*10
    bbox_cls_spec_conf = bbox_conf * bbox_prob  # 98*10
    bbox_cls_spec_conf[bbox_cls_spec_conf <= conf_thresh] = 0

    # for each class, sort the cls-spec-conf score
    for c in range(num_classes):
        rank = torch.sort(bbox_cls_spec_conf[:, c], descending=True).indices  # sort conf
        # for each bbox
        for i in range(bboxs.shape[0]):
            if bbox_cls_spec_conf[rank[i], c] == 0:
                continue
            for j in range(i + 1, bboxs.shape[0]):
                if bbox_cls_spec_conf[rank[j], c] != 0:
                    iou = calculate_iou(bboxs[rank[i], 0:4], bboxs[rank[j], 0:4])
                    if iou > iou_thresh:
                        bbox_cls_spec_conf[rank[j], c] = 0

    # exclude cls-specific confidence score=0
    bboxs = bboxs[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    bbox_cls_spec_conf = bbox_cls_spec_conf[torch.max(bbox_cls_spec_conf, dim=1).values > 0]

    ret = torch.ones((bboxs.size()[0], 6))

    # return null
    if bboxs.size()[0] == 0:
        return torch.tensor([])

    # bbox coord
    ret[:, 0:4] = bboxs[:, 0:4]
    # bbox class-specific confidence scores
    ret[:, 4] = torch.max(bbox_cls_spec_conf, dim=1).values
    # bbox class
    ret[:, 5] = torch.argmax(bboxs[:, 5:], dim=1).int()
    return ret


def calculate_iou(bbox1, bbox2):
    # bbox: x y w h
    bbox1, bbox2 = bbox1.cpu().detach().numpy().tolist(), bbox2.cpu().detach().numpy().tolist()

    area1 = bbox1[2] * bbox1[3]  # bbox1's area
    area2 = bbox2[2] * bbox2[3]  # bbox2's area

    max_left = max(bbox1[0] - bbox1[2] / 2, bbox2[0] - bbox2[2] / 2)
    min_right = min(bbox1[0] + bbox1[2] / 2, bbox2[0] + bbox2[2] / 2)
    max_top = max(bbox1[1] - bbox1[3] / 2, bbox2[1] - bbox2[3] / 2)
    min_bottom = min(bbox1[1] + bbox1[3] / 2, bbox2[1] + bbox2[3] / 2)

    if max_left >= min_right or max_top >= min_bottom:
        return 0
    else:
        # iou = intersect / union
        intersect = (min_right - max_left) * (min_bottom - max_top)
        return intersect / (area1 + area2 - intersect)


def parse_cfg(cfg_path):
    # cfg = {}
    # with open(cfg_path, 'r') as f:
    #     lines = f.readlines()
    #     for line in lines:
    #         if line[0] == '#' or line == '\n':
    #             continue
    #         line = line.strip().split(':')
    #         key, value = line[0].strip(), line[1].strip()
    #         cfg[key] = value

    with open(cfg_path, 'r') as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)  # dict
    print('Config:', cfg)
    return cfg


