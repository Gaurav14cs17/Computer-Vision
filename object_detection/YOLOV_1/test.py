import os

import cv2
import torch
import torch.nn as nn
import random
from torchvision import transforms
from PIL import Image
from torchvision.transforms import transforms

from model.model import YOLOv1
from utils.utils import nms


def decode_output(pred, S, number_classes, conf_thresh=0.1, iou_thresh=0.1):
    bboxs = torch.zeros((S * S, (5 + number_classes)))
    for x in range(S):
        for y in range(S):
            conf_1 = pred[x, y, 4]
            conf_2 = pred[x, y, 9]
            if conf_1 > conf_2:
                bboxs[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 0], pred[x, y, 1], pred[x, y, 2], pred[x, y, 3]])
                bboxs[(x * S + y), 4] = pred[x, y, 4]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]
            else:
                bboxs[(x * S + y), 0:4] = torch.Tensor([pred[x, y, 5], pred[x, y, 6], pred[x, y, 7], pred[x, y, 8]])
                bboxs[(x * S + y), 4] = pred[x, y, 9]
                bboxs[(x * S + y), 5:] = pred[x, y, 10:]


    xywhcc = nms(bboxs, number_classes, conf_thresh, iou_thresh)
    return xywhcc


COLORS = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]


def draw_bbox(img, bboxs, class_names):
    h, w = img.shape[0:2]
    n = bboxs.size()[0]
    bboxs = bboxs.detach().numpy()
    print(bboxs)
    for i in range(n):
        p1 = (int((bboxs[i, 0] - bboxs[i, 2] / 2) * w), int((bboxs[i, 1] - bboxs[i, 3] / 2) * h))
        p2 = (int((bboxs[i, 0] + bboxs[i, 2] / 2) * w), int((bboxs[i, 1] + bboxs[i, 3] / 2) * h))
        class_name = class_names[int(bboxs[i, 5])]
        cv2.rectangle(img, p1, p2, color=COLORS[int(bboxs[i, 5])], thickness=2)
        cv2.putText(img, class_name, p1, cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[int(bboxs[i, 5])])
    return img


def predict_img(img, model, input_size, S=7, num_classes=9, conf_thresh=0.10, iou_thresh=0.1):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pred_img = Image.fromarray(img).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])
    pred_img = transform(pred_img)
    pred_img.unsqueeze_(0)
    pred = model(pred_img)[0].detach().cpu()
    xywhcc = decode_output(pred, S, num_classes, conf_thresh, iou_thresh)
    return xywhcc


if __name__ == '__main__':
    num_classes = 9
    output = "./output_1/"
    weight_path = "/var/data/temp_exp/Computer-Vision/object_detection/YOLOV_1/temp/yolov1-pytorch/output/2021-08-19-14-44-46/epoch0.pth"
    source = "/var/data/vehicle_detection_data/anpr_dec_data/Process_data/tiny_yolov4_lpr_detection_data/valid/1593369335.972314.jpg"
    class_names = ['1' ,'2' ,'3' ,'4' ,'5' ,'6','7','8','9']
    input_size = 448
    S = 7

    model = YOLOv1(number_of_classes=num_classes)
    model.load_state_dict(torch.load(weight_path))
    print('Model loaded successfully!')
    if not os.path.exists(output):
        os.makedirs(output)
    img = cv2.imread(source)
    img_name = os.path.basename(source)
    xywhcc = predict_img(img.copy(), model, input_size, S,  num_classes)
    print(xywhcc)
    if xywhcc.size()[0] != 0:
        img = draw_bbox(img, xywhcc, class_names)
        cv2.imwrite(os.path.join(output, img_name), img)
