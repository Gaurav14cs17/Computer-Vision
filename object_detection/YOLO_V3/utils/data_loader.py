import os
import sys

sys.path.append("..")
sys.path.append("./utils")
sys.path.append("D:/labs/object_detection_model/Git_repo/Computer-Vision/object_detection/YOLO_V3/utils/")
import torch
from torch.utils.data import Dataset, DataLoader
import config.yolov3_config_voc as cfg
import cv2
import numpy as np
import random
import utils.data_augment as dataAug
import utils.tools as tools


class VocDataset(Dataset):
    def __init__(self, anno_file_type="train", img_size=416):
        self.img_size = img_size
        self.classes = cfg.DATA['CLASSES']
        self.num_classes = len(self.classes)
        self.classes_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)

    def __load_annotations(self, anno_type):
        assert anno_type in ['train', 'test'], "You must choice one of the 'train' or 'test' for anno_type parameter"
        anno_path = os.path.join(cfg.PROJECT_PATH, 'data', anno_type + "_annotation.txt")
        with open(anno_path, 'r') as f:
            x = f.readlines()
            annotations = list(filter(lambda n: len(n) > 0, x))
            # print(annotations)
        assert len(annotations) > 0, "NO image found {}".format(anno_path)
        return annotations

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        # cv2 image to PIL image
        # H W C --> C H W
        img_org = img_org.transpose(2, 0, 1)

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = dataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix

        label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes = self.__creat_label(bboxes)
        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        sbboxes = torch.from_numpy(sbboxes).float()
        mbboxes = torch.from_numpy(mbboxes).float()
        lbboxes = torch.from_numpy(lbboxes).float()
        return img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')
        image_path = anno[0]
        image = cv2.imread(image_path)
        assert image is not None, "File not found {}".format(image_path)
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        img, bboxes = dataAug.RandomHorizontalFilp()(np.copy(image), np.copy(bboxes))
        img, bboxes = dataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = dataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def __creat_label(self, bboxes):
        '''
        {"ANCHORS":[[(1.25, 1.625), (2.0, 3.75), (4.125, 2.875)],  # Anchors for small obj
            [(1.875, 3.8125), (3.875, 2.8125), (3.6875, 7.4375)],  # Anchors for medium obj
            [(3.625, 2.8125), (4.875, 6.1875), (11.65625, 10.1875)]] ,# Anchors for big obj
         "STRIDES":[8, 16, 32],
         "ANCHORS_PER_SCLAE":3
         }
         train_output_size = [52. 26. 13.]
        '''
        anchors = np.array(cfg.MODEL["ANCHORS"])
        strides = np.array(cfg.MODEL["STRIDES"])
        train_output_size = self.img_size / strides
        anchors_per_scale = cfg.MODEL["ANCHORS_PER_SCLAE"]

        label = [np.zeros(
            (int(train_output_size[i]), int(train_output_size[i]), int(anchors_per_scale), 6 + self.num_classes)) for i
            in range(3)]

        '''
        for i in range(3):
            print(label[i].shape)
        o/p:
           --> (52, 52, 3, 26)
           --> (26, 26, 3, 26)
           --> (13, 13, 3, 26)
        '''

        for i in range(3):
            label[i][..., 5] = 1.0

        '''
        Testing code 
        '''
        bboxes_xywh = [np.zeros((150, 4)) for _ in range(3)]  # Darknet the max_num is 30
        bbox_count = np.zeros((3,))

        for box in bboxes:
            box_coord = box[:4]
            classes_idx = int(box[4])
            bbox_mix = box[5]
            box_xywh = np.concatenate([(box_coord[2:] + box_coord[:2]) * 0.5, (box_coord[2:] - box_coord[:2])], axis=-1)

            # onehot
            one_hot = np.zeros(self.num_classes, dtype=np.float32)
            one_hot[classes_idx] = 1.0
            one_hot_smooth = dataAug.LabelSmooth()(one_hot, self.num_classes)

            iou_list = []
            check_matched = False
            bbox_xywh_scaled = 1.0 * box_xywh[np.newaxis, :]  # (1 , 4)
            '''
            # strides --> (3 , )
            # print(strides[: , np.newaxis].shape , strides[: , np.newaxis])
            '''
            bbox_xywh_scaled = bbox_xywh_scaled / strides[:, np.newaxis]
            #print(bbox_xywh_scaled)
            '''
            strides --> ( 3,1)
            bbox_xywh_scaled -->(3,4)
            '''
            for i in range(3):
                annchors_xywh = np.zeros((anchors_per_scale, 4))
                annchors_xywh[:, 0: 2] = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32) + 0.5
                annchors_xywh[:, 2:4] = anchors[i]
                scaled_iou = tools.iou_xywh_numpy(bbox_xywh_scaled[i][np.newaxis, :], annchors_xywh)
                iou_list.append(scaled_iou)
                iou_mask = scaled_iou > 0.3
                if np.any(iou_mask):
                    xind, yind = np.floor(bbox_xywh_scaled[i, 0:2]).astype(np.int32)
                    label[i][yind, xind, iou_mask, 0:4] = box_xywh
                    label[i][yind, xind, iou_mask, 4:5] = 1.0
                    label[i][yind, xind, iou_mask, 5:6] = bbox_mix
                    label[i][yind, xind, iou_mask, 6:] = one_hot_smooth
                    check_matched = True

                    '''
                    Testing code 
                    '''
                    bbox_ind = int(bbox_count[i] % 150)  # BUG : 150
                    bboxes_xywh[i][bbox_ind, :4] = box_xywh
                    bbox_count[i] += 1

            if not check_matched:
                max_iou_idx = np.argmax(np.array(iou_list).reshape(-1), axis=-1)
                best_detect = int(max_iou_idx / anchors_per_scale)
                best_anchor = int(max_iou_idx % anchors_per_scale)
                cx, cy = np.floor(bbox_xywh_scaled[best_detect, 0:2]).astype(np.int32)
                label[best_detect][cx, cy, best_anchor, 0:4] = box_xywh
                label[best_detect][cx, cy, best_anchor, 4:5] = 1.0
                label[best_detect][cx, cy, best_anchor, 5:6] = bbox_mix
                label[best_detect][cx, cy, best_anchor, 6:] = one_hot_smooth

                '''
                Testing code 
                '''
                bbox_ind = int(bbox_count[best_detect] % 150)
                bboxes_xywh[best_detect][bbox_ind, :4] = box_xywh
                bbox_count[best_detect] += 1

        label_sbbox, label_mbbox, label_lbbox = label
        sbboxes, mbboxes, lbboxes = bboxes_xywh
        return label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes


if __name__ == '__main__':
    voc_dataset = VocDataset(anno_file_type="train", img_size=448)
    dataloader = DataLoader(voc_dataset, shuffle=True, batch_size=1, num_workers=0)
    for i, (img, label_sbbox, label_mbbox, label_lbbox, sbboxes, mbboxes, lbboxes) in enumerate(dataloader):
        if i <= 10:
            print(img.shape, '\n')
            print(label_sbbox.shape)
            print(label_mbbox.shape)
            print(label_lbbox.shape, '\n')
            small_label = label_sbbox.reshape(-1, 26)
            medium_label = label_mbbox.reshape(-1, 26)
            large_label = label_lbbox.reshape(-1, 26)
            new_labels = np.concatenate([small_label, medium_label, large_label], axis=0)
            mask_label = new_labels[..., 4] > 0  # index 4 where is object
            object_label = new_labels[mask_label]
            bbox_labels = object_label[..., :4]  # ( n, 4 )
            print(bbox_labels)
            classes_labels = object_label[..., 6:]  # (n,20)
            classes_labels = np.argmax(classes_labels, axis=-1)  # (n)
            classes_labels = classes_labels.reshape(-1 , 1 ) #( n , 1)
            new_label_bbox_classes = np.concatenate([bbox_labels , classes_labels], axis=-1) #( n , 5 )
            tools.plot_box(new_label_bbox_classes, img, id=1)


