import torch
import os , cv2
import pandas as pd
from PIL import Image


class Custom_DataLoader(torch.utils.data.Dataset):
    def __init__(self, images_path, csv_path, transform=None, s=7, B=2, c=2):
        self.images_path = images_path
        self.csv_path = csv_path
        self.df = pd.read_csv(self.csv_path)
        self.images_names = os.listdir(self.images_path)
        print(self.images_names[:10])
        self.S = s
        self.B = B
        self.classes = c
        self.transform = transform
        self.labels = {'Bus': 0, 'Truck': 1}

    def __len__(self):
        return len(self.images_names)

    def __getitem__(self, index):
        csv_data = self.df.iloc[index]
        image_name = csv_data['ImageID'] + '.jpg'
        label_name = csv_data['LabelName']

        # XMin XMax YMin YMax
        xmin, xmax, ymin, ymax = csv_data['XMin'], csv_data['XMax'], csv_data['YMin'], csv_data['YMax']
        image_path = os.path.join(self.images_path,  image_name)
        cv_image = cv2.imread(image_path)
        img_h,img_w ,_ = cv_image.shape
        xmin , xmax  = xmin*img_w , xmax*img_w
        ymin , ymax = ymin*img_h , ymax*img_h

        # cv2.rectangle(cv_image , (int(xmin) , int(ymin)) , (int(xmax) , int(ymax)) , (0 , 0 , 255) , 1)
        # cv2.imshow("Imae" , cv_image)
        # cv2.waitKey(1000)

        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)

        w, h = abs(xmax - xmin), abs(ymax - ymin)
        cx, cy = (xmax + xmin) // 2, (ymax + ymin) // 2

        cx , w = cx /img_w , w/img_w
        cy , h = cy/img_h , h/img_h

        label_idx = self.labels[label_name]
        have_object = 1

        bbox = [[label_idx, cx, cy, w, h]]
        #print(bbox)
        bbox = torch.tensor(bbox)

        label_matrix = torch.zeros((self.S, self.S, self.classes + 5 * self.B))
        for box in bbox:
            classes_idx, x, y, w, h = box.tolist()
            classes_idx = int(classes_idx)
            i, j = int(self.S * y), int(self.S * x)
            x_cell, y_cell = self.S * x - j, self.S * y - i
            width_cell, height_cell = w * self.S, h * self.S
            if label_matrix[i, j, 2] == 0:
                label_matrix[i, j, 2] = 1
                bbox_coordinates = torch.tensor([x_cell, y_cell, width_cell, height_cell])
                label_matrix[i, j, 3:7] = bbox_coordinates
                label_matrix[i, j, classes_idx] = 1
        return image, label_matrix


if __name__ == '__main__':
    IMG_DIR = "D:/labs/object_detection_model/Git_repo/data_train/Data_set/images/images/"
    LABEL_DIR = "D:/labs/object_detection_model/Git_repo/data_train/Data_set/df.csv"
    images_path = "D:/labs/object_detection_model/Git_repo/Computer-Vision/object_detection/yolov_1_new/Data_set/images"
    csv_path = "D:/labs/object_detection_model/Git_repo/Computer-Vision/object_detection/yolov_1_new/Data_set/df.csv"
    data_load = Custom_DataLoader(IMG_DIR, LABEL_DIR )
    for idx in range(data_load.__len__()):
        data_load.__getitem__(idx)
