import os
import cv2
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from utils.utils import convert_to_yolo_labels


class YOLO_Dataset(Dataset):
    def __init__(self, Data_path, S, B, num_classes, transforms=None):
        self.Data_path = Data_path
        self.transforms = transforms
        self.filenames = list()
        for file_name in os.listdir( self.Data_path):
            if file_name.endswith('.txt'):
                self.filenames.append(file_name)

        self.filenames.sort()
        self.S = S
        self.B = B
        self.num_classes = num_classes
        self.s = set()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        file_name = self.filenames[idx]
        txt_file_name = file_name
        image_file_name = file_name.split('.txt')[0] + '.jpg'

        image_path = os.path.join(self.Data_path, image_file_name)
        txt_path = os.path.join(self.Data_path, txt_file_name)

        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert('RGB')
        if self.transforms is not None:
            img = self.transforms(img)

        bbox = []

        with open(txt_path, 'r') as f:
            lines = f.readlines()
            for line in lines:
                if line == "\n":
                    continue
                line = line.strip().split(' ')
                c , x, y, w, h  = int(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4])
                bbox.append((x, y, w, h, c))
        label = convert_to_yolo_labels(bbox, self.S, self.B, self.num_classes)
        label = torch.tensor(label)
        return img, label


def create_dataloader(Data_path, train_proportion, val_proportion, test_proportion, batch_size, input_size,
                      S, B, num_classes):
    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor()
    ])

    # create yolo dataset
    dataset = YOLO_Dataset(Data_path, S, B, num_classes, transforms=transform)

    dataset_size = len(dataset)
    train_size = int(dataset_size * train_proportion)
    val_size = int(dataset_size * val_proportion)
    # test_size = int(dataset_size * test_proportion)
    test_size = dataset_size - train_size - val_size

    # split dataset to train set, val set and test set three parts
    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

    # create data loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
