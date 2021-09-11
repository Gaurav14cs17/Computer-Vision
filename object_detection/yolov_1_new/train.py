import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torchvision.transforms.functional as FT
from tqdm import tqdm
from torch.utils.data import DataLoader
from model.model import *
from data_loader.datasets import Custom_DataLoader
from utils.utils import *
from utils.yolov1_loss import YoloLoss

seed = 123
torch.manual_seed(seed)

# Hyperparameters etc.
LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available else "cpu"
BATCH_SIZE = 8
WEIGHT_DECAY = 0
EPOCHS = 1000
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"

IMG_DIR = "D:/labs/object_detection_model/Git_repo/data_train/Data_set/images/images/"
LABEL_DIR = "D:/labs/object_detection_model/Git_repo/data_train/Data_set/df.csv"

input_size = 448
transform = transforms.Compose([
    transforms.Resize((input_size, input_size)),
    transforms.ToTensor()])


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []
    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update progress bar
        loop.set_postfix(loss=loss.item())
    print(f"Mean loss was {sum(mean_loss) / len(mean_loss)}")


def start_training():
    model = YOLOv1().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    loss_fn = YoloLoss()

    if LOAD_MODEL:
        load_checkpoint(torch.load(LOAD_MODEL_FILE), model, optimizer)

    train_dataset = Custom_DataLoader(IMG_DIR, LABEL_DIR, transform=transform)
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
        drop_last=True,
    )

    for epoch in range(EPOCHS):
        # pred_boxes, target_boxes = get_bboxes(
        #     train_loader, model, iou_threshold=0.5, threshold=0.4
        # )
        train_fn(train_loader, model, optimizer, loss_fn)


if __name__ == '__main__':
    start_training()
