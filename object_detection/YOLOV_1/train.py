import torch
import torch.nn as nn

import sys
sys.path.append("D:/labs/object_detection_model/Git_repo/Computer-Vision/object_detection/YOLOV_1/")

from model.model import YOLOv1
from data_loader.datasets import *
from utils.yolov1_loss import *
import matplotlib.pyplot as plt
import os
import time, argparse
from torch.optim import SGD
from torchvision import utils

output_path = "./output/"
yolo_loss = Yolo_loss()


def Train_model(model, train_loader, optimizer, epoch, device, train_loss_lst=None):
    if train_loss_lst is None:
        train_loss_lst = []
    model.train()
    train_loss = 0

    for batch_idx, (inputs, labels) in enumerate(train_loader):
        train_time = time.time()
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = yolo_loss(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        t_batch = time.time() - train_time
        if batch_idx == 0 and epoch == 0:
            fig = plt.figure()
            inputs = inputs.cpu()
            grid = utils.make_grid(inputs)
            plt.imshow(grid.numpy().transpose((1, 2, 0)))
            plt.savefig(os.path.join(output_path, 'batch0.png'))
            # plt.show()
            plt.close(fig)

        if batch_idx % 10 == 0:
            print('Train Epoch: {} [{}/{} ({:.1f}%)]  Time: {:.4f}s  Loss: {:.6f}'
                  .format(epoch, batch_idx * len(inputs), len(train_loader.dataset),
                          100. * batch_idx / len(train_loader), t_batch, loss.item()))

    train_loss = train_loss / len(train_loader)
    #train_loss_lst.append(train_loss)
    return train_loss


def validate(model, val_loader, device, val_loss_lst):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = yolo_loss(output, target)
            val_loss += loss.item()
    val_loss = val_loss / len(val_loader)
    print('Val set: Average loss: {:.4f}\n'.format(val_loss))
    val_loss_lst.append(val_loss)
    return val_loss_lst


def test(model, test_loader, device):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, targets in test_loader:
            data = data.to(device)
            targets = targets.to(device)
            outputs = model(data)
            loss = yolo_loss(outputs, targets)
            test_loss += loss.item()
    test_loss = test_loss / len(test_loader)
    print('Test set: Average loss: {:.4f}'.format(test_loss))


if __name__ == '__main__':
    Data_path  = "/var/data/vehicle_detection_data/anpr_dec_data/Process_data/tiny_yolov4_lpr_detection_data/valid/"
    batch_size  = 32
    input_size =  448
    S, B, num_classes = 7 , 2 , 9
    epochs = 100
    lr = 0.0001
    save_freq  = 2
    try:
        os.makedirs(output_path)
    except Exception as e :
        print("all Ready exit this file ")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = YOLOv1(number_of_classes=num_classes)
    model.to(device)
    train_loader, val_loader, test_loader = create_dataloader(Data_path, 0.8, 0.1, 0.1, batch_size,input_size, S, B, num_classes)
    print(len(train_loader) ,len( val_loader), len(test_loader))

    optimizer = SGD(model.parameters(), lr= lr, momentum=0.9, weight_decay=0.0005)
    # optimizer = Adam(model.parameters(), lr=lr)

    train_loss_lst, val_loss_lst = [], []

    # train epoch
    for epoch in range(epochs):
        #Train_model(model, train_loader, optimizer, epoch, device, train_loss_lst=[])
        train_loss_lst = Train_model(model, train_loader, optimizer, epoch, device,  train_loss_lst)
        val_loss_lst = validate(model, val_loader, device,  val_loss_lst)
        # save model weight every save_freq epoch
        if epoch % save_freq == 0 :
            torch.save(model.state_dict(), os.path.join(output_path, 'epoch' + str(epoch) + '.pth'))

    test(model, test_loader, device )

    # save model
    torch.save(model.state_dict(), os.path.join(output_path, 'last.pth'))
    # plot loss, save params change
    fig = plt.figure()
    plt.plot(range(epochs), train_loss_lst, 'g', label='train loss')
    plt.plot(range(epochs), val_loss_lst, 'k', label='val loss')
    plt.grid(True)
    plt.xlabel('epoch')
    plt.ylabel('acc-loss')
    plt.legend(loc="upper right")
    plt.savefig(os.path.join(output_path, 'loss_curve.jpg'))
    plt.show()
    plt.close(fig)