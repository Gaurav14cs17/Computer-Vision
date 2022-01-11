
import torch.nn as nn
import torch
import torch.nn.functional as F
from operator import __add__
from functools import reduce
import torchvision
from torch.autograd import Variable
from torchvision.datasets import FashionMNIST
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.optim as optim
import numpy as np, os
import matplotlib.pyplot as plt


class Conv2d_With_SamePadding(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super(Conv2d_With_SamePadding, self).__init__(*args, **kwargs)
        self.zero_pad_2d = nn.ZeroPad2d(reduce(__add__,
                                               [(k // 2 + (k - 2 * (k // 2)) - 1, k // 2) for k in
                                                self.kernel_size[::-1]]))

    def forward(self, input):
        return self._conv_forward(self.zero_pad_2d(input), self.weight , self.bias)


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv_layer_1 = Conv2d_With_SamePadding(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=0,
                                                    bias=False)
        self.activation_layer_1 = nn.ReLU()

        self.conv_layer_2 = Conv2d_With_SamePadding(in_channels=8, out_channels=16, kernel_size=3, stride=1,
                                                    padding=0, bias=False)
        self.activation_layer_2 = nn.ReLU()

        self.conv_layer_3 = Conv2d_With_SamePadding(in_channels=16, out_channels=32, kernel_size=3, stride=1,
                                                    padding=0, bias=False)
        self.activation_layer_3 = nn.ReLU()

        self.conv_layer_4 = Conv2d_With_SamePadding(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                                                    padding=1, bias=False)
        self.activation_layer_4 = nn.ReLU()

        self.avgpool = nn.AdaptiveAvgPool2d((15, 15))
        self.fc = nn.Linear(3600, 64)
        self.last_layer = nn.Linear(64, 10)

    def forward(self, x):
        x = self.conv_layer_1(x)
        x = self.activation_layer_1(x)
        x = self.conv_layer_2(x)
        x = self.activation_layer_2(x)
        x = self.conv_layer_3(x)
        x = self.activation_layer_3(x)
        x = self.conv_layer_4(x)
        x = self.activation_layer_4(x)
        x = self.avgpool(x)
        # print(x.shape)
        x = torch.flatten(x, 1)
        # print(x.shape)
        # exit()
        x = F.relu(self.fc(x))
        x = F.relu(self.last_layer(x))
        return x
        # print(x.shape)


# if __name__ == '__main__':
#     obj = Model()
#     image = torch.randn((1 , 1, 28 , 28 ))
#     print(image.shape)
#     obj.forward(image)


data_transform = transforms.ToTensor()
train_data = FashionMNIST(root='./data', train=True, download=True, transform=data_transform)
test_data = FashionMNIST(root='./data', train=False, download=True, transform=data_transform)
print('Train data, number of images: ', len(train_data))
print('Test data, number of images: ', len(test_data))
batch_size = 1
n_epochs = 5


class Train_model:
    def __init__(self):
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.criterion_loss = nn.CrossEntropyLoss()
        self.training_loss = []
        self.model = Model()
        self.model_dir = 'saved_models/'
        self.model_name = 'model_1.pt'
        try:
            os.makedirs(self.model_dir)
        except Exception as e:
            print("All Ready exits")

        print(self.model)
        self.optimizer = optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self, n_epochs=50):
        self.training_loss.clear()

        for epoch in range(n_epochs):
            check = False
            running_loss = 0.0
            for batch_i, data in enumerate(self.train_loader):
                inputs, labels = data
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion_loss(outputs, labels)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item()
                if batch_i % 1000 == 0:
                    # print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i + 1, running_loss / 1000))
                    self.training_loss.append(running_loss / 1000)
                    self.plot_graph()
                    running_loss = 0.0
                    if not epoch:
                        check = True
                        plt.show()

        print('Finished Training')
        torch.save(self.model.state_dict(), self.model_dir + self.model_name)
        return self.training_loss

    def plot_graph(self):
        plt.plot(self.training_loss)
        plt.xlabel('k batches')
        plt.ylabel('average loss per batch')
        plt.title('evolution of average training loss per batch')
    # plt.show()


class Testing_and_inference:
    def __init__(self):
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.criterion_loss = nn.CrossEntropyLoss()
        self.model_dir = 'saved_models/'
        self.model_name = 'model_1.pt'


        self.model = Model()
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name)))
        print(self.model)

    def test_acc(self):
        test_loss = torch.zeros(1)
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        self.model.eval()
        for batch_i, data in enumerate(self.test_loader):
            inputs, labels = data
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion_loss(outputs, labels)
                test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
                _, predicted = torch.max(outputs.data, 1)
                correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
                for i in range(batch_size):
                    label = labels.data[i]
                    class_correct[label] += correct[i].item()
                    class_total[label] += 1
        print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]),
                    np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    Train_obj = Train_model()
    Train_obj.train(n_epochs)
    Train_obj.plot_graph()

    Test_obj = Testing_and_inference()
    Test_obj.test_acc()








class Testing_and_inference:
    def __init__(self):
        self.test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)
        self.classes = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
                        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        self.criterion_loss = nn.CrossEntropyLoss()
        self.model_dir = './saved_models/'
        self.model_name = 'model_1.pt'


        self.model = Model()
        self.model.load_state_dict(torch.load(os.path.join(self.model_dir, self.model_name)))
        print(self.model)

    def test_acc(self):
        test_loss = torch.zeros(1)
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        self.model.eval()
        for batch_i, data in enumerate(self.test_loader):
            inputs, labels = data
            with torch.no_grad():
                outputs = self.model(inputs)
                loss = self.criterion_loss(outputs, labels)
                test_loss = test_loss + ((torch.ones(1) / (batch_i + 1)) * (loss.data - test_loss))
                _, predicted = torch.max(outputs.data, 1)
                correct = np.squeeze(predicted.eq(labels.data.view_as(predicted)))
                try:
                    for i in range(batch_size):
                      print(labels.data)
                      label = labels.data[i]
                      class_correct[label] += correct[i].item()
                      class_total[label] += 1
                except Exception as e:
                    print(e)

        print('Test Loss: {:.6f}\n'.format(test_loss.numpy()[0]))
        for i in range(10):
            if class_total[i] > 0:
                print('Test Accuracy of %5s: %2d%% (%2d/%2d)' % (
                    self.classes[i], 100 * class_correct[i] / class_total[i], np.sum(class_correct[i]),
                    np.sum(class_total[i])))
            else:
                print('Test Accuracy of %5s: N/A (no training examples)' % (self.classes[i]))
        print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
            100. * np.sum(class_correct) / np.sum(class_total), np.sum(class_correct), np.sum(class_total)))


if __name__ == '__main__':
    Test_obj = Testing_and_inference()
    Test_obj.test_acc()