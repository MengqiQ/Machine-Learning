import torch.utils.data
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable
import torchvision
import numpy as np

##TO-DO: Import data here:
transform = transforms.Compose([transforms.ToTensor(),
                                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])

data_train = datasets.MNIST(root="./data/",
                            transform=transform,
                            train=True,
                            download=True)

data_test = datasets.MNIST(root="./data/",
                           transform=transform,
                           train=False)

data_loader_train = torch.utils.data.DataLoader(dataset=data_train,
                                                batch_size=32,
                                                shuffle=True,
                                                num_workers=2)

data_loader_test = torch.utils.data.DataLoader(dataset=data_test,
                                               batch_size=32,
                                               shuffle=True,
                                               num_workers=2)


##


##TO-DO: Define your model:
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        ##Define layers making use of torch.nn functions:
        self.conv1 = torch.nn.Sequential(torch.nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2),
                                         torch.nn.Conv2d(6, 12
                                                         , kernel_size=3, stride=1, padding=1),
                                         torch.nn.ReLU(),
                                         torch.nn.MaxPool2d(stride=2, kernel_size=2))

        self.dense = torch.nn.Sequential(torch.nn.Linear(7 * 7 * 12, 128),
                                         torch.nn.ReLU(),
                                         torch.nn.Dropout(p=0.5),
                                         torch.nn.Linear(128, 10))


    def forward(self, x):
        ##Define how forward pass / inference is done:
        # print(x.shape)
        x = self.conv1(x)
        x = x.view(-1, 7 * 7 * 12)
        x = self.dense(x)
        # return out #return output
        # print(x.shape)
        return x


if __name__ == '__main__':
    my_net = Net()
    # my_net = my_net.cuda()
    print(my_net)
    ##TO-DO: Train your model:
    cost = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(my_net.parameters())
    # my_net.load_state_dict(torch.load('model.pkl'))
    n = 5
    for i in range(n):
        running_loss = 0.0
        running_correct = 0
        print("Epoch {}/{}".format(i, n))
        print("-" * 10)
        for data in data_loader_train:
            X_train, y_train = data
            X_train, y_train = Variable(X_train), Variable(y_train)
            outputs = my_net(X_train)
            _, pred = torch.max(outputs.data, 1)
            optimizer.zero_grad()
            loss = cost(outputs, y_train)

            loss.backward()
            optimizer.step()
            running_loss += loss.data[0]
            running_correct += torch.sum(pred == y_train.data)
        testing_correct = 0
        for data in data_loader_test:
            X_test, y_test = data
            X_test, y_test = Variable(X_test), Variable(y_test)
            outputs = my_net(X_test)
            _, pred = torch.max(outputs.data, 1)
            testing_correct += torch.sum(pred == y_test.data)
        print(
            "Loss is:{:.4f}, Train Accuracy is:{:.4f}%, Test Accuracy is:{:.4f}".format(running_loss / len(data_train),
                                                                                        100 * running_correct / len(
                                                                                            data_train),
                                                                                        100 * testing_correct / len(
                                                                                            data_test)))

torch.save(my_net.state_dict(), 'model.pkl')
