import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.autograd import Variable

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
class Net(nn.Module):
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
        x = self.conv1(x)
        x = x.view(-1, 7 * 7 * 12)
        x = self.dense(x)
        # return out #return output
        # print(x.shape)
        return x
        
        #return out #return output

my_net = Net()
my_net.load_state_dict(torch.load('model.pkl'))


