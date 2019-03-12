

import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 53 * 53, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 4)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 53 * 53)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.sigmoid(x)
        return x

tensor = torch.rand(64,3,224,224)
print(tensor.shape)
#tensor = tensor.unsqueeze(0)
print(tensor.shape)
net = Net()
x = net(tensor)
tensor2 = torch.rand(64,4)
#tensor2 = tensor2.unsqueeze(0)
print (tensor2.shape)
criterion = nn.functional.binary_cross_entropy
loss = criterion(x, tensor2)