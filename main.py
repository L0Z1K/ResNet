import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time

transform = transforms.Compose([
    transforms.Pad(4),
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32),
    transforms.ToTensor()
])

data_train = datasets.CIFAR10(root="CIFAR10/",
                              train=True,
                              download=True,
                              transform=transform)

data_test = datasets.CIFAR10(root="CIFAR10/",
                             train=False,
                             download=True,
                             transform=transforms.ToTensor())

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        if stride != 1: # When Downsampling,
            self.downsample = nn.Sequential(
                                nn.Conv2d(in_channels, out_channels, 3, 2, 1),
                                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = None
    
    def forward(self, x):
        out = self.bn1(self.conv1(x))
        out = F.relu(out)
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            x = self.downsample(x)
        out += x
        return F.relu(out)

class ResNet(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = []
        for _ in range(n):
            self.layer1.append(ResidualBlock(16, 16, 1))
        self.layer2 = [ResidualBlock(16, 32, 2)]
        for _ in range(n-1):
            self.layer2.append(ResidualBlock(32, 32, 1))
        self.layer3 = [ResidualBlock(32, 64, 2)]
        for _ in range(n-1):
            self.layer3.append(ResidualBlock(64, 64, 1))
        
        self.layer1 = nn.Sequential(*self.layer1)
        self.layer2 = nn.Sequential(*self.layer2)
        self.layer3 = nn.Sequential(*self.layer3)

        self.pool = nn.AvgPool2d(8)
        self.linear = nn.Linear(64, 10)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x

data_loader = DataLoader(dataset=data_train,
                         batch_size=256,
                         shuffle=True,
                         drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = ResNet(3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=0.0001, momentum=0.9)

total_epoch = 100
total_batch = len(data_loader)

print("[+] Train Start")
start = time.time()
for epoch in range(total_epoch):
    avg_cost = 0
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        y_pred = model(x)
        
        cost = criterion(y_pred, y)
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        avg_cost += cost
    avg_cost /= total_batch

    if (epoch+1) % 10 == 0:
      print("Epoch: %d/%d, Cost: %f" % (epoch+1, total_epoch, avg_cost))
end = time.time()
t = int(end - start)
print("[+] Training time: %dm %ds" %(t//60, t%60))

test_loader = DataLoader(dataset=data_test,
                         batch_size=256,
                         shuffle=False)

print("[+] Test Start")
def accuracy(output, target):
    batch_size = target.size(0)
    _, pred = output.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    error_1 = batch_size - correct[:1].view(-1).float().sum(0)
    error_5 = batch_size - correct[:5].view(-1).float().sum(0)
    return [error_1, error_5]

total_test = len(data_test)
error_1 = 0
error_5 = 0

for x, y in test_loader:
    x = x.to(device)
    y = y.to(device)
    y_pred = model(x)
    e1, e5 = accuracy(y_pred, y)
    error_1 += e1
    error_5 += e5

print("Top-1 Error: %d/%d (%.2f%%)" % (error_1, total_test, error_1/total_test*100))
print("Top-5 Error: %d/%d (%.2f%%)" % (error_5, total_test, error_5/total_test*100))
