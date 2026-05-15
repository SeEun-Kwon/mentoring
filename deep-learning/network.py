import torch
import torch.nn as nn
import torch.nn.functional as F

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, c_in, c_out, stride=1, dilation=1):
        super().__init__()
        self.c_in = c_in
        self.c_mid = c_out // 4
        self.c_out = c_out
        self.stride = stride

        self.conv1 = nn.Conv2d(c_in, self.c_mid, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(self.c_mid)

        self.conv2 = nn.Conv2d(self.c_mid, self.c_mid, kernel_size=3, stride=stride, dilation=dilation, padding=dilation)
        self.bn2 = nn.BatchNorm2d(self.c_mid)

        self.conv3 = nn.Conv2d(self.c_mid, c_out, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(c_out)

        if c_in != c_out or stride == 2:
            self.residual = nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(c_out)
            )

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))

        if self.c_in != self.c_out or self.stride == 2:
            residual = self.residual(residual)

        x += residual
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, layer_list, num_classes, AL):
        super().__init__()
        self.c_in = 64
        self.al = AL

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = nn.Sequential(
            Bottleneck(c_in=64, c_out=256, stride=1)
        )
        self.layer2 = nn.Sequential(
            Bottleneck(c_in=256, c_out=512, stride=2)
        )
        self.layer3_aux = nn.Sequential(
            Bottleneck(c_in=512, c_out=1024, stride=2)
        )
        self.layer3 = Bottleneck(c_in=1024, c_out=1024, stride=1)
        self.layer4 = nn.Sequential(
            Bottleneck(c_in=1024, c_out=2048, stride=2)
        )
        for i in range(layer_list[0] - 1):
            self.layer1.append(Bottleneck(c_in=256, c_out=256, stride=1))
        for i in range(layer_list[1] - 1):
            self.layer2.append(Bottleneck(c_in=512, c_out=512, stride=1))
        for i in range(layer_list[2] - 2):
            self.layer3_aux.append(Bottleneck(c_in=1024, c_out=1024, stride=1))
        for i in range(layer_list[3] - 1):
            self.layer4.append(Bottleneck(c_in=2048, c_out=2048, stride=1))

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        aux = self.layer3_aux(x)
        x = self.layer3(aux)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc(x)
        if self.al:
            return x, aux
        else:
            return x

def ResNet50(num_classes, AL):
    return ResNet([3, 4, 6, 3], num_classes, AL)

class CNN(nn.Module):
    def __init__(self, output_size=10):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.mlp = nn.Sequential(
            nn.Linear(1024, 4096),
            nn.ReLU(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Linear(4096, output_size)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.mlp(x)
        return x