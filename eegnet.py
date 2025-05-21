import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

#https://www.kaggle.com/code/xevhemary/eeg-pytorch/notebook
# EEGnet 4 temporal filters, 2 spatial filters
class EEGNet_4_2(nn.Module):
    def __init__(self):
        super(EEGNet_4_2, self).__init__()
        
        # Number of temporal filters
        self.F1 = 4

        # Number of pointwise filters
        self.F2 = 8
        # Number of spatial filters
        self.D = 2

        # Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        # depthwise conv2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (15, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        # separable conv2d
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(11*8, 1, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.conv3(x)
        print("After conv3:", x.shape)

        # Flatten the output for the linear layer
        x = x.view(-1, 11*8)
        print("After flattening:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x


# EEGNet 8 temporal filters, 2 spatial filters
#https://www.kaggle.com/code/xevhemary/eeg-pytorch/notebook
class EEGNet_8_2(nn.Module):
    def __init__(self):
        super(EEGNet_8_2, self).__init__()
        
        # Number of temporal filters
        self.F1 = 8

        # Number of pointwise filters
        self.F2 = 16
        # Number of spatial filters
        self.D = 2

        # Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        # depthwise conv2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (15, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        # separable conv2d
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(11*16, 1, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.conv3(x)
        print("After conv3:", x.shape)

        # Flatten the output for the linear layer
        x = x.view(-1, 11*16)
        print("After flattening:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x
    
#EEGNet 16 temporal filters, 2 spatial filters
#https://www.kaggle.com/code/xevhemary/eeg-pytorch/notebook
class EEGNet_16_2(nn.Module):
    def __init__(self):
        super(EEGNet_16_2, self).__init__()
        
        # Number of temporal filters
        self.F1 = 16

        # Number of pointwise filters
        self.F2 = 32
        # Number of spatial filters
        self.D = 2

        # Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        # depthwise conv2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (15, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        # separable conv2d
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(11*32, 1, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.conv3(x)
        print("After conv3:", x.shape)

        # Flatten the output for the linear layer
        x = x.view(-1, 11*32)
        print("After flattening:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x
    

#EEGNet 16 temporal filters, 4 spatial filters

#https://www.kaggle.com/code/xevhemary/eeg-pytorch/notebook
class EEGNet_16_4(nn.Module):
    def __init__(self):
        super(EEGNet_16_4, self).__init__()
        
        # Number of temporal filters
        self.F1 = 16

        # Number of pointwise filters
        self.F2 = 64
        # Number of spatial filters
        self.D = 4

        # Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, self.F1, (1, 64), padding=(0, 32), bias=False),
            nn.BatchNorm2d(self.F1)
        )
        # depthwise conv2d
        self.conv2 = nn.Sequential(
            nn.Conv2d(self.F1, self.D * self.F1, (15, 1), groups=self.F1, bias=False),
            nn.BatchNorm2d(self.D * self.F1),
            nn.ELU(),
            nn.AvgPool2d((1, 4)),
            nn.Dropout(0.5)
        )
        # separable conv2d
        self.conv3 = nn.Sequential(
            nn.Conv2d(self.D * self.F1, self.D * self.F1, (1, 16), padding=(0, 8), groups=self.D * self.F1, bias=False),
            nn.Conv2d(self.D * self.F1, self.F2, (1, 1), bias=False),
            nn.BatchNorm2d(self.F2),
            nn.ELU(),
            nn.AvgPool2d((1, 8)),
            nn.Dropout(0.5)
        )

        self.classifier = nn.Linear(11*64, 1, bias=True)


    def forward(self, x):
        x = self.conv1(x)
        print("After conv1:", x.shape)
        x = self.conv2(x)
        print("After conv2:", x.shape)
        x = self.conv3(x)
        print("After conv3:", x.shape)

        # Flatten the output for the linear layer
        x = x.view(-1, 11*64)
        print("After flattening:", x.shape)
        x = self.classifier(x)
        print("After classifier:", x.shape)
        return x