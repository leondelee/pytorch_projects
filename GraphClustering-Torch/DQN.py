#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import torch as t
from torch import nn
from torch.nn import functional as F

from mytorchtools.nn.BasicModel import BasicModel
from config import Configuration
cf = Configuration()


class DQN(BasicModel):

    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(cf.num_of_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        fc = t.nn.Linear(x.shape[1] * x.shape[2] * x.shape[3], cf.num_of_classes)
        return fc(x.view(cf.batch_size, -1))
