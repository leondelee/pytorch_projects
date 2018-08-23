#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : LeNet.py
File Description : Define LeNet
Author : Liangwei Li

"""
import torch as t

from .BasicModel import BasicModel


class LeNet(BasicModel):
    def __init__(self, in_channels, classes):
        super(LeNet, self).__init__()
        self.in_channels = in_channels
        self.classes = classes
        self.pre_layers_of_le_net = t.nn.Sequential(
            t.nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5),
            t.nn.AvgPool2d(kernel_size=2, stride=2),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            t.nn.AvgPool2d(kernel_size=2, stride=2),
            t.nn.ReLU()
        )

    def forward(self, x):
        x = self.pre_layers_of_le_net(x)
        shape_of_x = x.shape
        fc1 = t.nn.Linear(shape_of_x[-1], 120)
        fc2 = t.nn.Linear(120, self.classes)
        x = x.view(shape_of_x[0] * shape_of_x[1] * shape_of_x[2], shape_of_x[-1])
        print(x.shape)
        x = t.nn.functional.relu(fc1(x))
        return t.nn.functional.relu(fc2(x))


if __name__ == '__main__':
    from torchvision import datasets
    mnist = datasets.MNIST('mnist/', download=True, train=False)
    model = LeNet(3, 10)

