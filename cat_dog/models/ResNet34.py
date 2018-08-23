#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : ResNet34.py
File Description : Define my own ResNet34
Author : Liangwei Li

"""
import torch as t

from models.BasicModel import BasicModel


class BasicBlock(BasicModel):

    def __init__(self, block_in_channels, block_out_channels, stride=1, shortcut=None):
        super(BasicBlock, self).__init__()
        self.block_shortcut = shortcut
        self.block_residual = t.nn.Sequential(
            t.nn.Conv2d(in_channels=block_in_channels, out_channels=block_out_channels, kernel_size=3, stride=stride,
                        padding=1),
            t.nn.BatchNorm2d(num_features=block_out_channels),
            t.nn.ReLU(),
            t.nn.Conv2d(in_channels=block_out_channels, out_channels=block_out_channels, kernel_size=3, stride=1,
                        padding=1),
            t.nn.BatchNorm2d(num_features=block_out_channels)
        )

    def forward(self, x):
        residual = self.block_residual(x)
        shortcut = x if self.block_shortcut is None else self.block_shortcut(x)
        return t.nn.functional.relu(residual + shortcut)


class ResNet34(BasicModel):

    def __init__(self, in_channels, out_classes=2, shortcut=None):
        super(ResNet34, self).__init__()
        self.in_channels = in_channels
        self.out_classes = out_classes
        self.shortcut = shortcut
        self.pre_conv = t.nn.Sequential(
            t.nn.Conv2d(in_channels=self.in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            t.nn.BatchNorm2d(64),
            t.nn.ReLU(),
            t.nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layer1 = self.make_layer(in_channels=64, out_channels=128, num_of_blocks=3)
        self.layer2 = self.make_layer(in_channels=128, out_channels=256, num_of_blocks=4, stride=2)
        self.layer3 = self.make_layer(in_channels=256, out_channels=512, num_of_blocks=6, stride=2)
        self.layer4 = self.make_layer(in_channels=512, out_channels=512, num_of_blocks=3, stride=2)
        self.out_pooling = t.nn.AvgPool2d(kernel_size=7)
        self.out_fc = t.nn.Linear(512, self.out_classes)

    def make_layer(self, in_channels, out_channels, num_of_blocks, stride=1):
        assert num_of_blocks >= 2
        shortcut = t.nn.Sequential(
            t.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride),
            t.nn.BatchNorm2d(num_features=out_channels)
        )
        layers = []
        layers.append(BasicBlock(block_in_channels=in_channels, block_out_channels=out_channels, stride=stride,
                                 shortcut=shortcut))
        for cnt in range(1, num_of_blocks):
            layers.append(BasicBlock(block_in_channels=out_channels, block_out_channels=out_channels))
        return t.nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.out_pooling(x)
        x = x.view(x.size()[0], -1)
        y = self.out_fc(x)
        return y


if __name__ == '__main__':
    from config import training_dog
    resnet34 = ResNet34(in_channels=3)
    demox = training_dog
    for item in demox:
        #print(item[0].shape)
        print(resnet34(item[0]))
