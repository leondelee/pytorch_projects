#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : dataset.py
File Description : Define the class "DCDataSet()" to return a dataset object of pytorch, which will be used to feed in a
                   torch data_related loader.
Author : Liangwei Li

"""

import os

from PIL import Image
from torch.utils import data


class DCDataset(data.Dataset):

    def __init__(self, root, transformer=None, train=True, dev=False, test=False):
        """
        Get the root path of data_related images and exert different tranformation onto them according to their attributes
        (ie.train, test, dev)
        :param root: root path
        :param transformer: transformation to be used
        :param train: if true, it is training set
        :param dev: if true, it is development set
        :param test: if true, it is test set
        """
        self.train = train
        self.dev = dev
        self.test = test

        # path for each image
        imgs = [os.path.join(root, img) for img in os.listdir(root)]

        # sort images so that we can divide them without changing their relative positions
        imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))

        # divide 70% of training set as train data_related, the 30% else as development set
        if self.train:
            self.imgs = imgs[:int(0.7 * len(imgs))]
        elif self.dev:
            self.imgs = imgs[:int(0.3 * len(imgs))]
        else:
            self.imgs = imgs

        # if a user doesn't specify a transformer, define it here
        if transformer is None:
            from data_related.load_data import transform
            self.transformer = transform
        else:
            self.transformer = transformer

    def __getitem__(self, index):
        """
        return one element of the data images
        :param index: index to be indexed
        :return: data and its label
        """
        img_path = self.imgs[index]
        img_data = Image.open(img_path)
        img_data = self.transformer(img_data)
        label = 1 if 'dogs' in img_path.split('/') else 0
        return img_data, label

    def __len__(self):
        return len(self.imgs)


# define the paths respectfully
train_dog_path = '/home/speit/torch/cat_dog/data_related/training_set/dogs'
train_cat_path = '/home/speit/torch/cat_dog/data_related/training_set/cats'
test_dog_path = '/home/speit/torch/cat_dog/data_related/test_set/dogs'
test_cat_path = '/home/speit/torch/cat_dog/data_related/test_set/cats'

# get each dataset
training_dog = DCDataset(root=train_dog_path, train=True, dev=False, test=False)  # training set
training_cat = DCDataset(root=train_cat_path, train=True, dev=False, test=False)  # training set
dev_dog = DCDataset(root=train_dog_path, train=False, dev=True, test=False)  # development set
dev_cat = DCDataset(root=train_cat_path, train=False, dev=True, test=False)  # development set
test_dog = DCDataset(root=test_cat_path, train=False, dev=False, test=True)  # test set
test_cat = DCDataset(root=test_cat_path, train=False, dev=False, test=True)  # test set

if __name__ == '__main__':
    pass










