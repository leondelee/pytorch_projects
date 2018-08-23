#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from PIL import Image
from torch.utils import data
from torchvision import transforms as tm

from config import Configuration


transform = tm.Compose(
    [
        tm.Resize(224),
        tm.CenterCrop(224),
        tm.ToTensor(),
        tm.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5])
    ]
)

cf = Configuration()


def abstract_image_data(root):
    """
    This function is used to abstract image data_related set.
    param root: the path to the file in which contain different type of images
    return:  an image object
    """

    from torchvision.datasets import ImageFolder
    data_set = ImageFolder(root=root, transform=transform)
    return data_set


def load_data(dataset, shuffle=True, drop_last=False):
    """
    This function is used to load image data_related set.
    :param dataset: the data_related set you want to load
    :param batch_size: the number of images contained in a batch
    :param shuffle:  if True, the data_related set will be shuffled
    :param num_workers: the number of workers used to load data_related
    :param drop_last: drop the data_related which can not be composed as a batch
    :return: an iterable object
    """
    from torch.utils.data import DataLoader

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cf.batch_size,
        shuffle=shuffle,
        num_workers=cf.load_data_workers,
        drop_last=drop_last
    )
    return dataloader


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
        label = 1 if 'dog' in img_path.split('/')[-1].split('.')[0] else 0
        return img_data, label

    def __len__(self):
        return len(self.imgs)


# define the paths respectfully


# get each dataset
training_dog = DCDataset(root=cf.train_dog_path, train=True, dev=False, test=False)  # training set
training_cat = DCDataset(root=cf.train_cat_path, train=True, dev=False, test=False)  # training set
dev_dog = DCDataset(root=cf.train_dog_path, train=False, dev=True, test=False)  # development set
dev_cat = DCDataset(root=cf.train_cat_path, train=False, dev=True, test=False)  # development set
test_dog = DCDataset(root=cf.test_cat_path, train=False, dev=False, test=True)  # test set
test_cat = DCDataset(root=cf.test_cat_path, train=False, dev=False, test=True)  # test set

training_all = DCDataset(root=cf.train_all_path, train=True, dev=False, test=False)  # training set
dev_all = DCDataset(root=cf.train_all_path, train=False, dev=True, test=False)  # development set
test_all = DCDataset(root=cf.test_all_path, train=False, dev=False, test=True)  # test set


training_dog = load_data(training_dog)
training_cat = load_data(training_cat)
dev_dog = load_data(dev_dog)
dev_cat = load_data(dev_cat)
test_dog = load_data(test_dog)
test_cat = load_data(test_cat)

training_all = load_data(training_all)
dev_all = load_data(dev_all)
test_all = load_data(test_all)

if __name__ == '__main__':
    # from torchvision import transforms as tm
    # demo = abstract_image_data(root='/home/speit/torch/cat_dog/data_related/training_set/')
    # data_loader = load_data(demo)
    # for item, ite in enumerate(data_loader, 10):
    #     print(item)
    #     print(ite)
    #     print('*******')
    for data in training_all:
        print(data[1])




