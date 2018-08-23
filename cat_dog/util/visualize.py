#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name : Visualize.py
File Description : Packaging some visualization tools
Author : Liangwei Li

"""

import time

import numpy as np
import visdom


class Visualizer:
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        Plot multiple points at one time
        :param d:  dict = {name:value}
        :return:
        """
        for key, value in d.items():
            self.plot(key, value)

    def img_many(self, d):
        for key, value in d.items():
            self.img(key, value)

    def plot(self, key, value, **kwargs):
        x = self.index.get(key, 0)
        self.vis.line(X=np.array(x), Y=np.array(value), win=(key), update=None if x==0 else 'append', **kwargs)
        self.index[key] = x + 1

    def log(self, info, win='log_text'):
        self.log_text += ('[{time}] {info} '.format(time=time.strftime('%m%d_%H%M%S'), info=info))
        self.vis.text(self.log_text, win)

    def __getattr__(self, item, name):

        return getattr(self.vis, name)


