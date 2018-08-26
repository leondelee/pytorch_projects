#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import torch as t


class Configuration:
    global_task = 'classification'
    nodes = None
    memory_size = 1e5
    num_of_features = 128 * 128
    num_of_channels = 1
    num_of_nodes = 60
    num_of_classes = 3
    num_of_epochs = 1000
    num_of_episodes_in_one_epoch = 1000
    num_of_verbose_episodes = 10
    batch_size = 1
    gamma = 0.999
    eps_start = 0.9
    eps_end = 0.05
    eps_decay = 200
    target_update = 10
    device = t.device("cuda" if t.cuda.is_available() else "cpu")
    project_path = '/home/speit/cw-lab/GraphClustering-Torch/'
    log_dir_path = project_path + 'log/'
    time_format = "%m_%d_%H:%M:%S"  # time format used in this project for logging and saving model
    checkpoint_dir_path = project_path + 'checkpoints/'
