#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import torch as t

from DQN import DQN
from Agent import Agent
from config import Configuration
cf = Configuration()
from tools import check_previous_models


def train(policy_net, target_net):
    optimizer = t.optim.RMSprop(policy_net.parameters())
    target_net.eval()
    model = Agent(policy_net=policy_net, target_net=target_net, optimizer=optimizer)
    model.run()


if __name__ == '__main__':
    policy_net = DQN().to(cf.device)
    target_net = DQN().to(cf.device)
    model_flag = check_previous_models()  # check if there exist previous models
    if model_flag != None:
        target_net.load(model_flag)  # if true, it will let the the user
    train(policy_net, target_net)