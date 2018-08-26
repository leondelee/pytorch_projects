#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import random
import math
import time

import torch as t
import numpy as np
from tqdm import tqdm

from config import Configuration
cf = Configuration()
from ReplayMemory import ReplayMemory,Transition
memory = ReplayMemory(capacity=cf.memory_size)
from mytorchtools.utils.tools import mylog
from Env import Env
env = Env()


class Agent:
    def __init__(self, policy_net, target_net, optimizer):
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimizer = optimizer
        self.loss = 0
        self.episode_done = False
        self.current_player = None
        self.steps_done = 0

    def reset(self):
        self.steps_done = 0
        self.episode_done = False
        self.loss = 0
        self.current_player = None

    def run(self):
        current_time = time.strftime(cf.time_format)
        for iteration in range(cf.num_of_epochs):
            print('Epoch:', iteration)
            with tqdm(total=100) as pbar:
                for episode in range(cf.num_of_episodes_in_one_epoch):
                        self.one_episode()
                        if episode % cf.num_of_verbose_episodes == 0 and episode != 0:
                            pbar.update(100 / cf.num_of_episodes_in_one_epoch * cf.num_of_verbose_episodes)
                            print('\n')
                            log_content_ = 'Huber loss at epoch{epoch} episode{episode} is {loss}.\nScore is {score}'.format(
                                epoch=iteration,
                                episode=episode,
                                loss=self.loss,
                                score=env.last_score
                            )
                            print(log_content_)
                            mylog(current_time, log_content_)
            log_content = 'Huber loss at epoch{epoch} is {loss}.\nScore is {score}'.format(
                epoch=iteration,
                loss=self.loss,
                score=env.last_score
            )
            print(log_content)
            mylog(current_time, log_content)
            self.target_net.save()

    def one_episode(self):
        self.reset()
        env.reset()
        while not self.episode_done:
            current_state = env.current_state
            current_action = self.select_action(current_state)
            self.current_player = self.get_player()
            reward, self.episode_done = env.step(
                current_action,
                self.current_player
            )
            next_state = env.current_state
            memory.push(t.Tensor(current_state), t.tensor(current_action, dtype=t.float), t.Tensor(next_state), t.Tensor([reward]))
            self.optimize_model()

    def select_action(self, state):
        sample = random.random()
        eps_threshold = cf.eps_end + (cf.eps_start - cf.eps_end) * math.exp(-1. * self.steps_done / cf.eps_decay)
        self.steps_done += 1
        state = t.Tensor(state)
        if sample > eps_threshold:
            with t.no_grad():
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return t.tensor([[random.randrange(cf.num_of_classes)]], device=cf.device, dtype=t.long)

    def get_player(self):
        assert env.num_of_remain_nodes >= 1
        player_idx = np.random.randint(0, env.num_of_remain_nodes)
        return player_idx

    def optimize_model(self):
        if len(memory) < cf.batch_size:
            return
        transitions = memory.sample(cf.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = t.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=cf.device, dtype=t.uint8)
        non_final_next_states = t.cat([s for s in batch.next_state if s is not None])
        state_batch = t.cat(batch.state)
        action_batch = t.cat(batch.action)
        reward_batch = t.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch.long())

        # Compute V(s_{t+1}) for all next states.
        next_state_values = t.zeros(cf.batch_size)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * cf.gamma) + reward_batch

        # Compute Huber loss
        loss = t.nn.functional.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
        self.loss = loss
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()



