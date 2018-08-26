#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""

File Name :
File Description :
Author : Liangwei Li

"""
import numpy as np
import torch as t
from sklearn import metrics

from config import Configuration
cf = Configuration()


def generate_fake_data(seed=10, N=6):
    # x1 = np.random.normal(0, size=(N // 3, config.F))
    # x2 = np.random.normal(2, size=(N // 3, config.F))
    # x3 = np.random.normal(-2, size=(N // 3, config.F))
    x1 = np.ones((N // 3, cf.num_of_features)) * 1
    x2 = np.ones((N // 3, cf.num_of_features)) * 2
    x3 = np.ones((N // 3, cf.num_of_features)) * 3
    x = np.concatenate([x1, x2, x3])
    y = np.zeros([N])
    y[N // 3:N // 3 * 2] = 1
    y[N // 3 * 2:] = 2
    return x, y


class Env:
    def __init__(self):
        print('INITED~~~~~~~~~~~~~~~~~~~~~~~~')
        np.random.seed(0)
        self.raw_x, self.raw_y = generate_fake_data(10, cf.num_of_nodes)

        self.rand_idx = np.random.permutation(np.arange(cf.num_of_nodes))
        self.x = self.raw_x[self.rand_idx].reshape(1, cf.num_of_channels, self.raw_x.shape[0], -1)
        self.y = self.raw_y[self.rand_idx]
        
        self.current_state = self.x
        self.current_label = self.y[self.rand_idx]
        
        self.remain_nodes_idx = np.arange(cf.num_of_nodes)
        self.num_of_remain_nodes = cf.num_of_nodes
        self.last_score = 0
        self.score_cnt = 0
        self.episode_done = False
        self.reset()

    def reset(self):
        self.rand_idx = np.random.permutation(np.arange(cf.num_of_nodes))
        self.x = self.raw_x[self.rand_idx].reshape(1, cf.num_of_channels, self.raw_x.shape[0], -1)
        self.y = self.raw_y[self.rand_idx]
        
        self.current_state = self.x
        self.current_label = self.y[self.rand_idx]
        
        self.remain_nodes_idx = np.arange(cf.num_of_nodes)
        self.num_of_remain_nodes = cf.num_of_nodes

    def get_next_state(self, current_player_id=None):
        if current_player_id is None:
            return self.current_state
        else:
            state_shape = self.current_state.shape
            self.current_state[:, :, current_player_id] = t.zeros(1, state_shape[-1])
            self.remain_nodes_idx = np.delete(self.remain_nodes_idx, current_player_id, 0)
            return self.current_state

    def step(self, current_action, current_player_id=None):
        self.num_of_remain_nodes -= 1
        self.current_state = self.get_next_state(current_player_id)
        self.current_label[current_player_id] = current_action
        score = metrics.adjusted_mutual_info_score(self.y, self.current_label)
        reward = 0
        if self.score_cnt == 50:
            self.last_score = max(score, self.last_score)
            self.score_cnt = 0
        else:
            if score > self.last_score:
                print('last score is {ls}, current score is {cs}'.format(ls=self.last_score, cs=score))
                reward = 1
                self.score_cnt += 1
            else:
                reward = -1
        if self.num_of_remain_nodes == 0:
            self.episode_done = True
        return reward, self.episode_done


if __name__ == '__main__' :
    env = Env()
    idx = np.random.permutation(np.arange(cf.num_of_nodes))
    print(env.y[idx])
