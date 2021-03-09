#! /usr/bin/env python
#-*- coding:utf-8 -*-

import os
import datetime
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import gym

# state : (카트 위치, 카트 속력, 막대기 각도, 막대기 끝 속력)
# action : (좌, 우)
# reward : 막대가 넘어지지 않으면 +1
# terminal condition :
#   1. -12 < 막대의 각도 < 12
#   2. -2.4 < 카트의 위치 < 2.4
#   3. step > 200

env = gym.make("CartPole-v0")
algorithm = 'DQN'

state_size = 4
action_size = env.action_space.n

load_model = False
train_mode = True

batch_size = 32

gamma = 0.9
lr = 0.001

run_step = 40000
test_step = 10000

print_episode = 10
save_step = 100000

eps_init = 1.0
eps_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_models/" + date_time
load_path = "./saved_models/"

class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(object):
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.eps = eps_init
        if load_model == True:
            self.model.load_state_dict(torch.load(load_path))
            print("Model is loaded from %s"%load_path)
    
    def get_action(self, state):
        if train_mode:
            if self.eps > np.random.rand():
                return np.random.randint(0, action_size)
            else:
                with torch.no_grad():
                    Q = self.model(torch.FloatTensor(state).unsqueeze(0))
                    return np.argmax(Q.detach().numpy())
        else:
            with torch.no_grad():
                Q = self.model(torch.FloatTensor(state).unsqueeze(0))
                return np.argmax(Q.detach().numpy())

    def save_model(self, load_model, train_mode):
        # 새로운 모델
        if not load_model and train_mode:
            path_ = save_path + algorithm
            if not os.path.exists(path_):
                os.makedirs(path_)
            torch.save(self.model.state_dict(), path_ +'/model.pth')
            print("Save Model %s"%(save_path + algorithm))
        # 로딩한 모델
        elif load_model and train_mode:
            torch.save(self.model.state_dict(), load_path+'/model.pth')
            print("Save Model %s"%(load_path))
    
    def train_model(self, state, action, reward, next_state, done):
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)

        one_hot_action = torch.zeros(2)
        one_hot_action[action] = 1
        q = (self.model(state) * one_hot_action).sum()
        with torch.no_grad():
            max_Q = q.item()
            next_q = self.model(next_state)
            target_q = reward + next_q.max() * (gamma*(1-done))
        loss = F.smooth_l1_loss(q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q

if __name__ == "__main__":
    model = DQN()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    agent = DQNAgent(model, optimizer)
    model.train()
    step = 0
    episode = 0
    reward_list = []
    loss_list = []
    max_Q_list = []

    while step < run_step + test_step:
        state = env.reset()
        episode_rewards = 0
        done = False
        while not done:
            if step == run_step:
                train_mode = False
                model.eval()

            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_rewards += reward

            if train_mode == False:
                agent.eps = 0.0

            if train_mode == True:
                if agent.eps > eps_min:
                    agent.eps -= 1.0 / run_step
                loss, maxQ = agent.train_model(state, action, reward, next_state, done)
                loss_list.append(loss)
                max_Q_list.append(maxQ)
            
            if step % save_step == 0 and step != 0 and train_mode:
                agent.save_model(load_model, train_mode)

            state = next_state
            step += 1

        reward_list.append(episode_rewards)
        episode += 1

        if episode % print_episode == 0 and episode != 0:
            print("step: %d / episode: %d / reward: %.2f / loss%.4f / maxQ: %.2f / eps: %.4f"
                    %(step, episode, np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list), agent.eps))
            reward_list = []
            loss_list = []
            max_Q_list = []
    agent.save_model(load_model, train_mode)
    env.close()

