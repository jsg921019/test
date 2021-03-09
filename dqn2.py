#! /usr/bin/env python
#-*- coding:utf-8 -*-

#! /usr/bin/env python
#-*- coding:utf-8 -*-

import os
import datetime
import random
import numpy as np
from collections import deque

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
mem_maxlen = 10000

gamma = 0.99
lr = 0.00025

skip_frame = 1
stack_frame = 1

start_train_step = 10000
run_step = 50000
test_step = 10000

target_update_step = 1000
print_episode = 10
save_step = 100000

eps_init = 1.0
eps_min = 0.1

date_time = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
save_path = "./saved_models/" + date_time
load_path = "./saved_models/20210225-18-34-38DQN"

class DQN(nn.Module):
    def __init__(self, network_name):
        super(DQN, self).__init__()
        input_size = state_size*stack_frame
        self.network_name = network_name

        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQNAgent(object):
    def __init__(self, model, target_model, optimizer):
        self.model = model
        self.target_model = target_model
        self.optimizer = optimizer
        self.memory = deque(maxlen=mem_maxlen)
        self.obs_set = deque(maxlen=skip_frame*stack_frame)
        self.eps = eps_init
        self.update_target()
        if load_model == True:
            self.model.load_state_dict(torch.load(load_path+'/model.pth'))
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
    
    def skip_stack_frame(self, obs):
        self.obs_set.append(obs)
        state = np.zeros([state_size*stack_frame])
        for i in range(stack_frame):
            state[state_size*i : state_size*(i+1)] = self.obs_set[-1 -(skip_frame*i)]
        return state

    def append_sample(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update_target(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def train_model(self):

        batch = random.sample(self.memory, batch_size)

        state_batch = torch.FloatTensor(np.stack([b[0] for b in batch], axis=0))
        action_batch = torch.FloatTensor(np.stack([b[1] for b in batch], axis=0))
        reward_batch = torch.FloatTensor(np.stack([b[2] for b in batch], axis=0))
        next_state_batch = torch.FloatTensor(np.stack([b[3] for b in batch], axis=0))
        done_batch = torch.FloatTensor(np.stack([b[4] for b in batch], axis=0))

        eye = torch.eye(action_size)
        one_hot_action = eye[action_batch.view(-1).long()]
        q = (self.model(state_batch) * one_hot_action).sum(1)

        with torch.no_grad():
            max_Q = torch.max(q).item()
            next_q = self.target_model(next_state_batch)
            target_q = reward_batch + next_q.max(1).values * (gamma*(1-done_batch))

        loss = F.smooth_l1_loss(q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item(), max_Q

if __name__ == '__main__':
    model = DQN("main")
    target_model = DQN("target")
    optimizer = optim.Adam(model.parameters(), lr=lr)
    agent = DQNAgent(model, target_model, optimizer)
    model.train()
    step = 0
    episode = 0
    reward_list = []
    loss_list = []
    max_Q_list = []
    # 게임 진행 반복문
    while step < run_step + test_step:
        # 상태, episode reward, done 정보 초기화
        obs = env.reset()
        episode_rewards = 0
        done = False
        for i in range(skip_frame*stack_frame):
            agent.obs_set.append(obs)
        state = agent.skip_stack_frame(obs)
        # 에피소드를 위한 반복문
        while not done:
            if step == run_step:
                train_mode = False
                model.eval()
            # 행동 결정
            action = agent.get_action(state)
            # 다음 상태, 보상, 게임 종료 여부 정보 취득
            next_obs, reward, done, _ = env.step(action)
            episode_rewards += reward
            next_state = agent.skip_stack_frame(next_obs)
            # 카트가 최대한 중앙에서 벗어나지 않도록 학습
            reward -= abs(next_obs[0])
            # 리플레이 메모리에 데이터 저장
            if train_mode:
                agent.append_sample(state, action, reward, next_state, done)
            else:
                agent.eps = 0.0
                env.render()
            # 상태 정보 업데이트
            state = next_state
            step += 1
            if step > start_train_step and train_mode:
                # Epsilon 감소
                if agent.eps > eps_min:
                    agent.eps -= 1.0 / (run_step - start_train_step)
                # 모델 학습
                loss, maxQ = agent.train_model()
                loss_list.append(loss)
                max_Q_list.append(maxQ)
                # 타겟 네트워크 업데이트
                if step % target_update_step == 0:
                    agent.update_target()
            # 모델 저장
            if step % save_step == 0 and step != 0 and train_mode:
                agent.save_model(load_model, train_mode)
        reward_list.append(episode_rewards)
        episode += 1
        # 진행상황 출력
        if episode % print_episode == 0 and episode != 0:
            print("step: {} / episode: {} / reward: {:.2f} / loss: {:.4f} / maxQ: {:.2f} / epsilon: {:.4f}".format
                  (step, episode, np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list), agent.eps))
            reward_list = []
            loss_list = []
            max_Q_list = []
    agent.save_model(load_model, train_mode)
    env.close()

# if __name__ == "__main__":
#     model = DQN("main")
#     target_model = DQN("target")
#     optimizer = optim.Adam(model.parameters(), lr=lr)
#     agent = DQNAgent(model, target_model, optimizer)
#     model.train()
#     step = 0
#     episode = 0
#     reward_list = []
#     loss_list = []
#     max_Q_list = []

#     while step < run_step + test_step:
#         obs = env.reset()
#         episode_rewards = 0
#         done = False

#         for i in range(skip_frame*stack_frame):
#             agent.obs_set.append(obs)

#         state = agent.skip_stack_frame(obs)

#         while not done:
#             if step == run_step:
#                 train_mode = False
#                 model.eval()

#             action = agent.get_action(state)
#             next_obs, reward, done, _ = env.step(action)
#             episode_rewards += reward
#             next_state = agent.skip_stack_frame(next_obs)
            
#             # 카트가 중앙에서 벗어나지 않도록 학습
#             reward -= abs(next_obs[0])

#             if train_mode:
#                 agent.append_sample(state, action, reward, next_state, done)
#             else:
#                 agent.eps = 0.0
#                 env.render()
            
#             if step%target_update_step == 0:
#                 agent.update_target()
#                 if agent.eps > eps_min:
#                     agent.eps -= 1.0 / run_step
#                 loss, maxQ = agent.train_model(state, action, reward, next_state, done)
#                 loss_list.append(loss)
#                 max_Q_list.append(maxQ)
            
#             if step % save_step == 0 and step != 0 and train_mode:
#                 agent.save_model(load_model, train_mode)

#             state = next_state
#             step += 1

#         reward_list.append(episode_rewards)
#         episode += 1

#         if episode % print_episode == 0 and episode != 0:
#             print("step: %d / episode: %d / reward: %.2f / loss%.4f / maxQ: %.2f / eps: %.4f"
#                     %(step, episode, np.mean(reward_list), np.mean(loss_list), np.mean(max_Q_list), agent.eps))
#             reward_list = []
#             loss_list = []
#             max_Q_list = []
#     agent.save_model(load_model, train_mode)
#     env.close()

