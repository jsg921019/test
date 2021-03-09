#! /usr/bin/env python
#-*- coding:utf8 -*-

import numpy as np
import gym

env = gym.make('Taxi-v3')

action_size = env.action_space.n

gamma = 0.9
lr = 0.1

run_step = 500000
test_step = 10000

print_episode = 100

eps_init = 1.0
eps_min = 0.1

train_mode = True

class Agent(object):
    def __init__(self):
        self.Q_table = {}
        self.eps = eps_init

    def init_Q_table(self, state):
        if state not in self.Q_table:
            self.Q_table[state] = np.zeros(action_size)

    def get_action(self, state):
        if self.eps > np.random.rand():
            return np.random.randint(0, action_size)
        else:
            self.init_Q_table(state)
            return np.argmax(self.Q_table[state])
    
    def train_model(self, state, action, reward, next_state, done):

        # state가 Q_table에 없을시 초기화
        self.init_Q_table(state)
        self.init_Q_table(next_state)

        # 타깃 Q값
        target = reward + gamma * np.max(self.Q_table[next_state])

        # 현재 Q값
        Q_val = self.Q_table[state][action]

        if done:
            self.Q_table[state][action] =  (1-lr) * Q_val + lr * reward
        else:
            self.Q_table[state][action] = (1-lr) * Q_val + lr * target

if __name__ == "__main__":
    agent = Agent()

    step = 0
    episode = 0
    reward_list = []

    while step < run_step + test_step:

        # episode 시작시 초기화
        state = str(env.reset())
        episode_reward = 0
        done = False

        while not done:

            # train 종료 조건
            if step >= run_step:
                train_mode = False
                # env.render()

            # 다음 action get
            action = agent.get_action(state)

            # next state get
            next_state, reward, done, _ = env.step(action)
            next_state = str(next_state)
            episode_reward += reward

            # 훈련 모드일시 훈련시킴
            if train_mode:
                if agent.eps > eps_min:
                    agent.eps -= 1./run_step
                agent.train_model(state, action, reward, next_state, done)
            else:
                agent.eps = 0

            # 다음 step 준비
            state = next_state
            step += 1

        # 다음 episode 준비
        reward_list.append(episode_reward)
        episode += 1

        # 100번째마다 출력
        if episode != 0 and episode%print_episode == 0:
            print('Step: %d / Episode: %d / Epsilon: %.3f / Mean Rewards: %.3f'%(step, episode, agent.eps, np.mean(reward_list)))
            reward_list = []

    env.close()
