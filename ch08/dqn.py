import numpy as np
import torch
import gymnasium as gym
import random
from collections import deque
from copy import deepcopy
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size  =batch_size

    def __len__(self):
        return len(self.buffer)

    def add(self,state,action,reward,next_state,done):
        data = [state,action,reward,next_state,done]
        self.buffer.append(data)

    def get_batch(self):
        data = random.sample(self.buffer,self.batch_size)

        state = np.stack([x[0] for x in data])
        action = np.array([x[1]for x in data])
        reward = np.array([x[2] for x in data])
        next_state = np.stack([x[3] for x in data])
        done = np.array([x[4] for x in data]).astype(np.float32)

        return state,action,reward,next_state,done
    
    def reset(self):
        self.buffer.clear()


#Q関数をニューラルネットに置き換える
class QNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.Linear(4,100)
        self.l2 = nn.Linear(100,2)#行動のサイズ

    def forward(self,x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x

class QLearningAgent:
    def __init__(self):
        self.gamma = 0.9
        self.lr  =0.01
        self.epsilon = 0.1
        self.action_size = 2
        self.buffer_size = 10000
        self.batch_size = 32

        self.replay_buffer = ReplayBuffer(self.buffer_size,self.batch_size)
        self.qnet = QNet()
        self.target_net = deepcopy(self.qnet)
        self.optimizer = torch.optim.SGD(self.qnet.parameters(),self.lr)
        
    def get_action(self,state):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_size)
        else:
            state = torch.tensor(state,dtype=torch.float32).unsqueeze(0)
            qs = self.qnet(state).detach().cpu().numpy()
            return np.argmax(qs[0])
        
    def update_buffer(self,state,action,reward,next_state,done):
        self.replay_buffer.add(state,action,reward,next_state,done)
        
    def update(self):
        first = True
        if len(self.replay_buffer) <= self.batch_size:
            return 0
        states,actions,rewards,next_states,dones = self.replay_buffer.get_batch()
        target_q = self.target_net(torch.tensor(next_states,dtype=torch.float32).unsqueeze(0)).detach().max(1)[0]
        target = (torch.tensor(rewards,dtype=torch.float32).unsqueeze(1) + self.gamma*target_q*(1-torch.tensor(dones,dtype = torch.float32).unsqueeze(1)))
        #target_size:[32,2]
        q = self.qnet(torch.tensor(states))
        #q_size:[32,1]
        loss = F.mse_loss(target,q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return torch.tensor(loss).clone().detach().item()


#実装例------------------------------------
env = gym.make('CartPole-v1')
agent = QLearningAgent()

episodes = 100
loss_history = []
check_point= 100
for episode in range(episodes):
    state = env.reset()
    state = state[0]
    total_loss,cnt = 0,0
    done = False

    while not done:
        action = agent.get_action(state)
        next_state,reward,done,info,_ = env.step(action)
        agent.update_buffer(state,action,reward,next_state,done)
        total_loss += agent.update()
        if episode//check_point:
            env.render()
        state = next_state
    loss_history.append(total_loss)
env.close()

plt.plot([x for x in range(episodes)],loss_history)
plt.show()