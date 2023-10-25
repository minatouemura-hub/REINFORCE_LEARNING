#reinforce_lrを大枠は同じ
import numpy as np
import gymnasium as gym
import torch 
import torch.nn as nn 
import torch.nn.functional as F

#relu(fc(input_size=128,hidden_size)) 
#=> softmax(fc(hidden_size,action_size))
class Policy(nn.Module):
    def __init__(self,input_size,hidden_size,action_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.action_size = action_size
        
        self.fc1 = nn.Linear(input_size,hidden_size)
        self.fc2 = nn.Linear(hidden_size,action_size)

    def forward(self,x):#xはstate
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x),dim=1)
        return x

#BaseLineはNNによって求められる関数とする
class BaseLineNet(nn.Module):
    def __init__(self,input_size,output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        
        self.fc = nn.Linear(input_size,output_size)
    
    def forward(self,x):
        b_t = self.fc(x.detach())
        return b_t



from torch.optim import Adam
class Agent():
    def __init__(self):
        #パラメータを決める
        self.gamma = 0.98
        self.lr = 0.001
        self.action_size = 2

        self.input_size = 4
        self.hidden_size = 128
        self.output_size = 1#!
        self.baseline = BaseLineNet(self.input_size,self.output_size)

        #メモリを初期化
        self.memory = []
        #方策
        self.pi = Policy(self.input_size,self.hidden_size,self.action_size)
        self.optim = Adam(self.pi.parameters())

    def get_action(self,state):
        state = torch.Tensor(state[0])
        state = state.view(1,-1)#バッチ軸の追加
        probs = self.pi(state)
        probs = probs[0] #あるバッチを取り出す
        action = np.random.choice(len(probs),p = probs.detach().numpy())
        return action,probs[action]
    
    def calc_baseline(self,state):#!
        state = torch.Tensor(state[0])
        state = state.view(1,-1)#バッチ軸の追加
        b_t = self.baseline(state)
        return b_t
    
    def add(self,reward,prob,b_t):
        self.memory.append((reward,prob,b_t))
    
    def update(self):
        self.pi.zero_grad()

        G,loss = 0,0

        for reward,prob,b_t in reversed(self.memory):
            G += reward + self.gamma*(G-b_t)#!
            loss = -torch.log(prob)*G
        
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        self.memory = []

        
"""
#簡単なget_actionメソッドの実行
env = gym.make("CartPole-v1")
state = env.reset()
print(state)
agent = Agent()

action,probs = agent.get_action(state)
print("action:{}".format(action))
print("probs:{}".format(probs))
"""

episodes = 3000
env = gym.make("CartPole-v1")
agent = Agent()

reward_history = []
time_game_over = []

for episode in range(episodes):   

    state = env.reset()
    done = False
    total_reward = 0
    cnt = 0

    while not done:
        action,prob = agent.get_action(state)
        b_t = agent.calc_baseline(state)
        next_state,reward,done,info,_ = env.step(action)

        agent.add(reward,prob,b_t)
        total_reward += reward
        cnt += 1

    agent.update()
    reward_history.append(total_reward)
    time_game_over.append(cnt)

print("fastest:{}".format(min(time_game_over)))
print("lowest:{}".format(max(time_game_over)))

"""
fastest:8
lowest:113

"""









        

