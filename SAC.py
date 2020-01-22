import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from collections import namedtuple, deque
from torch.distributions import Normal, Categorical

TAU = 0.001 #1128 0.001 -- 0.005
LR = 3e-4 #1128 3e-4 -- 1e-3
GAMMA = 0.99
batch_size = 256
memory_capacity = 5000000
state_dim =126
act_dim = 20
num_agent = 5
load = False
random_step = 1000
steps = 0

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

class replay_memory:
    def __init__(self,capacity):
        self.memory = deque(maxlen = capacity)
        self.Experience = namedtuple('Experience',['state','action','next_state','reward','done'])
    
    def add(self,state,action,next_state,reward,done):
        e = self.Experience(state,action,next_state,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        experience = random.sample(self.memory,batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().cuda()

        return states,actions,next_states,rewards,dones
    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self,state_dim,act_dim,log_std_min = -20, log_std_max = 2):
        super(Actor,self).__init__()
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.f1 = nn.Linear(state_dim,512)
        self.f2 = nn.Linear(512,512)
        self.f3 = nn.Linear(512,128)
        self.f4 = nn.Linear(128,128)
        self.mean = nn.Linear(128,act_dim)
        self.log_std = nn.Linear(128,act_dim)

    def forward(self,state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = F.relu(self.f4(x))
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(self.log_std_min,self.log_std_max)

        return mean,log_std

    def evaluate(self,state,epsilon = 1e-6):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        normal = Normal(0,1)
        z = normal.sample()
        action = torch.tanh(mean + std*z.cuda())
        log_prob = Normal(mean,std).log_prob(mean+ std*z.cuda()) - torch.log(1 - action.pow(2) + epsilon)
        log_prob = log_prob.sum(dim = -1, keepdim = True)
        return action, log_prob, z, mean, log_std

    '''
    def get_action(self,states):
        actions = torch.zeros(num_agent,act_dim)
        states = torch.from_numpy(states).cuda().float()
        for i in range(num_agent):
            state = states[i,:]
            mean,log_std = self.forward(state.unsqueeze(0))
            std = log_std.exp()      
            normal = Normal(0, 1)
            z      = normal.sample().to(device)
            action = torch.tanh(mean + std*z)
            action += torch.from_numpy(np.random.randn(act_dim) * self.var[i]).float().cuda()
            if self.var[i] > 0.05:
                self.var[i] *= 0.999992
            action  = action.detach().cpu().numpy()
            actions[i,:] = action[0]
        return actions

    def sample_action(self,state):
        action = np.zeros((num_agent,act_dim),dtype = float)
        for i in range(num_agent):
            action[i,:] = random.random()
        return action
    '''
    

class Critic(nn.Module):
    def __init__(self,state_dim):
        super(Critic,self).__init__()
        self.f1 = nn.Linear(state_dim,512)
        self.f2 = nn.Linear(512,128)
        self.f3 = nn.Linear(128,128)
        self.f4 = nn.Linear(128,1)
    
    def forward(self,x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = self.f4(x)

        return x

class SoftQNetwork(nn.Module):
    def __init__(self,state_dim, act_dim):
        super(SoftQNetwork,self).__init__()
        self.f1 = nn.Linear(state_dim + act_dim, 512)
        self.f2 = nn.Linear(512,128)
        self.f3 = nn.Linear(128,128)
        self.f4 = nn.Linear(128,1)

    def forward(self,state,action):
        x = torch.cat([state,action], -1)
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = self.f4(x)

        return x

class SAC():
    def __init__(self):
        super(SAC,self).__init__()
        self.actor = Actor(state_dim,act_dim).cuda()
        self.critic = Critic(state_dim).cuda()
        self.target_cirtic = Critic(state_dim).cuda()
        self.softq1 = SoftQNetwork(state_dim,act_dim).cuda()
        self.softq2 = SoftQNetwork(state_dim,act_dim).cuda()
        self.actor_optim = optim.Adam(self.actor.parameters(),LR)
        self.critic_optim = optim.Adam(self.critic.parameters(),LR)
        self.softq1_optim = optim.Adam(self.softq1.parameters(),LR)
        self.softq2_optim = optim.Adam(self.softq2.parameters(),LR)
        self.memory = replay_memory(memory_capacity)
        self.steps = 0
        self.random_step = random_step
        self.load = load
        self.var = [1.0 for i in range(num_agent)]

    def select_action(self,states):
        actions = np.zeros((num_agent,act_dim),dtype = float)
        states = torch.from_numpy(states).cuda().float()
        self.steps += 1
        if self.steps < self.random_step:
            for i in range(num_agent):
                actions[i,:] = random.random()*0.56
            return actions
        else:
            for i in range(num_agent):
                state = states[i,:]
                mean,log_std = self.actor.forward(state.unsqueeze(0))
                std = log_std.exp()      
                normal = Normal(0, 1)
                z      = normal.sample().cuda()
                action = torch.tanh(mean + std*z)
                action += torch.from_numpy(np.random.randn(act_dim) * self.var[i]).float().cuda()
                if self.var[i] > 0.05:
                    self.var[i] *= 0.999992
                action  = action.detach().cpu().numpy()
                actions[i,:] = action[0]
            return actions
        
        
    def update(self):
        if self.load == True:
            return
        if self.steps < self.random_step:
            return
        state,action,next_state,reward,done = self.memory.sample(batch_size)
        reward = reward*10
        predict_q1 = self.softq1(state,action)
        predict_q2 = self.softq2(state,action)
        predict_v = self.critic(state)
        new_action,log_prob,_,mean,log_std = self.actor.evaluate(state)

        #Update Q Function
        target_value = self.target_cirtic(next_state)
        target_q_value = reward + (1 -done) * GAMMA*target_value
        q_value_loss1 = F.mse_loss(predict_q1,target_q_value.detach())
        q_value_loss2 = F.mse_loss(predict_q2,target_q_value.detach())

        self.softq1_optim.zero_grad()
        q_value_loss1.backward()
        self.softq1_optim.step()

        self.softq2_optim.zero_grad()
        q_value_loss2.backward()
        self.softq2_optim.step()

        #Update value function
        predict_new_q_value = torch.min(self.softq1(state,new_action),self.softq2(state,new_action))
        target_value_func = predict_new_q_value - log_prob
        value_loss = F.mse_loss(predict_v,target_value_func.detach())
        
        self.critic_optim.zero_grad()
        value_loss.backward()
        self.critic_optim.step()

        #Update Policy
        policy_loss = (log_prob - predict_new_q_value).mean()

        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()
        soft_update(self.target_cirtic,self.critic,TAU)
    def step(self,state,action,next_state,reward,done):
        if self.load == True:
            return
        for i in range(num_agent):
            self.memory.add(state[i],action[i],next_state[i],reward[i],done[i])
        self.update()




    