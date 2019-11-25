import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torch.distributions import Noraml
from collections import namedtuple, deque


TAU = 0.001
a_LR = 1e-4
c_LR = 1e-3
GAMMA = 0.95
#update_iteration = 100
batch_size = 128
memory_capacity = 2800000
state_dim = 126
act_dim = 20
load = False
seed = 13
torch.cuda.manual_seed(seed)


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
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.f1 = nn.Linear(state_dim, 1024)
        self.f2 = nn.Linear(1024,512)
        self.f3 = nn.Linear(512,act_dim)
        #self.f4 = nn.Linear(200,act_dim)
        self.max_action = 0.56

    def forward(self,x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        #x = F.relu(self.f3(x))
        x = torch.tanh(self.f3(x))
        return x*self.max_action

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.f1 = nn.Linear(act_dim+state_dim,512)
        self.f2 = nn.Linear(512,256)
        #self.f3 = nn.Linear(256,128)
        self.f4 = nn.Linear(256,1)
    def forward(self,x,u):
        x = F.relu(self.f1(torch.cat([x,u],1)))
        x = F.relu(self.f2(x))
        #x = F.relu(self.f3(x))
        x = self.f4(x)
        return x

class DDPG():
    def __init__(self):
        super(DDPG,self).__init__()
        self.actor = Actor(state_dim,act_dim).cuda()
        self.actor_target = Actor(state_dim,act_dim).cuda()
        self.critic = Critic(state_dim,act_dim).cuda()
        self.critic_target = Critic(state_dim,act_dim).cuda()
        self.actor_optim = optim.Adam(self.actor.parameters(),a_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(),c_LR)
        self.memory = replay_memory(memory_capacity)
        self.var = 1
        self.steps = 0
        self.random_step = 4396

    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).cuda()
        if load == True:
            self.actor.load_state_dict(torch.load('actor_1118_500k.pkl'))
            act = self.actor(state)
            return act.cpu().data.numpy()
        else:
            with torch.no_grad():
                act = self.actor(state)
                act += torch.from_numpy(np.random.randn(act_dim) * self.var).float().cuda()
                if self.var > 0.05:
                    self.var *= 0.99995
                #act = torch.clamp(act,-2,2)
                self.steps += 1
                return act.cpu().data.numpy()

    def update(self):
        if load == True:
            return
        if self.steps < self.random_step:
            return
        #for i in range(update_iteration):
        state,action,next_state,reward,done = self.memory.sample(batch_size)

        #compute target Q value
        target_Q = self.critic_target(next_state,self.actor_target(next_state))
        target_Q = reward + ((1-done)*GAMMA*target_Q).detach()

        #get current Q
        current_Q = self.critic(state,action)

        #computer cirtic loss
        critic_loss = F.mse_loss(current_Q,target_Q)
        #optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        #compute actor loss
        actor_loss = - self.critic(state,self.actor(state)).mean()
        #optimize the actor loss
        self.actor_optim.zero_grad()
        actor_loss.backward()
        self.actor_optim.step()

        #update targer network
        soft_update(self.critic_target,self.critic,TAU)
        soft_update(self.actor_target,self.actor,TAU)
        
    def step(self,state,action,next_state,reward,done):
        self.memory.add(state,action,next_state,reward,done)
        self.update()






        

