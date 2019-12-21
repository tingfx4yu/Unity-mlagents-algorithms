import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
#from torch.distributions import Noraml
from collections import namedtuple, deque


TAU = 0.001 #1128 0.001 -- 0.005
a_LR = 3e-4 #1128 3e-4 -- 1e-3
c_LR = 3e-4
GAMMA = 0.978
#update_iteration = 100
batch_size = 256
memory_capacity = 8000000
state_dim = 126
act_dim = 20
seed = 3
num_agents = 8
load = False
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
        self.f3 = nn.Linear(512,200)
        self.f4 = nn.Linear(200,act_dim)
        self.max_action = 0.50

    def forward(self,x):
        x = F.relu(self.f1(x))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        x = torch.tanh(self.f4(x))
        return x*self.max_action

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.f1 = nn.Linear(act_dim+state_dim,512)
        self.f2 = nn.Linear(512,256)
        self.f3 = nn.Linear(256,256)
        self.f4 = nn.Linear(256,1)

        self.l1 = nn.Linear(act_dim+state_dim,512)
        self.l2 = nn.Linear(512,256)
        self.l3 = nn.Linear(256,256)
        self.l4 = nn.Linear(256,1)
    def forward(self,x,u):
        sa = torch.cat([x,u],1)
        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = F.relu(self.f3(q1))
        q1 = self.f4(q1)

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = F.relu(self.l3(q2))
        q2 = self.f4(q2)


        return q1,q2

    def Q1(self,state,action):
        sa = torch.cat([state,action],1)

        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = F.relu(self.f3(q1))
        q1 = self.f4(q1)

        return q1

class TD3():
    def __init__(self):
        super(TD3,self).__init__()
        self.actor = Actor(state_dim,act_dim).cuda()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim,act_dim).cuda()
        self.critic_target = copy.deepcopy(self.critic)
        self.actor_optim = optim.Adam(self.actor.parameters(),a_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(),c_LR)
        self.memory = replay_memory(memory_capacity)
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.var = [1.0 for i in range(num_agents)]
        self.steps = 0
        self.total = 0
        self.max_action = 0.5
        self.random_step = 300
        self.load = load

    def select_action(self,states):
        actions = torch.zeros(num_agents, act_dim)
        states = torch.from_numpy(states).cuda().float()
        for i in range(num_agents):
            #state = torch.FloatTensor(state[i,:]).cuda()
            if self.load == True:
                self.actor.load_state_dict(torch.load('td3_actor_1212_180k.pkl'))
                #self.actor.load_state_dict(torch.load('td3_actor_1215_250k.pkl'))
                state = states[i,:]
                act = self.actor(state)
                actions[i,:] = act
                #return act.cpu().data.numpy()
            else:
                with torch.no_grad():
                    state = states[i,:]
                    act = self.actor(state.unsqueeze(0)).squeeze()
                    act += torch.from_numpy(np.random.randn(act_dim) * self.var[i]).float().cuda()
                    if self.var[i] > 0.05:
                        self.var[i] *= 0.999992
                    #act = torch.clamp(act,-2,2)
                    actions[i,:] = act
        self.steps += 1
        actions = actions.cpu().data.numpy()
        return actions

    def update(self):
        if self.load == True:
            return
        if self.steps < self.random_step:
            return
        #for i in range(update_iteration):
        state,action,next_state,reward,done = self.memory.sample(batch_size)
        self.total += 1
        with torch.no_grad():
            #compute target Q value
            noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip)
            next_action = (self.actor_target(next_state) + noise).clamp(-self.max_action,self.max_action)
            target_Q1, target_Q2 = self.critic_target(next_state,next_action)
            target_Q = torch.min(target_Q1,target_Q2)*0.95 + torch.max(target_Q1,target_Q2)*0.05
            target_Q = reward + ((1-done)*GAMMA*target_Q)

        #get current Q
        current_Q1, current_Q2 = self.critic(state,action)

        #computer cirtic loss
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
        #optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        if self.total % self.policy_freq == 0:      
            #compute actor loss
            actor_loss = - self.critic.Q1(state,self.actor(state)).mean()
            #optimize the actor loss
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            #update targer network
            soft_update(self.critic_target,self.critic,TAU)
            soft_update(self.actor_target,self.actor,TAU)
        
    def step(self,state,action,next_state,reward,done):
        if self.load == True:
            return
        for i in range(num_agents):
            self.memory.add(state[i],action[i],next_state[i],reward[i],done[i])
        self.update()






        

