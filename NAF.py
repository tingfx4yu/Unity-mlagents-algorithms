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
GAMMA = 0.99
#update_iteration = 100
batch_size = 128
memory_capacity = 800000
state_dim = 126
act_dim = 20

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

class Policy(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Policy,self).__init__()
        #self.b0 = nn.BatchNorm1d(state_dim)
        self.f1 = nn.Linear(state_dim,1024)
        self.b1 = nn.BatchNorm1d(1024)
        self.f2 = nn.Linear(1024,512)
        self.b2 = nn.BatchNorm1d(1024)
        self.f3 = nn.Linear(512,512)
        #self.b3 = nn.BatchNorm1d(512)

        self.V = nn.Linear(512,1)
        self.mu = nn.Linear(512,act_dim)
        self.L = nn.Linear(512,act_dim**2)

        self.max_mu = 0.56

        self.tril_mask = torch.tril(torch.ones(act_dim,act_dim),diagonal = -1).unsqueeze(0).cuda()
        self.diag_mask = torch.diag(torch.diag(torch.ones(act_dim,act_dim)).unsqueeze(0)).cuda()
    def forward(self,inputs,a = None):
        x = inputs
        #x = self.b0(x)
        x = F.relu(self.f1(x))
        #x = self.b1(x)
        x = F.relu(self.f2(x))
        #x = self.b2(x)
        x = F.relu(self.f3(x))
        #x = self.b3(x)
        V = self.V(x)
        mu = F.tanh(self.mu(x))
        Q = None

        if a is not None:
            L = self.L(x).view(-1,act_dim,act_dim).cuda()
            L = L*self.tril_mask.expand_as(L) + torch.exp(L) * self.diag_mask.expand_as(L)
            P = torch.bmm(L,L.transpose(2,1))

            u_mu = (a - mu).unsqueeze(2)
            A = -0.5 * torch.bmm(torch.bmm(u_mu.transpose(2,1),P),u_mu)[:,:,0]

            Q = A+V

        return mu*self.max_mu, Q, V

class NAF():
    def __init__(self):
        super(NAF,self).__init__()
        self.model = Policy(state_dim,act_dim).cuda()
        self.target_model = Policy(state_dim,act_dim).cuda()
        self.optim = optim.Adam(self.model.parameters(),a_LR)
        self.memory = replay_memory(memory_capacity)
        self.var = 1
        self.steps = 0
        self.random_step = 4396

    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).cuda()
        if self.steps < self.random_step:
            self.model.eval()
        with torch.no_grad():     
            act,_,_ = self.model(state)
        act += torch.from_numpy(np.random.randn(act_dim) * self.var).float().cuda()
        if self.var > 0.05:
            self.var *= 0.999994
        #act = torch.clamp(act,-2,2)
        self.steps += 1
        return act.cpu().data.numpy()

    def update(self):
        if self.steps < self.random_step:
            return
        #for i in range(update_iteration):
        state,action,next_state,reward,done = self.memory.sample(batch_size)
        with torch.no_grad():
            _,_,Q_target_ = self.target_model(next_state)
            Q_target = reward + GAMMA * (1-done)* Q_target_
        
        _,current_Q,_ = self.model(state,action)
        loss = F.mse_loss(current_Q,Q_target)
        self.optim.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(),1)
        self.optim.step()

        #update targer network
        soft_update(self.target_model,self.model,TAU)
        
    def step(self,state,action,next_state,reward,done):
        self.memory.add(state,action,next_state,reward,done)
        self.update()






        

