import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
#from torch.distributions import Noraml
from collections import namedtuple, deque


TAU = 0.001
a_LR = 3e-4
c_LR = 3e-4
GAMMA = 0.99
#update_iteration = 100
batch_size = 128
memory_capacity = 2800000
state_dim = 126
act_dim = 20
load = False

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
        self.f1 = nn.Linear(state_dim + act_dim, 1024)
        self.f2 = nn.Linear(1024,512)
        self.f3 = nn.Linear(512,act_dim)
        #self.f4 = nn.Linear(200,act_dim)
        self.max_action = 0.56

    def forward(self,state, action):
        a = F.relu(self.f1(torch.cat([state, action], 1)))
        a = F.relu(self.f2(a))
        a = 0.05 * self.max_action * torch.tanh(self.f3(a))
        return (a + action).clamp(-self.max_action, self.max_action)

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()
        self.f1 = nn.Linear(act_dim+state_dim,512)
        self.f2 = nn.Linear(512,256)
        #self.f3 = nn.Linear(256,128)
        self.f3 = nn.Linear(256,1)

        self.l1 = nn.Linear(act_dim+state_dim,512)
        self.l2 = nn.Linear(512,256)
        self.l3 = nn.Linear(256,1)
    def forward(self,x,u):
        sa = torch.cat([x,u],1)
        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = self.f3(q1)

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = self.l3(q2)


        return q1,q2

    def Q1(self,state,action):
        sa = torch.cat([state,action],1)

        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = self.f3(q1)

        return q1

class VAE(nn.Module):
    def __init__(self,state_dim,action_dim,latent_dim, max_action):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim , 750)
        self.e2 = nn.Linear(750,750)

        self.mean = nn.Linear(750,latent_dim)
        self.log_std = nn.Linear(750,latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750,750)
        self.d3 = nn.Linear(750,action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim

    def forward(self,state,action):
        z = F.relu(self.e1(torch.cat([state,action],1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        log_std = self.log_std(z).clamp(-4,15)
        std = torch.exp(log_std)

        z = mean + std* torch.FloatTensor(np.random.normal(0,1,size = (std.size()))).cuda()

        u = self.decode(state,z)
        return u,mean,std

    def decode(self,state,z = None):
        if z is None:
            z = torch.FloatTensor(np.random.normal(0,1,size = (state.size(0),self.latent_dim))).cuda().clamp(-0.56,0.56)
        a = F.relu(self.d1(torch.cat([state,z],1)))
        a = F.relu(self.d2(a))

        return self.max_action * torch.tanh(self.d3(a))
 


class BCQ():
    def __init__(self):
        super(BCQ,self).__init__()
        latent_dim = act_dim * 2
        max_action = 0.56
        self.actor = Actor(state_dim,act_dim).cuda()
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(state_dim,act_dim).cuda()
        self.critic_target = copy.deepcopy(self.critic)
        self.vae = VAE(state_dim,act_dim,latent_dim,max_action).cuda()
        self.vae_optimizer = optim.Adam(self.vae.parameters())
        self.actor_optim = optim.Adam(self.actor.parameters(),a_LR)
        self.critic_optim = optim.Adam(self.critic.parameters(),c_LR)
        self.memory = replay_memory(memory_capacity)
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_freq = 2
        self.var = 1
        self.steps = 0
        self.total = 0
        self.random_step = 4396

    def select_action(self,state):
        state = torch.FloatTensor(state.reshape(1,-1)).cuda()
        if load == True:
            self.actor.load_state_dict(torch.load('actor_1118_500k.pkl'))
            act = self.actor(state)
            return act.cpu().data.numpy()
        else:
            self.steps += 1
            with torch.no_grad():
                
                act = self.actor(state,self.vae.decode(state))
                '''
                q1 = self.critic.Q1(state,act)
                idx = q1.max(0)[1]
                ac = act[idx].cpu().data.numpy()
                return ac
                '''
                
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
        self.total += 1
        recon , mean, std = self.vae(state,action)
        recon_loss = F.mse_loss(recon,action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        vae_loss = recon_loss + 0.5 * KL_loss
        self.vae_optimizer.zero_grad()
        vae_loss.backward()
        self.vae_optimizer.step()

        with torch.no_grad():
        #compute target Q value
            target_Q1, target_Q2 = self.critic_target(next_state,self.actor_target(next_state,self.vae.decode(next_state)))
            target_Q = torch.min(target_Q1,target_Q2)*0.75 + torch.max(target_Q1,target_Q2)*0.25
            target_Q = reward + ((1-done)*GAMMA*target_Q)

        #get current Q
        current_Q1, current_Q2 = self.critic(state,action)

        #computer cirtic loss
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
        #optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()   
            #compute actor loss
        sampled_actions = self.vae.decode(state)
        perturbed_actions = self.actor(state,sampled_actions)
        actor_loss = - self.critic.Q1(state,perturbed_actions).mean()
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






        

