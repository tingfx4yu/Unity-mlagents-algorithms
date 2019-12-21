import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
from torch.distributions import Normal
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
hidden_layer = 256
random_step = 300
delay = 0.999992
torch.cuda.manual_seed(seed)

def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)

class replay_memory:
    def __init__(self,capacity):
        self.memory = deque(maxlen = capacity)
        self.Experience = namedtuple('Experience',['h_in','c_in','h_out','c_out','state','action','last_action','next_state','reward','done'])
    
    def add(self,h_in,c_in,h_out,c_out,state,action,last_action,next_state,reward,done):
        e = self.Experience(h_in,c_in,h_out,c_out,state,action,last_action,next_state,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        experience = random.sample(self.memory,batch_size)
        #np.concatenate
        h_in_c = np.concatenate([e.h_in for e in experience if e is not None],-2)
        h_in = torch.from_numpy(h_in_c).float().cuda().detach()
        h_out = torch.from_numpy(np.concatenate([e.h_out for e in experience if e is not None],-2)).float().cuda().detach()
        c_in = torch.from_numpy(np.concatenate([e.c_in for e in experience if e is not None],-2)).float().cuda().detach()
        c_out = torch.from_numpy(np.concatenate([e.c_out for e in experience if e is not None],-2)).float().cuda().detach()
        #h_in = torch.cat([e.h_in for e in experience if e is not None],-2).float().cuda()
        #c_in = torch.cat([e.c_in for e in experience if e is not None],-2).float().cuda()
        #h_out = torch.cat([e.h_out for e in experience if e is not None],-2).float().cuda()
        #c_out = torch.cat([e.c_out for e in experience if e is not None],-2).float().cuda()

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        last_actions = torch.from_numpy(np.vstack([e.last_action for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().cuda().view(batch_size,-1)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().cuda()


        hidden_in = (h_in,c_in)
        hidden_out = (h_out,c_out)


        return hidden_in,hidden_out,states,actions,last_actions,next_states,rewards,dones
    def __len__(self):
        return len(self.memory)

class Actor(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Actor,self).__init__()
        self.f1 = nn.Linear(state_dim, hidden_layer)
        self.f2 = nn.Linear(state_dim + act_dim,hidden_layer)
        self.lstm1 = nn.LSTM(hidden_layer,hidden_layer)
        self.f3 = nn.Linear(hidden_layer*2,hidden_layer)
        self.f4 = nn.Linear(hidden_layer,act_dim)
        self.max_action = 0.50

    def forward(self,state,last_action,hidden_in):
        state = state.permute(1,0,2)
        last_action = last_action.permute(1,0,2)
        fc_branch = F.selu(self.f1(state))
        fc_branch = fc_branch.permute(1,0,2)
        lstm_branch = torch.cat([state,last_action],-1)
        lstm_branch = F.selu(self.f2(lstm_branch))
        lstm_branch = lstm_branch.permute(1,0,2)
        self.lstm1.flatten_parameters()

        lstm_branch, lstm_hidden = self.lstm1(lstm_branch,hidden_in)

        merged_branch = torch.cat([fc_branch,lstm_branch],-1)
        x = F.selu(self.f3(merged_branch))
        x = torch.tanh(self.f4(x)).clone() * self.max_action
        x = x.permute(1,0,2)

        return x, lstm_hidden

    def evaluate(self,state,last_action,hidden_in,noise_scale = 0.0):
        #normal = Normal(0,1)
        action, hidden_out = self.forward(state,last_action,hidden_in)
        #noise = noise_scale * normal.sample()
        action = self.max_action * action
        return action, hidden_out

        

class Critic(nn.Module):
    def __init__(self,state_dim,act_dim):
        super(Critic,self).__init__()

        self.f1 = nn.Linear(act_dim + state_dim,hidden_layer)
        self.f2 = nn.Linear(state_dim + act_dim,hidden_layer)
        self.lstm1 = nn.LSTM(hidden_layer,hidden_layer)
        self.f3 = nn.Linear(hidden_layer*2,hidden_layer)
        self.f4 = nn.Linear(hidden_layer,1)

        self.l1 = nn.Linear(act_dim + state_dim,hidden_layer)
        self.l2 = nn.Linear(state_dim + act_dim,hidden_layer)
        self.lstm2 = nn.LSTM(hidden_layer,hidden_layer)
        self.l3 = nn.Linear(hidden_layer*2,hidden_layer)
        self.l4 = nn.Linear(hidden_layer,1)

    def forward(self,state,action,last_action,hidden_in):
        state = state.permute(1,0,2)
        #action = action.permute(1,0,2)
        last_action = last_action.permute(1,0,2)

        fc_branch1 = torch.cat([state,action],-1)
        fc_branch1 = F.selu(self.f1(fc_branch1)).permute(1,0,2)
        

        lstm_branch1 = torch.cat([state,last_action],-1)
        lstm_branch1 = F.selu(self.f2(lstm_branch1))
        lstm_branch1 = lstm_branch1.permute(1,0,2)
        self.lstm1.flatten_parameters()

        lstm_branch1,lstm_hidden1 = self.lstm1(lstm_branch1,hidden_in)

        merged_branch = torch.cat([fc_branch1,lstm_branch1],-1)

        Q1 = F.selu(self.f3(merged_branch))
        Q1 = self.f4(Q1)
        Q1 = Q1.permute(1,0,2)

        fc_branch2 = torch.cat([state,action],-1)
        fc_branch2 = F.relu(self.f1(fc_branch2)).permute(1,0,2)

        lstm_branch2 = torch.cat([state,last_action],-1)
        lstm_branch2 = F.relu(self.f2(lstm_branch2))
        lstm_branch2 = lstm_branch2.permute(1,0,2)
        self.lstm2.flatten_parameters()

        lstm_branch2,lstm_hidden2 = self.lstm2(lstm_branch2,hidden_in)

        merged_branch2 = torch.cat([fc_branch2,lstm_branch2],-1)

        Q2 = F.relu(self.f3(merged_branch2))
        Q2 = self.f4(Q2)
        Q2 = Q2.permute(1,0,2)

        return Q1,Q2, lstm_hidden1,lstm_hidden2
        '''
        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = F.relu(self.f3(q1))
        q1 = self.f4(q1)

        q2 = F.relu(self.l1(sa))
        q2 = F.relu(self.l2(q2))
        q2 = F.relu(self.l3(q2))
        q2 = self.f4(q2)


        return q1,q2
        '''
    '''
    def Q1(self,state,action):
        sa = torch.cat([state,action],1)

        q1 = F.relu(self.f1(sa))
        q1 = F.relu(self.f2(q1))
        q1 = F.relu(self.f3(q1))
        q1 = self.f4(q1)

        return q1
    ''' 
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
        #self.max_action = 0.5
        self.random_step = random_step
        self.load = load

    def select_action(self,states,last_action,hidden_in):
        actions = np.zeros((num_agents,act_dim),dtype = 'float')
        last_action = torch.from_numpy(last_action).cuda().float()
        hidden_out_1 = torch.zeros(num_agents,1,1,hidden_layer).cuda().float()
        hidden_out_2 = torch.zeros(num_agents,1,1,hidden_layer).cuda().float()
        hidden_in_1 = hidden_in[0]
        hidden_in_2 = hidden_in[1]
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
                    state = states[i,:].unsqueeze(0).unsqueeze(0)
                    last_act = last_action[i,:].unsqueeze(0).unsqueeze(0)
                    hidden_in_11 = hidden_in_1[i,:]
                    hidden_in_22 = hidden_in_2[i,:]
                    hidden_in = (hidden_in_11,hidden_in_22)
                    #act = self.actor(state.unsqueeze(0)).squeeze()
                    act, hid_out = self.actor.forward(state,last_act,hidden_in)
                    act = act.cpu().data.numpy()
                    #act += torch.from_numpy(np.random.randn(act_dim) * self.var[i])
                    act += np.random.randn(act_dim) * self.var[i]
                    if self.var[i] > 0.05:
                        self.var[i] *= delay
                    #act = torch.clamp(act,-2,2)
                    actions[i,:] = act
                    hidden_out_1[i,:] = hid_out[0]
                    hidden_out_2[i,:] = hid_out[1]
        self.steps += 1
        hidden_out = (hidden_out_1,hidden_out_2)
        #actions = actions.cpu().data.numpy()
        return actions, hidden_out

    def update(self):
        if self.load == True:
            return
        if self.steps < self.random_step:
            return
        #for i in range(update_iteration):
        hidden_in,hidden_out,state,action,last_action,next_state,reward,done = self.memory.sample(batch_size)
        self.total += 1
        state = state.unsqueeze(0)
        action = action.unsqueeze(0).permute(1,0,2)
        next_state = next_state.unsqueeze(0)
        last_action = last_action.unsqueeze(0)
        with torch.no_grad():
            #compute target Q value
            #noise = (torch.randn_like(action)*self.policy_noise).clamp(-self.noise_clip,self.noise_clip)
            new_action, _ = self.actor.evaluate(state,last_action,hidden_in)
            #new_action = new_action.squeeze(1)

            next_action,_ = self.actor_target.evaluate(next_state,last_action,hidden_in)
            #next_action = next_action.squeeze(1)

            target_Q1, target_Q2,_,_ = self.critic_target(next_state,next_action,last_action,hidden_in)
            target_Q = torch.min(target_Q1,target_Q2)*0.95 + torch.max(target_Q1,target_Q2)*0.05
            target_Q = reward.unsqueeze(1) + ((1-done.unsqueeze(1))*GAMMA*target_Q)

        #get current Q

        current_Q1, current_Q2,_,_ = self.critic(state,action,last_action,hidden_in)

        #computer cirtic loss
        critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
        #optimize the critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        if self.total % self.policy_freq == 0:      
            #compute actor loss
            actor_loss,_,_,_ =  self.critic(state,new_action,last_action,hidden_in)
            actor_loss = -1 * actor_loss.mean()
            #optimize the actor loss
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            #update targer network
            soft_update(self.critic_target,self.critic,TAU)
            soft_update(self.actor_target,self.actor,TAU)
        
    def step(self,hidden_in,hidden_out,state,action,last_action,next_state,reward,done):
        if self.load == True:
            return
        hidden_in_1 = hidden_in[0]
        hidden_in_2 = hidden_in[1]
        hidden_out_1 = hidden_out[0]
        hidden_out_2 = hidden_out[1]

        for i in range(num_agents):
            h_in = hidden_in_1[i].cpu().numpy()
            c_in = hidden_in_2[i].cpu().numpy()
            #hidden_in = (hidden_in1,hidden_in2)

            h_out = hidden_out_1[i].cpu().numpy()
            c_out = hidden_out_2[i].cpu().numpy()
            #hidden_out = (hidden_in1,hidden_in2)
            self.memory.add(h_in,c_in,h_out,c_out,state[i],action[i],last_action[i],next_state[i],reward[i],done[i])
        self.update()






        

