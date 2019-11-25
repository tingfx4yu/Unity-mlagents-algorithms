import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import namedtuple, deque
import random
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from torch.optim import Adam


#-------hyperParameter-----------

num_agents = 2
batch_size = 256
dim_obs = 126
dim_act = 20
capacity = 2500000
explore_step = 4396
GAMMA = 0.99
tau = 0.001
scale_reward = 0.1
use_cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor

#----------Basic Function-----------

# -----------Net Structure---------------

class Critic(nn.Module):
    def __init__(self,num_agents,dim_o,dim_a):
        super(Critic,self).__init__()
        self.num_agents = num_agents
        self.dim_o = dim_o
        self.dim_a = dim_a
        obs_dim = dim_o * num_agents
        act_dim = dim_a * num_agents

        self.fc1 = nn.Linear(obs_dim,1024+obs_dim)
        self.fc2 = nn.Linear(1024+act_dim+obs_dim,1024)
        self.fc3 = nn.Linear(1024,300)
        self.fc4 = nn.Linear(300,1)
    def forward(self,input,acts):
        output = F.relu(self.fc1(input))
        output = torch.cat([output,acts],1)
        output = F.relu(self.fc2(output))
        output = F.relu(self.fc3(output))
        output = self.fc4(output)
        return output

class Actor(nn.Module):
    def __init__(self,dim_o,dim_a):
        super(Actor,self).__init__()
        self.fc1 = nn.Linear(dim_o,1024)
        self.fc2 = nn.Linear(1024,512)
        self.fc3 = nn.Linear(512,512)
        self.fc4 = nn.Linear(512,dim_a)
        self.max_a = 0.56

    def forward(self,obs):
        result = F.relu(self.fc1(obs))
        result = F.relu(self.fc2(result))
        result = F.relu(self.fc3(result))
        result = torch.tanh(self.fc4(result))
        return result*self.max_a


#--------------------Buffer------------------------

class replay_memory:
    def __init__(self,capacity):
        self.memory = deque(maxlen = capacity)
        self.Experience = namedtuple('Experience',['state','action','next_state','reward','done'])
    
    def add(self,state,action,reward,next_state,done):
        e = self.Experience(state,action,next_state,reward,done)
        self.memory.append(e)
    def sample(self,batch_size):
        experience = random.sample(self.memory,batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experience if e is not None])).float().cuda().view(batch_size,num_agents,-1)
        actions = torch.from_numpy(np.vstack([e.action for e in experience if e is not None])).float().cuda().view(batch_size,num_agents,-1)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experience if e is not None])).float().cuda().view(batch_size,num_agents,-1)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experience if e is not None])).float().cuda().view(batch_size,num_agents,-1)
        dones = torch.from_numpy(np.vstack([e.done for e in experience if e is not None]).astype(np.uint8)).float().cuda()

        return states,actions,rewards,next_states,dones
    def __len__(self):
        return len(self.memory)
        

#--------------------MADDPG------------------------

class MADDPG():
    def __init__(self):
        self.actors = [Actor(dim_obs,dim_act) for i in range(num_agents)]
        self.critics = [Critic(num_agents,dim_obs,dim_act) for i in range(num_agents)]
        self.actors_target = deepcopy(self.actors)
        self.critics_target = deepcopy(self.critics)
        self.steps = 0
        self.memory = replay_memory(capacity)
        self.var = [1 for i in range(num_agents)]
        #self.random_number = [random.uniform(-0.5,0.5) for i in range(num_agents)]
        self.critic_optimizer = [Adam(x.parameters(),lr=1e-3) for x in self.critics]
        self.actor_optimizer = [Adam(x.parameters(),lr=1e-4) for x in self.actors]

        if torch.cuda.is_available():
            for x in self.actors:
                x.cuda()
            for x in self.critics:
                x.cuda()
            for x in self.actors_target:
                x.cuda()
            for x in self.critics_target:
                x.cuda()

    def learn(self):
        if self.steps <= explore_step:
            return
        ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        
        a_loss = []
        c_loss = []
        for agent in range(num_agents):
            states, actions, rewards,next_states,dones = self.memory.sample(batch_size)
            #batch = self.memory.Experience(*zip(*transitions))
            non_final_mask = ByteTensor(list(map(lambda s: s is not None, next_states)))
            '''
            state_batch = torch.stack(batch.state).type(FloatTensor)
            action_batch = torch.stack(batch.action).type(FloatTensor)
            reward_batch = torch.stack(batch.reward).type(FloatTensor)
            '''
            non_final_next_states =torch.stack([s for s in next_states if s is not None])

            whole_state = states.view(batch_size,-1)
            whole_action = actions.view(batch_size,-1)
            self.critic_optimizer[agent].zero_grad()
            current_Q = self.critics[agent](whole_state,whole_action)
            non_final_next_actions = [self.actors_target[i](non_final_next_states[:,i,:]) for i in range(num_agents)]
            non_final_next_actions = torch.stack(non_final_next_actions)
            non_final_next_actions = (non_final_next_actions.transpose(0,1).contiguous())

            target_Q = torch.zeros(batch_size).type(FloatTensor)

            target_Q[non_final_mask] = self.critics_target[agent](
                non_final_next_states.view(-1, num_agents * dim_obs),
                non_final_next_actions.view(-1,
                                            num_agents * dim_act)
            ).squeeze()
            # scale_reward: to scale reward in Q functions
            reward_1 = rewards[:,agent]

            target_Q = (target_Q.unsqueeze(1) * GAMMA) + (
                rewards[:, agent] * scale_reward)

            loss_Q = nn.MSELoss()(current_Q, target_Q.detach())
            loss_Q.backward()
            self.critic_optimizer[agent].step()

            self.actor_optimizer[agent].zero_grad()
            state_i = states[:, agent, :]
            action_i = self.actors[agent](state_i)
            ac = actions.clone()
            ac[:, agent, :] = action_i
            whole_action = ac.view(batch_size, -1)
            actor_loss = -self.critics[agent](whole_state, whole_action)
            actor_loss = actor_loss.mean()
            actor_loss.backward()
            self.actor_optimizer[agent].step()
            c_loss.append(loss_Q)
            a_loss.append(actor_loss)      
            '''
            target_Q = self.critics_target[agent](next_states[:,agent,:],self.actors_target[agent](next_states[:,agent,:]))
            target_Q = reward[:,agent,:] + (1 - done[:,agent]*GAMMA*target_Q).detach()

            current_Q = self.critics[agent]()
            '''
            for i in range(num_agents):
                soft_update(self.critics_target[i],self.critics[i],tau)
                soft_update(self.actors_target[i],self.actors[i],tau)
        #if self.steps % 50000 == 0:
            #torch.save(self.actors.state_dict(),'Critic.pkl')
            #torch.save(self.critics.state_dict(),'Actor.pkl')


    def select_action(self, state_batch):
        # state_batch: n_agents x state_dim
        actions = torch.zeros(num_agents, dim_act)
        FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
        state_batch = torch.from_numpy(state_batch).cuda().float()
        for i in range(num_agents):
            sb = state_batch[i, :].detach().cuda()
            act = self.actors[i](sb.unsqueeze(0)).squeeze()

            act += torch.from_numpy(
                np.random.randn(dim_act) * self.var[i]).type(FloatTensor)

            if (self.steps > explore_step) and (self.var[i] > 0.05):
                self.var[i] *= 0.99999
            #act = torch.clamp(act, -3, 3)
            actions[i, :] = act
        self.steps += 1
        actions = actions.cpu().detach().numpy()

        return actions


    def step(self,state,action,next_state,reward,done):
        self.memory.add(state,action,reward,next_state,done)
        self.learn()


def soft_update(target, source, t):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(
            (1 - t) * target_param.data + t * source_param.data)


def hard_update(target, source):
    for target_param, source_param in zip(target.parameters(),
                                          source.parameters()):
        target_param.data.copy_(source_param.data)






        





    

