#env_name = "env/Crawler/Crawler_6agents_4points_o"
#env_name = "env/Crawler/Crawler_r60agents_4points"
env_name = "env/Crawler/Crawler_8agents_4points"
train_model = True
import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
import copy
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import namedtuple,deque
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.distributions import Normal, Categorical
from mlagents.envs.environment import UnityEnvironment
#from DDPG import MADDPG
from TD3_mlagents_LSTM import TD3


id = random.randint(1,10)
#Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
env = UnityEnvironment(file_name= env_name, worker_id=0, seed =1)
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env_info = env.reset(train_mode = True)[default_brain]
max_step = 1000
num_agents = 8
act_dim = 20
#maddpg = MADDPG()
td3 = TD3()
rewards = []
save_mode = True

for eps in range(1):
    env_info = env.reset(train_mode = True)[default_brain]
    done = False
    eps_reward = 0
    state = env_info.vector_observations
    #state = torch.from_numpy(state).float()
    score = 0
    #running_reward = -1000
    hidden_out = (torch.zeros(num_agents,1,1,256).cuda().float(),torch.zeros(num_agents,1,1,256).cuda().float() )
    last_action = np.zeros((num_agents,act_dim),dtype = float)
    while True:
        hidden_in = hidden_out
        action,hidden_out = td3.select_action(state,last_action,hidden_in)
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        #hidden_in = hidden_in.cpu().numpy()
        #hidden_out = hidden_out.cpu().numpy()
        td3.step(hidden_in,hidden_out,state,action,last_action,next_state,reward,done)
        state = next_state
        last_action = action
        score = np.mean(reward)
        rewards.append(score)
        '''
        if td3.steps  == 200000:
            torch.save(td3.actor.state_dict(),'actor_1127_200k.pkl')
            torch.save(td3.critic.state_dict(),'critic_1127_200k.pkl')
        '''
        if save_mode and not td3.load:
            if td3.steps == 250000:
                torch.save(td3.actor.state_dict(),'td3lstm_actor_1220_250k.pkl')
                #torch.save(td3.critic.state_dict(),'td3_critic_1212_180k.pkl')
        
        if td3.steps == 1000000:
            torch.save(td3.actor.state_dict(),'td3lstm_actor_1220_1m.pkl')
            #torch.save(td3.critic.state_dict(),'critic_1118_1m.pkl')
         
        if td3.steps % 1000000 == 0 and not td3.load:
            with open('1215_TD3_1m',mode = 'w') as f:
                for x in rewards:
                    f.write(str(x))
                    f.write('\n')
                rewards.clear()
        print ('step:',td3.steps)
        print("Score:",score)
    '''
    while True:
        action,action_log_prob = agent.select_action(state)
        env_info = env.step(action)[default_brain]
        next_state, reward,done = env_info.vector_observations,env_info.rewards,env_info.local_done
        trans = Transition(state,action,reward,action_log_prob,next_state)
        if agent.store_transition(trans):
            agent.update()
        score += reward[0]
        state = next_state
        print('Step',agent.step)
        print(score)
    '''
env.close()
