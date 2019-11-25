env_name = "env/Crawler/Crawler_1agents_2points_s"
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
from DDPG import DDPG




#Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
env = UnityEnvironment(file_name= env_name, worker_id=1, seed =1)
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env_info = env.reset(train_mode = True)[default_brain]
max_step = 1000
#maddpg = MADDPG()
ddpg = DDPG()
rewards = []

for eps in range(1):
    env_info = env.reset(train_mode = True)[default_brain]
    done = False
    eps_reward = 0
    state = env_info.vector_observations
    #state = torch.from_numpy(state).float()
    score = 0
    #running_reward = -1000

    while True:
        action = ddpg.select_action(state)
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        ddpg.step(state,action,next_state,reward,done)
        state = next_state
        score += reward[0]
        rewards.append(score)
        if ddpg.steps  == 200000:
            torch.save(ddpg.actor.state_dict(),'actor_1118_200k.pkl')
            torch.save(ddpg.critic.state_dict(),'critic_1118_200k.pkl')
        if ddpg.steps  == 500000:
            torch.save(ddpg.actor.state_dict(),'actor_1118_500k.pkl')
            torch.save(ddpg.critic.state_dict(),'critic_1118_500k.pkl')
        if ddpg.steps == 1000000:
            torch.save(ddpg.actor.state_dict(),'actor_1118_1m.pkl')
            torch.save(ddpg.critic.state_dict(),'critic_1118_1m.pkl')    
        if ddpg.steps % 3000000 == 0:
            with open('data_1118_3m',mode = 'w') as f:
                for x in rewards:
                    f.write(str(x))
                    f.write('\n')
                rewards.clear()
                f.close()
        print ('step:',ddpg.steps)
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
