env_name = "env/Crawler/Crawler_2agents_2points"
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
from MADDPG import MADDPG




#Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
env = UnityEnvironment(file_name= env_name, worker_id=1, seed =1)
default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env_info = env.reset(train_mode = True)[default_brain]
#max_step = 1000
#maddpg = MADDPG()
maddpg = MADDPG()
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
        action = maddpg.select_action(state)
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        maddpg.step(state,action,next_state,reward,done)
        state = next_state
        score += reward[0]
        rewards.append(score)
        if maddpg.steps % 1000000 == 0:
            with open('data_11_14',mode = 'w') as f:
                for x in rewards:
                    f.write(str(x))
                    f.write('\n')
                rewards.clear()
                f.close()
        print ('step:',maddpg.steps)
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
