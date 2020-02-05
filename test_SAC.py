#env_name = "env/Crawler/Push_5agents"
#env_name = "env/Crawler/Crawler_5agents_4points_o_ft"
#env_name = "env/Crawler/Crawler_36agents_2points_fs"
#env_name = "env/Crawler/Crawler_30agents_2points_o_ft"
#env_name = "env/Crawler/Crawler_5agents_co_navi"
#env_name = "env/Crawler/246input_5agents"
#env_name = "env/Crawler/53input_3output_5agents_foodcollect"
env_name = "env/Crawler/140input_2output_5agents_push"
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
from mlagents.envs.environment import UnityEnvironment
#from DDPG import MADDPG
from SAC import SAC



#Transition = namedtuple('Transition',['state', 'action', 'reward', 'a_log_prob', 'next_state'])
env = UnityEnvironment(file_name= env_name, worker_id=random.randint(0,10), seed =1)

default_brain = env.brain_names[0]
brain = env.brains[default_brain]
env_info = env.reset(train_mode = True)[default_brain]
max_step = 1000
#maddpg = MADDPG()
sac = SAC()
#env = NormalizedAction(env)
rewards = []
steps = 0
def f_mean(x):
    a = 0
    for i in range(len(x)):
        a += i
    return a/len(x)



for eps in range(1):
    env_info = env.reset(train_mode = True)[default_brain]
    done = False
    eps_reward = 0
    state = env_info.vector_observations
    #state = torch.from_numpy(state).float()
    score = 0
    #running_reward = -1000

    while True:
        #steps +=1
        action = sac.select_action(state)
        env_info = env.step(action)[default_brain]
        next_state = env_info.vector_observations
        reward = env_info.rewards
        done = env_info.local_done
        sac.step(state,action,next_state,reward,done)
        state = next_state
        score += max(reward)
        rewards.append(score)

        '''
        if sac.steps  == 200000:
            torch.save(sac.actor.state_dict(),'actor_1126_200k.pkl')
            torch.save(sac.critic.state_dict(),'critic_1136_200k.pkl')
        '''
        if sac.steps  == 250000:
            torch.save(sac.actor.state_dict(),'actor_250k_0205.pkl')
            #torch.save(sac.critic.state_dict(),'critic_500k_0201.pkl')
            #torch.save(sac.softq1.state_dict(),'softq1_500k_0201.pkl')
            #torch.save(sac.softq2.state_dict(),'softq2_500k_0201.pkl')
            #torch.save(sac.critic.state_dict(),'critic_1126_500k.pkl')
        if sac.steps == 700000:
            torch.save(sac.actor.state_dict(),'actor_700k_0205.pkl')
            #torch.save(sac.critic.state_dict(),'critic_1126_1m.pkl')
            with open('SAC_0205_700k',mode = 'w') as f:
                for x in rewards:
                    f.write(str(x))
                    f.write('\n')
                rewards.clear()
                f.close()
        if sac.steps == 1000000:
            torch.save(sac.actor.state_dict(),'actor_1m_0201.pkl')
        print ('step:',sac.steps)
        print("Score:",score)
env.close()
